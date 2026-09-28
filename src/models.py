"""Whiplash equations (1)--(10), with explicit masks for unavailable priors.

Memory keys use the current shared encoder and training windows only. The runner
refreshes detached keys every epoch and before evaluation; no frozen auxiliary
encoder or handcrafted-pattern retrieval is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_impl import MambaStack


def _masked_softmax(
    scores: torch.Tensor, valid: torch.Tensor, dim: int
) -> torch.Tensor:
    """Normalize valid entries; an entirely unavailable row returns zeros."""
    weights = torch.softmax(
        scores.masked_fill(~valid, torch.finfo(scores.dtype).min), dim=dim
    )
    weights = weights * valid.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).tiny
    )


class EventTokenizer(nn.Module):
    """Shared event-token FFN applied independently to local states."""

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, h_local: torch.Tensor) -> torch.Tensor:
        return self.net(h_local)


class RecentQueryPool(nn.Module):
    """Equation (3): learned softmax pooling over the last T_q event tokens."""

    def __init__(self, d_model: int, query_len: int = 6):
        super().__init__()
        if query_len < 1:
            raise ValueError("query_len must be positive")
        self.query_len = query_len
        self.score = nn.Linear(d_model, 1)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        recent = events[:, -min(self.query_len, events.size(1)) :]
        return (torch.softmax(self.score(recent), dim=1) * recent).sum(dim=1)


class HistoricalPatternMemory(nn.Module):
    """Equation (4): same-link cosine top-k and future aggregation.

    Keys are [B,R,L,D] or shared [R,L,D]. Corresponding futures are
    [B,R,H,L,1] or [R,H,L,1], with the last dimension optional.
    True in [B,R] / [B,R,L] memory_mask means available. Shared futures
    are expanded as views, and only the selected top-k futures are gathered.
    memory_coarse_top_k/dropout are retained for constructor compatibility;
    there is a single cosine top-k, without a learned reranker.
    """

    def __init__(
        self,
        d_model: int,
        memory_top_k: int | None = 4,
        memory_coarse_top_k: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if memory_top_k is not None and memory_top_k < 1:
            raise ValueError("memory_top_k must be positive or None")
        self.memory_top_k = memory_top_k

    def forward(
        self,
        recent_query: torch.Tensor,
        memory_tokens: torch.Tensor,
        future_targets: torch.Tensor | None = None,
        memory_ages: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ):
        bsz, links, dim = recent_query.shape
        shared = memory_tokens.ndim == 3
        valid = torch.isfinite(memory_tokens).all(dim=-1)
        normalized_tokens = F.normalize(torch.nan_to_num(memory_tokens), dim=-1)
        if shared:
            memory_tokens = memory_tokens.unsqueeze(0).expand(bsz, -1, -1, -1)
            valid = valid.unsqueeze(0).expand(bsz, -1, -1)
        if (
            memory_tokens.ndim != 4
            or memory_tokens.shape[0] != bsz
            or memory_tokens.shape[2:] != (links, dim)
        ):
            raise ValueError("memory_tokens must have shape [R,L,D] or [B,R,L,D]")
        count = memory_tokens.size(1)
        if memory_mask is not None:
            mask = memory_mask.to(device=recent_query.device, dtype=torch.bool)
            if mask.ndim == 1 and mask.shape == (count,):
                mask = mask[None, :, None].expand(bsz, -1, links)
            elif mask.ndim == 2 and mask.shape == (bsz, count):
                mask = mask[:, :, None].expand(-1, -1, links)
            if mask.shape != valid.shape:
                raise ValueError("memory_mask must have shape [B,R] or [B,R,L]")
            valid = valid & mask

        futures = None
        if future_targets is not None:
            futures = future_targets
            if shared:
                if futures.ndim == 4 and futures.size(-1) == 1:
                    futures = futures.squeeze(-1)
                if (
                    futures.ndim != 3
                    or futures.size(0) != count
                    or futures.size(2) != links
                ):
                    raise ValueError(
                        "Shared futures must have shape [R,H,L,1] or [R,H,L]"
                    )
                finite_futures = torch.isfinite(futures).all(dim=1).unsqueeze(0)
                futures = futures.unsqueeze(0).expand(bsz, -1, -1, -1)
            else:
                if futures.ndim == 5 and futures.size(-1) == 1:
                    futures = futures.squeeze(-1)
                if (
                    futures.ndim != 4
                    or futures.shape[:2] != (bsz, count)
                    or futures.size(3) != links
                ):
                    raise ValueError(
                        "Batched futures must have shape [B,R,H,L,1] or [B,R,H,L]"
                    )
                finite_futures = torch.isfinite(futures).all(dim=2)
            valid = valid & finite_futures

        # Keep the shared bank unbatched in the contraction: expanding B first
        # can make einsum materialize B copies when it reshapes for matmul.
        equation = "bld,rld->brl" if shared else "bld,brld->brl"
        scores = torch.einsum(
            equation, F.normalize(recent_query, dim=-1), normalized_tokens
        )
        k = count if self.memory_top_k is None else min(self.memory_top_k, count)
        values, indices = scores.masked_fill(
            ~valid, torch.finfo(scores.dtype).min
        ).topk(k, dim=1)
        selected_valid = valid.gather(1, indices)
        weights = _masked_softmax(values, selected_valid, dim=1) if k else values
        selected = memory_tokens.gather(1, indices[..., None].expand(-1, -1, -1, dim))
        history = (weights[..., None] * torch.nan_to_num(selected)).sum(dim=1)
        dense = torch.zeros_like(scores).scatter(1, indices, weights)
        aux = {
            "memory_weights": dense,
            "memory_indices": indices,
            "selected_memory_weights": weights,
            "memory_valid": valid.any(dim=1),
        }
        if futures is not None:
            horizon = futures.size(2)
            selected_future = futures.permute(0, 1, 3, 2).gather(
                1, indices[..., None].expand(-1, -1, -1, horizon)
            )
            aux["retrieved_future"] = (
                (weights[..., None] * torch.nan_to_num(selected_future))
                .sum(dim=1)
                .permute(0, 2, 1)
            )
            aux["future_indices"] = indices
        if memory_ages is not None:
            ages = memory_ages
            if ages.ndim == 1 and ages.shape == (count,):
                ages = ages[None, :, None].expand(bsz, -1, links)
            elif ages.ndim == 2 and ages.shape == (bsz, count):
                ages = ages[:, :, None].expand(-1, -1, links)
            if ages.shape != valid.shape:
                raise ValueError("memory_ages must have shape [R], [B,R], or [B,R,L]")
            aux["selected_memory_age"] = ages.gather(1, indices)
        return history, aux


class SparseTemporalRouter(nn.Module):
    """Equations (5)--(7): cosine candidates, leading lags, sparse routing."""

    def __init__(
        self,
        d_model: int,
        top_k: int = 8,
        coarse_top_k: int = 32,
        max_lag: float = 12,
        query_len: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        if top_k < 1 or coarse_top_k < 1 or max_lag < 0:
            raise ValueError(
                "Routing k values must be positive and max_lag nonnegative"
            )
        self.top_k, self.coarse_top_k, self.max_lag = (
            top_k,
            coarse_top_k,
            float(max_lag),
        )
        self.query_pool = RecentQueryPool(d_model, query_len)
        self.lag_mlp = nn.Sequential(
            nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(4 * d_model + 1, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    @staticmethod
    def _gather_sources(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        bsz, links, dim = source.shape
        return (
            source[:, None]
            .expand(-1, links, -1, -1)
            .gather(2, indices[..., None].expand(bsz, links, -1, dim))
        )

    @staticmethod
    def _aligned_pair_features(
        target: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        target = target[:, :, None].expand(-1, -1, source.size(2), -1)
        return torch.cat(
            (target, source, (target - source).abs(), target * source), dim=-1
        )

    @staticmethod
    def _interpolate_selected_source_states(
        h_local: torch.Tensor, lag: torch.Tensor, source_index: torch.Tensor
    ) -> torch.Tensor:
        # Paper uses 1-based T-delta: zero-based tensor position is (T-1)-delta.
        bsz, steps, links, dim = h_local.shape
        position = ((steps - 1) - lag).clamp(0, steps - 1)
        low = position.floor().long()
        high = (low + 1).clamp(max=steps - 1)
        fraction = (position - low.to(position.dtype)).unsqueeze(-1)
        source = h_local.permute(0, 2, 1, 3).reshape(bsz * links * steps, dim)
        batch = torch.arange(bsz, device=h_local.device)[:, None, None]
        base = (batch * links + source_index) * steps
        low_state = source[(base + low).reshape(-1)].reshape(*lag.shape, dim)
        high_state = source[(base + high).reshape(-1)].reshape(*lag.shape, dim)
        return low_state * (1 - fraction) + high_state * fraction

    def forward(
        self,
        h_local: torch.Tensor,
        events: torch.Tensor,
        recent_query: torch.Tensor | None = None,
        return_aux: bool = False,
        route_mask: torch.Tensor | None = None,
        source_index_override: torch.Tensor | None = None,
    ):
        bsz, _, links, dim = h_local.shape
        query = self.query_pool(events) if recent_query is None else recent_query
        if source_index_override is None:
            k_coarse = min(self.coarse_top_k, max(links - 1, 0))
            normalized = F.normalize(query, dim=-1)
            scores = normalized @ normalized.transpose(1, 2)
            self_mask = torch.eye(links, dtype=torch.bool, device=query.device)[None]
            coarse = (
                scores.masked_fill(self_mask, torch.finfo(scores.dtype).min)
                .topk(k_coarse, dim=-1)
                .indices
            )
        else:
            coarse = source_index_override.to(device=query.device, dtype=torch.long)
            if coarse.ndim != 3 or coarse.shape[:2] != (bsz, links):
                raise ValueError("source_index_override must have shape [B,L,Kc]")
            if torch.any((coarse < 0) | (coarse >= links)):
                raise ValueError("source_index_override contains an out-of-range link")
            k_coarse = coarse.size(2)
        valid = coarse != torch.arange(links, device=query.device)[None, :, None]
        source_queries = self._gather_sources(query, coarse)
        lag = self.max_lag * torch.sigmoid(
            self.lag_mlp(self._aligned_pair_features(query, source_queries))
        ).squeeze(-1)
        aligned = self._interpolate_selected_source_states(h_local, lag, coarse)
        pair = self._aligned_pair_features(query, aligned)
        scores = self.score_mlp(
            torch.cat((pair, (lag / max(self.max_lag, 1.0))[..., None]), dim=-1)
        ).squeeze(-1)
        k = min(self.top_k, k_coarse)
        values, selected = scores.masked_fill(
            ~valid, torch.finfo(scores.dtype).min
        ).topk(k, dim=-1)
        selected_valid = valid.gather(2, selected)
        if route_mask is not None:
            if route_mask.shape != selected_valid.shape:
                raise ValueError("route_mask must match selected routes [B,L,K]")
            selected_valid = selected_valid & route_mask.to(
                device=query.device, dtype=torch.bool
            )
        weights = _masked_softmax(values, selected_valid, dim=-1) if k else values
        selected_aligned = aligned.gather(
            2, selected[..., None].expand(-1, -1, -1, dim)
        )
        routed = (weights[..., None] * selected_aligned).sum(dim=2)
        if not return_aux:
            return routed
        indices = coarse.gather(2, selected)
        aux = {
            "route_weights": weights,
            "route_indices": indices,
            "estimated_lags": lag.gather(2, selected),
            "coarse_indices": coarse,
            "candidate_indices": coarse,
            "selected_source_indices": indices,
            "routing_scores": values,
            "routing_weights": weights,
            "aligned_sources": selected_aligned,
            "routed_context": routed,
            "route_valid": selected_valid.any(dim=-1),
        }
        return routed, aux


class ConfidenceFusion(nn.Module):
    """Equation (8): two-way local/routed context softmax."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(
        self,
        local_ctx: torch.Tensor,
        routed_ctx: torch.Tensor,
        route_valid: torch.Tensor,
    ):
        logits = self.gate(torch.cat((local_ctx, routed_ctx), dim=-1))
        valid = torch.stack((torch.ones_like(route_valid), route_valid), dim=-1)
        weights = _masked_softmax(logits, valid, dim=-1)
        return weights[..., :1] * local_ctx + weights[..., 1:] * routed_ctx, weights


class StatisticalPriorMixer(nn.Module):
    """Equation (10): three branch logits independently for every horizon."""

    def __init__(self, d_model: int, pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * 3),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context).reshape(*context.shape[:2], self.pred_len, 3)


class Whiplash(nn.Module):
    """Paper-aligned selective SSM, dual retrieval, residual and prior fusion.

    Legacy statistical feature / pattern retrieval inputs are accepted only
    for call compatibility. Older architecture checkpoints are incompatible.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        pred_len: int,
        num_links: int,
        mamba_layers: int = 2,
        d_state: int = 16,
        top_k: int = 8,
        coarse_top_k: int = 32,
        max_lag: float = 12,
        query_len: int = 6,
        memory_top_k: int | None = 4,
        memory_coarse_top_k: int = 128,
        use_historical_memory: bool = True,
        use_mamba: bool = True,
        use_nonlocal_router: bool = True,
        use_pattern_retrieval: bool = True,
        use_statistical_prior: bool = True,
        dropout: float = 0.1,
        residual_forecast: bool = True,
        seq_len: int = 12,
        expand: int = 2,
        conv_kernel: int = 4,
        eps: float = 1e-4,
    ):
        super().__init__()
        if min(in_features, d_model, pred_len, num_links, seq_len) < 1:
            raise ValueError("All model dimensions must be positive")
        if not 0 < eps < 0.5:
            raise ValueError("eps must lie in (0,0.5)")
        self.in_features, self.d_model = in_features, d_model
        self.pred_len, self.num_links, self.seq_len = pred_len, num_links, seq_len
        self.eps = eps
        self.use_historical_memory, self.use_mamba = use_historical_memory, use_mamba
        self.use_nonlocal_router, self.use_statistical_prior = (
            use_nonlocal_router,
            use_statistical_prior,
        )
        self.residual_forecast = residual_forecast
        self.embed = nn.Linear(in_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, 1, d_model))
        self.local_encoder = (
            MambaStack(
                d_model,
                mamba_layers,
                d_state,
                expand=expand,
                conv_kernel=conv_kernel,
                dropout=dropout,
            )
            if use_mamba
            else nn.Identity()
        )
        self.local_norm = nn.LayerNorm(d_model)
        self.event_tokenizer = EventTokenizer(d_model, dropout)
        self.recent_query_pool = RecentQueryPool(d_model, query_len)
        self.historical_memory = HistoricalPatternMemory(d_model, memory_top_k)
        self.temporal_router = SparseTemporalRouter(
            d_model, top_k, coarse_top_k, max_lag, query_len, dropout
        )
        self.fusion = ConfidenceFusion(d_model, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.prior_mixer = StatisticalPriorMixer(d_model, pred_len, dropout)

    def _encode_sequence(self, x: torch.Tensor):
        if x.ndim != 4 or x.size(2) != self.num_links or x.size(3) != self.in_features:
            raise ValueError("Input must have shape [B,T,num_links,in_features]")
        bsz, steps, links, _ = x.shape
        if not 1 <= steps <= self.seq_len:
            raise ValueError(
                f"Input length must be between 1 and seq_len={self.seq_len}"
            )
        encoded = self.embed(x) + self.pos_embed[:, :steps]
        local = encoded.permute(0, 2, 1, 3).reshape(bsz * links, steps, self.d_model)
        local = self.local_norm(self.local_encoder(local))
        local = local.reshape(bsz, links, steps, self.d_model).permute(0, 2, 1, 3)
        return local, self.event_tokenizer(local)

    def build_memory_tokens(self, x_memory: torch.Tensor) -> torch.Tensor:
        """Encode [R,T,L,F] or [B,R,T,L,F] with the current shared encoder."""
        if x_memory.ndim == 4:
            if x_memory.size(0) == 0:
                return x_memory.new_empty(0, self.num_links, self.d_model)
            _, events = self._encode_sequence(x_memory)
            return self.recent_query_pool(events)
        if x_memory.ndim != 5:
            raise ValueError("Memory windows must have shape [R,T,L,F] or [B,R,T,L,F]")
        bsz, count, steps, links, feats = x_memory.shape
        if not count:
            return x_memory.new_empty(bsz, 0, links, self.d_model)
        _, events = self._encode_sequence(
            x_memory.reshape(bsz * count, steps, links, feats)
        )
        return self.recent_query_pool(events).reshape(bsz, count, links, self.d_model)

    def forward(
        self,
        x_recent: torch.Tensor,
        x_memory: torch.Tensor | None = None,
        memory_tokens: torch.Tensor | None = None,
        memory_futures: torch.Tensor | None = None,
        memory_stat_features: torch.Tensor | None = None,
        memory_ages: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        stat_prior: torch.Tensor | None = None,
        stat_std: torch.Tensor | None = None,
        stat_features: torch.Tensor | None = None,
        return_aux: bool = False,
        return_analysis: bool = False,
        route_mask: torch.Tensor | None = None,
        source_index_override: torch.Tensor | None = None,
    ):
        local, events = self._encode_sequence(x_recent)
        query = self.recent_query_pool(events)
        bsz, links, _ = query.shape
        if self.use_nonlocal_router:
            routed, route_aux = self.temporal_router(
                local, events, query, True, route_mask, source_index_override
            )
            route_valid = route_aux["route_valid"]
        else:
            routed, route_aux = torch.zeros_like(query), {}
            route_valid = torch.zeros(bsz, links, dtype=torch.bool, device=query.device)
        context, fusion_weights = self.fusion(query, routed, route_valid)
        if (
            memory_tokens is None
            and x_memory is not None
            and self.use_historical_memory
        ):
            memory_tokens = self.build_memory_tokens(x_memory)
        memory_aux = {}
        if self.use_historical_memory and memory_tokens is not None:
            _, memory_aux = self.historical_memory(
                query, memory_tokens, memory_futures, memory_ages, memory_mask
            )

        raw = self.head(context)
        baseline = (
            x_recent[:, -1, :, 0, None]
            .expand(-1, -1, self.pred_len)
            .clamp(self.eps, 1 - self.eps)
        )
        residual = (
            torch.sigmoid(torch.logit(baseline) + raw)
            if self.residual_forecast
            else torch.sigmoid(raw)
        )
        stat = torch.zeros_like(residual)
        stat_valid = torch.zeros_like(residual, dtype=torch.bool)
        if self.use_statistical_prior and stat_prior is not None:
            if stat_prior.ndim == 4 and stat_prior.size(-1) == 1:
                stat_prior = stat_prior.squeeze(-1)
            if stat_prior.shape != (bsz, self.pred_len, links):
                raise ValueError("stat_prior must have shape [B,H,L] or [B,H,L,1]")
            stat = stat_prior.permute(0, 2, 1)
            stat_valid = torch.isfinite(stat)
            stat = torch.nan_to_num(stat).clamp(0, 1)
        future = torch.zeros_like(residual)
        future_valid = torch.zeros_like(residual, dtype=torch.bool)
        if "retrieved_future" in memory_aux:
            retrieved = memory_aux["retrieved_future"]
            if retrieved.shape != (bsz, self.pred_len, links):
                raise ValueError("Memory future horizon does not match pred_len")
            future = retrieved.permute(0, 2, 1).clamp(0, 1)
            future_valid = memory_aux["memory_valid"][..., None].expand_as(future)
        available = torch.stack(
            (stat_valid, torch.ones_like(stat_valid), future_valid), dim=-1
        )
        prior_weights = _masked_softmax(self.prior_mixer(context), available, dim=-1)
        branches = torch.stack((stat, residual, future), dim=-1)
        prediction = (
            (prior_weights * branches).sum(dim=-1).permute(0, 2, 1).unsqueeze(-1)
        )
        if not (return_aux or return_analysis):
            return prediction
        weights = prior_weights.permute(
            0, 2, 1, 3
        )  # [B,H,L,3]: statistical, residual, future.
        return {
            "prediction": prediction,
            "final_prediction": prediction,
            "residual_prediction": residual.permute(0, 2, 1).unsqueeze(-1),
            "stat_prior": stat.permute(0, 2, 1),
            "prior_weights": weights,
            "alpha": weights[..., 2:3],
            "future_alpha": weights[..., 2:3],
            "fusion_weights": fusion_weights,
            "recent_query": query,
            "fused_context": context,
            **route_aux,
            **memory_aux,
        }


def build_whiplash(
    in_features: int,
    d_model: int,
    pred_len: int,
    num_links: int,
    top_k: int = 8,
    coarse_top_k: int = 32,
    max_lag: float = 12,
    query_len: int = 6,
    memory_top_k: int | None = 4,
    memory_coarse_top_k: int = 128,
    dropout: float = 0.1,
    seq_len: int = 12,
    mamba_layers: int = 2,
    d_state: int = 16,
    expand: int = 2,
    conv_kernel: int = 4,
    eps: float = 1e-4,
) -> Whiplash:
    """Construct equations (1)--(10); unspecified sizes are implementation defaults."""
    return Whiplash(
        in_features=in_features,
        d_model=d_model,
        pred_len=pred_len,
        num_links=num_links,
        top_k=top_k,
        coarse_top_k=coarse_top_k,
        max_lag=max_lag,
        query_len=query_len,
        memory_top_k=memory_top_k,
        memory_coarse_top_k=memory_coarse_top_k,
        dropout=dropout,
        seq_len=seq_len,
        mamba_layers=mamba_layers,
        d_state=d_state,
        expand=expand,
        conv_kernel=conv_kernel,
        eps=eps,
    )


@torch.no_grad()
def build_memory_bank(
    model: Whiplash,
    x_memory: torch.Tensor,
    future_targets: torch.Tensor | None = None,
    batch_size: int = 32,
    device: torch.device | str | None = None,
):
    """Refresh keys from the current encoder in eval mode, returning detached CPU keys.

    Supports shared [R,T,L,F] or legacy batched [N,R,T,L,F] windows.
    The caller owns training-only selection/exclusion and refresh scheduling.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if x_memory.ndim not in (4, 5):
        raise ValueError("Memory windows must have four or five dimensions")
    device = torch.device(
        device if device is not None else next(model.parameters()).device
    )
    was_training = model.training
    model.eval()
    try:
        chunks = [
            model.build_memory_tokens(
                x_memory[start : start + batch_size].to(device)
            ).cpu()
            for start in range(0, x_memory.size(0), batch_size)
        ]
        shape = (
            (*x_memory.shape[:2], model.num_links, model.d_model)
            if x_memory.ndim == 5
            else (0, model.num_links, model.d_model)
        )
        tokens = (
            torch.cat(chunks, dim=0)
            if chunks
            else torch.empty(shape, dtype=x_memory.dtype)
        )
    finally:
        model.train(was_training)
    return tokens if future_targets is None else (tokens, future_targets)
