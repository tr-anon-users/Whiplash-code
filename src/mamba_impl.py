from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveScan(nn.Module):
    """Causal, linear-time PyTorch selective SSM (not a fused mamba-ssm kernel).

    Delta, B, and C depend on each input. Negative diagonal A makes the
    exponential state decay stable. Delta starts log-uniformly in [0.001,0.1].
    """

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        if d_inner < 1 or d_state < 1:
            raise ValueError("d_inner and d_state must be positive")
        self.d_inner = d_inner
        self.d_state = d_state

        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1)
        )
        self.d = nn.Parameter(torch.ones(d_inner))
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.b_proj = nn.Linear(d_inner, d_state)
        self.c_proj = nn.Linear(d_inner, d_state)
        nn.init.uniform_(self.dt_proj.weight, -(d_inner**-0.5), d_inner**-0.5)
        initial_dt = torch.exp(
            torch.empty(d_inner).uniform_(
                torch.log(torch.tensor(0.001)).item(),
                torch.log(torch.tensor(0.1)).item(),
            )
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(initial_dt + torch.log(-torch.expm1(-initial_dt)))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, T, D)
        bsz, steps, dim = u.shape
        a = -torch.exp(self.a_log).to(dtype=u.dtype)  # (D, N)
        state = u.new_zeros(bsz, dim, self.d_state)
        outputs = []

        dt = F.softplus(self.dt_proj(u)) + 1e-4  # (B, T, D)
        b_t = self.b_proj(u)  # (B, T, N)
        c_t = self.c_proj(u)  # (B, T, N)

        for t in range(steps):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, D, 1)
            decay = torch.exp(dt_t * a.unsqueeze(0))
            write = dt_t * b_t[:, t, :].unsqueeze(1) * u[:, t, :].unsqueeze(-1)
            state = decay * state + write
            y_t = (state * c_t[:, t, :].unsqueeze(1)).sum(dim=-1)
            y_t = y_t + self.d * u[:, t, :]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    """A compact Mamba-style block with causal depthwise convolution and selective scan."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if expand < 1 or conv_kernel < 1:
            raise ValueError("expand and conv_kernel must be positive")
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            padding=conv_kernel - 1,
        )
        self.scan = SelectiveScan(self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        u, gate = self.in_proj(x).chunk(2, dim=-1)

        conv = self.conv(u.transpose(1, 2))[..., : u.size(1)].transpose(1, 2)
        conv = F.silu(conv)
        scanned = self.scan(conv)
        y = scanned * F.silu(gate)
        y = self.out_proj(y)
        return residual + self.dropout(y)


class MambaStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.0,
        conv_kernel: int = 4,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
