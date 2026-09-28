"""Whiplash Eq. 1-10 training with train-only memory and untouched final test data."""

from __future__ import annotations
import argparse
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.data import load_traffic_data
from src.metrics import horizon_metrics
from src.models import build_whiplash
from src.utils import count_parameters, make_run_dir, save_json, set_seed


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", required=True)
    p.add_argument("--link-path", default=None)
    p.add_argument("--output-dir", default="outputs/runs")
    integers = dict(
        seq_len=12,
        pred_len=12,
        d_model=16,
        mamba_layers=2,
        d_state=16,
        expand=2,
        conv_kernel=4,
        top_k=8,
        coarse_top_k=32,
        query_len=6,
        memory_top_k=4,
        memory_stride=12,
        memory_lookback_steps=8064,
        max_memory_candidates=672,
        memory_encode_batch_size=32,
        epochs=50,
        batch_size=32,
        patience=5,
        fold=0,
        n_folds=5,
        seed=42,
        num_workers=0,
        cpu_threads=4,
        interval_minutes=5,
        max_train_batches=0,
        max_val_batches=0,
        max_test_batches=0,
    )
    floats = dict(
        max_lag=12.0,
        dropout=0.1,
        eps=1e-4,
        lr=1e-3,
        weight_decay=1e-4,
        change_loss_weight=0.2,
        residual_loss_weight=0.1,
        grad_clip=1.0,
        train_ratio=0.7,
        val_ratio=0.1,
        mape_threshold=1.0,
    )
    for key, value in integers.items():
        p.add_argument("--" + key.replace("_", "-"), type=int, default=value)
    for key, value in floats.items():
        p.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    p.add_argument("--split-mode", choices=["holdout", "expanding"], default="holdout")
    p.add_argument("--speed-unit", choices=["kmh", "mph"], default="kmh")
    p.add_argument("--zero-is-missing", action="store_true")
    p.add_argument("--hdf-key", default=None)
    p.add_argument(
        "--start-time",
        default=None,
        help="Explicit timestamp origin for timestamp-free wide CSV only",
    )
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p


def validate_args(args):
    for name in (
        "epochs",
        "batch_size",
        "patience",
        "memory_encode_batch_size",
        "cpu_threads",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if (
        min(
            args.change_loss_weight,
            args.residual_loss_weight,
            args.weight_decay,
            args.mape_threshold,
        )
        < 0
    ):
        raise ValueError(
            "Loss weights, weight decay and MAPE threshold must be nonnegative"
        )
    if args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("Learning rate and gradient clip must be positive")
    if min(args.max_train_batches, args.max_val_batches, args.max_test_batches) < 0:
        raise ValueError("Batch limits must be nonnegative (0 means all batches)")


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    return args


def resolve_device(choice):
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; install a matching CUDA PyTorch build or use --device cpu"
        )
    return torch.device(choice)


class ForecastLoss(nn.Module):
    """Masked MSE(final) + .2 L1(adjacent changes) + .1 MSE(residual).

    Changes require two observed endpoints; H=1 has zero change loss.
    """

    def __init__(self, change_loss_weight=0.2, residual_loss_weight=0.1):
        super().__init__()
        self.change_loss_weight = change_loss_weight
        self.residual_loss_weight = residual_loss_weight

    @staticmethod
    def masked_mean(values, mask):
        return torch.where(
            mask, values, torch.zeros_like(values)
        ).sum() / mask.sum().clamp_min(1)

    def components(self, pred, target, mask=None):
        final = pred["prediction"] if isinstance(pred, dict) else pred
        valid = torch.isfinite(target)
        if mask is not None:
            valid = valid & mask.bool()
        # Sanitize BEFORE arithmetic; masked NaNs must not poison gradients.
        y = torch.where(valid, target, torch.zeros_like(target))
        f = torch.where(valid, final, torch.zeros_like(final))
        mse = self.masked_mean((f - y).square(), valid)
        delta = mse * 0.0
        if target.shape[1] > 1:
            pair = valid[:, 1:] & valid[:, :-1]
            delta = self.masked_mean(
                ((f[:, 1:] - f[:, :-1]) - (y[:, 1:] - y[:, :-1])).abs(), pair
            )
        residual = mse * 0.0
        if isinstance(pred, dict) and "residual_prediction" in pred:
            r = torch.where(valid, pred["residual_prediction"], torch.zeros_like(final))
            residual = self.masked_mean((r - y).square(), valid)
        return dict(mse=mse, change=delta, residual=residual)

    def forward(self, pred, target, mask=None):
        v = self.components(pred, target, mask)
        return (
            v["mse"]
            + self.change_loss_weight * v["change"]
            + self.residual_loss_weight * v["residual"]
        )


@torch.no_grad()
def refresh_memory(model, data, device, batch_size):
    """Detached key snapshot from current weights, once per shared training window."""
    training = model.training
    model.eval()
    chunks = []
    for start in range(0, len(data.memory_windows), batch_size):
        x = torch.as_tensor(
            data.memory_windows[start : start + batch_size], device=device
        )
        chunks.append(model.build_memory_tokens(x).detach())
    model.train(training)
    if not chunks:
        return None
    return dict(
        tokens=torch.cat(chunks),
        futures=torch.as_tensor(data.memory_futures, device=device),
    )


def forward_batch(model, batch, memory, device, return_aux=True):
    kwargs = dict(stat_prior=batch["stat_prior"].to(device), return_aux=return_aux)
    if memory is not None:
        indices = batch["memory_indices"].to(device)
        valid = batch["memory_mask"].to(device)
        # Share futures across B; only the eligibility mask is dense B*R*L.
        mask = torch.zeros(
            (len(indices), len(memory["tokens"]), valid.shape[-1]),
            dtype=torch.bool,
            device=device,
        )
        bi, ci, li = valid.nonzero(as_tuple=True)
        mask[bi, indices[bi, ci], li] = True
        kwargs.update(
            memory_tokens=memory["tokens"],
            memory_futures=memory["futures"],
            memory_mask=mask,
        )
    return model(batch["x"].to(device), **kwargs)


def run_epoch(
    model,
    loader,
    memory,
    criterion,
    device,
    optimizer=None,
    max_batches=0,
    grad_clip=1.0,
):
    model.train(optimizer is not None)
    totals = dict(mse=0.0, change=0.0, residual=0.0)
    counts = dict(mse=0, change=0, residual=0)
    with torch.set_grad_enabled(optimizer is not None):
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            target, mask = batch["y"].to(device), batch["target_mask"].to(device)
            if not mask.any():
                continue
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            pred = forward_batch(model, batch, memory, device)
            parts = criterion.components(pred, target, mask)
            loss = (
                parts["mse"]
                + criterion.change_loss_weight * parts["change"]
                + criterion.residual_loss_weight * parts["residual"]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite forecasting loss")
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
            sizes = dict(
                mse=int(mask.sum()),
                residual=int(mask.sum()),
                change=int((mask[:, 1:] & mask[:, :-1]).sum()),
            )
            for key in totals:
                totals[key] += float(parts[key].detach()) * sizes[key]
                counts[key] += sizes[key]
    if not counts["mse"]:
        raise ValueError("No observed targets in this split or selected batches")
    means = {key: totals[key] / max(1, counts[key]) for key in totals}
    return (
        means["mse"]
        + criterion.change_loss_weight * means["change"]
        + criterion.residual_loss_weight * means["residual"]
    )


@torch.no_grad()
def collect_predictions(model, loader, memory, device, max_batches=0):
    model.eval()
    predictions, targets, masks = [], [], []
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        predictions.append(
            forward_batch(model, batch, memory, device, False).cpu().numpy()
        )
        targets.append(batch["y"].numpy())
        masks.append(batch["target_mask"].numpy())
    if not predictions:
        raise ValueError("No evaluation samples")
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)


def run_training(args):
    validate_args(args)
    args.data_path = str(Path(args.data_path).resolve())
    if args.link_path is not None:
        args.link_path = str(Path(args.link_path).resolve())
    set_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)
    run_dir = make_run_dir(args.output_dir)
    data_keys = (
        "data_path",
        "link_path",
        "seq_len",
        "pred_len",
        "batch_size",
        "train_ratio",
        "val_ratio",
        "num_workers",
        "seed",
        "interval_minutes",
        "split_mode",
        "fold",
        "n_folds",
        "memory_stride",
        "memory_lookback_steps",
        "max_memory_candidates",
        "zero_is_missing",
        "hdf_key",
        "speed_unit",
        "start_time",
    )
    data = load_traffic_data(**{k: getattr(args, k) for k in data_keys})
    model_keys = (
        "d_model",
        "pred_len",
        "top_k",
        "coarse_top_k",
        "max_lag",
        "query_len",
        "memory_top_k",
        "dropout",
        "mamba_layers",
        "d_state",
        "expand",
        "conv_kernel",
        "seq_len",
        "eps",
    )
    model_config = {k: getattr(args, k) for k in model_keys}
    model_config.update(
        in_features=len(data.feature_names), num_links=len(data.link_ids)
    )
    model = build_whiplash(**model_config).to(device)
    smoke = bool(
        args.max_train_batches or args.max_val_batches or args.max_test_batches
    )
    config = vars(args) | dict(
        device_resolved=str(device),
        parameters=count_parameters(model),
        model_config=model_config,
        feature_names=data.feature_names,
        run_dir=str(run_dir),
        run_kind="smoke_test" if smoke else "experiment",
        environment=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            numpy=np.__version__,
            pandas=pd.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        ),
    )
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "data_metadata.json", data.metadata)
    criterion = ForecastLoss(args.change_loss_weight, args.residual_loss_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history, best, stale, best_epoch = [], float("inf"), 0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        memory = refresh_memory(model, data, device, args.memory_encode_batch_size)
        train_loss = run_epoch(
            model,
            data.train_loader,
            memory,
            criterion,
            device,
            optimizer,
            args.max_train_batches,
            args.grad_clip,
        )
        memory = refresh_memory(model, data, device, args.memory_encode_batch_size)
        val_loss = run_epoch(
            model,
            data.val_loader,
            memory,
            criterion,
            device,
            max_batches=args.max_val_batches,
        )
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"epoch {epoch:03d} | train {train_loss:.6f} | validation {val_loss:.6f}",
            flush=True,
        )
        if val_loss < best - 1e-9:
            best, stale, best_epoch = val_loss, 0, epoch
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    model_config=model_config,
                    config=config,
                    data_metadata=data.metadata,
                    epoch=epoch,
                    best_val_loss=best,
                ),
                run_dir / "best_model.pth",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(
        run_dir / "best_model.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    memory = refresh_memory(model, data, device, args.memory_encode_batch_size)
    scope = "test" if args.split_mode == "holdout" else "validation"
    loader = data.test_loader if scope == "test" else data.val_loader
    limit = args.max_test_batches if scope == "test" else args.max_val_batches
    pred, true, mask = collect_predictions(model, loader, memory, device, limit)
    metrics = horizon_metrics(pred, true, data.speed_scaler, mask, args.mape_threshold)
    for name, values in [("preds", pred), ("trues", true), ("observed_mask", mask)]:
        np.save(run_dir / f"{name}_{scope}_scaled.npy", values)
    np.savez(
        run_dir / "preprocessing.npz",
        speed_scale=data.speed_scaler.scale_,
        speed_min=data.speed_scaler.min_,
        speed_data_min=data.speed_scaler.data_min_,
        speed_data_max=data.speed_scaler.data_max_,
        delta_scale=data.delta_scaler.scale_,
        delta_min=data.delta_scaler.min_,
    )
    summary = dict(
        model="Whiplash_AISTATS",
        scope=scope,
        run_kind=config["run_kind"],
        seed=args.seed,
        pred_len=args.pred_len,
        fold=args.fold if scope == "validation" else None,
        best_epoch=best_epoch,
        best_val_loss=best,
        elapsed_seconds=round(time.time() - start, 3),
        run_dir=str(run_dir),
        avg=metrics["avg"],
        last=metrics["last"],
    )
    save_json(run_dir / "metrics.json", metrics)
    save_json(run_dir / "summary.json", summary)
    print(
        f"{scope} horizon-average metrics: {metrics['avg']}\nrun_dir: {run_dir}",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run_training(parse_args())
