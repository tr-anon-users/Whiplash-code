"""Observed-label metrics in km/h; undefined statistics are JSON null."""

from __future__ import annotations
import numpy as np


def inverse_speed(values, scaler):
    values = values[..., 0] if values.ndim == 4 else values
    n, h, l = values.shape
    return scaler.inverse_transform(values.reshape(n * h, l)).reshape(n, h, l)


def metric_block(pred, true, mask=None, mape_threshold=1.0):
    valid = np.isfinite(true)
    if mask is not None:
        valid &= mask.astype(bool)
    if not np.isfinite(pred[valid]).all():
        raise ValueError("Non-finite predictions on observed targets")
    mape_valid = valid & (np.abs(true) > mape_threshold)
    error = pred[valid] - true[valid]
    mse = float(np.mean(error**2)) if error.size else None
    return {
        "MAPE(%)": float(
            np.mean(np.abs((pred[mape_valid] - true[mape_valid]) / true[mape_valid]))
            * 100
        )
        if mape_valid.any()
        else None,
        "MAE": float(np.mean(np.abs(error))) if error.size else None,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)) if mse is not None else None,
        "valid_count": int(valid.sum()),
        "mape_valid_count": int(mape_valid.sum()),
    }


def horizon_metrics(preds, trues, scaler, mask=None, mape_threshold=1.0):
    pred, true = inverse_speed(preds, scaler), inverse_speed(trues, scaler)
    if mask is not None and mask.ndim == 4:
        mask = mask[..., 0]
    per = {
        f"h{h + 1}": metric_block(
            pred[:, h], true[:, h], None if mask is None else mask[:, h], mape_threshold
        )
        for h in range(pred.shape[1])
    }
    avg = {}
    for key in ("MAPE(%)", "MAE", "MSE", "RMSE"):
        values = [m[key] for m in per.values() if m[key] is not None]
        avg[key] = float(np.mean(values)) if values else None
    return dict(
        avg=avg,
        last=per[f"h{pred.shape[1]}"],
        per_horizon=per,
        units={"MAE": "km/h", "MSE": "(km/h)^2", "RMSE": "km/h", "MAPE(%)": "percent"},
        mape_threshold_kmh=mape_threshold,
        aggregation="arithmetic mean of defined per-horizon metrics",
    )
