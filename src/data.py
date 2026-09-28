"""Training-only preprocessing, chronological splits, and lazy Whiplash data.

Forecast origins and memory future ends are exclusive. Training retrieval
requires memory_future_end <= query_start; evaluation memory is frozen to the
training split. Every memory window is stored once, shared by all queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SplitBounds:
    train_end: int
    val_end: int
    test_end: int
    development_end: int
    mode: str = "holdout"
    fold: int | None = None


def make_temporal_splits(
    n_steps: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    mode: str = "holdout",
    n_folds: int = 5,
) -> list[SplitBounds]:
    """Expanding CV uses development data only and exposes no test samples.

    Development is the first train_ratio+val_ratio fraction, partitioned into
    n_folds+1 blocks. Fold i trains on its first i+1 blocks and validates on the
    following block. The remaining final holdout stays inaccessible to CV loaders.
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 - train_ratio):
        raise ValueError("Split ratios must be positive and sum to less than 1")
    if n_steps < 3:
        raise ValueError("At least three time steps are required")
    # Avoid losing an entire row to 1440 * .7 == 1007.9999999999999.
    dev_end = int(round(n_steps * (train_ratio + val_ratio), 9))
    if mode == "holdout":
        return [
            SplitBounds(int(round(n_steps * train_ratio, 9)), dev_end, n_steps, dev_end)
        ]
    if mode != "expanding":
        raise ValueError("split mode must be 'holdout' or 'expanding'")
    if n_folds < 2:
        raise ValueError("Expanding CV requires at least two folds")
    edges = np.linspace(0, dev_end, n_folds + 2, dtype=int)
    return [
        SplitBounds(
            int(edges[i + 1]), int(edges[i + 2]), int(edges[i + 2]), dev_end, mode, i
        )
        for i in range(n_folds)
    ]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    speed_scaler: MinMaxScaler
    delta_scaler: MinMaxScaler
    link_ids: list[Any]
    feature_names: list[str]
    train_samples: int
    val_samples: int
    test_samples: int
    memory_windows: torch.Tensor
    memory_futures: torch.Tensor
    memory_valid: torch.Tensor
    memory_starts: np.ndarray
    memory_future_ends: np.ndarray
    metadata: dict[str, Any]


class TrafficWindowDataset(Dataset):
    """Slices one feature array instead of duplicating all sliding windows."""

    def __init__(
        self,
        features,
        targets,
        observed,
        stat_table,
        time_slots,
        origins,
        seq_len,
        pred_len,
        memory_future_ends,
        memory_valid,
        train_end,
        training,
        memory_lookback_steps,
        max_memory_candidates,
    ):
        self.features, self.targets, self.observed = features, targets, observed
        self.stat_table, self.time_slots, self.origins = stat_table, time_slots, origins
        self.seq_len, self.pred_len = seq_len, pred_len
        self.memory_future_ends, self.memory_valid = memory_future_ends, memory_valid
        self.train_end, self.training = train_end, training
        self.memory_lookback_steps = memory_lookback_steps
        self.candidate_count = min(max_memory_candidates, len(memory_future_ends))
        self._evaluation_candidates = self._candidates(train_end)

    def __len__(self):
        return len(self.origins)

    def _candidates(self, boundary):
        right = int(np.searchsorted(self.memory_future_ends, boundary, side="right"))
        left = int(
            np.searchsorted(
                self.memory_future_ends,
                boundary - self.memory_lookback_steps,
                side="left",
            )
        )
        left = max(left, right - self.candidate_count)
        count = right - left
        indices = np.zeros(self.candidate_count, dtype=np.int64)
        mask = np.zeros((self.candidate_count, self.features.shape[1]), dtype=bool)
        if count:
            indices[:count] = np.arange(left, right)
            mask[:count] = self.memory_valid[left:right]
        return indices, mask

    def __getitem__(self, index):
        origin = int(self.origins[index])
        start, end = origin - self.seq_len, origin + self.pred_len
        indices, mask = (
            self._candidates(start) if self.training else self._evaluation_candidates
        )
        return {
            "x": torch.from_numpy(self.features[start:origin]),
            "y": torch.from_numpy(self.targets[origin:end, :, None]),
            "target_mask": torch.from_numpy(self.observed[origin:end, :, None]),
            "stat_prior": torch.from_numpy(
                self.stat_table[self.time_slots[origin:end], :, None]
            ),
            "forecast_origin": torch.tensor(origin, dtype=torch.long),
            "query_start": torch.tensor(start, dtype=torch.long),
            "memory_indices": torch.from_numpy(indices),
            "memory_mask": torch.from_numpy(mask),
        }


def _read_legacy_numeric_hdf(path, key):
    """Read old pandas fixed numeric frames without changing source metadata.

    Original DCRNN benchmark files have bytes-valued pandas 0.15 attributes
    that modern pandas cannot decode. Read only the plain numeric arrays and
    datetime axis; do not deserialize their old pickled frequency metadata.
    """
    import tables

    def text(value):
        return (
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
        )

    with tables.open_file(path, mode="r") as handle:
        if key is None:
            groups = [
                group
                for group in handle.walk_groups("/")
                if "pandas_type" in group._v_attrs._v_attrnames
                and text(group._v_attrs.pandas_type) == "frame"
            ]
            if len(groups) != 1:
                raise ValueError(
                    "Legacy HDF requires one fixed DataFrame, or an explicit hdf_key"
                )
            group = groups[0]
        else:
            group = handle.get_node("/" + key.strip("/"))
        attrs = group._v_attrs
        required = {"axis0", "axis1", "block0_items", "block0_values"}
        if (
            text(getattr(attrs, "pandas_type", "")) != "frame"
            or int(getattr(attrs, "nblocks", 0)) != 1
            or not required.issubset(group._v_children)
        ):
            raise ValueError(
                "Unsupported legacy HDF: expected one numeric fixed-frame block"
            )
        if text(getattr(group.axis1._v_attrs, "kind", "")) != "datetime64":
            raise ValueError(
                "Legacy HDF axis must explicitly declare datetime64 nanoseconds"
            )
        if "tz" in group.axis1._v_attrs._v_attrnames:
            raise ValueError(
                "Timezone-bearing legacy HDF should be exported to a modern datetime CSV/HDF"
            )
        columns, block_columns = group.axis0.read(), group.block0_items.read()
        if not np.array_equal(columns, block_columns):
            raise ValueError(
                "Legacy HDF column axis and numeric block order do not match"
            )
        times, values = group.axis1.read(), group.block0_values.read()
        if times.dtype.kind != "i" or values.dtype.kind not in "fiu":
            raise ValueError(
                "Legacy HDF must contain an integer datetime axis and numeric values"
            )
        if values.shape != (len(times), len(columns)):
            raise ValueError("Legacy HDF value shape does not match time and link axes")
        columns = [
            text(value) if isinstance(value, (bytes, np.bytes_)) else value
            for value in columns
        ]
        return pd.DataFrame(
            values, index=pd.to_datetime(times, unit="ns"), columns=columns
        )


def _read_speed_frame(data_path, link_path, hdf_key, start_time, interval_minutes):
    path = Path(data_path)
    generated_timestamps = False
    if path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        try:
            frame = pd.read_hdf(path, key=hdf_key)
        except TypeError:
            frame = _read_legacy_numeric_hdf(path, hdf_key)
        except ImportError as exc:
            raise ImportError(
                "HDF input requires the optional 'tables' package"
            ) from exc
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("HDF must contain a time-by-link pandas DataFrame")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("HDF must have a DatetimeIndex")
        source_format = "wide_hdf"
    else:
        required = {
            "PRCS_YEAR",
            "PRCS_MON",
            "PRCS_DAY",
            "PRCS_HH",
            "PRCS_MIN",
            "LINK_ID",
            "PRCS_SPD",
        }
        header = pd.read_csv(path, nrows=0)
        # Existing TOPIS exports include precomputed/possibly noncausal features.
        # Read only original speed/time columns, and derive fresh safe features.
        frame = (
            pd.read_csv(path, usecols=lambda col: col in required)
            if required.issubset(header.columns)
            else pd.read_csv(path)
        )
        if required.issubset(frame.columns):
            timestamps = pd.to_datetime(
                frame[
                    ["PRCS_YEAR", "PRCS_MON", "PRCS_DAY", "PRCS_HH", "PRCS_MIN"]
                ].rename(
                    columns={
                        "PRCS_YEAR": "year",
                        "PRCS_MON": "month",
                        "PRCS_DAY": "day",
                        "PRCS_HH": "hour",
                        "PRCS_MIN": "minute",
                    }
                )
            )
            frame = frame.assign(datetime=timestamps)
            frame["PRCS_SPD"] = pd.to_numeric(frame["PRCS_SPD"], errors="coerce")
            # Keep rows containing no observed speeds so the original extent survives.
            frame = (
                frame.groupby(["datetime", "LINK_ID"], observed=True)["PRCS_SPD"]
                .mean()
                .unstack()
            )
            source_format = "topis_long_csv"
        else:
            time_columns = [
                col
                for col in frame.columns
                if str(col).lower() in {"datetime", "timestamp", "date", "time"}
            ]
            if (
                not time_columns
                and len(frame.columns)
                and str(frame.columns[0]).startswith("Unnamed:")
            ):
                time_columns = [frame.columns[0]]
            if len(time_columns) == 1:
                frame.index = pd.to_datetime(frame.pop(time_columns[0]), errors="raise")
            elif not time_columns and start_time is not None:
                frame.index = pd.date_range(
                    start_time, periods=len(frame), freq=f"{interval_minutes}min"
                )
                generated_timestamps = True
            else:
                raise ValueError(
                    "Wide CSV needs one datetime/timestamp/date/time column; timestamp-free CSV requires explicit start_time"
                )
            source_format = "wide_csv"
    if frame.index.hasnans:
        raise ValueError("Input contains missing timestamps")
    frame = frame.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    frame = frame.groupby(level=0).mean().sort_index()
    if link_path is not None:
        links = pd.read_csv(link_path)
        if "s_link" not in links.columns:
            raise ValueError("link file must contain an s_link column")

        # Link CSVs sometimes serialize integral IDs as floating point.
        def identifier(value):
            try:
                number = float(value)
                return str(int(number)) if number.is_integer() else str(value)
            except (ValueError, TypeError):
                return str(value)

        requested = {identifier(value) for value in links["s_link"].dropna()}
        frame = frame.loc[
            :, [col for col in frame.columns if identifier(col) in requested]
        ]
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("No speed observations remain after loading/filtering")
    return frame, source_format, generated_timestamps


def _date_range(index, start, end):
    if end <= start:
        return None
    return {
        "start": index[start].isoformat(),
        "end_inclusive": index[end - 1].isoformat(),
        "start_index": start,
        "end_index_exclusive": end,
        "steps": end - start,
    }


def _scaler_metadata(scaler):
    return {
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "scale": scaler.scale_.tolist(),
        "min": scaler.min_.tolist(),
    }


def load_traffic_data(
    data_path: str | Path,
    link_path: str | Path | None = None,
    seq_len: int = 12,
    pred_len: int = 12,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    interval_minutes: int = 5,
    split_mode: str = "holdout",
    fold: int = 0,
    n_folds: int = 5,
    memory_stride: int = 12,
    memory_lookback_steps: int = 8064,
    max_memory_candidates: int = 672,
    zero_is_missing: bool = False,
    hdf_key: str | None = None,
    speed_unit: str = "kmh",
    start_time: str | None = None,
) -> DataBundle:
    """Load TOPIS or wide speed data with preprocessing fitted on training only.

    Inputs are clipped to [0,1], while targets remain unclipped so inverse-transform
    metrics preserve real unseen extremes. Labels retain their raw observation
    mask. Missing timestamps are inserted; no backward fill/interpolation is used.
    """
    if (
        min(
            seq_len,
            pred_len,
            batch_size,
            interval_minutes,
            memory_stride,
            memory_lookback_steps,
            max_memory_candidates,
        )
        <= 0
    ):
        raise ValueError(
            "Lengths, batch size, interval, and memory settings must be positive"
        )
    if 1440 % interval_minutes:
        raise ValueError("interval_minutes must evenly divide one day")
    if speed_unit not in {"kmh", "mph"}:
        raise ValueError("speed_unit must be 'kmh' or 'mph'")
    frame, source_format, generated_timestamps = _read_speed_frame(
        data_path, link_path, hdf_key, start_time, interval_minutes
    )
    original_timestamps = len(frame)
    grid = pd.date_range(frame.index[0], frame.index[-1], freq=f"{interval_minutes}min")
    if not frame.index.isin(grid).all():
        raise ValueError("Timestamps do not lie on the requested regular sampling grid")
    frame = frame.reindex(grid).where(lambda values: values >= 0)
    if zero_is_missing:
        frame = frame.mask(frame == 0)
    if speed_unit == "mph":
        frame = frame * 1.609344
    splits = make_temporal_splits(
        len(frame), train_ratio, val_ratio, split_mode, n_folds
    )
    if not 0 <= fold < len(splits):
        raise ValueError(f"fold must be between 0 and {len(splits) - 1}")
    bounds = splits[fold]
    if (
        bounds.train_end < seq_len + pred_len
        or bounds.val_end - bounds.train_end < pred_len
    ):
        raise ValueError(
            "Insufficient time steps for the selected training and validation split"
        )
    if split_mode == "holdout" and bounds.test_end - bounds.val_end < pred_len:
        raise ValueError("Insufficient time steps in final test split")
    train_available = frame.iloc[: bounds.train_end].notna().any(axis=0)
    dropped_links = [str(col) for col in frame.columns[~train_available]]
    frame = frame.loc[:, train_available]
    if frame.shape[1] == 0:
        raise ValueError("No links have observed speed in the training split")
    observed = frame.notna().to_numpy(dtype=bool, copy=True)
    training = frame.iloc[: bounds.train_end]
    fallback = training.median()
    filled = frame.ffill().fillna(fallback)
    speed_scaler = MinMaxScaler().fit(training.to_numpy())
    targets = speed_scaler.transform(filled.to_numpy()).astype(np.float32)
    delta = filled.diff().fillna(0.0).to_numpy()
    delta_scaler = MinMaxScaler().fit(delta[: bounds.train_end])
    features = np.stack(
        [np.clip(targets, 0, 1), np.clip(delta_scaler.transform(delta), 0, 1)], axis=-1
    ).astype(np.float32)
    slots = ((frame.index.hour * 60 + frame.index.minute) // interval_minutes).to_numpy(
        dtype=np.int64
    )
    train_scaled_observed = pd.DataFrame(
        speed_scaler.transform(training.to_numpy()), index=slots[: bounds.train_end]
    )
    tod = train_scaled_observed.groupby(level=0).mean()
    overall = train_scaled_observed.mean(axis=0).to_numpy(dtype=np.float32)
    stat_table = np.broadcast_to(
        overall, (1440 // interval_minutes, len(overall))
    ).copy()
    stat_table[tod.index.to_numpy(dtype=int)] = tod.fillna(pd.Series(overall)).to_numpy(
        dtype=np.float32
    )
    stat_table = np.clip(stat_table, 0, 1).astype(np.float32)

    memory_starts = np.arange(
        0, bounds.train_end - seq_len - pred_len + 1, memory_stride, dtype=np.int64
    )
    memory_origins = memory_starts + seq_len
    memory_future_ends = memory_origins + pred_len
    memory_windows = torch.from_numpy(
        np.stack([features[s : s + seq_len] for s in memory_starts])
    )
    memory_futures = torch.from_numpy(
        np.stack([targets[o : o + pred_len, :, None] for o in memory_origins])
    )
    memory_valid_array = np.stack(
        [observed[o : o + pred_len].all(axis=0) for o in memory_origins]
    )
    datasets = {}
    counts = np.r_[0, np.cumsum(observed.sum(axis=1), dtype=np.int64)]
    for name, label_start, label_end in (
        ("train", seq_len, bounds.train_end),
        ("val", bounds.train_end, bounds.val_end),
        ("test", bounds.val_end, bounds.test_end),
    ):
        first = max(seq_len, label_start)
        origins = np.arange(first, max(first, label_end - pred_len + 1), dtype=np.int64)
        origins = origins[(counts[origins + pred_len] - counts[origins]) > 0]
        datasets[name] = TrafficWindowDataset(
            features,
            targets,
            observed,
            stat_table,
            slots,
            origins,
            seq_len,
            pred_len,
            memory_future_ends,
            memory_valid_array,
            bounds.train_end,
            name == "train",
            memory_lookback_steps,
            max_memory_candidates,
        )
    if not len(datasets["train"]) or not len(datasets["val"]):
        raise ValueError("Training or validation has no windows with observed targets")
    if split_mode == "holdout" and not len(datasets["test"]):
        raise ValueError("Test split has no windows with observed targets")
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }
    link_ids = [v.item() if isinstance(v, np.generic) else v for v in frame.columns]
    metadata = {
        "source_path": str(Path(data_path).resolve()),
        "source_format": source_format,
        "source_speed_unit": speed_unit,
        "model_and_metric_speed_unit": "km/h",
        "timestamps_generated_from_start_time": generated_timestamps,
        "interval_minutes": interval_minutes,
        "total_steps": len(frame),
        "inserted_missing_steps": len(frame) - original_timestamps,
        "dataset_range": _date_range(frame.index, 0, len(frame)),
        "split_mode": split_mode,
        "fold": bounds.fold,
        "n_folds": n_folds if split_mode == "expanding" else None,
        "split_boundaries": {
            "train_end": bounds.train_end,
            "val_end": bounds.val_end,
            "test_end": bounds.test_end,
        },
        "split_ranges": {
            "train": _date_range(frame.index, 0, bounds.train_end),
            "validation": _date_range(frame.index, bounds.train_end, bounds.val_end),
            "test": _date_range(frame.index, bounds.val_end, bounds.test_end),
        },
        "final_holdout_range": _date_range(
            frame.index, bounds.development_end, len(frame)
        ),
        "split_ratios_requested": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": 1 - train_ratio - val_ratio,
        },
        "samples": {name: len(dataset) for name, dataset in datasets.items()},
        "links": link_ids,
        "link_count": len(link_ids),
        "dropped_no_training_observations": dropped_links,
        "features": ["speed", "delta"],
        "seq_len": seq_len,
        "pred_len": pred_len,
        "observed_fraction": {
            "train": float(observed[: bounds.train_end].mean()),
            "validation": float(observed[bounds.train_end : bounds.val_end].mean()),
            "test": float(observed[bounds.val_end : bounds.test_end].mean())
            if split_mode == "holdout"
            else None,
        },
        "preprocessing": {
            "missing": "nonfinite or negative; zero optionally missing",
            "zero_is_missing": zero_is_missing,
            "imputation": "causal forward fill, then per-link training median for initial holes",
            "fallback_median_kmh": fallback.tolist(),
            "label_mask": "original observed values only",
            "speed_scaler": _scaler_metadata(speed_scaler),
            "delta_scaler": _scaler_metadata(delta_scaler),
            "clip_input_features": True,
            "clip_targets": False,
            "statistical_prior": "observed training speed mean at same time-of-day; per-link observed training mean fallback",
        },
        "memory": {
            "entry_count": len(memory_starts),
            "stride_steps": memory_stride,
            "lookback_steps": memory_lookback_steps,
            "max_candidates": max_memory_candidates,
            "valid_link_entries": int(memory_valid_array.sum()),
            "training_exclusion": "memory future end (exclusive) <= current query start",
            "evaluation": "fixed latest training-only candidates; lookback anchored at training end",
            "missing_future": "exclude same-link entry unless every target step was observed",
            "first_start": int(memory_starts[0]),
            "last_future_end_exclusive": int(memory_future_ends[-1]),
        },
    }
    return DataBundle(
        loaders["train"],
        loaders["val"],
        loaders["test"],
        speed_scaler,
        delta_scaler,
        link_ids,
        ["speed", "delta"],
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]),
        memory_windows,
        memory_futures,
        torch.from_numpy(memory_valid_array),
        memory_starts,
        memory_future_ends,
        metadata,
    )
