#!/usr/bin/env python
"""Standalone resampler for the dexterous grasp stability dataset.

Edit CONFIG below, then run:

    python resample.py

The script builds fixed-window, fixed-resolution, training-ready HDF5 samples
from the raw dataset stored in grasp_data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import copy
import csv
import hashlib
import io
import json
import random
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


CONFIG: dict[str, Any] = {
    "source_root": "data/grasp_data",
    "output_root": "data/resampled_grasp_data",
    "write_policy": "skip",  # "skip" | "overwrite"
    "max_trials": 0,  # 0 = all trials
    "seed": 369,
    "verify_after_build": True,
    "verify_samples": 64,
    "dataset": {
        "require_all_4_fingers": True,
        "fingers": ["thumb", "index", "middle", "ring"],
        "window": {
            # Available phases: reach, grasp, lift, post_lift
            "mode": "duration",  # "duration" | "phase_range"
            # mode="duration": t0 = anchor_phase start; t1 = t0 + duration_s
            "anchor_phase": "grasp",
            "duration_s": 3.0,
            # mode="phase_range": t0 = start_phase start; t1 is end_phase end.
            # end_phase end is next phase start minus end_phase_margin_s.
            # Example grasp -> lift means [grasp_start, post_lift_start - margin].
            "start_phase": "grasp",
            "end_phase": "lift",
            "end_phase_margin_s": 0.1,
            # Used when end_phase has no following phase timestamp.
            "missing_end_policy": "skip",  # "skip" | "fallback_duration"
            "fallback_duration_s": 3.0,
            
        },
        "realsense": {
            "frames": 16,
            "image_size": 256,
        },
        "touch": {
            "camera_frames": 16,
            "camera_image_size": 256,
            "seq_len": 224,
            "audio_bins": 256,
            "imu_preference": ["imu_raw_acc.csv"],
            "imu_abs_reject": 32768,
            "imu_min_valid_rows": 8,
        },
        "proprio": {
            "seq_len": 100,
        },
        "storage": {
            "dtype_numeric": "float16",  # "float16" | "float32"
            "compression": "lzf",  # "none" | "lzf" | "gzip"
        },
    },
}


SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 3
TOUCH_MODALITIES = ("cam", "audio", "imu", "pressure")
ALL_TOUCH_MODALITIES = dict.fromkeys(TOUCH_MODALITIES, True)
DEFAULT_TOUCH_FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring")
DEFAULT_PROPRIO_FINGER_ORDER: tuple[str, ...] = ("thumb", "index", "middle", "ring")
DEFAULT_TILBURG_JOINT_INDICES_BY_FINGER: dict[str, tuple[int, ...]] = {
    "thumb": (0, 1, 2, 3),
    "index": (4, 5, 6, 7),
    "middle": (8, 9, 10, 11),
    "ring": (12, 13, 14, 15),
}
PROPRIO_XARM_DIM: int = 6
TRIAL_INDEX_FIELDS = [
    "trial_id",
    "raw_trial_index",
    "object_name",
    "session_name",
    "label",
    "t0",
    "t1",
    "source_h5_relpath",
    "sample_key",
    "sample_relpath",
]
TOUCH_CAM_VARIANTS = {"raw", "bgsub"}


@dataclass(frozen=True)
class ProprioLayout:
    fingers: tuple[str, ...]
    tilburg_joint_indices: tuple[int, ...]
    tilburg_dim: int
    xarm_dim: int
    proprio_dim: int


@dataclass(frozen=True)
class TrialRecord:
    h5_path: Path
    session_name: str
    object_name: str
    label: int
    t0: float
    t1: float


@dataclass(frozen=True)
class TrialIndexRow:
    trial_id: int
    raw_trial_index: int
    object_name: str
    session_name: str
    label: int
    t0: float
    t1: float
    source_h5_relpath: str
    sample_key: str
    sample_relpath: str

    def to_csv_row(self) -> dict[str, str]:
        return {
            "trial_id": str(int(self.trial_id)),
            "raw_trial_index": str(int(self.raw_trial_index)),
            "object_name": str(self.object_name),
            "session_name": str(self.session_name),
            "label": str(int(self.label)),
            "t0": f"{float(self.t0):.12f}",
            "t1": f"{float(self.t1):.12f}",
            "source_h5_relpath": str(self.source_h5_relpath),
            "sample_key": str(self.sample_key),
            "sample_relpath": str(self.sample_relpath),
        }

    @staticmethod
    def from_csv_row(row: Mapping[str, Any]) -> TrialIndexRow:
        return TrialIndexRow(
            trial_id=int(row["trial_id"]),
            raw_trial_index=int(row["raw_trial_index"]),
            object_name=str(row["object_name"]),
            session_name=str(row["session_name"]),
            label=int(row["label"]),
            t0=float(row["t0"]),
            t1=float(row["t1"]),
            source_h5_relpath=str(row["source_h5_relpath"]),
            sample_key=str(row["sample_key"]),
            sample_relpath=str(row["sample_relpath"]),
        )


def _resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path


def _to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_text(value: object) -> str:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return str(value)


def _to_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _read_csv_dataset(ds: h5py.Dataset) -> pd.DataFrame:
    if ds.dtype.names is None:
        raise ValueError(f"Expected structured table dataset, got dtype={ds.dtype} at {ds.name}")
    arr = ds[()]
    df = pd.DataFrame.from_records(arr)
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].apply(_decode_text)
        elif df[column].dtype.kind in ("S", "U"):
            df[column] = df[column].astype(str)
    return df


def _read_times(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.asarray([], dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").astype(float).to_numpy()


def _latest_indices(times: np.ndarray, queries: np.ndarray) -> np.ndarray:
    if times.size == 0:
        return np.zeros_like(queries, dtype=np.int64)
    idx = np.searchsorted(times, queries, side="right") - 1
    idx = np.clip(idx, 0, len(times) - 1)
    return idx.astype(np.int64)


def _sample_latest(times: np.ndarray, values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    if times.size == 0 or values.size == 0:
        out_dim = int(values.shape[1]) if values.ndim >= 2 else 1
        return np.zeros((queries.size, out_dim), dtype=np.float32)
    idx = _latest_indices(times, queries)
    sampled = values[idx]
    sampled = np.asarray(sampled, dtype=np.float32)
    sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
    if sampled.ndim == 1:
        sampled = sampled.reshape(-1, 1)
    return sampled


def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    out = (arr - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _bytes_from_vlen(ds: h5py.Dataset) -> bytes:
    if len(ds) <= 0:
        return b""
    raw = ds[0]
    if isinstance(raw, np.ndarray):
        return raw.tobytes()
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return np.asarray(raw, dtype=np.uint8).tobytes()


def _decode_jpeg(frames_ds: h5py.Dataset, idx: int, image_size: int) -> np.ndarray | None:
    try:
        data = np.asarray(frames_ds[int(idx)], dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))


def _sanitize_component(value: object) -> str:
    raw = str(value).strip()
    if not raw:
        return "unknown"
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    text = "".join(out).strip("._")
    return text if text else "unknown"


def _normalize_finger_names(value: object, *, field_name: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list/tuple of finger names, got {type(value).__name__}.")
    out: list[str] = []
    for i, raw in enumerate(value):
        name = str(raw).strip().lower()
        if not name:
            raise ValueError(f"{field_name}[{i}] must be a non-empty string.")
        out.append(name)
    if not out and not bool(allow_empty):
        raise ValueError(f"{field_name} must include at least one finger.")
    if len(set(out)) != len(out):
        raise ValueError(f"{field_name} contains duplicates: {out}")
    return out


def _resolve_proprio_layout(dataset_cfg: Mapping[str, Any]) -> ProprioLayout:
    fingers = _normalize_finger_names(
        dataset_cfg.get("fingers", list(DEFAULT_PROPRIO_FINGER_ORDER)),
        field_name="dataset.fingers",
        allow_empty=True,
    )
    tilburg_indices: list[int] = []
    for finger in fingers:
        if finger not in DEFAULT_TILBURG_JOINT_INDICES_BY_FINGER:
            supported = ", ".join(DEFAULT_PROPRIO_FINGER_ORDER)
            raise ValueError(f"Unsupported finger {finger!r} in dataset.fingers. Supported: {supported}")
        tilburg_indices.extend(int(v) for v in DEFAULT_TILBURG_JOINT_INDICES_BY_FINGER[finger])

    tilburg_dim = int(len(tilburg_indices))
    xarm_dim = int(PROPRIO_XARM_DIM)
    return ProprioLayout(
        fingers=tuple(fingers),
        tilburg_joint_indices=tuple(int(v) for v in tilburg_indices),
        tilburg_dim=tilburg_dim,
        xarm_dim=xarm_dim,
        proprio_dim=tilburg_dim + xarm_dim,
    )


def _discover_h5_paths(root: Path) -> list[Path]:
    base = Path(root).expanduser()
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*.h5") if path.is_file())


def _phase_events(grasp_group: h5py.Group) -> list[tuple[str, float]]:
    phases_ds = grasp_group.get("grasp_phases.csv")
    if not isinstance(phases_ds, h5py.Dataset):
        return []
    df = _read_csv_dataset(phases_ds)
    if "phase" not in df.columns or "perf_time" not in df.columns:
        return []
    events: list[tuple[str, float]] = []
    for _, row in df.iterrows():
        phase = str(row.get("phase", "")).strip().lower()
        t = _to_float(row.get("perf_time"))
        if phase and t is not None:
            events.append((phase, float(t)))
    return sorted(events, key=lambda item: item[1])


def _resolve_window_times(grasp_group: h5py.Group, window_cfg: Mapping[str, Any]) -> tuple[float | None, float | None, str | None]:
    events = _phase_events(grasp_group)
    if not events:
        return None, None, "missing_phase_events"

    mode = str(window_cfg.get("mode", "duration")).strip().lower() or "duration"
    if mode == "duration":
        anchor = str(window_cfg.get("anchor_phase", "grasp")).strip().lower()
        duration_s = max(1e-6, float(window_cfg.get("duration_s", 3.0)))
        for phase, t0 in events:
            if phase == anchor:
                return float(t0), float(t0 + duration_s), None
        return None, None, "missing_anchor_phase"

    if mode != "phase_range":
        return None, None, "invalid_window_mode"

    start_phase = str(window_cfg.get("start_phase", "grasp")).strip().lower()
    end_phase = str(window_cfg.get("end_phase", "lift")).strip().lower()
    margin_s = max(0.0, float(window_cfg.get("end_phase_margin_s", 0.1)))
    fallback_duration_s = max(1e-6, float(window_cfg.get("fallback_duration_s", 3.0)))
    missing_end_policy = str(window_cfg.get("missing_end_policy", "skip")).strip().lower()
    if missing_end_policy not in {"skip", "fallback_duration"}:
        return None, None, "invalid_missing_end_policy"

    start_idx = None
    for i, (phase, _t) in enumerate(events):
        if phase == start_phase:
            start_idx = i
            break
    if start_idx is None:
        return None, None, "missing_start_phase"

    t0 = float(events[start_idx][1])
    end_idx = None
    for i in range(start_idx, len(events)):
        if events[i][0] == end_phase:
            end_idx = i
            break
    if end_idx is None:
        return None, None, "missing_end_phase"

    if end_idx + 1 < len(events):
        t1 = float(events[end_idx + 1][1] - margin_s)
    elif missing_end_policy == "fallback_duration":
        t1 = float(t0 + fallback_duration_s)
    else:
        return None, None, "missing_following_phase"

    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return None, None, "invalid_window"
    return float(t0), float(t1), None


def _manual_label(grasp_group: h5py.Group) -> int | None:
    ds = grasp_group.get("grasp_label.csv")
    if not isinstance(ds, h5py.Dataset):
        return None
    arr = ds[()]
    names = arr.dtype.names or ()
    if "method" not in names or "label" not in names:
        return None
    for rec in arr:
        method = _decode_text(rec["method"]).strip().lower()
        if method != "manual":
            continue
        try:
            lv = int(rec["label"])
        except Exception:
            text = _decode_text(rec["label"]).strip()
            if text in {"0", "1"}:
                lv = int(text)
            else:
                continue
        if lv in (0, 1):
            return int(lv)
    return None


def _session_groups(h5: h5py.File) -> list[tuple[str, h5py.Group]]:
    out: list[tuple[str, h5py.Group]] = []
    for name in sorted(h5.keys()):
        obj = h5.get(name)
        if isinstance(obj, h5py.Group):
            out.append((name, obj))
    return out


class MultiModalStabPredDataset:
    """Trial-level raw dataset used to build resampled samples."""

    def __init__(
        self,
        *,
        h5_paths: list[Path],
        cfg: Mapping[str, Any],
        load_rgb: bool,
        load_touch: bool,
        touch_modality_enabled: Mapping[str, Any] | None,
        load_proprio: bool,
    ) -> None:
        self.cfg = cfg
        dcfg = cfg["dataset"]
        self.load_rgb = bool(load_rgb)
        self.load_touch = bool(load_touch)
        self.load_proprio = bool(load_proprio)

        self.fingers = [str(v).strip().lower() for v in dcfg.get("fingers", list(DEFAULT_TOUCH_FINGERS))]
        proprio_layout = _resolve_proprio_layout(dcfg)
        self.proprio_tilburg_joint_indices = tuple(int(v) for v in proprio_layout.tilburg_joint_indices)
        self.proprio_tilburg_dim = int(proprio_layout.tilburg_dim)
        self.proprio_in_dim = int(proprio_layout.proprio_dim)
        self.require_all_4 = bool(dcfg.get("require_all_4_fingers", True))
        self.window_cfg = dcfg.get("window", {})
        if not isinstance(self.window_cfg, Mapping):
            self.window_cfg = {}

        rcfg = dcfg["realsense"]
        self.rgb_frames = int(rcfg.get("frames", 16))
        self.rgb_image_size = int(rcfg.get("image_size", 256))

        tcfg = dcfg["touch"]
        self.touch_cam_frames = int(tcfg.get("camera_frames", 16))
        self.touch_camera_image_size = int(tcfg.get("camera_image_size", self.rgb_image_size))
        self.touch_seq_len = int(tcfg.get("seq_len", 224))
        self.touch_audio_bins = int(tcfg.get("audio_bins", 256))
        self.imu_preference = [str(v) for v in tcfg.get("imu_preference", [])]
        self.imu_abs_reject = max(0.0, float(tcfg.get("imu_abs_reject", 32768.0)))
        self.imu_min_valid_rows = max(1, int(tcfg.get("imu_min_valid_rows", 8)))
        self.touch_modality_enabled: dict[str, bool] = dict.fromkeys(TOUCH_MODALITIES, False)
        if self.load_touch:
            if len(self.fingers) <= 0:
                raise ValueError("load_touch=True requires at least one configured finger.")
            if not isinstance(touch_modality_enabled, Mapping):
                raise ValueError("touch_modality_enabled must be provided when load_touch=True.")
            missing = [name for name in TOUCH_MODALITIES if name not in touch_modality_enabled]
            if missing:
                raise ValueError("touch_modality_enabled is missing required keys: " + ", ".join(missing))
            self.touch_modality_enabled = {name: bool(touch_modality_enabled.get(name, False)) for name in TOUCH_MODALITIES}
            if not any(self.touch_modality_enabled.values()):
                raise ValueError("load_touch=True but all touch modalities are disabled.")

        self.prop_seq_len = int(dcfg["proprio"].get("seq_len", 100))
        self.records, self.skip_stats = self._scan_records([Path(p) for p in h5_paths])
        if not self.records:
            raise ValueError("No valid manual-labeled grasp trials found.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        rec = self.records[int(idx)]
        with h5py.File(rec.h5_path, "r") as h5:
            session = h5.get(rec.session_name)
            if not isinstance(session, h5py.Group):
                raise KeyError(f"Session not found: {rec.session_name} in {rec.h5_path}")

            inputs: dict[str, torch.Tensor] = {}
            if self.load_rgb:
                rgb_frames, rgb_mask = self._extract_rgb(session, rec.t0, rec.t1)
                inputs["rgb_frames"] = torch.from_numpy(rgb_frames)
                inputs["rgb_mask"] = torch.from_numpy(rgb_mask)

            if self.load_touch:
                touch_pack = self._extract_touch(session, rec.t0, rec.t1)
                inputs["touch_cam"] = torch.from_numpy(touch_pack["cam"])
                inputs["touch_audio"] = torch.from_numpy(touch_pack["audio"])
                inputs["touch_imu"] = torch.from_numpy(touch_pack["imu"])
                inputs["touch_pressure"] = torch.from_numpy(touch_pack["pressure"])
                inputs["touch_modality_mask"] = torch.from_numpy(touch_pack["mask"])

            if self.load_proprio:
                proprio_seq, proprio_mask = self._extract_proprio(session, rec.t0, rec.t1)
                inputs["proprio_seq"] = torch.from_numpy(proprio_seq)
                inputs["proprio_modality_mask"] = torch.from_numpy(proprio_mask)

        label = torch.tensor(float(rec.label), dtype=torch.float32)
        return inputs, label

    def _scan_records(self, h5_paths: Iterable[Path]) -> tuple[list[TrialRecord], dict[str, int]]:
        records: list[TrialRecord] = []
        skipped: dict[str, int] = {}

        for h5_path in h5_paths:
            try:
                with h5py.File(h5_path, "r") as h5:
                    for session_name, session in _session_groups(h5):
                        grasp = session.get("grasp")
                        if not isinstance(grasp, h5py.Group):
                            skipped["missing_grasp"] = skipped.get("missing_grasp", 0) + 1
                            continue

                        t0, t1, window_reason = _resolve_window_times(grasp, self.window_cfg)
                        if t0 is None or t1 is None:
                            key = window_reason or "missing_window"
                            skipped[key] = skipped.get(key, 0) + 1
                            continue

                        label = _manual_label(grasp)
                        if label is None:
                            skipped["missing_manual_label"] = skipped.get("missing_manual_label", 0) + 1
                            continue

                        if self.require_all_4 and self.load_touch and not self._has_all_fingers(session):
                            skipped["missing_required_fingers"] = skipped.get("missing_required_fingers", 0) + 1
                            continue

                        records.append(
                            TrialRecord(
                                h5_path=Path(h5_path),
                                session_name=str(session_name),
                                object_name=str(h5_path.parent.name),
                                label=int(label),
                                t0=float(t0),
                                t1=float(t1),
                            )
                        )
            except Exception:
                skipped["h5_open_error"] = skipped.get("h5_open_error", 0) + 1

        return records, skipped

    def _has_all_fingers(self, session: h5py.Group) -> bool:
        digit = self._digit360_root(session)
        if not isinstance(digit, h5py.Group):
            return False
        keys = {str(k).strip().lower() for k in digit}
        return all(finger in keys for finger in self.fingers)

    @staticmethod
    def _digit360_root(session: h5py.Group) -> h5py.Group | None:
        opentouch = session.get("opentouch")
        if not isinstance(opentouch, h5py.Group):
            return None
        digit360 = opentouch.get("digit360")
        return digit360 if isinstance(digit360, h5py.Group) else None

    def _extract_rgb(self, session: h5py.Group, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        zeros = np.zeros((self.rgb_frames, 3, self.rgb_image_size, self.rgb_image_size), dtype=np.float32)
        mask = np.ones((1,), dtype=np.float32)

        rs = session.get("realsense")
        if not isinstance(rs, h5py.Group):
            return zeros, mask
        rgb_group = rs.get("rgb")
        if not isinstance(rgb_group, h5py.Group):
            return zeros, mask

        meta_ds = rgb_group.get("rgb.csv")
        frames_ds = rgb_group.get("frames")
        if not isinstance(meta_ds, h5py.Dataset) or not isinstance(frames_ds, h5py.Dataset):
            return zeros, mask

        meta = _read_csv_dataset(meta_ds)
        times = _read_times(meta, "host_perf")
        if times.size == 0:
            return zeros, mask

        queries = np.linspace(float(t0), float(t1), num=self.rgb_frames, endpoint=True, dtype=np.float64)
        idx = _latest_indices(times, queries)

        frames: list[np.ndarray] = []
        ok = 0
        for i in idx:
            img = _decode_jpeg(frames_ds, int(i), self.rgb_image_size)
            if img is None:
                frames.append(np.zeros((3, self.rgb_image_size, self.rgb_image_size), dtype=np.float32))
            else:
                ok += 1
                frames.append(img.astype(np.float32))

        out = np.stack(frames, axis=0).astype(np.float32)
        mask[...] = 0.0 if ok > 0 else 1.0
        return out, mask

    def _empty_touch_arrays(self) -> dict[str, np.ndarray]:
        nf = len(self.fingers)
        return {
            "cam": np.zeros(
                (nf, self.touch_cam_frames, 3, self.touch_camera_image_size, self.touch_camera_image_size),
                dtype=np.float32,
            ),
            "audio": np.zeros((nf, self.touch_seq_len, self.touch_audio_bins), dtype=np.float32),
            "imu": np.zeros((nf, self.touch_seq_len, 3), dtype=np.float32),
            "pressure": np.zeros((nf, self.touch_seq_len, 1), dtype=np.float32),
            "mask": np.ones((nf, self.touch_seq_len, 4), dtype=np.float32),
        }

    def _fill_finger_touch(
        self,
        fg: h5py.Group,
        fi: int,
        out: dict[str, np.ndarray],
        *,
        t0: float,
        t1: float,
        cam_queries: np.ndarray,
        seq_queries: np.ndarray,
    ) -> None:
        """Fill row `fi` of the touch arrays from a single fingertip group.

        The mask channel is 1.0 where a modality is missing, so an all-ones row means
        the fingertip contributed nothing.
        """
        if self.touch_modality_enabled["cam"]:
            cam_meta, cam_frames = self._load_touch_camera(fg)
            values, valid = self._touch_cam_sequence(cam_meta, cam_frames, t0=t0, t1=t1)
            out["cam"][fi] = values
            if np.any(valid):
                cam_idx = _latest_indices(cam_queries, seq_queries)
                out["mask"][fi, :, TOUCH_MODALITIES.index("cam")] = (~valid[cam_idx]).astype(np.float32)

        if self.touch_modality_enabled["audio"]:
            chunks_df, wav_audio = self._load_touch_audio(fg)
            values, present = self._touch_audio_sequence(chunks_df, wav_audio, t0=t0, t1=t1)
            out["audio"][fi] = values
            out["mask"][fi, :, TOUCH_MODALITIES.index("audio")] = 1.0 if present else 0.0

        if self.touch_modality_enabled["imu"]:
            values, present = self._touch_series_sequence(
                self._load_touch_imu(fg),
                t0=t0,
                t1=t1,
                seq_len=self.touch_seq_len,
                value_cols=("x", "y", "z"),
                abs_reject=self.imu_abs_reject,
                min_valid_rows=self.imu_min_valid_rows,
            )
            out["imu"][fi] = values
            out["mask"][fi, :, TOUCH_MODALITIES.index("imu")] = 1.0 if present else 0.0

        if self.touch_modality_enabled["pressure"]:
            values, present = self._touch_series_sequence(
                self._load_touch_pressure(fg),
                t0=t0,
                t1=t1,
                seq_len=self.touch_seq_len,
                value_cols=("pressure",),
            )
            out["pressure"][fi] = values
            out["mask"][fi, :, TOUCH_MODALITIES.index("pressure")] = 1.0 if present else 0.0

    def _extract_touch(self, session: h5py.Group, t0: float, t1: float) -> dict[str, np.ndarray]:
        out = self._empty_touch_arrays()
        digit = self._digit360_root(session)
        if not isinstance(digit, h5py.Group):
            return out

        cam_queries = np.linspace(float(t0), float(t1), num=self.touch_cam_frames, endpoint=True, dtype=np.float64)
        seq_queries = np.linspace(float(t0), float(t1), num=self.touch_seq_len, endpoint=True, dtype=np.float64)
        for fi, finger in enumerate(self.fingers):
            fg = digit.get(finger)
            if isinstance(fg, h5py.Group):
                self._fill_finger_touch(
                    fg, fi, out, t0=t0, t1=t1, cam_queries=cam_queries, seq_queries=seq_queries
                )
        return {key: value.astype(np.float32) for key, value in out.items()}

    def _load_touch_camera(self, finger_group: h5py.Group) -> tuple[pd.DataFrame | None, h5py.Dataset | None]:
        camera = finger_group.get("camera")
        if not isinstance(camera, h5py.Group):
            return None, None
        meta_ds = camera.get("camera.csv")
        frames_ds = camera.get("frames")
        if not isinstance(meta_ds, h5py.Dataset) or not isinstance(frames_ds, h5py.Dataset):
            return None, None
        return _read_csv_dataset(meta_ds), frames_ds

    def _load_touch_audio(self, finger_group: h5py.Group) -> tuple[pd.DataFrame | None, np.ndarray | None]:
        audio = finger_group.get("audio")
        if not isinstance(audio, h5py.Group):
            return None, None
        chunks_ds = audio.get("chunks.csv")
        wav_ds = audio.get("wav")
        if not isinstance(chunks_ds, h5py.Dataset) or not isinstance(wav_ds, h5py.Dataset):
            return None, None

        chunks_df = _read_csv_dataset(chunks_ds)
        wav_bytes = _bytes_from_vlen(wav_ds)
        if not wav_bytes:
            return chunks_df, None

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                nch = max(1, int(wf.getnchannels()))
                sampwidth = int(wf.getsampwidth())
                nframes = int(wf.getnframes())
                raw = wf.readframes(nframes)
            if sampwidth != 2:
                return chunks_df, None
            pcm = np.frombuffer(raw, dtype=np.int16)
            if pcm.size == 0:
                return chunks_df, None
            if nch > 1:
                pcm = pcm.reshape(-1, nch).mean(axis=1)
            audio_arr = (pcm.astype(np.float32) / 32768.0).reshape(-1)
            return chunks_df, audio_arr
        except Exception:
            return chunks_df, None

    def _load_touch_imu(self, finger_group: h5py.Group) -> pd.DataFrame | None:
        serial = finger_group.get("serial")
        if not isinstance(serial, h5py.Group):
            return None

        for name in self.imu_preference or ["imu_raw_acc.csv"]:
            ds = serial.get(str(name))
            if isinstance(ds, h5py.Dataset):
                return _read_csv_dataset(ds)

        for name in sorted(serial.keys()):
            key = str(name).strip().lower()
            if "imu_raw_acc" in key and key.endswith(".csv"):
                ds = serial.get(str(name))
                if isinstance(ds, h5py.Dataset):
                    return _read_csv_dataset(ds)
        return None

    def _load_touch_pressure(self, finger_group: h5py.Group) -> pd.DataFrame | None:
        serial = finger_group.get("serial")
        if not isinstance(serial, h5py.Group):
            return None
        ds = serial.get("pressure.csv")
        if not isinstance(ds, h5py.Dataset):
            return None
        return _read_csv_dataset(ds)

    def _touch_cam_sequence(
        self,
        cam_meta: pd.DataFrame | None,
        frames_ds: h5py.Dataset | None,
        *,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        zeros = np.zeros(
            (self.touch_cam_frames, 3, self.touch_camera_image_size, self.touch_camera_image_size),
            dtype=np.float32,
        )
        valid = np.zeros((self.touch_cam_frames,), dtype=bool)
        if cam_meta is None or frames_ds is None:
            return zeros, valid

        times = _read_times(cam_meta, "time_perf")
        if times.size == 0:
            return zeros, valid

        queries = np.linspace(float(t0), float(t1), num=self.touch_cam_frames, endpoint=True, dtype=np.float64)
        idx = _latest_indices(times, queries)

        for qi, end_idx in enumerate(idx):
            frame = _decode_jpeg(frames_ds, int(end_idx), self.touch_camera_image_size)
            if frame is None:
                continue
            zeros[qi] = frame.astype(np.float32)
            valid[qi] = True

        return zeros, valid

    def _touch_audio_sequence(
        self,
        chunks_df: pd.DataFrame | None,
        wav_audio: np.ndarray | None,
        *,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, bool]:
        zeros = np.zeros((self.touch_seq_len, self.touch_audio_bins), dtype=np.float32)
        if chunks_df is None or wav_audio is None:
            return zeros, True
        req_cols = {"time_perf", "start_sample", "end_sample"}
        if not req_cols.issubset(set(chunks_df.columns)):
            return zeros, True

        times = pd.to_numeric(chunks_df["time_perf"], errors="coerce").to_numpy(dtype=float)
        start_idx = pd.to_numeric(chunks_df["start_sample"], errors="coerce").to_numpy(dtype=float)
        end_idx = pd.to_numeric(chunks_df["end_sample"], errors="coerce").to_numpy(dtype=float)

        keep = np.isfinite(times) & np.isfinite(start_idx) & np.isfinite(end_idx) & (times >= float(t0)) & (times <= float(t1))
        if not np.any(keep):
            return zeros, True

        s = int(np.clip(np.nanmin(start_idx[keep]), 0, max(len(wav_audio) - 1, 0)))
        e = int(np.clip(np.nanmax(end_idx[keep]), s + 1, len(wav_audio)))
        if e <= s:
            return zeros, True

        seg = wav_audio[s:e]
        if seg.size < 64:
            seg = np.pad(seg, (0, 64 - seg.size), mode="constant")

        wav_t = torch.tensor(seg, dtype=torch.float32)
        win = torch.hann_window(512, dtype=torch.float32)
        spec = torch.stft(
            wav_t,
            n_fft=512,
            hop_length=128,
            win_length=512,
            window=win,
            return_complex=True,
        )
        mag = torch.abs(spec).clamp_min(1e-6).log()
        mag = mag.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            mag,
            size=(self.touch_audio_bins, self.touch_seq_len),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        feat = resized.transpose(0, 1)
        feat = (feat - feat.mean()) / feat.std().clamp_min(1e-6)
        return feat.cpu().numpy().astype(np.float32), False

    def _touch_series_sequence(
        self,
        df: pd.DataFrame | None,
        *,
        t0: float,
        t1: float,
        seq_len: int,
        value_cols: tuple[str, ...],
        abs_reject: float | None = None,
        min_valid_rows: int = 1,
    ) -> tuple[np.ndarray, bool]:
        zeros = np.zeros((seq_len, len(value_cols)), dtype=np.float32)
        if df is None:
            return zeros, True
        if "time_perf" not in df.columns:
            return zeros, True
        for col in value_cols:
            if col not in df.columns:
                return zeros, True

        times = pd.to_numeric(df["time_perf"], errors="coerce").to_numpy(dtype=float)
        vals = np.stack(
            [pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in value_cols],
            axis=1,
        )

        values_finite = np.isfinite(vals).all(axis=1)
        keep = np.isfinite(times) & values_finite & (times >= float(t0)) & (times <= float(t1))
        if abs_reject is not None:
            reject = float(abs_reject)
            if np.isfinite(reject) and reject > 0.0:
                keep = keep & (np.max(np.abs(vals), axis=1) <= reject)
        if not np.any(keep):
            return zeros, True
        if int(np.sum(keep)) < max(1, int(min_valid_rows)):
            return zeros, True

        queries = np.linspace(float(t0), float(t1), num=seq_len, endpoint=True, dtype=np.float64)
        sampled = _sample_latest(times[keep], vals[keep], queries)
        sampled = _zscore(sampled)
        return sampled.astype(np.float32), False

    def _extract_proprio(self, session: h5py.Group, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        out = np.zeros((self.prop_seq_len, self.proprio_in_dim), dtype=np.float32)
        mask = np.ones((2,), dtype=np.float32)
        queries = np.linspace(float(t0), float(t1), num=self.prop_seq_len, endpoint=True, dtype=np.float64)

        til_vals, til_ok = self._extract_tilburg(session, queries)
        if til_ok:
            out[:, : self.proprio_tilburg_dim] = til_vals
            mask[0] = 0.0

        xarm_vals, xarm_ok = self._extract_xarm_ee(session, queries)
        if xarm_ok:
            out[:, self.proprio_tilburg_dim :] = xarm_vals
            mask[1] = 0.0

        return out, mask

    def _extract_tilburg(self, session: h5py.Group, queries: np.ndarray) -> tuple[np.ndarray, bool]:
        zeros = np.zeros((queries.size, self.proprio_tilburg_dim), dtype=np.float32)
        tilburg = session.get("tilburg")
        if not isinstance(tilburg, h5py.Group):
            return zeros, False

        ds = None
        for name in sorted(tilburg.keys()):
            if str(name).startswith("tilburg_pos"):
                candidate = tilburg.get(name)
                if isinstance(candidate, h5py.Dataset):
                    ds = candidate
                    break
        if not isinstance(ds, h5py.Dataset):
            return zeros, False

        df = _read_csv_dataset(ds)
        if "perf_time" not in df.columns:
            return zeros, False

        joint_cols = [f"finger_{i}_deg" for i in range(16)]
        if not all(col in df.columns for col in joint_cols):
            return zeros, False

        times = _read_times(df, "perf_time")
        vals_full = np.stack([pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in joint_cols], axis=1)
        vals = vals_full[:, list(self.proprio_tilburg_joint_indices)]
        keep = np.isfinite(times) & np.isfinite(vals).all(axis=1)
        if not np.any(keep):
            return zeros, False
        sampled = _sample_latest(times[keep], vals[keep], queries)
        sampled = _zscore(sampled)
        return sampled.astype(np.float32), True

    def _extract_xarm_ee(self, session: h5py.Group, queries: np.ndarray) -> tuple[np.ndarray, bool]:
        zeros = np.zeros((queries.size, 6), dtype=np.float32)
        xarm = session.get("xarm")
        if not isinstance(xarm, h5py.Group):
            return zeros, False

        ds = xarm.get("xarm_eepos.csv")
        if not isinstance(ds, h5py.Dataset):
            return zeros, False

        df = _read_csv_dataset(ds)
        req_cols = ["perf_time", "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]
        if not all(col in df.columns for col in req_cols):
            return zeros, False

        times = _read_times(df, "perf_time")
        vals = np.stack(
            [
                pd.to_numeric(df["x_mm"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(df["y_mm"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(df["z_mm"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(df["roll_deg"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(df["pitch_deg"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(df["yaw_deg"], errors="coerce").to_numpy(dtype=float),
            ],
            axis=1,
        )
        keep = np.isfinite(times) & np.isfinite(vals).all(axis=1)
        if not np.any(keep):
            return zeros, False
        sampled = _sample_latest(times[keep], vals[keep], queries)
        sampled = _zscore(sampled)
        return sampled.astype(np.float32), True


def _resolve_storage_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.get("dataset", {})
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}
    storage_cfg = data_cfg.get("storage", {})
    if not isinstance(storage_cfg, Mapping):
        storage_cfg = {}

    dtype_numeric = str(storage_cfg.get("dtype_numeric", "float16")).strip().lower() or "float16"
    if dtype_numeric not in {"float16", "float32"}:
        raise ValueError(f"dataset.storage.dtype_numeric must be float16|float32, got {dtype_numeric!r}")

    compression = str(storage_cfg.get("compression", "lzf")).strip().lower() or "lzf"
    if compression in {"off", "none", "null"}:
        compression = "none"
    if compression not in {"none", "lzf", "gzip"}:
        raise ValueError(f"dataset.storage.compression must be none|lzf|gzip, got {compression!r}")

    write_policy = str(cfg.get("write_policy", "skip")).strip().lower() or "skip"
    if write_policy not in {"skip", "overwrite"}:
        raise ValueError(f"write_policy must be one of skip|overwrite, got {write_policy!r}")

    return {
        "root": _resolve_path(cfg.get("output_root", "data/resampled_grasp_data")),
        "dtype_numeric": dtype_numeric,
        "compression": compression,
        "write_policy": write_policy,
        "verify_after_build": _to_bool(cfg.get("verify_after_build", True), default=True),
    }


def _compute_config_fingerprint(cfg: Mapping[str, Any]) -> str:
    dataset_cfg = cfg.get("dataset", {})
    if not isinstance(dataset_cfg, Mapping):
        dataset_cfg = {}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "require_all_4_fingers": dataset_cfg.get("require_all_4_fingers"),
            "fingers": dataset_cfg.get("fingers"),
            "window": dataset_cfg.get("window"),
            "realsense": dataset_cfg.get("realsense"),
            "touch": dataset_cfg.get("touch"),
            "proprio": dataset_cfg.get("proprio"),
            "storage": dataset_cfg.get("storage"),
        },
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_sample_key(
    *,
    source_h5_relpath: str,
    session_name: str,
    t0: float,
    t1: float,
    label: int,
) -> str:
    payload = {
        "source_h5_relpath": str(source_h5_relpath),
        "session_name": str(session_name),
        "t0": f"{float(t0):.9f}",
        "t1": f"{float(t1):.9f}",
        "label": int(label),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sample_relpath(*, object_name: str, session_name: str, sample_key: str) -> Path:
    obj = _sanitize_component(object_name)
    sess = _sanitize_component(session_name)
    skey = _sanitize_component(str(sample_key).lower())
    return Path("samples") / obj / f"{sess}__{skey}.h5"


def _write_trial_index_csv(root: Path, rows: list[TrialIndexRow]) -> Path:
    path = root / "trial_index.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_INDEX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())
    return path


def _write_meta_json(root: Path, payload: Mapping[str, Any]) -> Path:
    path = root / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=True)
    return path


def _meta_payload(
    *,
    cfg: Mapping[str, Any],
    source_dataset_root: Path,
    root: Path,
    num_trials: int,
) -> dict[str, Any]:
    dataset_cfg = cfg.get("dataset", {})
    if not isinstance(dataset_cfg, Mapping):
        dataset_cfg = {}
    dataset_fingers = _normalize_finger_names(
        dataset_cfg.get("fingers", list(DEFAULT_TOUCH_FINGERS)),
        field_name="dataset.fingers",
    )
    proprio_layout = _resolve_proprio_layout(dataset_cfg)
    realsense_cfg = dataset_cfg.get("realsense", {})
    touch_cfg = dataset_cfg.get("touch", {})
    if not isinstance(realsense_cfg, Mapping):
        realsense_cfg = {}
    if not isinstance(touch_cfg, Mapping):
        touch_cfg = {}
    return {
        "schema_version": int(SCHEMA_VERSION),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_trials": int(num_trials),
        "source_dataset_root": str(source_dataset_root),
        "resampled_root": str(root),
        "config_fingerprint": _compute_config_fingerprint(cfg),
        "config": cfg,
        "dataset_fingers": list(dataset_fingers),
        "dataset_tilburg_joint_indices": [int(v) for v in proprio_layout.tilburg_joint_indices],
        "dataset_proprio_xarm_dim": int(proprio_layout.xarm_dim),
        "dataset_proprio_dim": int(proprio_layout.proprio_dim),
        "rgb_image_size": int(realsense_cfg.get("image_size", 256)),
        "touch_camera_image_size": int(touch_cfg.get("camera_image_size", realsense_cfg.get("image_size", 256))),
        "window": dict(dataset_cfg.get("window", {})) if isinstance(dataset_cfg.get("window", {}), Mapping) else {},
    }


def _compression_kwargs(compression: str) -> dict[str, Any]:
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    return {"compression": "lzf", "shuffle": True}


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    arr = value.detach().cpu().numpy()
    return np.asarray(arr)


def _encode_uint8_image_tensor(arr: np.ndarray) -> np.ndarray:
    clipped = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return np.asarray(np.round(clipped * 255.0), dtype=np.uint8)


def _encode_mask_uint8(arr: np.ndarray) -> np.ndarray:
    return np.asarray((arr.astype(np.float32) >= 0.5), dtype=np.uint8)


def _encode_numeric(arr: np.ndarray, dtype_numeric: str) -> np.ndarray:
    dt = np.float16 if dtype_numeric == "float16" else np.float32
    out = np.asarray(arr, dtype=dt)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _touch_cam_background_subtraction(touch_cam_raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(touch_cam_raw, dtype=np.float32)
    if arr.ndim != 5:
        raise ValueError(f"touch_cam must be rank-5 [F,T,C,H,W], got shape={tuple(arr.shape)}")
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.shape[1] <= 0:
        return out
    for fi in range(arr.shape[0]):
        bg = arr[fi, 0]
        out[fi] = np.abs(arr[fi] - bg[None, ...])
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def _write_sample_h5(
    *,
    path: Path,
    inputs: Mapping[str, torch.Tensor],
    label: torch.Tensor,
    object_name: str,
    session_name: str,
    source_h5_relpath: str,
    t0: float,
    t1: float,
    dtype_numeric: str,
    compression: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    cmp_kw = _compression_kwargs(compression)
    utf8 = h5py.string_dtype(encoding="utf-8")

    rgb_frames = _encode_uint8_image_tensor(_to_numpy(inputs["rgb_frames"]))
    rgb_mask = _encode_mask_uint8(_to_numpy(inputs["rgb_mask"]))
    touch_cam_raw = _to_numpy(inputs["touch_cam"])
    touch_cam_bgsub = _touch_cam_background_subtraction(touch_cam_raw)
    touch_cam_raw_u8 = _encode_uint8_image_tensor(touch_cam_raw)
    touch_cam_bgsub_u8 = _encode_uint8_image_tensor(touch_cam_bgsub)
    touch_audio = _encode_numeric(_to_numpy(inputs["touch_audio"]), dtype_numeric=dtype_numeric)
    touch_imu = _encode_numeric(_to_numpy(inputs["touch_imu"]), dtype_numeric=dtype_numeric)
    touch_pressure = _encode_numeric(_to_numpy(inputs["touch_pressure"]), dtype_numeric=dtype_numeric)
    touch_modality_mask = _encode_mask_uint8(_to_numpy(inputs["touch_modality_mask"]))
    proprio_seq = _encode_numeric(_to_numpy(inputs["proprio_seq"]), dtype_numeric=dtype_numeric)
    proprio_modality_mask = _encode_mask_uint8(_to_numpy(inputs["proprio_modality_mask"]))

    label_val = int(float(label.detach().item()) >= 0.5)

    with h5py.File(tmp_path, "w") as h5:
        data = h5.create_group("data")
        data.create_dataset("rgb_frames", data=rgb_frames, **cmp_kw)
        data.create_dataset("rgb_mask", data=rgb_mask, **cmp_kw)
        data.create_dataset("touch_cam_raw", data=touch_cam_raw_u8, **cmp_kw)
        data.create_dataset("touch_cam_bgsub", data=touch_cam_bgsub_u8, **cmp_kw)
        data.create_dataset("touch_audio", data=touch_audio, **cmp_kw)
        data.create_dataset("touch_imu", data=touch_imu, **cmp_kw)
        data.create_dataset("touch_pressure", data=touch_pressure, **cmp_kw)
        data.create_dataset("touch_modality_mask", data=touch_modality_mask, **cmp_kw)
        data.create_dataset("proprio_seq", data=proprio_seq, **cmp_kw)
        data.create_dataset("proprio_modality_mask", data=proprio_modality_mask, **cmp_kw)

        meta = h5.create_group("meta")
        meta.create_dataset("label", data=np.int8(label_val))
        meta.create_dataset("t0", data=np.float64(t0))
        meta.create_dataset("t1", data=np.float64(t1))
        meta.create_dataset("object_name", data=np.asarray(str(object_name), dtype=utf8))
        meta.create_dataset("session_name", data=np.asarray(str(session_name), dtype=utf8))
        meta.create_dataset("source_h5_relpath", data=np.asarray(str(source_h5_relpath), dtype=utf8))

    tmp_path.replace(path)


def _normalize_touch_cam_variant(value: object, *, field_name: str) -> str:
    raw = str(value).strip().lower()
    alias = {
        "raw": "raw",
        "bgsub": "bgsub",
        "background_subtraction": "bgsub",
        "background-subtraction": "bgsub",
    }
    out = alias.get(raw)
    if out is None:
        allowed = ", ".join(sorted(TOUCH_CAM_VARIANTS))
        raise ValueError(f"{field_name} must be one of {{{allowed}}}, got {value!r}")
    return out


def _read_sample_h5(
    *,
    path: Path,
    load_rgb: bool,
    load_touch: bool,
    touch_cam_variant: str,
    touch_modality_enabled: Mapping[str, Any] | None,
    load_proprio: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing resampled sample: {path}")

    variant = _normalize_touch_cam_variant(touch_cam_variant, field_name="touch_cam_variant")
    inputs: dict[str, torch.Tensor] = {}
    with h5py.File(path, "r") as h5:
        data = h5["data"]
        if load_rgb:
            rgb_frames = np.asarray(data["rgb_frames"][()], dtype=np.float32) / 255.0
            rgb_mask = np.asarray(data["rgb_mask"][()], dtype=np.uint8).astype(np.float32)
            inputs["rgb_frames"] = torch.from_numpy(rgb_frames)
            inputs["rgb_mask"] = torch.from_numpy(rgb_mask)

        if load_touch:
            cam_key = f"touch_cam_{variant}"
            if cam_key not in data:
                raise KeyError(f"Missing dataset '{cam_key}' in resampled sample: {path}")
            touch_cam = np.asarray(data[cam_key][()], dtype=np.float32) / 255.0
            touch_audio = np.asarray(data["touch_audio"][()], dtype=np.float32)
            touch_imu = np.asarray(data["touch_imu"][()], dtype=np.float32)
            touch_pressure = np.asarray(data["touch_pressure"][()], dtype=np.float32)
            touch_modality_mask = np.asarray(data["touch_modality_mask"][()], dtype=np.uint8).astype(np.float32)

            enabled = dict.fromkeys(TOUCH_MODALITIES, True)
            if isinstance(touch_modality_enabled, Mapping):
                for key in enabled:
                    enabled[key] = bool(touch_modality_enabled.get(key, enabled[key]))

            if not enabled["cam"]:
                touch_cam[...] = 0.0
                touch_modality_mask[..., 0] = 1.0
            if not enabled["audio"]:
                touch_audio[...] = 0.0
                touch_modality_mask[..., 1] = 1.0
            if not enabled["imu"]:
                touch_imu[...] = 0.0
                touch_modality_mask[..., 2] = 1.0
            if not enabled["pressure"]:
                touch_pressure[...] = 0.0
                touch_modality_mask[..., 3] = 1.0

            inputs["touch_cam"] = torch.from_numpy(touch_cam)
            inputs["touch_audio"] = torch.from_numpy(touch_audio)
            inputs["touch_imu"] = torch.from_numpy(touch_imu)
            inputs["touch_pressure"] = torch.from_numpy(touch_pressure)
            inputs["touch_modality_mask"] = torch.from_numpy(touch_modality_mask)

        if load_proprio:
            proprio_seq = np.asarray(data["proprio_seq"][()], dtype=np.float32)
            proprio_modality_mask = np.asarray(data["proprio_modality_mask"][()], dtype=np.uint8).astype(np.float32)
            inputs["proprio_seq"] = torch.from_numpy(proprio_seq)
            inputs["proprio_modality_mask"] = torch.from_numpy(proprio_modality_mask)

        label_raw = h5["meta"]["label"][()]
        label = torch.tensor(float(int(label_raw)), dtype=torch.float32)

    return inputs, label


def _inputs_all_finite(inputs: dict[str, torch.Tensor]) -> tuple[bool, str | None]:
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if bool(torch.isfinite(value).all().item()):
            continue
        return False, name
    return True, None


def _compare_inputs(raw: dict[str, torch.Tensor], sample: dict[str, torch.Tensor]) -> tuple[bool, str | None]:
    for key, raw_tensor in raw.items():
        if key not in sample:
            return False, f"missing_key:{key}"
        sample_tensor = sample[key]
        if tuple(raw_tensor.shape) != tuple(sample_tensor.shape):
            return False, f"shape:{key}"

        if key in {"rgb_frames", "touch_cam"}:
            delta = (raw_tensor.float() - sample_tensor.float()).abs().max().item()
            if float(delta) > (1.5 / 255.0):
                return False, f"image_delta:{key}:{delta:.6f}"
            continue

        if key.endswith("_mask"):
            if not torch.equal((raw_tensor >= 0.5), (sample_tensor >= 0.5)):
                return False, f"mask:{key}"
            continue

        delta = (raw_tensor.float() - sample_tensor.float()).abs().max().item()
        if float(delta) > 5e-3:
            return False, f"numeric_delta:{key}:{delta:.6f}"

    return True, None


def _verify_samples(
    *,
    rows: list[TrialIndexRow],
    raw_dataset: MultiModalStabPredDataset,
    root: Path,
    n_samples: int,
) -> None:
    if not rows:
        return

    n = max(1, min(int(n_samples), len(rows)))
    picks = random.sample(rows, n) if len(rows) > n else list(rows)

    for row in picks:
        raw_inputs, raw_label = raw_dataset[row.raw_trial_index]
        sample_inputs, sample_label = _read_sample_h5(
            path=root / Path(row.sample_relpath),
            load_rgb=True,
            load_touch=True,
            touch_cam_variant="raw",
            touch_modality_enabled=ALL_TOUCH_MODALITIES,
            load_proprio=True,
        )

        raw_ok, raw_bad_key = _inputs_all_finite(raw_inputs)
        sample_ok, sample_bad_key = _inputs_all_finite(sample_inputs)
        if not raw_ok:
            raise ValueError(f"verify failed: raw non-finite at key={raw_bad_key}, row={row.trial_id}")
        if not sample_ok:
            raise ValueError(f"verify failed: resampled non-finite at key={sample_bad_key}, row={row.trial_id}")

        same, reason = _compare_inputs(raw_inputs, sample_inputs)
        if not same:
            raise ValueError(f"verify failed: {reason}, row={row.trial_id}")

        if abs(float(raw_label.item()) - float(sample_label.item())) > 1e-6:
            raise ValueError(f"verify failed: label mismatch, row={row.trial_id}")


@dataclass
class WriteCounts:
    """Tally of what happened to each sample file during a run."""

    written_new: int = 0
    rewritten: int = 0
    skipped_existing: int = 0


def _relative_source_path(h5_path: str | Path, source_root: Path) -> str:
    """Source trial path relative to the dataset root, falling back to the raw path."""
    try:
        return str(Path(h5_path).resolve().relative_to(source_root.resolve()))
    except Exception:
        return str(h5_path)


def _print_startup_banner(
    *, source_root: Path, num_h5: int, out_root: Path, write_policy: str, storage_cfg: Mapping[str, Any]
) -> None:
    print(f"[resample] source_root={source_root}")
    print(f"[resample] source_h5={num_h5}")
    print(f"[resample] out_root={out_root}")
    print("[resample] modalities=rgb,touch(cam/audio/imu/pressure),proprio")
    print(f"[resample] write_policy={write_policy}")
    print(
        "[resample] storage="
        f"rgb/touch_cam(raw+bgsub):uint8 numeric:{storage_cfg['dtype_numeric']} "
        f"masks:uint8 compression:{storage_cfg['compression']}"
    )


def _emit_trial(
    *,
    raw_ds: MultiModalStabPredDataset,
    raw_idx: int,
    trial_id: int,
    source_root: Path,
    out_root: Path,
    storage_cfg: Mapping[str, Any],
    write_policy: str,
    counts: WriteCounts,
) -> TrialIndexRow:
    """Write one resampled sample (unless an existing file may be kept) and return its index row."""
    rec = raw_ds.records[raw_idx]
    source_rel = _relative_source_path(rec.h5_path, source_root)
    sample_key = _build_sample_key(
        source_h5_relpath=source_rel,
        session_name=rec.session_name,
        t0=float(rec.t0),
        t1=float(rec.t1),
        label=int(rec.label),
    )
    sample_relpath = _sample_relpath(
        object_name=rec.object_name,
        session_name=rec.session_name,
        sample_key=sample_key,
    )
    sample_path = out_root / sample_relpath
    sample_exists = sample_path.is_file()

    if write_policy == "skip" and sample_exists:
        counts.skipped_existing += 1
    else:
        inputs, label = raw_ds[raw_idx]
        finite, bad_key = _inputs_all_finite(inputs)
        if not finite:
            raise ValueError(
                f"Non-finite inputs before write at raw_idx={raw_idx}, key={bad_key}, "
                f"session={rec.session_name}"
            )
        _write_sample_h5(
            path=sample_path,
            inputs=inputs,
            label=label,
            object_name=rec.object_name,
            session_name=rec.session_name,
            source_h5_relpath=source_rel,
            t0=float(rec.t0),
            t1=float(rec.t1),
            dtype_numeric=str(storage_cfg["dtype_numeric"]),
            compression=str(storage_cfg["compression"]),
        )
        if sample_exists:
            counts.rewritten += 1
        else:
            counts.written_new += 1

    return TrialIndexRow(
        trial_id=trial_id,
        raw_trial_index=raw_idx,
        object_name=str(rec.object_name),
        session_name=str(rec.session_name),
        label=int(rec.label),
        t0=float(rec.t0),
        t1=float(rec.t1),
        source_h5_relpath=source_rel,
        sample_key=str(sample_key),
        sample_relpath=str(sample_relpath),
    )


def main() -> None:
    cfg = copy.deepcopy(CONFIG)
    if not isinstance(cfg.get("dataset", {}), Mapping):
        raise ValueError("CONFIG['dataset'] must be a mapping.")
    _set_seed(int(cfg.get("seed", 42)))

    source_root = _resolve_path(cfg.get("source_root", "data/grasp_data"))
    storage_cfg = _resolve_storage_config(cfg)
    out_root = Path(storage_cfg["root"]).expanduser()
    write_policy = str(storage_cfg["write_policy"])
    verify = bool(storage_cfg.get("verify_after_build", True))

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "samples").mkdir(parents=True, exist_ok=True)

    h5_paths = _discover_h5_paths(source_root)
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found under {source_root}")
    _print_startup_banner(
        source_root=source_root,
        num_h5=len(h5_paths),
        out_root=out_root,
        write_policy=write_policy,
        storage_cfg=storage_cfg,
    )

    raw_ds = MultiModalStabPredDataset(
        h5_paths=h5_paths,
        cfg=cfg,
        load_rgb=True,
        load_touch=True,
        touch_modality_enabled=ALL_TOUCH_MODALITIES,
        load_proprio=True,
    )
    print(f"[resample] usable_trials={len(raw_ds)}")
    if raw_ds.skip_stats:
        print(f"[resample] skipped={raw_ds.skip_stats}")

    max_trials = max(0, int(cfg.get("max_trials", 0) or 0))
    indices = list(range(len(raw_ds)))
    if max_trials > 0:
        indices = indices[:max_trials]
        print(f"[resample] max_trials={max_trials}")

    started = time.perf_counter()
    counts = WriteCounts()
    rows: list[TrialIndexRow] = []
    for n, raw_idx in enumerate(indices, start=1):
        rows.append(
            _emit_trial(
                raw_ds=raw_ds,
                raw_idx=raw_idx,
                trial_id=len(rows),
                source_root=source_root,
                out_root=out_root,
                storage_cfg=storage_cfg,
                write_policy=write_policy,
                counts=counts,
            )
        )
        if n % 100 == 0 or n == len(indices):
            elapsed = time.perf_counter() - started
            print(
                f"[resample] progress {n}/{len(indices)} "
                f"({100.0 * n / max(1, len(indices)):.1f}%) rate={n / max(1e-6, elapsed):.2f} trials/s"
            )

    _write_trial_index_csv(out_root, rows)
    _write_meta_json(
        out_root,
        _meta_payload(cfg=cfg, source_dataset_root=source_root, root=out_root, num_trials=len(rows)),
    )

    total_elapsed = time.perf_counter() - started
    print(
        f"[resample] done trials={len(rows)} elapsed={total_elapsed:.1f}s "
        f"({(len(rows) / max(1e-6, total_elapsed)):.2f} trials/s)"
    )
    print(
        "[resample] write_summary "
        f"new={counts.written_new} rewritten={counts.rewritten} skipped_existing={counts.skipped_existing}"
    )
    print(f"[resample] wrote {out_root / 'trial_index.csv'}")
    print(f"[resample] wrote {out_root / 'meta.json'}")

    if verify:
        verify_n = max(1, int(cfg.get("verify_samples", 64)))
        print(f"[resample] verify start (samples={verify_n})")
        _verify_samples(rows=rows, raw_dataset=raw_ds, root=out_root, n_samples=verify_n)
        print("[resample] verify ok")


if __name__ == "__main__":
    main()
