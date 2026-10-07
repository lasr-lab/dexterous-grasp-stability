#!/usr/bin/env python
"""Standalone viewer for the dexterous grasp stability prediction dataset."""

from __future__ import annotations

import contextlib
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from collections.abc import Sequence

import cv2
import h5py
import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk


# ==== Dataset-local settings ====
SCRIPT_DIR = Path(__file__).resolve().parent
GRASP_DATA_ROOT = SCRIPT_DIR / "data" / "grasp_data"
OBJECT_IMAGE_ROOT = SCRIPT_DIR / "data" / "object_pictures_jpg_square"
OBJECT_PROPERTIES_CSV = SCRIPT_DIR / "data" / "dataset.csv"

AUDIO_FINGER: str | None = "middle"
PRESSURE_FINGER: str | None = "thumb"
PLAYBACK_FPS: float = 60.0
WINDOW_SCALE: float = 1.15
TILE_SIZE: tuple[int, int] = (800, 450)
VIEWER_HEADER_HEIGHT_PX: int = 88
GUI_FONT_SIZE: int = 13
GUI_HEADING_FONT_SIZE: int = 14
# Layout is expressed in font metrics -- characters wide, lines tall -- rather than
# pixels, so the window keeps the same apparent size on high-DPI and standard displays.
GUI_ROW_HEIGHT_LINES: float = 1.5
GUI_MIN_CHARS: tuple[int, int] = (104, 30)
GUI_INITIAL_CHARS: tuple[int, int] = (150, 48)
TRIAL_TABLE_TAB_CHARS: tuple[int, int] = (34, 50)
SUMMARY_TAB_CHARS: int = 30
PREVIEW_LINES: int = 7

LABEL_TAG_COLORS = {"success": "#2E7D32", "failure": "#B71C1C", "unknown": "black"}
SELECTED_ROW_COLOR = "#d9e8fb"
STATUS_COLOR = "#1f4f99"

INTERNAL_PLAY_ENV = "DEXGRASP_STABPRED_PLAY_H5"
INTERNAL_OBJECT_ENV = "DEXGRASP_STABPRED_OBJECT"
INTERNAL_SESSION_ENV = "DEXGRASP_STABPRED_SESSION"
INTERNAL_AUTO_LABEL_ENV = "DEXGRASP_STABPRED_AUTO_LABEL"
INTERNAL_MANUAL_LABEL_ENV = "DEXGRASP_STABPRED_MANUAL_LABEL"

XARM_UNIT_LABEL = "deg"
TILBURG_UNIT_LABEL = "deg"
FT_FORCE_UNIT_LABEL = "N"
FT_TORQUE_UNIT_LABEL = "Nm"
PRESSURE_UNIT_LABEL = "Pa"
TEMPERATURE_UNIT_LABEL = "C"

TIMELINE_MARGIN_PX = 20
TIMELINE_BOTTOM_OFFSET_PX = 20
TIMELINE_HEIGHT_PX = 6
TIMELINE_HIT_PADDING_PX = 12
TIMELINE_PLAYED_COLOR = (48, 48, 230)  # red in BGR
TIMELINE_UNPLAYED_COLOR = (90, 90, 90)
TIMELINE_KNOB_RADIUS_PX = 7

DEFAULT_DEPTH_VIS_SETTINGS: dict[str, object] = {
    "normalize_mode": "fixed",
    "depth_min_mm": 200.0,
    "depth_max_mm": 1200.0,
    "percentile_low": 5.0,
    "percentile_high": 95.0,
    "grayscale_near_white": True,
    "invalid_to_black": True,
    "heatmap_colormap": "JET",
}


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _get_colormap(name: str) -> int:
    table = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    }
    return table.get(str(name).strip().upper(), cv2.COLORMAP_JET)


def _depth_to_visuals(depth_raw: np.ndarray, depth_cfg: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    valid = depth_raw > 0
    valid_vals = depth_raw[valid]
    mode = str(depth_cfg.get("normalize_mode", "fixed")).strip().lower()

    if valid_vals.size == 0:
        lo, hi = 0.0, 1.0
    elif mode == "per_frame_minmax":
        lo, hi = float(valid_vals.min()), float(valid_vals.max())
    elif mode == "per_frame_percentile":
        p_lo = _as_float(depth_cfg.get("percentile_low"), 5.0)
        p_hi = _as_float(depth_cfg.get("percentile_high"), 95.0)
        lo, hi = np.percentile(valid_vals, [p_lo, p_hi]).astype(float).tolist()
    else:
        lo = _as_float(depth_cfg.get("depth_min_mm"), 200.0)
        hi = _as_float(depth_cfg.get("depth_max_mm"), 1200.0)

    if hi <= lo:
        hi = lo + 1.0

    norm = np.clip((depth_raw.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    depth8 = (norm * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(depth8, _get_colormap(str(depth_cfg.get("heatmap_colormap", "JET"))))

    if bool(depth_cfg.get("grayscale_near_white", True)):
        gray_u8 = cv2.bitwise_not(depth8)
    else:
        gray_u8 = depth8

    if bool(depth_cfg.get("invalid_to_black", True)):
        gray_u8[~valid] = 0
        heatmap[~valid] = 0

    gray_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    return heatmap, gray_bgr


def _bytes_from_vlen(ds: h5py.Dataset, idx: int = 0) -> bytes:
    arr = np.asarray(ds[idx])
    try:
        return arr.tobytes()
    except Exception:
        return bytes(arr.tolist())


def _decode_text(value: object) -> str:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return str(value)


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


def _resize_with_padding(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    if img is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _timeline_rect(tile_size: tuple[int, int] = TILE_SIZE) -> tuple[int, int, int, int]:
    tw, th = tile_size
    x0 = TIMELINE_MARGIN_PX
    x1 = 2 * tw - TIMELINE_MARGIN_PX
    y0 = VIEWER_HEADER_HEIGHT_PX + 2 * th - TIMELINE_BOTTOM_OFFSET_PX
    y1 = y0 + TIMELINE_HEIGHT_PX
    return x0, y0, x1, y1


def _timeline_hit(x: int, y: int, tile_size: tuple[int, int] = TILE_SIZE) -> bool:
    x0, y0, x1, y1 = _timeline_rect(tile_size)
    return (
        (x0 - TIMELINE_HIT_PADDING_PX) <= x <= (x1 + TIMELINE_HIT_PADDING_PX)
        and (y0 - TIMELINE_HIT_PADDING_PX) <= y <= (y1 + TIMELINE_HIT_PADDING_PX)
    )


def _time_from_timeline_x(x: int, bounds: tuple[float, float], tile_size: tuple[int, int] = TILE_SIZE) -> float:
    t0, t1 = bounds
    if t1 <= t0:
        return t0
    x0, _, x1, _ = _timeline_rect(tile_size)
    p = np.clip((x - x0) / max(x1 - x0, 1), 0.0, 1.0)
    return t0 + float(p) * (t1 - t0)


def _latest_index(times: np.ndarray, t: float) -> int | None:
    if times.size == 0:
        return None
    idx = np.searchsorted(times, t, side="right") - 1
    return None if idx < 0 else int(idx)


class FrameStream:
    def __init__(self, times: Sequence[float]):
        self.times = np.asarray(times, dtype=float)
        self._last_idx: int | None = None
        self._last_frame: np.ndarray | None = None

    @property
    def min_time(self) -> float | None:
        return float(self.times[0]) if self.times.size else None

    @property
    def max_time(self) -> float | None:
        return float(self.times[-1]) if self.times.size else None

    def _decode(self, idx: int) -> np.ndarray | None:
        raise NotImplementedError

    def get(self, t: float) -> np.ndarray | None:
        idx = _latest_index(self.times, t)
        if idx is None:
            return None
        if idx != self._last_idx:
            self._last_frame = self._decode(idx)
            self._last_idx = idx
        return self._last_frame


class RealSenseColorStream(FrameStream):
    def __init__(self, group: h5py.Group):
        meta_ds = group.get("rgb.csv")
        if not isinstance(meta_ds, h5py.Dataset):
            raise KeyError(f"{group.name}/rgb.csv is required by the current schema")
        meta = _read_csv_dataset(meta_ds)
        super().__init__(meta["host_perf"].astype(float).to_numpy())
        self.frames = group["frames"]

    def _decode(self, idx: int) -> np.ndarray | None:
        data = np.asarray(self.frames[idx], dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)


class RealSenseDepthStream(FrameStream):
    def __init__(
        self,
        group: h5py.Group,
        render_mode: Literal["grayscale", "heatmap"] = "grayscale",
        vis_cfg: dict[str, object] | None = None,
    ):
        meta_ds = group.get("depth.csv")
        if not isinstance(meta_ds, h5py.Dataset):
            raise KeyError(f"{group.name}/depth.csv is required by the current schema")
        meta = _read_csv_dataset(meta_ds)
        super().__init__(meta["host_perf"].astype(float).to_numpy())
        self.raw_ds = group.get("raw_mm")
        if self.raw_ds is None:
            raise KeyError(f"{group.name}/raw_mm is required by the current schema")
        self.vis_cfg = dict(vis_cfg) if vis_cfg is not None else dict(DEFAULT_DEPTH_VIS_SETTINGS)
        mode = str(render_mode).strip().lower()
        self.render_mode: Literal["grayscale", "heatmap"] = "heatmap" if mode == "heatmap" else "grayscale"

    def _decode(self, idx: int) -> np.ndarray | None:
        depth = np.asarray(self.raw_ds[idx])
        heatmap, gray = _depth_to_visuals(depth, self.vis_cfg)
        return gray if self.render_mode == "grayscale" else heatmap


class DigitCameraStream(FrameStream):
    def __init__(self, digit360_group: h5py.Group, finger: str):
        fg = digit360_group[finger]
        camera_group = fg.get("camera")
        if not isinstance(camera_group, h5py.Group):
            raise KeyError(f"Digit360 camera group is missing for finger '{finger}'")
        meta_ds = camera_group.get("camera.csv")
        frames_ds = camera_group.get("frames")
        if not isinstance(meta_ds, h5py.Dataset) or not isinstance(frames_ds, h5py.Dataset):
            raise KeyError(f"Digit360 camera stream is missing for finger '{finger}'")
        meta = _read_csv_dataset(meta_ds)
        super().__init__(meta.get("time_perf", pd.Series(dtype=float)).astype(float).to_numpy())
        self.frames = frames_ds

    def _decode(self, idx: int) -> np.ndarray | None:
        data = np.asarray(self.frames[idx], dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)


class AudioChunkStream:
    def __init__(self, chunk_df: pd.DataFrame):
        self.df = chunk_df
        self.times = chunk_df.get("time_perf", pd.Series(dtype=float)).astype(float).to_numpy()

    @property
    def min_time(self) -> float | None:
        return float(self.times[0]) if self.times.size else None

    @property
    def max_time(self) -> float | None:
        return float(self.times[-1]) if self.times.size else None


class PressureStream:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.times = df.get("time_perf", pd.Series(dtype=float)).astype(float).to_numpy()
        self.press = df.get("pressure", pd.Series(dtype=float)).astype(float).to_numpy()
        self.temp = df.get("temperature", pd.Series(dtype=float)).astype(float).to_numpy()
        self._last_idx: int | None = None
        self._last_press: float | None = None
        self._last_temp: float | None = None
        n_base = min(10, len(self.press))
        finite_press = self.press[:n_base][np.isfinite(self.press[:n_base])]
        self.baseline = float(np.mean(finite_press)) if finite_press.size else None

    @property
    def min_time(self) -> float | None:
        return float(self.times[0]) if self.times.size else None

    @property
    def max_time(self) -> float | None:
        return float(self.times[-1]) if self.times.size else None

    def values(self, t: float) -> tuple[float | None, float | None, float | None, float | None]:
        idx = _latest_index(self.times, t)
        if idx is None:
            return None, None, self.baseline, None
        if idx != self._last_idx:
            self._last_press = float(self.press[idx]) if idx < len(self.press) else None
            self._last_temp = float(self.temp[idx]) if idx < len(self.temp) else None
            self._last_idx = idx
        delta = None
        if self._last_press is not None and self.baseline is not None:
            delta = self._last_press - self.baseline
        return self._last_press, self._last_temp, self.baseline, delta


class TimedTableStream:
    def __init__(self, df: pd.DataFrame, time_col: str):
        self.time_col = time_col
        if time_col in df.columns:
            self.df = df.sort_values(time_col).reset_index(drop=True)
            self.times = self.df[time_col].astype(float).to_numpy()
        else:
            self.df = df
            self.times = np.array([], dtype=float)
        self._last_idx: int | None = None
        self._last_row: pd.Series | None = None

    @property
    def min_time(self) -> float | None:
        return float(self.times[0]) if self.times.size else None

    @property
    def max_time(self) -> float | None:
        return float(self.times[-1]) if self.times.size else None

    def row(self, t: float) -> pd.Series | None:
        idx = _latest_index(self.times, t)
        if idx is None:
            return None
        if idx != self._last_idx:
            self._last_row = self.df.iloc[int(idx)]
            self._last_idx = idx
        return self._last_row

    def value(self, t: float, col: str) -> object | None:
        row = self.row(t)
        if row is None or col not in row.index:
            return None
        val = row[col]
        return None if pd.isna(val) else val


@dataclass
class Streams:
    color: RealSenseColorStream | None
    depth: RealSenseDepthStream | None
    digit_cams: dict[str, DigitCameraStream]
    audio: AudioChunkStream | None
    audio_spec: np.ndarray | None
    audio_finger: str | None
    pressure: PressureStream | None
    pressure_finger: str | None
    xarm: TimedTableStream | None
    xarm_ee: TimedTableStream | None
    xarm_ft: TimedTableStream | None
    tilburg: TimedTableStream | None
    phases: TimedTableStream | None

    def time_bounds(self) -> tuple[float, float]:
        digit_list = list(self.digit_cams.values())
        timed_streams = [
            self.color,
            self.depth,
            self.audio,
            self.pressure,
            self.xarm,
            self.xarm_ee,
            self.xarm_ft,
            self.tilburg,
            *digit_list,
        ]
        mins = [s.min_time for s in timed_streams if s and s.min_time is not None]
        maxs = [s.max_time for s in timed_streams if s and s.max_time is not None]
        if not mins or not maxs:
            raise ValueError("No time-bearing streams found.")
        return float(min(mins)), float(max(maxs))


@dataclass
class TrialInfo:
    object_name: str
    path: Path
    session_name: str
    automatic_label: str
    manual_label: str


def _pick_session(h5: h5py.File, session: str | None = None) -> h5py.Group:
    if session and session in h5:
        return h5[session]
    for name in h5:
        obj = h5[name]
        if isinstance(obj, h5py.Group):
            return obj
    raise ValueError("No session group found in HDF5.")


def _session_name(path: Path) -> str:
    try:
        with h5py.File(path, "r") as h5:
            return _pick_session(h5).name.strip("/")
    except Exception:
        return path.stem


def _status_from_label_value(raw: object) -> str:
    text = _decode_text(raw).strip()
    try:
        label = int(float(text))
    except Exception:
        return "unknown"
    if label == 1:
        return "success"
    if label == 0:
        return "failure"
    return "unknown"


def _read_label_statuses(path: Path) -> tuple[str, str]:
    try:
        with h5py.File(path, "r") as h5:
            session_group = _pick_session(h5)
            grasp = session_group.get("grasp")
            if not isinstance(grasp, h5py.Group):
                return "unknown", "unknown"
            label_ds = grasp.get("grasp_label.csv")
            if not isinstance(label_ds, h5py.Dataset) or label_ds.dtype.names is None:
                return "unknown", "unknown"
            arr = label_ds[()]
    except Exception:
        return "unknown", "unknown"

    auto_status = "unknown"
    manual_status = "unknown"
    names = arr.dtype.names or ()
    for rec in arr:
        method = _decode_text(rec["method"]).strip().lower() if "method" in names else ""
        label_status = _status_from_label_value(rec["label"]) if "label" in names else "unknown"
        if method in {"automatic", "auto"} or method.startswith("auto"):
            auto_status = label_status
        elif method == "manual":
            manual_status = label_status
    return auto_status, manual_status


def _load_trials_for_object(object_dir: Path) -> list[TrialInfo]:
    trials: list[TrialInfo] = []
    for h5_path in sorted(object_dir.glob("*.h5")):
        automatic, manual = _read_label_statuses(h5_path)
        trials.append(
            TrialInfo(
                object_name=object_dir.name,
                path=h5_path,
                session_name=_session_name(h5_path),
                automatic_label=automatic,
                manual_label=manual,
            )
        )
    return trials


def _list_object_dirs() -> list[Path]:
    if not GRASP_DATA_ROOT.is_dir():
        return []
    return sorted(path for path in GRASP_DATA_ROOT.iterdir() if path.is_dir())


def _format_joint_lines(prefix: str, row: pd.Series | None, unit: str | None = None, max_chars: int = 60) -> list[str]:
    if row is None:
        return [f"{prefix}: --"]
    vals = [float(v) for v in row.values[1:]]
    parts = [f"{v:+.2f}" for v in vals]
    suffix = f" [{unit}]" if unit else ""
    if prefix.lower().startswith("tilburg") and len(parts) % 4 == 0:
        grouped = [parts[i : i + 4] for i in range(0, len(parts), 4)]
        finger_names = ["thumb", "index", "middle", "ring"]
        base_prefix = f"{prefix}{suffix}: "
        pad = " " * len(base_prefix)
        lines = []
        for i, group in enumerate(grouped):
            fname = finger_names[i] if i < len(finger_names) else f"finger{i + 1}"
            line_prefix = base_prefix if i == 0 else pad
            lines.append(f"{line_prefix}{fname} " + " ".join(group))
        return lines

    lines: list[str] = []
    current = f"{prefix}{suffix}:"
    for part in parts:
        sep = " " if current else ""
        if len(current) + len(sep) + len(part) <= max_chars:
            current = f"{current}{sep}{part}"
        else:
            lines.append(current)
            current = f"   {part}"
    if current:
        lines.append(current)
    return lines


def _format_xarm_ft_lines(row: pd.Series | None) -> list[str]:
    if row is None:
        return ["XArmFT: --"]
    vals: list[float] = []
    if {"fx", "fy", "fz", "mx", "my", "mz"}.issubset(set(row.index)):
        vals = [float(row[name]) for name in ["fx", "fy", "fz", "mx", "my", "mz"]]
    else:
        numeric = [float(v) for v in row.values[1:] if pd.notna(v)]
        if len(numeric) >= 6:
            vals = numeric[:6]
    if len(vals) < 6:
        return ["XArmFT: --"]
    fx, fy, fz, mx, my, mz = vals
    return [f"XArmFT F:{fx:+.2f},{fy:+.2f},{fz:+.2f} M:{mx:+.3f},{my:+.3f},{mz:+.3f} [{FT_FORCE_UNIT_LABEL}/{FT_TORQUE_UNIT_LABEL}]"]


def _format_xarm_ee_lines(row: pd.Series | None) -> list[str]:
    if row is None:
        return ["XArmEE: --"]
    vals: list[float] = []
    if {"x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"}.issubset(set(row.index)):
        vals = [float(row[name]) for name in ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]]
    else:
        numeric = [float(v) for v in row.values[1:] if pd.notna(v)]
        if len(numeric) >= 6:
            vals = numeric[:6]
    if len(vals) < 6:
        return ["XArmEE: --"]
    x, y, z, roll, pitch, yaw = vals
    return [
        f"XArmEE xyz:{x:+.1f},{y:+.1f},{z:+.1f} [mm]",
        f"       rpy:{roll:+.1f},{pitch:+.1f},{yaw:+.1f} [deg]",
    ]


def _format_tilburg_compact_lines(row: pd.Series | None, unit: str | None = None) -> list[str]:
    if row is None:
        return ["Tilburg: --"]
    vals = [float(v) for v in row.values[1:]]
    if not vals:
        return ["Tilburg: --"]
    suffix = f" [{unit}]" if unit else ""
    if len(vals) % 4 != 0:
        return _format_joint_lines("Tilburg", row, unit=unit, max_chars=34)
    groups = [vals[i : i + 4] for i in range(0, len(vals), 4)]
    finger_names = ["th", "in", "mi", "ri"]
    lines = [f"Tilburg{suffix}:"]
    for i, group in enumerate(groups):
        fname = finger_names[i] if i < len(finger_names) else f"f{i + 1}"
        lines.append(f"{fname}: " + " ".join(f"{v:+.1f}" for v in group))
    return lines


OverlayLine = tuple[str, tuple[int, int, int]]


def _text_width_px(text: str, font_scale: float) -> int:
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    return int(width)


def _wrap_text(text: str, max_px: int, font_scale: float) -> list[str]:
    """Wrap on spaces, hard-splitting any single word that is still too wide."""
    out: list[str] = []
    current = ""
    for word in str(text).split(" "):
        candidate = word if not current else f"{current} {word}"
        if _text_width_px(candidate, font_scale) <= max_px:
            current = candidate
            continue
        if current:
            out.append(current)
        current = word
        while _text_width_px(current, font_scale) > max_px and len(current) > 1:
            cut = len(current) - 1
            while cut > 1 and _text_width_px(current[:cut], font_scale) > max_px:
                cut -= 1
            out.append(current[:cut])
            current = current[cut:]
    if current:
        out.append(current)
    return out or [str(text)]


def _wrap_block(lines: list[OverlayLine], max_px: int, font_scale: float) -> list[OverlayLine]:
    return [(segment, color) for text, color in lines for segment in _wrap_text(text, max_px, font_scale)]


def _draw_spectrogram(
    tile: np.ndarray,
    spec_img: np.ndarray | None,
    audio_stream: AudioChunkStream | None,
    now_t: float,
    audio_finger: str | None,
    spec_h: int,
) -> None:
    """Fill the top band of the tile with the spectrogram and its playhead."""
    h, w = tile.shape[:2]
    if spec_img is not None:
        tile[0:spec_h, 0:w] = _resize_with_padding(spec_img, (w, spec_h))
    else:
        cv2.putText(tile, "No spectrogram", (20, spec_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    if (
        audio_stream
        and audio_stream.min_time is not None
        and audio_stream.max_time is not None
        and audio_stream.max_time > audio_stream.min_time
    ):
        p = np.clip((now_t - audio_stream.min_time) / (audio_stream.max_time - audio_stream.min_time), 0.0, 1.0)
        x = int(p * (w - 1))
        cv2.line(tile, (x, 0), (x, spec_h - 1), (0, 255, 255), 2)
        cv2.putText(tile, f"{now_t:.2f}", (min(x + 5, w - 120), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    audio_name = str(audio_finger).strip() if audio_finger else "--"
    cv2.putText(tile, f"Audio ({audio_name})", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)


def _overlay_state_lines(
    pressure: float | None,
    temperature: float | None,
    delta: float | None,
    pressure_finger: str | None,
    xarm_row: pd.Series | None,
    xarm_ee_row: pd.Series | None,
    xarm_ft_row: pd.Series | None,
    til_row: pd.Series | None,
) -> tuple[list[OverlayLine], list[OverlayLine]]:
    """Sensor/robot state text, split into the left and right overlay columns."""
    left: list[OverlayLine] = []
    right: list[OverlayLine] = []
    pressure_name = str(pressure_finger).strip() if pressure_finger else "--"
    if pressure is not None:
        delta_txt = f" dP={delta:+.1f}" if delta is not None else ""
        left.append((f"Pressure ({pressure_name})[{PRESSURE_UNIT_LABEL}]: {pressure:.1f}{delta_txt}", (220, 220, 255)))
    if temperature is not None:
        left.append((f"Temp ({pressure_name}): {temperature:.2f} [{TEMPERATURE_UNIT_LABEL}]", (220, 220, 255)))
    for line in _format_joint_lines("XArm", xarm_row, unit=XARM_UNIT_LABEL, max_chars=42):
        left.append((line, (220, 220, 220)))
    for line in _format_xarm_ee_lines(xarm_ee_row):
        left.append((line, (220, 220, 220)))
    for line in _format_xarm_ft_lines(xarm_ft_row):
        left.append((line, (220, 220, 220)))
    for line in _format_tilburg_compact_lines(til_row, unit=TILBURG_UNIT_LABEL):
        right.append((line, (220, 220, 220)))
    return left, right


def _build_audio_tile(
    spec_img: np.ndarray | None,
    audio_stream: AudioChunkStream | None,
    now_t: float,
    audio_finger: str | None,
    pressure: float | None,
    temperature: float | None,
    delta: float | None,
    pressure_finger: str | None,
    xarm_row: pd.Series | None,
    xarm_ee_row: pd.Series | None,
    xarm_ft_row: pd.Series | None,
    til_row: pd.Series | None,
    size: tuple[int, int],
) -> np.ndarray:
    w, h = size
    spec_h = int(h * 0.42)
    text_top = spec_h + 24
    text_bottom = h - 8
    text_left_x = 20
    text_right_x = int(w * 0.56)

    tile = np.zeros((h, w, 3), dtype=np.uint8)
    _draw_spectrogram(tile, spec_img, audio_stream, now_t, audio_finger, spec_h)

    # Darken the lower half so the state text stays readable over the spectrogram colours.
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, spec_h), (w, h), (0, 0, 0), thickness=-1)
    tile = cv2.addWeighted(overlay, 0.45, tile, 0.55, 0)

    left_lines, right_lines = _overlay_state_lines(
        pressure, temperature, delta, pressure_finger, xarm_row, xarm_ee_row, xarm_ft_row, til_row
    )

    gutter = 20
    left_max_w = max(80, text_right_x - text_left_x - gutter)
    right_max_w = max(80, w - text_right_x - gutter)

    # Shrink the font until both columns fit, then stop at the readable minimum.
    font_scale = 0.68
    min_font_scale = 0.48
    while True:
        line_step = max(16, int(36 * font_scale))
        max_lines = max(1, (text_bottom - text_top) // line_step + 1)
        left_wrapped = _wrap_block(left_lines, left_max_w, font_scale)
        right_wrapped = _wrap_block(right_lines, right_max_w, font_scale)
        if len(left_wrapped) <= max_lines and len(right_wrapped) <= max_lines:
            break
        if font_scale <= min_font_scale:
            break
        font_scale -= 0.03

    def draw_block(lines: list[OverlayLine], x: int) -> None:
        y = text_top
        for text, color in lines:
            if y > text_bottom:
                break
            cv2.putText(tile, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
            y += line_step

    draw_block(left_wrapped, text_left_x)
    draw_block(right_wrapped, text_right_x)
    return tile


def _build_empty_tile(message: str, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    tile = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(tile, message, (14, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (180, 180, 180), 1, cv2.LINE_AA)
    return tile


def _compose_digit_grid(digit_streams: dict[str, DigitCameraStream], t: float, tile_size: tuple[int, int] = TILE_SIZE) -> np.ndarray | None:
    if not digit_streams:
        return None
    tw, th = tile_size
    sub_w, sub_h = tw // 2, th // 2
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    positions = [(0, 0), (sub_w, 0), (0, sub_h), (sub_w, sub_h)]
    for (finger, stream), (x0, y0) in zip(sorted(digit_streams.items()), positions, strict=False):
        frame = stream.get(t)
        tile = _resize_with_padding(frame, (sub_w, sub_h)) if frame is not None else _build_empty_tile(f"no {finger}", (sub_w, sub_h))
        cv2.putText(tile, finger, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
        canvas[y0 : y0 + sub_h, x0 : x0 + sub_w] = tile
    return canvas


def _draw_fit_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    max_width: int,
    scale: float,
    min_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    fit_scale = scale
    while fit_scale > min_scale:
        (text_w, _), _baseline = cv2.getTextSize(text, font, fit_scale, thickness)
        if text_w <= max_width:
            break
        fit_scale -= 0.04

    display = text
    (text_w, _), _baseline = cv2.getTextSize(display, font, fit_scale, thickness)
    if text_w > max_width:
        suffix = "..."
        while display:
            candidate = display[:-1].rstrip() + suffix
            (text_w, _), _baseline = cv2.getTextSize(candidate, font, fit_scale, thickness)
            if text_w <= max_width:
                display = candidate
                break
            display = display[:-1]

    cv2.putText(canvas, display, origin, font, fit_scale, color, thickness, cv2.LINE_AA)


def _draw_header(
    canvas: np.ndarray,
    trial_info: TrialInfo,
    current_phase: str | None,
    now_t: float,
    bounds: tuple[float, float],
) -> None:
    h, w = canvas.shape[:2]
    del h
    cv2.rectangle(canvas, (0, 0), (w, VIEWER_HEADER_HEIGHT_PX), (20, 22, 24), thickness=-1)
    cv2.line(canvas, (0, VIEWER_HEADER_HEIGHT_PX - 1), (w, VIEWER_HEADER_HEIGHT_PX - 1), (70, 70, 70), 1)
    phase_text = f"phase: {current_phase}" if current_phase else "phase: --"
    label_text = (
        f"{trial_info.object_name} / {trial_info.path.name}    "
        f"auto={trial_info.automatic_label}    manual={trial_info.manual_label}"
    )
    _draw_fit_text(
        canvas,
        label_text,
        (22, 34),
        max_width=w - 44,
        scale=0.92,
        min_scale=0.64,
        color=(245, 245, 245),
        thickness=2,
    )
    _draw_fit_text(
        canvas,
        f"{phase_text}    perf_t={now_t:.3f}    range={bounds[0]:.3f}-{bounds[1]:.3f}",
        (22, 68),
        max_width=w - 44,
        scale=0.74,
        min_scale=0.56,
        color=(120, 235, 255),
        thickness=1,
    )


def _compose_canvas(
    color: np.ndarray | None,
    depth: np.ndarray | None,
    digit: np.ndarray | None,
    audio_tile: np.ndarray,
    now_t: float,
    bounds: tuple[float, float],
    current_phase: str | None,
    trial_info: TrialInfo,
    tile_size: tuple[int, int] = TILE_SIZE,
) -> np.ndarray:
    tw, th = tile_size
    header_h = VIEWER_HEADER_HEIGHT_PX
    canvas = np.zeros((header_h + th * 2, tw * 2, 3), dtype=np.uint8)
    _draw_header(canvas, trial_info, current_phase, now_t, bounds)
    canvas[header_h : header_h + th, 0:tw] = (
        _resize_with_padding(color, tile_size) if color is not None else _build_empty_tile("No RGB frame", tile_size)
    )
    canvas[header_h : header_h + th, tw : 2 * tw] = (
        _resize_with_padding(depth, tile_size) if depth is not None else _build_empty_tile("No depth", tile_size)
    )
    canvas[header_h + th : header_h + 2 * th, 0:tw] = (
        _resize_with_padding(digit, tile_size) if digit is not None else _build_empty_tile("No Digit frame", tile_size)
    )
    canvas[header_h + th : header_h + 2 * th, tw : 2 * tw] = audio_tile

    t0, t1 = bounds
    p = 0.0 if t1 <= t0 else np.clip((now_t - t0) / (t1 - t0), 0.0, 1.0)
    bar_x0, bar_y0, bar_x1, bar_y1 = _timeline_rect(tile_size)
    filled_x = int(np.clip(bar_x0 + int((bar_x1 - bar_x0) * p), bar_x0, bar_x1))
    knob_y = (bar_y0 + bar_y1) // 2
    cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x1, bar_y1), TIMELINE_UNPLAYED_COLOR, thickness=-1)
    if filled_x > bar_x0:
        cv2.rectangle(canvas, (bar_x0, bar_y0), (filled_x, bar_y1), TIMELINE_PLAYED_COLOR, thickness=-1)
    cv2.circle(canvas, (filled_x, knob_y), TIMELINE_KNOB_RADIUS_PX, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (filled_x, knob_y), max(1, TIMELINE_KNOB_RADIUS_PX - 2), TIMELINE_PLAYED_COLOR, thickness=-1, lineType=cv2.LINE_AA)
    return canvas


def _subgroup(parent: h5py.Group | None, name: str) -> h5py.Group | None:
    """Return parent[name] when it exists and is a group."""
    if not isinstance(parent, h5py.Group):
        return None
    child = parent.get(name)
    return child if isinstance(child, h5py.Group) else None


def _dataset(parent: h5py.Group | None, name: str) -> h5py.Dataset | None:
    """Return parent[name] when it exists and is a dataset."""
    if not isinstance(parent, h5py.Group):
        return None
    child = parent.get(name)
    return child if isinstance(child, h5py.Dataset) else None


def _timed_table(group: h5py.Group | None, name: str) -> TimedTableStream | None:
    """Build a perf_time-indexed stream from group[name], or None when absent."""
    dataset = _dataset(group, name)
    return None if dataset is None else TimedTableStream(_read_csv_dataset(dataset), "perf_time")


def _load_realsense(session_group: h5py.Group) -> tuple[RealSenseColorStream | None, RealSenseDepthStream | None]:
    rs = _subgroup(session_group, "realsense")
    rgb_group = _subgroup(rs, "rgb")
    depth_group = _subgroup(rs, "depth")
    color = RealSenseColorStream(rgb_group) if rgb_group is not None else None
    depth = (
        RealSenseDepthStream(depth_group, render_mode="grayscale", vis_cfg=DEFAULT_DEPTH_VIS_SETTINGS)
        if depth_group is not None
        else None
    )
    return color, depth


def _load_digit360(
    session_group: h5py.Group, audio_finger: str | None, pressure_finger: str | None
) -> tuple[
    dict[str, DigitCameraStream],
    AudioChunkStream | None,
    np.ndarray | None,
    str | None,
    PressureStream | None,
    str | None,
]:
    """Load the per-fingertip camera streams plus the single audio and pressure channel shown in the overlay."""
    digit_cams: dict[str, DigitCameraStream] = {}
    root = _subgroup(_subgroup(session_group, "opentouch"), "digit360")
    if root is None:
        return digit_cams, None, None, None, None, None

    fingers = list(root.keys())
    for finger in fingers:
        try:
            digit_cams[finger] = DigitCameraStream(root, finger)
        except Exception as exc:
            print(f"[viewer] skip digit camera '{finger}': {exc}")
    if not fingers:
        return digit_cams, None, None, None, None, None

    picked_audio = str(audio_finger if audio_finger in fingers else fingers[0])
    audio_group = _subgroup(root[picked_audio], "audio")
    chunks_ds = _dataset(audio_group, "chunks.csv")
    spec_ds = _dataset(audio_group, "spectrogram.jpg")
    audio = AudioChunkStream(_read_csv_dataset(chunks_ds)) if chunks_ds is not None else None
    spec = (
        cv2.imdecode(np.frombuffer(_bytes_from_vlen(spec_ds), dtype=np.uint8), cv2.IMREAD_COLOR)
        if spec_ds is not None
        else None
    )

    picked_pressure = str(pressure_finger if pressure_finger in fingers else fingers[0])
    pressure_ds = _dataset(_subgroup(root[picked_pressure], "serial"), "pressure.csv")
    pressure = PressureStream(_read_csv_dataset(pressure_ds)) if pressure_ds is not None else None

    return digit_cams, audio, spec, picked_audio, pressure, picked_pressure


def load_streams(session_group: h5py.Group, audio_finger: str | None, pressure_finger: str | None) -> Streams:
    color, depth = _load_realsense(session_group)
    digit_cams, audio, audio_spec, picked_audio, pressure, picked_pressure = _load_digit360(
        session_group, audio_finger, pressure_finger
    )

    xarm = _subgroup(session_group, "xarm")
    phase_ds = _dataset(_subgroup(session_group, "grasp"), "grasp_phases.csv")
    phase_stream = None
    if phase_ds is not None:
        phase_df = _read_csv_dataset(phase_ds)
        if "perf_time" in phase_df.columns and "phase" in phase_df.columns:
            phase_stream = TimedTableStream(phase_df, "perf_time")

    return Streams(
        color=color,
        depth=depth,
        digit_cams=digit_cams,
        audio=audio,
        audio_spec=audio_spec,
        audio_finger=picked_audio,
        pressure=pressure,
        pressure_finger=picked_pressure,
        xarm=_timed_table(xarm, "xarm_jpos.csv"),
        xarm_ee=_timed_table(xarm, "xarm_eepos.csv"),
        xarm_ft=_timed_table(xarm, "xarm_ft.csv"),
        tilburg=_timed_table(_subgroup(session_group, "tilburg"), "tilburg_pos.csv"),
        phases=phase_stream,
    )


def _render_canvas(streams: Streams, now_t: float, bounds: tuple[float, float], trial_info: TrialInfo) -> np.ndarray:
    color = streams.color.get(now_t) if streams.color else None
    depth = streams.depth.get(now_t) if streams.depth else None
    digit = _compose_digit_grid(streams.digit_cams, now_t, TILE_SIZE)
    pressure_val, temp_val, _base_val, delta_val = (None, None, None, None)
    if streams.pressure:
        pressure_val, temp_val, _base_val, delta_val = streams.pressure.values(now_t)
    xarm_row = streams.xarm.row(now_t) if streams.xarm else None
    xarm_ee_row = streams.xarm_ee.row(now_t) if streams.xarm_ee else None
    xarm_ft_row = streams.xarm_ft.row(now_t) if streams.xarm_ft else None
    til_row = streams.tilburg.row(now_t) if streams.tilburg else None
    phase_value = streams.phases.value(now_t, "phase") if streams.phases else None
    phase_label = str(phase_value) if phase_value is not None else None
    audio_tile = _build_audio_tile(
        streams.audio_spec,
        streams.audio,
        now_t,
        streams.audio_finger,
        pressure_val,
        temp_val,
        delta_val,
        streams.pressure_finger,
        xarm_row,
        xarm_ee_row,
        xarm_ft_row,
        til_row,
        TILE_SIZE,
    )
    return _compose_canvas(color, depth, digit, audio_tile, now_t, bounds, phase_label, trial_info, TILE_SIZE)


@dataclass
class PlaybackInput:
    """Mouse state for the playback window: timeline seeks and click-to-pause."""

    bounds: tuple[float, float]
    dragging: bool = False
    seek_t: float | None = None
    toggle_pause: bool = False

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            if _timeline_hit(x, y, TILE_SIZE):
                self.dragging = True
                self.seek_t = _time_from_timeline_x(x, self.bounds, TILE_SIZE)
            else:
                self.toggle_pause = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.seek_t = _time_from_timeline_x(x, self.bounds, TILE_SIZE)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.seek_t = _time_from_timeline_x(x, self.bounds, TILE_SIZE)
            self.dragging = False

    def take_seek(self) -> float | None:
        """Consume a pending seek target, if any."""
        seek, self.seek_t = self.seek_t, None
        return seek

    def take_pause_toggle(self) -> bool:
        """Consume a pending click-to-pause request."""
        toggled, self.toggle_pause = self.toggle_pause, False
        return toggled


def _open_playback_window(name: str) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    base_w = TILE_SIZE[0] * 2
    base_h = VIEWER_HEADER_HEIGHT_PX + TILE_SIZE[1] * 2
    scale = max(0.1, float(WINDOW_SCALE))
    cv2.resizeWindow(name, int(base_w * scale), int(base_h * scale))


def _window_closed(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def run_viewer(trial_info: TrialInfo) -> None:
    if not trial_info.path.exists():
        raise FileNotFoundError(trial_info.path)

    window_name = f"{trial_info.object_name} - {trial_info.path.name}"
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    try:
        with h5py.File(trial_info.path, "r") as h5:
            session_group = _pick_session(h5, trial_info.session_name)
            streams = load_streams(session_group, AUDIO_FINGER, PRESSURE_FINGER)
            bounds = t_start, t_end = streams.time_bounds()

            dt = 1.0 / max(PLAYBACK_FPS, 1e-3)
            now_t = t_start
            next_tick = time.perf_counter() + dt
            paused_by_user = False
            last_canvas: np.ndarray | None = None
            controls = PlaybackInput(bounds=bounds)

            _open_playback_window(window_name)
            cv2.setMouseCallback(window_name, controls.on_mouse)
            print(
                f"Playing {trial_info.path} session={session_group.name} "
                f"from perf {t_start:.3f} to {t_end:.3f} at {PLAYBACK_FPS:.1f} Hz "
                "(q: quit, p: pause/resume, click video: pause/resume, drag timeline: seek)"
            )

            while now_t <= t_end:
                seek_t = controls.take_seek()
                seek_applied = seek_t is not None
                if seek_t is not None:
                    now_t = float(np.clip(seek_t, t_start, t_end))
                    last_canvas = _render_canvas(streams, now_t, bounds, trial_info)
                    next_tick = time.perf_counter() + dt

                if controls.take_pause_toggle():
                    paused_by_user = not paused_by_user
                    next_tick = time.perf_counter() + dt

                if not (paused_by_user or controls.dragging) and not seek_applied:
                    last_canvas = _render_canvas(streams, now_t, bounds, trial_info)
                    now_t += dt

                if last_canvas is not None:
                    cv2.imshow(window_name, last_canvas)
                if _window_closed(window_name):
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused_by_user = not paused_by_user
                    next_tick = time.perf_counter() + dt

                if paused_by_user or controls.dragging:
                    time.sleep(0.01)
                else:
                    sleep_time = max(0.0, next_tick - time.perf_counter())
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    next_tick += dt
    finally:
        with contextlib.suppress(cv2.error):
            cv2.destroyWindow(window_name)
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def _to_ppm_bytes(rgb_frame: np.ndarray) -> bytes:
    h, w = rgb_frame.shape[:2]
    return f"P6\n{w} {h}\n255\n".encode("ascii") + rgb_frame.tobytes()


@dataclass
class ObjectProperties:
    material: str
    compliance: str
    weight: str
    size: str


def _load_object_properties() -> dict[str, ObjectProperties]:
    """Read per-object properties from dataset.csv.

    The file is optional: a partial download without it simply shows no properties.
    """
    if not OBJECT_PROPERTIES_CSV.is_file():
        return {}
    try:
        with OBJECT_PROPERTIES_CSV.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return {}

    def cell(row: dict[str, str], key: str) -> str:
        value = (row.get(key) or "").strip()
        return "" if value in {"", "-"} else value

    properties: dict[str, ObjectProperties] = {}
    for row in rows:
        name = cell(row, "object_name")
        if not name:
            continue
        material = cell(row, "primary_material") or "n/a"
        secondary = cell(row, "secondary_materials")
        if secondary:
            material = f"{material} (+{secondary})"
        compliance = {"0": "rigid", "1": "deformable"}.get(cell(row, "compliance"), "n/a")
        weight = cell(row, "weight_g")
        dims = [cell(row, key) for key in ("width_mm", "depth_mm", "height_mm")]
        properties[name] = ObjectProperties(
            material=material,
            compliance=compliance,
            weight=f"{weight} g" if weight else "n/a",
            size=" x ".join(dims) + " mm" if all(dims) else "n/a",
        )
    return properties


def _configure_label_tags(widget: tk.Text) -> None:
    """Apply the shared success/failure/unknown colours to a text widget."""
    for tag, color in LABEL_TAG_COLORS.items():
        widget.tag_configure(tag, foreground=color)


def _find_object_image(object_name: str) -> Path | None:
    candidates = [object_name, object_name.replace(" ", "_")]
    suffixes = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    for stem in candidates:
        for suffix in suffixes:
            path = OBJECT_IMAGE_ROOT / f"{stem}{suffix}"
            if path.is_file():
                return path
    normalized = object_name.replace(" ", "_").lower()
    if OBJECT_IMAGE_ROOT.is_dir():
        for path in OBJECT_IMAGE_ROOT.iterdir():
            if path.is_file() and path.stem.lower() == normalized:
                return path
    return None


def _load_preview_photo(object_name: str, size: tuple[int, int] = (220, 220)) -> tk.PhotoImage | None:
    image_path = _find_object_image(object_name)
    if image_path is None:
        return None
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    preview = _resize_with_padding(frame, size)
    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    return tk.PhotoImage(data=_to_ppm_bytes(rgb), format="PPM")


class DataViewerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dexterous Grasp Stability Data Viewer")
        self.char_w = 8      # replaced with real font metrics in _build_ui
        self.line_h = 18

        self.object_dirs: list[Path] = []
        self.filtered_object_dirs: list[Path] = []
        self.trials: list[TrialInfo] = []
        self.selected_trial_index: int | None = None
        self.selected_object: Path | None = None
        self.preview_photo: tk.PhotoImage | None = None
        self.object_properties = _load_object_properties()
        self.worker: subprocess.Popen[bytes] | None = None

        self.search_text = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready")

        self.object_list: tk.Listbox
        self.trial_text: tk.Text
        self.preview_label: ttk.Label
        self.object_summary: tk.Text
        self.play_button: ttk.Button
        self.refresh_button: ttk.Button

        self._build_ui()
        self._load_objects()
        self.root.after_idle(self._set_initial_geometry)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _set_initial_geometry(self) -> None:
        self.root.update_idletasks()
        min_w = GUI_MIN_CHARS[0] * self.char_w
        min_h = GUI_MIN_CHARS[1] * self.line_h
        want_w = GUI_INITIAL_CHARS[0] * self.char_w
        want_h = GUI_INITIAL_CHARS[1] * self.line_h
        # winfo_screen* spans every attached monitor, so cap the window at one
        # screen's worth of the smaller dimension rather than the full desktop.
        screen_w = max(min_w, int(self.root.winfo_screenwidth()))
        screen_h = max(min_h, int(self.root.winfo_screenheight()))
        avail_w = max(min_w, min(screen_w, int(screen_h * 16 / 9)) - 80)
        avail_h = max(min_h, screen_h - 80)
        win_w = min(want_w, avail_w)
        win_h = min(want_h, avail_h)
        pos_x = max(0, (min(screen_w, avail_w + 80) - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        self._configure_fonts(style)
        self._measure_fonts()
        self.root.minsize(GUI_MIN_CHARS[0] * self.char_w, GUI_MIN_CHARS[1] * self.line_h)

        outer = ttk.Frame(self.root, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        self._build_header(outer)

        body = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")
        self._build_object_panel(body)

        right = ttk.Frame(body)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        body.add(right, weight=4)
        self._build_summary_panel(right)
        self._build_controls(right)
        self._build_trial_table(right)

        ttk.Label(outer, textvariable=self.status_text, foreground=STATUS_COLOR).grid(
            row=2, column=0, sticky="w", pady=(10, 0)
        )

    def _build_header(self, outer: ttk.Frame) -> None:
        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=f"Data root: {GRASP_DATA_ROOT}").grid(row=0, column=0, sticky="w")
        self.refresh_button = ttk.Button(header, text="Refresh", command=self._load_objects)
        self.refresh_button.grid(row=0, column=1, sticky="e")

    def _build_object_panel(self, body: ttk.PanedWindow) -> None:
        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        body.add(left, weight=1)

        ttk.Label(left, text="Objects").grid(row=0, column=0, sticky="w")
        search = ttk.Entry(left, textvariable=self.search_text)
        search.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        self.search_text.trace_add("write", lambda *_args: self._filter_objects())

        object_frame = ttk.Frame(left)
        object_frame.grid(row=2, column=0, sticky="nsew")
        object_frame.columnconfigure(0, weight=1)
        object_frame.rowconfigure(0, weight=1)
        self.object_list = tk.Listbox(
            object_frame,
            exportselection=False,
            activestyle="none",
            font=tkfont.nametofont("TkDefaultFont"),
            height=18,
        )
        object_scroll = ttk.Scrollbar(object_frame, orient=tk.VERTICAL, command=self.object_list.yview)
        self.object_list.configure(yscrollcommand=object_scroll.set)
        self.object_list.grid(row=0, column=0, sticky="nsew")
        object_scroll.grid(row=0, column=1, sticky="ns")
        self.object_list.bind("<<ListboxSelect>>", lambda _event: self._on_object_selected())

    def _build_summary_panel(self, right: ttk.Frame) -> None:
        top_right = ttk.Frame(right)
        top_right.grid(row=0, column=0, sticky="ew")
        top_right.columnconfigure(1, weight=1)
        self.preview_label = ttk.Label(top_right, text="No object selected", anchor="center", width=28)
        self.preview_label.grid(row=0, column=0, sticky="nw", padx=(0, 14))
        self.object_summary = tk.Text(
            top_right,
            height=12,
            width=68,
            tabs=SUMMARY_TAB_CHARS * self.char_w,
            borderwidth=0,
            highlightthickness=0,
            bg=self.root.cget("bg"),
            fg="black",
            wrap="word",
            font=tkfont.nametofont("TkDefaultFont"),
        )
        self.object_summary.grid(row=0, column=1, sticky="nsew")
        self.object_summary.configure(state="disabled")
        _configure_label_tags(self.object_summary)

    def _build_controls(self, right: ttk.Frame) -> None:
        controls = ttk.Frame(right)
        controls.grid(row=1, column=0, sticky="ew", pady=(10, 6))
        controls.columnconfigure(0, weight=1)
        self.play_button = ttk.Button(controls, text="Play Selected Trial", command=self._start_playback)
        self.play_button.grid(row=0, column=1, sticky="e")
        self.play_button.configure(state="disabled")

    def _build_trial_table(self, right: ttk.Frame) -> None:
        trial_frame = ttk.Frame(right)
        trial_frame.grid(row=2, column=0, sticky="nsew")
        trial_frame.columnconfigure(0, weight=1)
        trial_frame.rowconfigure(1, weight=1)
        tabs = tuple(n * self.char_w for n in TRIAL_TABLE_TAB_CHARS)

        trial_header = tk.Text(
            trial_frame,
            height=1,
            borderwidth=0,
            highlightthickness=0,
            bg=self.root.cget("bg"),
            fg="black",
            cursor="arrow",
            tabs=tabs,
            font=tkfont.nametofont("TkHeadingFont"),
        )
        trial_header.grid(row=0, column=0, sticky="ew", padx=(4, 0))
        trial_header.insert(tk.END, "trial\tautomatic\tmanual")
        trial_header.configure(state="disabled")

        self.trial_text = tk.Text(
            trial_frame,
            height=18,
            borderwidth=1,
            relief="solid",
            highlightthickness=0,
            bg="white",
            fg="black",
            cursor="arrow",
            exportselection=False,
            takefocus=True,
            wrap="none",
            tabs=tabs,
            font=tkfont.nametofont("TkDefaultFont"),
            padx=4,
            pady=4,
        )
        trial_scroll = ttk.Scrollbar(trial_frame, orient=tk.VERTICAL, command=self.trial_text.yview)
        self.trial_text.configure(yscrollcommand=trial_scroll.set)
        self.trial_text.grid(row=1, column=0, sticky="nsew")
        trial_scroll.grid(row=1, column=1, sticky="ns")
        _configure_label_tags(self.trial_text)
        self.trial_text.tag_configure("selected", background=SELECTED_ROW_COLOR)
        self.trial_text.bind("<Button-1>", self._on_trial_clicked)
        self.trial_text.bind("<Double-1>", self._on_trial_double_clicked)
        self.trial_text.bind("<Up>", lambda _event: self._move_trial_selection(-1))
        self.trial_text.bind("<Down>", lambda _event: self._move_trial_selection(1))
        self.trial_text.bind("<Return>", self._on_trial_return_pressed)
        self.trial_text.configure(state="disabled")

    def _configure_fonts(self, style: ttk.Style) -> None:
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkTooltipFont"):
            try:
                font = tkfont.nametofont(name)
            except tk.TclError:
                continue
            font.configure(size=max(int(font.cget("size")), GUI_FONT_SIZE))
        try:
            heading_font = tkfont.nametofont("TkHeadingFont")
            heading_font.configure(size=max(int(heading_font.cget("size")), GUI_HEADING_FONT_SIZE), weight="bold")
        except tk.TclError:
            pass

        style.configure(".", font=tkfont.nametofont("TkDefaultFont"))
        style.configure("TButton", padding=(10, 6))
        row_height = int(round(tkfont.nametofont("TkDefaultFont").metrics("linespace") * GUI_ROW_HEIGHT_LINES))
        style.configure("Treeview", rowheight=row_height, font=tkfont.nametofont("TkDefaultFont"))
        style.configure("Treeview.Heading", font=tkfont.nametofont("TkHeadingFont"))

    def _measure_fonts(self) -> None:
        """Cache the width of one character and the height of one line."""
        font = tkfont.nametofont("TkDefaultFont")
        self.char_w = max(1, font.measure("0"))
        self.line_h = max(1, font.metrics("linespace"))

    def _load_objects(self) -> None:
        if self._playback_is_running():
            self.status_text.set("Playback is running. Close the OpenCV window first.")
            return
        self.object_dirs = _list_object_dirs()
        total_trials = sum(1 for _ in GRASP_DATA_ROOT.rglob("*.h5")) if GRASP_DATA_ROOT.is_dir() else 0
        self.status_text.set(f"Detected {len(self.object_dirs)} objects and {total_trials} H5 trials.")
        self._filter_objects()

    def _filter_objects(self) -> None:
        query = self.search_text.get().strip().lower()
        self.filtered_object_dirs = [
            path for path in self.object_dirs if not query or query in path.name.lower()
        ]
        self.object_list.delete(0, tk.END)
        for path in self.filtered_object_dirs:
            self.object_list.insert(tk.END, path.name)
        if self.filtered_object_dirs:
            self.object_list.selection_set(0)
            self._on_object_selected()
        else:
            self._clear_trials()

    def _on_object_selected(self) -> None:
        selection = self.object_list.curselection()
        if not selection:
            return
        idx = int(selection[0])
        if idx >= len(self.filtered_object_dirs):
            return
        object_dir = self.filtered_object_dirs[idx]
        self.selected_object = object_dir
        self.status_text.set(f"Loading trials for {object_dir.name}...")
        self.root.update_idletasks()
        self.trials = _load_trials_for_object(object_dir)
        self._populate_trials()
        self._update_preview(object_dir.name)
        self._update_object_summary()
        self.status_text.set(f"Selected {object_dir.name}: {len(self.trials)} trials.")

    def _clear_trials(self) -> None:
        self.selected_object = None
        self.trials = []
        self.selected_trial_index = None
        self._clear_trial_table()
        self.preview_photo = None
        self.preview_label.configure(image="", text="No object selected")
        self._set_object_summary([])
        self.play_button.configure(state="disabled")

    def _populate_trials(self) -> None:
        self.selected_trial_index = None
        self.trial_text.configure(state="normal")
        self.trial_text.delete("1.0", tk.END)
        for trial in self.trials:
            self.trial_text.insert(tk.END, f"{trial.path.name}\t")
            self._insert_trial_status(trial.automatic_label)
            self.trial_text.insert(tk.END, "\t")
            self._insert_trial_status(trial.manual_label)
            self.trial_text.insert(tk.END, "\n")
        self.trial_text.configure(state="disabled")
        self._select_trial_index(0 if self.trials else None)

    def _clear_trial_table(self) -> None:
        self.trial_text.configure(state="normal")
        self.trial_text.delete("1.0", tk.END)
        self.trial_text.configure(state="disabled")

    def _insert_trial_status(self, status: str) -> None:
        tag = status if status in {"success", "failure"} else "unknown"
        self.trial_text.insert(tk.END, status, tag)

    def _trial_index_from_event(self, event: tk.Event) -> int | None:
        line_text = self.trial_text.index(f"@{event.x},{event.y}").split(".", maxsplit=1)[0]
        try:
            idx = int(line_text) - 1
        except ValueError:
            return None
        return idx if 0 <= idx < len(self.trials) else None

    def _select_trial_index(self, idx: int | None) -> None:
        if idx is not None and not (0 <= idx < len(self.trials)):
            idx = None
        self.selected_trial_index = idx
        self.trial_text.configure(state="normal")
        self.trial_text.tag_remove("selected", "1.0", tk.END)
        if idx is not None:
            line_no = idx + 1
            self.trial_text.tag_add("selected", f"{line_no}.0", f"{line_no}.end+1c")
            self.trial_text.see(f"{line_no}.0")
        self.trial_text.configure(state="disabled")
        self._update_play_state()

    def _on_trial_clicked(self, event: tk.Event) -> str:
        self.trial_text.focus_set()
        self._select_trial_index(self._trial_index_from_event(event))
        return "break"

    def _on_trial_double_clicked(self, event: tk.Event) -> str:
        self._select_trial_index(self._trial_index_from_event(event))
        self._start_playback()
        return "break"

    def _move_trial_selection(self, delta: int) -> str:
        if not self.trials:
            self._select_trial_index(None)
            return "break"
        if self.selected_trial_index is None:
            new_idx = 0
        else:
            new_idx = min(max(self.selected_trial_index + delta, 0), len(self.trials) - 1)
        self._select_trial_index(new_idx)
        return "break"

    def _on_trial_return_pressed(self, _event: tk.Event) -> str:
        self._start_playback()
        return "break"

    def _update_preview(self, object_name: str) -> None:
        side = PREVIEW_LINES * self.line_h
        self.preview_photo = _load_preview_photo(object_name, (side, side))
        if self.preview_photo is None:
            self.preview_label.configure(image="", text="No preview image")
        else:
            self.preview_label.configure(image=self.preview_photo, text="")

    def _update_object_summary(self) -> None:
        if self.selected_object is None:
            self._set_object_summary([])
            return
        manual_success = sum(1 for trial in self.trials if trial.manual_label == "success")
        manual_failure = sum(1 for trial in self.trials if trial.manual_label == "failure")
        manual_unknown = len(self.trials) - manual_success - manual_failure
        auto_success = sum(1 for trial in self.trials if trial.automatic_label == "success")
        auto_failure = sum(1 for trial in self.trials if trial.automatic_label == "failure")
        auto_unknown = len(self.trials) - auto_success - auto_failure
        manual_known = manual_success + manual_failure
        success_rate = None if manual_known == 0 else manual_success / manual_known * 100.0
        success_rate_text = "n/a" if success_rate is None else f"{success_rate:.1f}% ({manual_success}/{manual_known})"
        parts = [
            [("Object: ", None), (self.selected_object.name, None), ("\n", None)],
        ]
        properties = self.object_properties.get(self.selected_object.name)
        if properties is not None:
            parts += [
                [("\n", None)],
                [("Material: ", None), (properties.material, None),
                 ("\tCompliance: ", None), (properties.compliance, None), ("\n", None)],
                [("Weight: ", None), (properties.weight, None),
                 ("\tSize: ", None), (properties.size, None), ("\n", None)],
            ]
        parts += [
            [("\n", None)],
            [("Trials: ", None), (str(len(self.trials)), None), ("\n", None)],
            [
                ("Automatic labels: ", None),
                ("success", "success"),
                (f"={auto_success}, ", None),
                ("failure", "failure"),
                (f"={auto_failure}, unknown={auto_unknown}\n", None),
            ],
            [
                ("Manual labels: ", None),
                ("success", "success"),
                (f"={manual_success}, ", None),
                ("failure", "failure"),
                (f"={manual_failure}, unknown={manual_unknown}\n", None),
            ],
            [("\nGrasp success rate: ", None), (success_rate_text, None), ("\n", None)],
            [("\nOpenCV playback: q quits, p pauses, click toggles pause, drag timeline seeks.", None)],
        ]
        self._set_object_summary(parts)

    def _set_object_summary(self, parts: list[list[tuple[str, str | None]]]) -> None:
        self.object_summary.configure(state="normal")
        self.object_summary.delete("1.0", tk.END)
        for line in parts:
            for text, tag in line:
                if tag:
                    self.object_summary.insert(tk.END, text, tag)
                else:
                    self.object_summary.insert(tk.END, text)
        self.object_summary.configure(state="disabled")

    def _selected_trial(self) -> TrialInfo | None:
        if self.selected_trial_index is None:
            return None
        if not (0 <= self.selected_trial_index < len(self.trials)):
            return None
        return self.trials[self.selected_trial_index]

    def _update_play_state(self) -> None:
        state = "normal" if self._selected_trial() is not None and not self._playback_is_running() else "disabled"
        self.play_button.configure(state=state)

    def _playback_is_running(self) -> bool:
        return self.worker is not None and self.worker.poll() is None

    def _start_playback(self) -> None:
        trial = self._selected_trial()
        if trial is None:
            return
        if self._playback_is_running():
            self.status_text.set("Playback is already running.")
            return

        self.play_button.configure(state="disabled")
        self.refresh_button.configure(state="disabled")
        self.status_text.set(f"Playing {trial.object_name}/{trial.path.name}. Press q in the OpenCV window to stop.")
        env = os.environ.copy()
        env[INTERNAL_PLAY_ENV] = str(trial.path)
        env[INTERNAL_OBJECT_ENV] = trial.object_name
        env[INTERNAL_SESSION_ENV] = trial.session_name
        env[INTERNAL_AUTO_LABEL_ENV] = trial.automatic_label
        env[INTERNAL_MANUAL_LABEL_ENV] = trial.manual_label
        self.worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], cwd=str(SCRIPT_DIR), env=env)
        self.root.after(150, self._poll_worker)

    def _poll_worker(self) -> None:
        if not self.worker:
            return
        return_code = self.worker.poll()
        if return_code is None:
            self.root.after(150, self._poll_worker)
            return
        self.worker = None
        self.refresh_button.configure(state="normal")
        self._update_play_state()
        if return_code != 0:
            self.status_text.set("Playback failed. Check the terminal output for details.")
            messagebox.showerror("Playback failed", f"Playback exited with code {return_code}.", parent=self.root)
        else:
            self.status_text.set("Playback finished.")

    def _on_close(self) -> None:
        if self._playback_is_running():
            self.status_text.set("Close the OpenCV playback window first.")
            messagebox.showinfo("Playback is running", "Press q in the OpenCV window before closing this viewer.", parent=self.root)
            return
        cv2.destroyAllWindows()
        self.root.destroy()


def _trial_from_internal_env() -> TrialInfo | None:
    path_text = os.environ.get(INTERNAL_PLAY_ENV)
    if not path_text:
        return None
    path = Path(path_text).expanduser().resolve()
    return TrialInfo(
        object_name=os.environ.get(INTERNAL_OBJECT_ENV, path.parent.name),
        path=path,
        session_name=os.environ.get(INTERNAL_SESSION_ENV, "") or _session_name(path),
        automatic_label=os.environ.get(INTERNAL_AUTO_LABEL_ENV, "unknown"),
        manual_label=os.environ.get(INTERNAL_MANUAL_LABEL_ENV, "unknown"),
    )


def main() -> None:
    internal_trial = _trial_from_internal_env()
    if internal_trial is not None:
        run_viewer(internal_trial)
        return

    root = tk.Tk()
    DataViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
