#!/usr/bin/env python3
"""MPC Silence Split & Wavetable Builder

Loads a WAV file containing multiple sounds separated by silence, removes the
silent areas, exports each sound as an individual WAV, and creates a
pitch-synchronous wavetable WAV from every sound.

Designed for Akai MPC 3.9+ user oscillator workflows. The wavetable output is a
mono WAV made from consecutive equal-length single-cycle frames.
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import traceback
import webbrowser
import zipfile
import shutil
import tempfile
import wave

try:
    import winsound
except ImportError:  # Windows-only preview backend
    winsound = None
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: numpy. Install with: pip install numpy") from exc

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: soundfile. Install with: pip install soundfile") from exc

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None


APP_TITLE = "MPC Silence Split & Wavetable Builder"


@dataclass
class Segment:
    index: int
    start: int
    end: int
    peak: float
    rms_db: float
    enabled: bool = True
    duplicate_of: Optional[int] = None
    similarity: float = 0.0

    def duration(self, sample_rate: int) -> float:
        return max(0, self.end - self.start) / float(sample_rate)


def amplitude_to_db(value: float, floor_db: float = -120.0) -> float:
    if value <= 0.0:
        return floor_db
    return max(floor_db, 20.0 * math.log10(value))


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=1, dtype=np.float32)
    raise ValueError(f"Unsupported audio shape: {audio.shape}")


def moving_rms(audio: np.ndarray, window_samples: int) -> np.ndarray:
    window_samples = max(1, int(window_samples))
    squared = np.square(audio, dtype=np.float64)
    kernel = np.ones(window_samples, dtype=np.float64) / window_samples
    return np.sqrt(np.convolve(squared, kernel, mode="same")).astype(np.float32)


def detect_segments(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float,
    min_silence_ms: float,
    min_sound_ms: float,
    padding_ms: float,
    analysis_window_ms: float = 10.0,
) -> List[Segment]:
    """Detect non-silent regions and merge gaps shorter than min_silence_ms."""
    if len(audio) == 0:
        return []

    window = max(1, round(sample_rate * analysis_window_ms / 1000.0))
    rms = moving_rms(audio, window)
    threshold = 10.0 ** (threshold_db / 20.0)
    active = rms >= threshold

    changes = np.flatnonzero(np.diff(active.astype(np.int8))) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(active)]
    values = active[starts]
    raw_regions = [(int(s), int(e)) for s, e, on in zip(starts, ends, values) if on]
    if not raw_regions:
        return []

    max_gap = round(sample_rate * min_silence_ms / 1000.0)
    merged: List[List[int]] = [[raw_regions[0][0], raw_regions[0][1]]]
    for start, end in raw_regions[1:]:
        if start - merged[-1][1] < max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    pad = round(sample_rate * padding_ms / 1000.0)
    min_sound = round(sample_rate * min_sound_ms / 1000.0)
    segments: List[Segment] = []
    for start, end in merged:
        start = max(0, start - pad)
        end = min(len(audio), end + pad)
        if end - start < min_sound:
            continue
        chunk = audio[start:end]
        peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
        rms_value = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)))) if len(chunk) else 0.0
        segments.append(
            Segment(
                index=len(segments) + 1,
                start=start,
                end=end,
                peak=peak,
                rms_db=amplitude_to_db(rms_value),
            )
        )
    return segments


def fade_edges(audio: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    result = np.array(audio, dtype=np.float32, copy=True)
    fade = min(len(result) // 2, max(0, round(sample_rate * fade_ms / 1000.0)))
    if fade > 0:
        result[:fade] *= np.linspace(0.0, 1.0, fade, endpoint=True, dtype=np.float32)
        result[-fade:] *= np.linspace(1.0, 0.0, fade, endpoint=True, dtype=np.float32)
    return result


def normalise_peak(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 1e-12:
        return np.array(audio, dtype=np.float32, copy=True)
    target = 10.0 ** (target_db / 20.0)
    return (audio * (target / peak)).astype(np.float32)


def rising_zero_crossings(audio: np.ndarray) -> np.ndarray:
    if len(audio) < 2:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero((audio[:-1] <= 0.0) & (audio[1:] > 0.0)) + 1


def estimate_period_autocorrelation(
    audio: np.ndarray,
    sample_rate: int,
    min_freq: float,
    max_freq: float,
) -> Optional[int]:
    if len(audio) < 8:
        return None
    min_lag = max(2, int(sample_rate / max_freq))
    max_lag = min(len(audio) - 2, int(sample_rate / min_freq))
    if max_lag <= min_lag:
        return None

    x = np.asarray(audio, dtype=np.float64)
    x = x - np.mean(x)
    window = np.hanning(len(x))
    x *= window
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    if corr[0] <= 1e-15:
        return None
    corr /= corr[0]
    search = corr[min_lag : max_lag + 1]
    if len(search) == 0:
        return None
    lag = int(np.argmax(search)) + min_lag
    return lag


def linear_resample(audio: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Resample length must be positive")
    if len(audio) == length:
        return np.array(audio, dtype=np.float32, copy=True)
    if len(audio) < 2:
        return np.zeros(length, dtype=np.float32)
    old_x = np.linspace(0.0, 1.0, len(audio), endpoint=True)
    new_x = np.linspace(0.0, 1.0, length, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def high_quality_resample(audio: np.ndarray, length: int) -> np.ndarray:
    if resample_poly is None or len(audio) < 2:
        return linear_resample(audio, length)
    gcd = math.gcd(len(audio), length)
    up = length // gcd
    down = len(audio) // gcd
    output = resample_poly(audio, up, down)
    if len(output) < length:
        output = np.pad(output, (0, length - len(output)))
    return np.asarray(output[:length], dtype=np.float32)


def extract_cycle_near(
    audio: np.ndarray,
    center: int,
    sample_rate: int,
    min_freq: float,
    max_freq: float,
) -> Optional[np.ndarray]:
    """Extract one likely waveform cycle around center using rising crossings."""
    if len(audio) < 8:
        return None

    max_period = max(4, int(sample_rate / min_freq))
    left = max(0, center - max_period * 3)
    right = min(len(audio), center + max_period * 3)
    local = audio[left:right]
    crossings = rising_zero_crossings(local) + left
    if len(crossings) >= 2:
        before = crossings[crossings <= center]
        start_idx = int(before[-1]) if len(before) else int(crossings[0])
        after = crossings[crossings > start_idx]
        if len(after):
            periods = after - start_idx
            min_period = int(sample_rate / max_freq)
            max_period_allowed = int(sample_rate / min_freq)
            valid = after[(periods >= min_period) & (periods <= max_period_allowed)]
            if len(valid):
                end_idx = int(valid[0])
                cycle = audio[start_idx:end_idx]
                if len(cycle) >= 4:
                    return cycle

    analysis_radius = min(max_period * 2, max(16, len(audio) // 2))
    a = max(0, center - analysis_radius)
    b = min(len(audio), center + analysis_radius)
    period = estimate_period_autocorrelation(audio[a:b], sample_rate, min_freq, max_freq)
    if period is None:
        return None
    start = max(0, min(len(audio) - period, center - period // 2))
    return audio[start : start + period]


def remove_dc_and_close_cycle(cycle: np.ndarray) -> np.ndarray:
    cycle = np.asarray(cycle, dtype=np.float32)
    if len(cycle) == 0:
        return cycle
    cycle = cycle - np.mean(cycle)
    # Remove a small endpoint mismatch without applying a full fade that would
    # alter the waveform too much.
    mismatch = float(cycle[-1] - cycle[0])
    cycle = cycle - np.linspace(0.0, mismatch, len(cycle), dtype=np.float32)
    return cycle


def build_wavetable(
    audio: np.ndarray,
    sample_rate: int,
    frame_count: int,
    frame_size: int,
    min_freq: float,
    max_freq: float,
    normalise_each_frame: bool,
) -> Tuple[np.ndarray, dict]:
    """Create a concatenated single-cycle wavetable from a sound segment."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < 16:
        raise ValueError("Segment is too short for wavetable creation")

    # Avoid fades and silence padding when selecting cycles.
    abs_audio = np.abs(audio)
    active_threshold = max(1e-5, float(np.max(abs_audio)) * 0.015)
    active_indices = np.flatnonzero(abs_audio >= active_threshold)
    if len(active_indices):
        active_start = int(active_indices[0])
        active_end = int(active_indices[-1]) + 1
    else:
        active_start, active_end = 0, len(audio)

    margin = max(1, int((active_end - active_start) * 0.02))
    active_start = min(active_end - 1, active_start + margin)
    active_end = max(active_start + 1, active_end - margin)
    positions = np.linspace(active_start, active_end - 1, frame_count, dtype=np.int64)

    frames = []
    extracted_periods = []
    previous: Optional[np.ndarray] = None
    for center in positions:
        cycle = extract_cycle_near(audio, int(center), sample_rate, min_freq, max_freq)
        if cycle is None or len(cycle) < 4:
            if previous is None:
                # Last-resort local window. It still gives a valid frame and
                # keeps the batch export from failing on noisy/transient audio.
                radius = max(2, min(len(audio) // 2, int(sample_rate / 110.0)))
                a = max(0, int(center) - radius // 2)
                cycle = audio[a : min(len(audio), a + radius)]
            else:
                frames.append(previous.copy())
                extracted_periods.append(None)
                continue

        extracted_periods.append(len(cycle))
        cycle = remove_dc_and_close_cycle(cycle)
        frame = high_quality_resample(cycle, frame_size)
        frame = remove_dc_and_close_cycle(frame)
        if normalise_each_frame:
            peak = float(np.max(np.abs(frame)))
            if peak > 1e-9:
                frame = frame / peak
        previous = frame.astype(np.float32)
        frames.append(previous.copy())

    table = np.concatenate(frames).astype(np.float32)
    table = normalise_peak(table, -1.0)
    valid_periods = [p for p in extracted_periods if p]
    metadata = {
        "format": "concatenated-single-cycle-wavetable",
        "frame_count": frame_count,
        "frame_size": frame_size,
        "total_samples": int(len(table)),
        "sample_rate": int(sample_rate),
        "cycle_detection_hz": [float(min_freq), float(max_freq)],
        "median_source_cycle_samples": float(np.median(valid_periods)) if valid_periods else None,
        "normalised": True,
    }
    return table, metadata



def duplicate_fingerprint(audio: np.ndarray, length: int = 2048) -> np.ndarray:
    """Create a level-independent fingerprint for duplicate detection."""
    x = np.asarray(audio, dtype=np.float32)
    if len(x) < 8:
        return np.zeros(length // 2 + 1, dtype=np.float32)
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x)))
    if peak > 1e-9:
        x = x / peak
    x = linear_resample(x, length)
    windowed = x * np.hanning(len(x)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windowed)).astype(np.float32)
    spectrum = np.log1p(spectrum)
    norm = float(np.linalg.norm(spectrum))
    if norm > 1e-12:
        spectrum /= norm
    return spectrum


def mark_duplicate_segments(audio: np.ndarray, segments: Sequence[Segment], threshold: float) -> None:
    """Mark later, highly similar segments as duplicates and disable them."""
    fingerprints: list[np.ndarray] = []
    accepted_indices: list[int] = []
    for segment in segments:
        segment.duplicate_of = None
        segment.similarity = 0.0
        chunk = audio[segment.start:segment.end]
        fp = duplicate_fingerprint(chunk)
        best_similarity = 0.0
        best_segment_index: Optional[int] = None
        for accepted_pos, accepted_fp in zip(accepted_indices, fingerprints):
            similarity = float(np.dot(fp, accepted_fp))
            candidate = segments[accepted_pos]
            duration_ratio = min(len(chunk), candidate.end-candidate.start) / max(len(chunk), candidate.end-candidate.start)
            similarity *= float(duration_ratio)
            if similarity > best_similarity:
                best_similarity = similarity
                best_segment_index = candidate.index
        segment.similarity = best_similarity
        if best_segment_index is not None and best_similarity >= threshold:
            segment.duplicate_of = best_segment_index
            segment.enabled = False
        else:
            segment.enabled = True
            accepted_indices.append(segment.index - 1)
            fingerprints.append(fp)

def safe_stem(text: str) -> str:
    valid = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    cleaned = "".join(ch if ch in valid else "_" for ch in text.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "sound"



class WaveformCanvas(tk.Canvas):
    BG = "#11151b"
    WAVE = "#52b7ff"
    MID = "#303845"
    ACTIVE = "#ff9d2e"
    ENABLED = "#43d17d"
    DISABLED = "#69727e"
    HANDLE = "#ffffff"

    def __init__(self, master, on_select=None, on_boundary_change=None, **kwargs):
        super().__init__(master, background=self.BG, highlightthickness=0, cursor="crosshair", **kwargs)
        self.audio: Optional[np.ndarray] = None
        self.segments: Sequence[Segment] = []
        self.selected_index: Optional[int] = None
        self.on_select = on_select
        self.on_boundary_change = on_boundary_change
        self._drag_side: Optional[str] = None
        self.bind("<Configure>", lambda _event: self.redraw())
        self.bind("<Button-1>", self._mouse_down)
        self.bind("<B1-Motion>", self._mouse_drag)
        self.bind("<ButtonRelease-1>", self._mouse_up)

    def set_data(self, audio: Optional[np.ndarray], segments: Sequence[Segment], selected_index: Optional[int] = None) -> None:
        self.audio = audio
        self.segments = segments
        self.selected_index = selected_index
        self.redraw()

    def _selected_segment(self) -> Optional[Segment]:
        if self.selected_index is None:
            return None
        return next((s for s in self.segments if s.index == self.selected_index), None)

    def _sample_from_x(self, x: float) -> int:
        if self.audio is None or len(self.audio) == 0:
            return 0
        width = max(1, self.winfo_width())
        return max(0, min(len(self.audio) - 1, int(round(x / width * len(self.audio)))))

    def _mouse_down(self, event) -> None:
        if self.audio is None or not self.segments:
            return
        width = max(1, self.winfo_width())
        total = len(self.audio)
        selected = self._selected_segment()
        if selected is not None:
            sx = selected.start / total * width
            ex = selected.end / total * width
            if abs(event.x - sx) <= 10:
                self._drag_side = "start"
                self.configure(cursor="sb_h_double_arrow")
                return
            if abs(event.x - ex) <= 10:
                self._drag_side = "end"
                self.configure(cursor="sb_h_double_arrow")
                return
        sample = self._sample_from_x(event.x)
        hit = next((s for s in self.segments if s.start <= sample <= s.end), None)
        if hit is not None and self.on_select:
            self.on_select(hit.index)

    def _mouse_drag(self, event) -> None:
        segment = self._selected_segment()
        if segment is None or self.audio is None or self._drag_side is None:
            return
        sample = self._sample_from_x(event.x)
        if self._drag_side == "start":
            segment.start = max(0, min(sample, segment.end - 1))
        else:
            segment.end = min(len(self.audio), max(sample, segment.start + 1))
        if self.on_boundary_change:
            self.on_boundary_change(segment, False)
        self.redraw()

    def _mouse_up(self, _event) -> None:
        segment = self._selected_segment()
        changed = self._drag_side is not None
        self._drag_side = None
        self.configure(cursor="crosshair")
        if changed and segment is not None and self.on_boundary_change:
            self.on_boundary_change(segment, True)

    def redraw(self) -> None:
        self.delete("all")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        self.create_line(0, height / 2, width, height / 2, fill=self.MID)
        if self.audio is None or len(self.audio) == 0:
            self.create_text(width / 2, height / 2, text="Drop or open a WAV file", fill="#9aa6b2", font=("Segoe UI", 12))
            return

        samples_per_pixel = max(1, int(math.ceil(len(self.audio) / width)))
        for x in range(width):
            a = x * samples_per_pixel
            b = min(len(self.audio), a + samples_per_pixel)
            if a >= len(self.audio):
                break
            peak = float(np.max(np.abs(self.audio[a:b]))) if b > a else 0.0
            y1 = height / 2 - peak * height * 0.42
            y2 = height / 2 + peak * height * 0.42
            self.create_line(x, y1, x, y2, fill=self.WAVE)

        total = len(self.audio)
        for segment in self.segments:
            x1 = segment.start / total * width
            x2 = segment.end / total * width
            is_selected = self.selected_index == segment.index
            outline = self.ACTIVE if is_selected else (self.ENABLED if segment.enabled else self.DISABLED)
            self.create_rectangle(x1, 3, x2, height - 3, outline=outline, width=3 if is_selected else 2)
            self.create_text(x1 + 6, 8, text=str(segment.index), fill=outline, anchor="nw", font=("Segoe UI Semibold", 10))
            if is_selected:
                self.create_line(x1, 0, x1, height, fill=self.HANDLE, width=2)
                self.create_line(x2, 0, x2, height, fill=self.HANDLE, width=2)
                self.create_rectangle(x1 - 4, height / 2 - 10, x1 + 4, height / 2 + 10, fill=self.ACTIVE, outline=self.HANDLE)
                self.create_rectangle(x2 - 4, height / 2 - 10, x2 + 4, height / 2 + 10, fill=self.ACTIVE, outline=self.HANDLE)


class App(tk.Tk):
    WEBSITE = "https://mpctoolkit.com"

    def __init__(self) -> None:
        super().__init__()
        self.title("MPC Wavetable Studio")
        self.geometry("1100x720")
        self.minsize(820, 600)
        self.configure(bg="#0d1117")

        self.audio: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.source_path: Optional[Path] = None
        self.segments: List[Segment] = []
        self.selected_segment_index: Optional[int] = None
        self.message_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self.preview_temp: Optional[Path] = None
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.threshold_db = tk.DoubleVar(value=-48.0)
        self.min_silence_ms = tk.DoubleVar(value=250.0)
        self.min_sound_ms = tk.DoubleVar(value=80.0)
        self.padding_ms = tk.DoubleVar(value=5.0)
        self.fade_ms = tk.DoubleVar(value=3.0)
        self.frame_count = tk.IntVar(value=64)
        self.frame_size = tk.IntVar(value=2048)
        self.min_freq = tk.DoubleVar(value=35.0)
        self.max_freq = tk.DoubleVar(value=2000.0)
        self.normalise_segments = tk.BooleanVar(value=True)
        self.normalise_frames = tk.BooleanVar(value=True)
        self.export_cleaned_wavs = tk.BooleanVar(value=False)
        self.detect_duplicates = tk.BooleanVar(value=True)
        self.duplicate_threshold = tk.DoubleVar(value=0.96)
        self.set_name = tk.StringVar(value="Wavetable")
        self.start_ms = tk.DoubleVar(value=0.0)
        self.end_ms = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar(value="Ready")

        self._configure_style()
        self._build_ui()
        self.after(100, self._poll_messages)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background="#171c24", foreground="#e8edf2", fieldbackground="#10151c", bordercolor="#303844", font=("Segoe UI", 10))
        style.configure("TFrame", background="#171c24")
        style.configure("Card.TFrame", background="#171c24", relief="flat")
        style.configure("TLabel", background="#171c24", foreground="#dfe7ee")
        style.configure("Muted.TLabel", foreground="#92a0ae", font=("Segoe UI", 9))
        style.configure("Title.TLabel", foreground="#ffffff", font=("Segoe UI Semibold", 21))
        style.configure("Accent.TLabel", foreground="#ff9d2e", font=("Segoe UI Semibold", 10))
        style.configure("TLabelframe", background="#171c24", foreground="#ffffff", bordercolor="#303844")
        style.configure("TLabelframe.Label", background="#171c24", foreground="#ffffff", font=("Segoe UI Semibold", 11))
        style.configure("TButton", background="#28313d", foreground="#ffffff", padding=(9, 5), borderwidth=0)
        style.map("TButton", background=[("active", "#354252"), ("pressed", "#202832")])
        style.configure("Accent.TButton", background="#f08b23", foreground="#111111", font=("Segoe UI Semibold", 10), padding=(12, 6))
        style.map("Accent.TButton", background=[("active", "#ffa43b"), ("pressed", "#d87919")])
        style.configure("Treeview", background="#11161d", fieldbackground="#11161d", foreground="#dce4eb", rowheight=24, borderwidth=0)
        style.configure("Treeview.Heading", background="#242c36", foreground="#ffffff", font=("Segoe UI Semibold", 9), relief="flat")
        style.map("Treeview", background=[("selected", "#8b561f")], foreground=[("selected", "#ffffff")])
        style.configure("TEntry", padding=4)
        style.configure("TCheckbutton", background="#171c24", foreground="#dfe7ee")
        style.configure("Horizontal.TProgressbar", troughcolor="#0e1319", background="#ff9d2e", bordercolor="#0e1319")

    def _build_ui(self) -> None:
        # Compact layout: the footer always stays visible and the settings are
        # split into tabs instead of being stacked vertically.
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 6))
        title_box = ttk.Frame(header)
        title_box.pack(side="left", fill="x", expand=True)
        ttk.Label(title_box, text="MPC Wavetable Studio", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_box, text="Silence splitter and MPC wavetable ZIP creator", style="Muted.TLabel").pack(anchor="w")
        ttk.Button(header, text="mpctoolkit.com", command=lambda: webbrowser.open(self.WEBSITE)).pack(side="right")

        # Two compact toolbar rows remain usable on narrower windows.
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 6))
        topbar = ttk.Frame(toolbar)
        topbar.pack(fill="x")
        ttk.Button(topbar, text="Open WAV", command=self.load_wav).pack(side="left")
        ttk.Button(topbar, text="Detect Sounds", command=self.detect).pack(side="left", padx=5)
        ttk.Button(topbar, text="Export MPC ZIP", style="Accent.TButton", command=self.export).pack(side="right")

        previewbar = ttk.Frame(toolbar)
        previewbar.pack(fill="x", pady=(5, 0))
        ttk.Button(previewbar, text="Play Full", command=self.preview_full).pack(side="left")
        ttk.Button(previewbar, text="Play Selected", command=self.preview_selected_sequence).pack(side="left", padx=5)
        ttk.Button(previewbar, text="Play Segment", command=self.preview_segment).pack(side="left")
        ttk.Button(previewbar, text="Stop", command=self.stop_preview).pack(side="left", padx=5)
        ttk.Button(previewbar, text="Select All", command=lambda: self.set_all(True)).pack(side="right")
        ttk.Button(previewbar, text="Select None", command=lambda: self.set_all(False)).pack(side="right", padx=5)

        self.waveform = WaveformCanvas(
            outer, height=165,
            on_select=self.select_segment_from_waveform,
            on_boundary_change=self.boundary_changed_from_waveform,
        )
        self.waveform.pack(fill="x", pady=(0, 7))

        content = ttk.Panedwindow(outer, orient="horizontal")
        content.pack(fill="both", expand=True)

        left = ttk.Frame(content, padding=(0, 0, 5, 0))
        right = ttk.Frame(content, padding=(5, 0, 0, 0))
        content.add(left, weight=1)
        content.add(right, weight=3)

        settings = ttk.Notebook(left)
        settings.pack(fill="both", expand=True)

        export_card = ttk.Frame(settings, padding=9)
        settings.add(export_card, text="Export")
        ttk.Label(export_card, text="Wavetable name prefix").grid(row=0, column=0, sticky="w")
        ttk.Entry(export_card, textvariable=self.set_name).grid(row=1, column=0, sticky="ew", pady=(3, 5))
        ttk.Label(
            export_card,
            text="Creates Prefix1, Prefix2 … directly in Oscillators/Wavetables.",
            style="Muted.TLabel", wraplength=260,
        ).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(
            export_card, text="Include cleaned split WAV files",
            variable=self.export_cleaned_wavs,
        ).grid(row=3, column=0, sticky="w", pady=(7, 0))
        export_card.columnconfigure(0, weight=1)

        detect_card = ttk.Frame(settings, padding=9)
        settings.add(detect_card, text="Silence")
        self._setting_row(detect_card, 0, "Threshold", self.threshold_db, "dBFS")
        self._setting_row(detect_card, 1, "Minimum silence", self.min_silence_ms, "ms")
        self._setting_row(detect_card, 2, "Minimum sound", self.min_sound_ms, "ms")
        self._setting_row(detect_card, 3, "Edge padding", self.padding_ms, "ms")
        self._setting_row(detect_card, 4, "Fade", self.fade_ms, "ms")
        ttk.Checkbutton(
            detect_card, text="Deselect duplicate sounds",
            variable=self.detect_duplicates,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self._setting_row(detect_card, 6, "Duplicate similarity", self.duplicate_threshold, "0–1")
        detect_card.columnconfigure(1, weight=1)

        wt_card = ttk.Frame(settings, padding=9)
        settings.add(wt_card, text="Wavetable")
        self._setting_row(wt_card, 0, "Frames", self.frame_count, "")
        self._setting_row(wt_card, 1, "Samples per frame", self.frame_size, "")
        self._setting_row(wt_card, 2, "Lowest pitch", self.min_freq, "Hz")
        self._setting_row(wt_card, 3, "Highest pitch", self.max_freq, "Hz")
        ttk.Checkbutton(wt_card, text="Normalise source segments", variable=self.normalise_segments).grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(wt_card, text="Normalise wavetable frames", variable=self.normalise_frames).grid(row=5, column=0, columnspan=3, sticky="w")
        wt_card.columnconfigure(1, weight=1)

        table_card = ttk.LabelFrame(right, text="Detected Sounds", padding=7)
        table_card.pack(fill="both", expand=True)
        columns = ("use", "number", "start", "end", "duration", "duplicate", "peak", "rms")
        self.tree = ttk.Treeview(table_card, columns=columns, show="headings", selectmode="browse")
        labels = {"use":"Use", "number":"#", "start":"Start", "end":"End", "duration":"Length", "duplicate":"Duplicate", "peak":"Peak", "rms":"RMS"}
        widths = {"use":42, "number":34, "start":72, "end":72, "duration":68, "duplicate":130, "peak":62, "rms":62}
        for col in columns:
            self.tree.heading(col, text=labels[col])
            self.tree.column(col, width=widths[col], anchor="center", stretch=col in {"start", "end", "duplicate"})
        yscroll = ttk.Scrollbar(table_card, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(table_card, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_card.rowconfigure(0, weight=1)
        table_card.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Double-1>", self.toggle_selected)
        self.tree.bind("<space>", self.toggle_selected)

        edit_card = ttk.LabelFrame(right, text="Selected Segment", padding=7)
        edit_card.pack(fill="x", pady=(6, 0))
        ttk.Label(edit_card, text="Start").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(edit_card, textvariable=self.start_ms, from_=0.0, to=99999999.0, increment=1.0, width=10, command=self.apply_boundaries).grid(row=0, column=1, padx=(4, 3))
        ttk.Label(edit_card, text="ms").grid(row=0, column=2, sticky="w")
        ttk.Label(edit_card, text="End").grid(row=0, column=3, sticky="w", padx=(10, 0))
        ttk.Spinbox(edit_card, textvariable=self.end_ms, from_=0.0, to=99999999.0, increment=1.0, width=10, command=self.apply_boundaries).grid(row=0, column=4, padx=(4, 3))
        ttk.Label(edit_card, text="ms").grid(row=0, column=5, sticky="w")
        ttk.Button(edit_card, text="Apply", command=self.apply_boundaries).grid(row=0, column=6, padx=(10, 3))
        ttk.Label(
            edit_card,
            text="Drag the white waveform handles or enter exact millisecond values.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=7, sticky="w", pady=(4, 0))

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(6, 0))
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(footer, mode="determinate", length=150)
        self.progress.pack(side="right", padx=(6, 0))
        link = tk.Label(
            footer, text="MPC Toolkit", bg="#171c24", fg="#ff9d2e",
            cursor="hand2", font=("Segoe UI Semibold", 9),
        )
        link.pack(side="right", padx=(6, 0))
        link.bind("<Button-1>", lambda _e: webbrowser.open(self.WEBSITE))

    @staticmethod
    def _setting_row(parent, row: int, label: str, variable: tk.Variable, unit: str) -> None:
        ranges = {
            "Threshold": (-120.0, 0.0, 1.0),
            "Minimum silence": (1.0, 10000.0, 10.0),
            "Minimum sound": (1.0, 10000.0, 10.0),
            "Edge padding": (0.0, 2000.0, 1.0),
            "Fade": (0.0, 1000.0, 1.0),
            "Duplicate similarity": (0.50, 0.9999, 0.01),
            "Frames": (1, 256, 1),
            "Samples per frame": (32, 8192, 32),
            "Lowest pitch": (1.0, 5000.0, 1.0),
            "Highest pitch": (2.0, 20000.0, 10.0),
        }
        low, high, step = ranges.get(label, (-999999.0, 999999.0, 1.0))
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Spinbox(parent, textvariable=variable, from_=low, to=high, increment=step, width=11).grid(row=row, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(parent, text=unit, style="Muted.TLabel").grid(row=row, column=2, sticky="w")

    def load_wav(self) -> None:
        filename = filedialog.askopenfilename(title="Select source WAV", filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")])
        if not filename:
            return
        try:
            data, sr = sf.read(filename, dtype="float32", always_2d=False)
            mono = ensure_mono(data)
            if len(mono) == 0:
                raise ValueError("The selected WAV file is empty")
            self.audio, self.sample_rate, self.source_path = mono, int(sr), Path(filename)
            self.segments = []
            self.selected_segment_index = None
            default_name = safe_stem(self.source_path.stem)
            if default_name:
                self.set_name.set(default_name)
            self._refresh_table()
            self.waveform.set_data(self.audio, self.segments)
            self.status.set(f"Loaded {self.source_path.name} · {len(mono) / sr:.2f} s · {sr} Hz · mono")
            self.detect()
        except Exception as exc:
            messagebox.showerror("MPC Wavetable Studio", f"Could not load the WAV file:\n\n{exc}")

    def detect(self) -> None:
        if self.audio is None or self.sample_rate is None:
            messagebox.showinfo("MPC Wavetable Studio", "Open a WAV file first.")
            return
        try:
            self.segments = detect_segments(self.audio, self.sample_rate, self.threshold_db.get(), self.min_silence_ms.get(), self.min_sound_ms.get(), self.padding_ms.get())
            if self.detect_duplicates.get() and self.segments:
                threshold = min(0.9999, max(0.5, float(self.duplicate_threshold.get())))
                mark_duplicate_segments(self.audio, self.segments, threshold)
            self.selected_segment_index = self.segments[0].index if self.segments else None
            self._refresh_table()
            if self.selected_segment_index:
                self.tree.selection_set(str(self.selected_segment_index))
                self.on_tree_select()
            self.waveform.set_data(self.audio, self.segments, self.selected_segment_index)
            duplicates = sum(1 for segment in self.segments if segment.duplicate_of is not None)
            self.status.set(f"Detected {len(self.segments)} sound segment(s) · {duplicates} duplicate(s) deselected.")
        except Exception as exc:
            messagebox.showerror("MPC Wavetable Studio", f"Detection failed:\n\n{exc}")

    def _refresh_table(self) -> None:
        if not hasattr(self, "tree"):
            return
        for item in self.tree.get_children():
            self.tree.delete(item)
        if self.sample_rate is None:
            return
        for segment in self.segments:
            duplicate_text = "—" if segment.duplicate_of is None else f"#{segment.duplicate_of} · {segment.similarity * 100:.1f}%"
            self.tree.insert("", "end", iid=str(segment.index), values=("Yes" if segment.enabled else "No", segment.index, f"{segment.start/self.sample_rate:.3f}s", f"{segment.end/self.sample_rate:.3f}s", f"{segment.duration(self.sample_rate):.3f}s", duplicate_text, f"{amplitude_to_db(segment.peak):.1f} dB", f"{segment.rms_db:.1f} dB"))

    def on_tree_select(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected or self.sample_rate is None:
            return
        idx = int(selected[0]) - 1
        if not (0 <= idx < len(self.segments)):
            return
        segment = self.segments[idx]
        self.selected_segment_index = segment.index
        self.start_ms.set(round(segment.start / self.sample_rate * 1000.0, 2))
        self.end_ms.set(round(segment.end / self.sample_rate * 1000.0, 2))
        self.waveform.set_data(self.audio, self.segments, self.selected_segment_index)

    def apply_boundaries(self) -> None:
        if self.sample_rate is None or self.audio is None or self.selected_segment_index is None:
            return
        idx = self.selected_segment_index - 1
        try:
            start = int(round(self.start_ms.get() / 1000.0 * self.sample_rate))
            end = int(round(self.end_ms.get() / 1000.0 * self.sample_rate))
            start = max(0, min(start, len(self.audio)-1))
            end = max(start + 1, min(end, len(self.audio)))
            segment = self.segments[idx]
            segment.start, segment.end = start, end
            self._recalculate_segment_stats(segment)
            self._refresh_table()
            self.tree.selection_set(str(segment.index))
            self.waveform.set_data(self.audio, self.segments, self.selected_segment_index)
            self.status.set(f"Updated boundaries for segment {segment.index}.")
        except Exception as exc:
            messagebox.showerror("MPC Wavetable Studio", f"Invalid segment boundaries:\n\n{exc}")

    def select_segment_from_waveform(self, segment_index: int) -> None:
        if not (1 <= segment_index <= len(self.segments)):
            return
        self.tree.selection_set(str(segment_index))
        self.tree.see(str(segment_index))
        self.on_tree_select()

    def boundary_changed_from_waveform(self, segment: Segment, final: bool) -> None:
        if self.sample_rate is None or self.audio is None:
            return
        self.selected_segment_index = segment.index
        self.start_ms.set(round(segment.start / self.sample_rate * 1000.0, 2))
        self.end_ms.set(round(segment.end / self.sample_rate * 1000.0, 2))
        if final:
            self._recalculate_segment_stats(segment)
            self._refresh_table()
            self.tree.selection_set(str(segment.index))
            self.status.set(f"Updated boundaries for segment {segment.index} with the mouse.")

    def _recalculate_segment_stats(self, segment: Segment) -> None:
        if self.audio is None:
            return
        chunk = self.audio[segment.start:segment.end]
        segment.peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)))) if len(chunk) else 0.0
        segment.rms_db = amplitude_to_db(rms)

    def _write_preview_wav(self, audio: np.ndarray) -> Path:
        if self.sample_rate is None:
            raise ValueError("No sample rate available")
        self.stop_preview()
        fd, name = tempfile.mkstemp(prefix="mpc_wavetable_preview_", suffix=".wav")
        os.close(fd)
        path = Path(name)
        sf.write(path, np.asarray(audio, dtype=np.float32), self.sample_rate, subtype="PCM_16")
        self.preview_temp = path
        return path

    def _play_audio(self, audio: np.ndarray, label: str) -> None:
        if winsound is None:
            messagebox.showerror("MPC Wavetable Studio", "Audio preview is available in the Windows build.")
            return
        if len(audio) == 0:
            return
        try:
            path = self._write_preview_wav(audio)
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.status.set(f"Playing {label}…")
        except Exception as exc:
            messagebox.showerror("MPC Wavetable Studio", f"Could not play audio:\n\n{exc}")

    def preview_full(self) -> None:
        if self.audio is None:
            messagebox.showinfo("MPC Wavetable Studio", "Open a WAV file first.")
            return
        self._play_audio(self.audio, "the full source")

    def preview_segment(self) -> None:
        if self.audio is None or self.selected_segment_index is None:
            messagebox.showinfo("MPC Wavetable Studio", "Select a sound segment first.")
            return
        segment = self.segments[self.selected_segment_index - 1]
        self._play_audio(self.audio[segment.start:segment.end], f"segment {segment.index}")

    def preview_selected_sequence(self) -> None:
        if self.audio is None or self.sample_rate is None:
            messagebox.showinfo("MPC Wavetable Studio", "Open a WAV file first.")
            return
        selected = [s for s in self.segments if s.enabled]
        if not selected:
            messagebox.showinfo("MPC Wavetable Studio", "No sound segments are selected.")
            return
        gap = np.zeros(max(1, int(self.sample_rate * 0.08)), dtype=np.float32)
        parts = []
        for segment in selected:
            parts.append(self.audio[segment.start:segment.end])
            parts.append(gap)
        self._play_audio(np.concatenate(parts), "all selected segments")

    def stop_preview(self) -> None:
        if winsound is not None:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except RuntimeError:
                pass
        if self.preview_temp is not None:
            try:
                self.preview_temp.unlink(missing_ok=True)
            except OSError:
                pass
            self.preview_temp = None
        if hasattr(self, "status"):
            self.status.set("Preview stopped")

    def _on_close(self) -> None:
        self.stop_preview()
        self.destroy()

    def toggle_selected(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        idx = int(selected[0]) - 1
        if 0 <= idx < len(self.segments):
            self.segments[idx].enabled = not self.segments[idx].enabled
            self._refresh_table()
            self.tree.selection_set(str(idx + 1))
            self.on_tree_select()

    def set_all(self, enabled: bool) -> None:
        for segment in self.segments:
            segment.enabled = enabled
        self._refresh_table()
        self.waveform.set_data(self.audio, self.segments, self.selected_segment_index)

    def export(self) -> None:
        if self.audio is None or self.sample_rate is None or self.source_path is None:
            messagebox.showinfo("MPC Wavetable Studio", "Open and analyse a WAV file first.")
            return
        selected = [s for s in self.segments if s.enabled]
        if not selected:
            messagebox.showinfo("MPC Wavetable Studio", "No sound segments are selected.")
            return
        prefix = safe_stem(self.set_name.get())
        if not prefix:
            messagebox.showinfo("MPC Wavetable Studio", "Enter a set name or wavetable prefix.")
            return
        filename = filedialog.asksaveasfilename(title="Save MPC wavetable ZIP", defaultextension=".zip", initialfile=f"{prefix}_MPC_Wavetables.zip", filetypes=[("ZIP archive", "*.zip")])
        if not filename:
            return
        try:
            settings = {
                "prefix": prefix,
                "frame_count": int(self.frame_count.get()),
                "frame_size": int(self.frame_size.get()),
                "min_freq": float(self.min_freq.get()),
                "max_freq": float(self.max_freq.get()),
                "fade_ms": float(self.fade_ms.get()),
                "normalise_segments": bool(self.normalise_segments.get()),
                "normalise_frames": bool(self.normalise_frames.get()),
                "export_cleaned_wavs": bool(self.export_cleaned_wavs.get()),
            }
            if not 1 <= settings["frame_count"] <= 256:
                raise ValueError("Frames must be between 1 and 256")
            if not 32 <= settings["frame_size"] <= 8192:
                raise ValueError("Samples per frame must be between 32 and 8192")
            if settings["min_freq"] <= 0 or settings["max_freq"] <= settings["min_freq"]:
                raise ValueError("Pitch range is invalid")
        except Exception as exc:
            messagebox.showerror("MPC Wavetable Studio", str(exc))
            return
        self.progress.configure(maximum=len(selected), value=0)
        self.status.set("Creating MPC ZIP…")
        threading.Thread(target=self._export_worker, args=(Path(filename), selected, settings), daemon=True).start()

    def _export_worker(self, zip_path: Path, selected: List[Segment], settings: dict) -> None:
        assert self.audio is not None and self.sample_rate is not None
        work_dir = zip_path.parent / f".{zip_path.stem}_build"
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir)
            root = work_dir / "Oscillators" / "Wavetables"
            root.mkdir(parents=True, exist_ok=True)
            cleaned_dir = work_dir / "Cleaned WAVs"
            if settings["export_cleaned_wavs"]:
                cleaned_dir.mkdir(parents=True, exist_ok=True)

            for out_index, segment in enumerate(selected, start=1):
                wt_name = f"{settings['prefix']}{out_index}"
                chunk = np.array(self.audio[segment.start:segment.end], dtype=np.float32, copy=True)
                chunk = fade_edges(chunk, self.sample_rate, settings["fade_ms"])
                if settings["normalise_segments"]:
                    chunk = normalise_peak(chunk, -1.0)
                if settings["export_cleaned_wavs"]:
                    sf.write(cleaned_dir / f"{wt_name}.wav", chunk, self.sample_rate, subtype="PCM_24")

                table, _metadata = build_wavetable(chunk, self.sample_rate, settings["frame_count"], settings["frame_size"], settings["min_freq"], settings["max_freq"], settings["normalise_frames"])
                wt_dir = root / wt_name
                wt_dir.mkdir(parents=True, exist_ok=True)
                sf.write(wt_dir / f"{wt_name}.wav", table, self.sample_rate, subtype="PCM_24")
                format_json = {"formatInfo": {"numSamplesPerSingleCycle": settings["frame_size"], "numSingleCycles": settings["frame_count"]}}
                (wt_dir / "format.json").write_text(json.dumps(format_json, indent=2), encoding="utf-8")
                self.message_queue.put(("progress", out_index))

            zip_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for file in work_dir.rglob("*"):
                    if file.is_file():
                        zf.write(file, file.relative_to(work_dir))
            self.message_queue.put(("done", (len(selected), zip_path)))
        except Exception as exc:
            self.message_queue.put(("error", f"{exc}\n\n{traceback.format_exc()}"))
        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.message_queue.get_nowait()
                if kind == "progress":
                    self.progress.configure(value=int(payload))
                    self.status.set(f"Creating wavetable {payload}…")
                elif kind == "done":
                    count, zip_path = payload
                    self.status.set(f"Finished · {count} wavetable(s) · {zip_path.name}")
                    messagebox.showinfo("MPC Wavetable Studio", f"MPC ZIP created successfully.\n\nWavetables: {count}\nFile: {zip_path}")
                elif kind == "error":
                    self.status.set("Export failed")
                    messagebox.showerror("MPC Wavetable Studio", f"Export failed:\n\n{payload}")
        except queue.Empty:
            pass
        self.after(100, self._poll_messages)


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
