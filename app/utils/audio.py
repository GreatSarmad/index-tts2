"""Utility helpers for manipulating audio files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def apply_speed_and_pitch(
    source: Path,
    destination: Path,
    speed: float = 1.0,
    pitch: float = 0.0,
) -> Tuple[Path, float]:
    """Apply speed and pitch adjustments to an audio file."""

    source = Path(source)
    destination = Path(destination)

    y, sr = librosa.load(source, sr=None)
    if y.size == 0:
        raise ValueError("Audio prompt is empty.")

    if not math.isclose(speed, 1.0, rel_tol=1e-3):
        y = librosa.effects.time_stretch(y, rate=float(speed))

    if not math.isclose(pitch, 0.0, abs_tol=1e-3):
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(pitch))

    sf.write(destination, y.astype(np.float32), sr, subtype="PCM_16")
    return destination, sr
