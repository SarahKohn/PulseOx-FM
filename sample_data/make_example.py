"""Generate a synthetic example pulse-oximetry recording for PulseOx-FM.

The output ``example_ppg.npy`` is a single 120-second, 3-channel segment shaped
``(1, 3, 15000)`` at 125 Hz, matching the model's expected input. The data is
entirely synthetic (generated from deterministic noise + sinusoids with a fixed
seed) and is **not** derived from any human cohort. It exists only so that the
forward pass and figure code can be exercised without access to real data.

Channels (relative index): 0 = SpO2, 1 = HR, 2 = PAT/PPG waveform.

Reproduce with:
    python sample_data/make_example.py
"""
from __future__ import annotations

import os

import numpy as np

FS = 125              # Hz
DURATION_S = 120      # seconds
N = FS * DURATION_S   # 15000 samples
SEED = 20260629       # fixed for reproducibility


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def make_example() -> np.ndarray:
    """Return a synthetic (1, 3, 15000) float32 array, z-scored per channel."""
    rng = np.random.default_rng(SEED)
    t = np.arange(N) / FS

    # Channel 0 — SpO2: ~flat near-ceiling trace with slow drift + small noise.
    spo2 = 97.0 + 0.8 * np.sin(2 * np.pi * t / 90.0) + rng.normal(0, 0.3, N)

    # Channel 1 — HR: ~60 bpm baseline with respiratory-sinus-arrhythmia wobble.
    hr = 60.0 + 4.0 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 1.0, N)

    # Channel 2 — PAT/PPG waveform: ~1 Hz pulse with a dicrotic-notch harmonic.
    pulse = 1.0 * np.sin(2 * np.pi * 1.0 * t) + 0.35 * np.sin(2 * np.pi * 2.0 * t + 0.6)
    ppg = pulse + rng.normal(0, 0.05, N)

    seg = np.stack([_zscore(spo2), _zscore(hr), _zscore(ppg)], axis=0)
    return seg[np.newaxis, :, :].astype(np.float32)  # (1, 3, 15000)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "example_ppg.npy")
    arr = make_example()
    np.save(out, arr)
    print(f"Wrote synthetic example: {out}  shape={arr.shape} dtype={arr.dtype}")
