# beatsync

**Dual-scale cardiac event cross-correlation and cycle-slip-resilient clock synchronization for multi-device wearables.**

`beatsync` aligns unsynchronized chest electrocardiography (ECG) and wrist photoplethysmography (PPG) recordings without requiring shared hardware master clocks, synchronization taps, or motion-rich intervals.

## Installation

```bash
pip install beatsync
```

For local development and running the test suite:

```bash
pip install -e .[dev]
pytest -v
```

## Three-Stage Architecture

1. **Stage 1 (`events.py`)**: Zero-phase Butterworth bandpass filtering (`5-25 Hz` ECG; `0.7-3.5 Hz` PPG), prominence-guarded peak extraction, non-interpolating short-IBI artifact pruning (`< 0.50 * median_IBI`), and dual-scale Gaussian impulse encoding (`sigma_w = 120 ms` coarse, `sigma_n = 35 ms` fine).
2. **Stage 2 (`xcorr.py`)**: Overlap-normalized power-of-two real FFT cross-correlation (`fft_xcorr`) for global minute-scale coarse locking (`+/-1800 s`), followed by PTT-compensated sliding windows extracting top-`K` parabolic sub-sample candidate peaks.
3. **Stage 3 (`trajectory.py`, `segments.py`)**: Cross-half anchor consensus cycle-slip rescue (`rescue_subperiod_peaks`), iterative MAD Theil-Sen drift regression (`fit_theil_sen_drift`) with bootstrap confidence intervals, and automatic packet-drop / clock-reset segmentation (`synchronize_segmented_streams`).
4. **Bundled Real WESAD Data & Parity Checks (`datasets.py`, `segments.py`)**: Ships with a real 360 s WESAD `S10` dual-device waveform slice, a 5-subject 15-minute event cohort, 75-segment and 15-session benchmark summaries, and `verify_scipy_equivalence()`.
