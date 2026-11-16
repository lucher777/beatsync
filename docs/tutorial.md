# Tutorial: Synchronizing Multi-Device ECG & PPG Streams

This tutorial demonstrates end-to-end synchronization using bundled real WESAD recordings, segmented jump recovery, and SciPy parity verification.

## 1. Real WESAD S10 Waveform Synchronization

```python
from beatsync import (
    SyncConfig,
    evaluate_alignment_accuracy,
    inject_waveform_distortion,
    load_wesad_waveform_slice,
    synchronize_clocks,
)

sl = load_wesad_waveform_slice()
cfg = SyncConfig(ptt_prior_s=0.250, bootstrap_draws=100)
native = synchronize_clocks(
    ecg=sl["ecg"],
    ppg=sl["bvp"],
    fs_ecg_hz=sl["fs_ecg_hz"],
    fs_ppg_hz=sl["fs_bvp_hz"],
    config=cfg,
)

ecg_in, ppg_dist, t_eval = inject_waveform_distortion(
    sl["ecg"], sl["bvp"], sl["fs_ecg_hz"], sl["fs_bvp_hz"],
    offset_s=-48.5, drift_ppm=240.0,
)
result = synchronize_clocks(
    ecg=ecg_in,
    ppg=ppg_dist,
    fs_ecg_hz=sl["fs_ecg_hz"],
    fs_ppg_hz=sl["fs_bvp_hz"],
    config=cfg,
)
print(result.summary())
print(evaluate_alignment_accuracy(result, t_eval, -48.5, 240.0, baseline_result=native))
```

## 2. Five-Subject Event Cohort & Piecewise Jump Segmentation

```python
import numpy as np
from beatsync import load_wesad_event_cohort, synchronize_segmented_streams

cohort = load_wesad_event_cohort()
print(cohort[0]["events"].summary())

# Map target timestamps and resample a continuous signal onto the reference grid
t_target = np.linspace(0.0, 300.0, 19200)
aligned = result.align_series(t_target, ppg_dist[:19200], t_ref_grid_s=np.linspace(0.0, 300.0, 19200))
```

For recordings with packet drops or backward clock resets, `synchronize_segmented_streams(t_ecg_s, ecg, t_ppg_s, ppg, config=cfg)` detects timestamp discontinuities via `detect_timestamp_jumps` and fits piecewise trajectories (`SegmentSyncResult.map_piecewise`).

## 3. Command-Line Interface & SciPy Equivalence

```bash
beatsync demo --offset -48.5 --drift-ppm 240
beatsync benchmark
beatsync equiv
```
