# beatsync

**Dual-scale cardiac event cross-correlation and cycle-slip-resilient clock synchronization for asynchronous multi-device wearables.**

`beatsync` recovers large startup clock offsets (up to +/-30 minutes), linear oscillator crystal drift (ppm), and hardware filter latencies between independent chest electrocardiography (ECG) and peripheral wrist photoplethysmography (PPG) devices without requiring shared hardware master clocks, synchronization taps, or motion-rich intervals.

## Core Methodology

1. **Stage 1 — Cardiac Event Extraction & Dual-Scale Gaussian Impulse Encoding (`events.py`)**
   Zero-phase Butterworth bandpass filtering (`5-25 Hz` ECG, `0.7-3.5 Hz` PPG), non-interpolating short-IBI artifact suppression (`< 0.50 * median_IBI`), and projection onto uniform impulse grids convolved with both a drift-tolerant wide Gaussian kernel (`sigma_w = 120 ms`) and a sub-sample narrow kernel (`sigma_n = 35 ms`).
2. **Stage 2 — Coarse Global FFT Lock & Sub-Period Multi-Peak Rescue (`xcorr.py`, `trajectory.py`)**
   Overlap-normalized power-of-two real FFT cross-correlation locks global startup offset `delta_0`, followed by pulse transit time (`tau_PTT = 0.25 s`) compensated sliding windows extracting top-K parabolic sub-sample peaks. When wrist motion induces integer-period cycle slips (`+/- 1 IBI`), cross-half anchor consensus scoring and secondary-peak rescue recover the true physiological trajectory.
3. **Stage 3 — Iterative-MAD Theil-Sen Drift Fitting & Jump Segmentation (`trajectory.py`, `segments.py`)**
   Pairwise median slope estimation with iterative Median Absolute Deviation outlier pruning, physical oscillator bounds (`+/- 1000 ppm`), percentile bootstrap 95% confidence intervals, and automatic timestamp-jump segmentation (`synchronize_segmented_streams`).

## Installation

```bash
pip install beatsync
```

For local development with test and plotting extras:

```bash
pip install -e .[dev]
pytest -v
```

## Five-Minute Quickstart (Real Bundled WESAD Data)

```python
from beatsync import (
    SyncConfig,
    evaluate_alignment_accuracy,
    inject_waveform_distortion,
    load_wesad_waveform_slice,
    synchronize_clocks,
)

sl = load_wesad_waveform_slice()
cfg = SyncConfig(ptt_prior_s=0.250)
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

## Command-Line Interface

```bash
beatsync demo --offset -48.5 --drift-ppm 240
beatsync benchmark
beatsync equiv
beatsync sync --ecg chest.csv --ppg wrist.csv --fs-ecg 250 --fs-ppg 64 --out-json sync.json
```

## Cross-Package Numerical Verification

`beatsync.verify_scipy_equivalence()` verifies:
- `fft_xcorr` agrees with `scipy.signal.correlate(..., method="direct")` within `< 1e-12` while achieving `> 100x` speedup on multi-minute recordings.
- `fit_theil_sen_drift` matches `scipy.stats.theilslopes` within `< 1e-14`.
- `bandpass_zero_phase` matches `scipy.signal.butter` + `filtfilt` identically (`< 1e-12`).

## Documentation & Contributing

Full tutorials and API documentation are maintained in `docs/` (`mkdocs.yml`). See `CONTRIBUTING.md` for developer setup and test verification instructions.

## Citation

If you use `beatsync` in academic work, please cite `CITATION.cff` and the accompanying dataset attribution (Schmidt et al., 2018, doi:10.1145/3242969.3242985).
