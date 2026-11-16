# API Reference

## Configuration & Containers (`beatsync.types`)

- `SyncConfig`: Frozen configuration dataclass (`grid_fs_hz`, `coarse_sigma_s`, `fine_sigma_s`, `ptt_prior_s`, `max_coarse_lag_s`, `window_s`, `step_s`, `top_k_peaks`, `slip_threshold_s`, `max_drift_ppm`, `mad_iterations`, `bootstrap_draws`).
- `CardiacEvents`: Detected ECG R-peaks and PPG systolic peak container with `.mean_hr_bpm()` and `.summary()`.
- `CandidatePeak` & `WindowEstimate`: Per-window cross-correlation candidates and rescue status.
- `DriftFitResult`: Affine model parameters (`intercept_s`, `slope`, `drift_ppm`, bootstrap bounds, `n_inliers`, `mad_residual_s`) with `.predict(t_s)`.
- `SyncResult`: Complete synchronization output with `.predict_offset(t_ref_s)`, `.map_target_to_ref(t_target_s)`, `.map_ref_to_target(t_ref_s)`, `.align_series(...)`, `.window_dataframe()`, and `.summary()`.
- `SegmentBoundary` & `SegmentSyncResult`: Piecewise segmented synchronization output with `.map_piecewise(t_target_s)` and `.to_dataframe()`.
- `EquivalenceReport`: Cross-package parity report relative to SciPy reference routines.

## Event Extraction & Impulse Encoding (`beatsync.events`)

- `bandpass_zero_phase(signal, fs_hz, low_hz, high_hz, order=2)`
- `detect_ecg_rpeaks(ecg, fs_ecg_hz, min_rr_s=0.33, percentile_q=80.0, height_scale=0.35, refine_subsample=True)`
- `detect_ppg_peaks(ppg, fs_ppg_hz, min_ibi_s=0.33, ectopic_ratio=0.50, filter_short_ibi=True, refine_subsample=True)`
- `detect_cardiac_events(ecg, ppg, fs_ecg_hz, fs_ppg_hz, subject_id="recording", filter_short_ibi=True, refine_subsample=True)`
- `encode_gaussian_impulses(event_times_s, duration_s, grid_fs_hz=64.0, sigma_s=0.035)`
- `extract_motion_envelope(accel, fs_in_hz, grid_fs_hz=64.0)`

## Cross-Correlation & Peak Extraction (`beatsync.xcorr`)

- `fft_xcorr(ref_series, target_series, min_lag_s, max_lag_s, grid_fs_hz=64.0)`
- `direct_xcorr_reference(ref_series, target_series, min_lag_s, max_lag_s, grid_fs_hz=64.0)`
- `compute_prominence_quality(corr, exclusion_bins)`
- `extract_topk_peaks(lags_s, corr, grid_fs_hz, ptt_prior_s, top_k=5, min_peak_sep_s=0.32)`
- `dual_scale_xcorr(events, config)`

## Cycle-Slip Rescue & Trajectory Fitting (`beatsync.trajectory`)

- `fit_theil_sen_drift(times_s, offsets_s, ...)`
- `score_subperiod_trajectory(intercept_s, slope, times_s, offset_matrix, weight_matrix, quality_weights, kernel_width_s=0.075)`
- `rescue_subperiod_peaks(raw_windows, config)`
- `synchronize_clocks(ecg=None, ppg=None, fs_ecg_hz=250.0, fs_ppg_hz=64.0, events=None, config=None, subject_id="recording")`

## Segmentation, Baselines, Datasets & Parity (`beatsync.segments`, `beatsync.datasets`, `beatsync.synthetic`)

- `detect_timestamp_jumps(timestamps_s, expected_dt_s=None, gap_threshold_s=2.0)`
- `synchronize_segmented_streams(t_ecg_s, ecg, t_ppg_s, ppg, config=None, gap_threshold_s=2.0, min_segment_s=45.0)`
- `evaluate_alignment_accuracy(result, eval_times_s, true_offset_s, true_drift_ppm, baseline_result=None, heldout_anchor_offset_s=None)`
- `fit_ibi_ols_baseline`, `fit_single_peak_baseline`, `fit_coarse_ols_baseline`, `fit_motion_xcorr_baseline`
- `verify_scipy_equivalence(events=None, seed=2026, tol=1e-12)`
- `load_wesad_waveform_slice()`, `load_wesad_event_cohort()`, `load_wesad_benchmark_tables()`
- `DualDeviceSimulator`, `inject_clock_distortion`, `inject_waveform_distortion`
