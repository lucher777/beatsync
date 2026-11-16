"""Multi-segment timestamp jump handling, comparative baselines, and SciPy parity checks."""
from __future__ import annotations

import time

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import theilslopes

from .events import bandpass_zero_phase, encode_gaussian_impulses
from .trajectory import fit_theil_sen_drift, synchronize_clocks
from .types import (
    CardiacEvents,
    DriftFitResult,
    EquivalenceReport,
    SegmentBoundary,
    SegmentSyncResult,
    SyncConfig,
    SyncResult,
)
from .xcorr import (
    compute_prominence_quality,
    direct_xcorr_reference,
    dual_scale_xcorr,
    extract_topk_peaks,
    fft_xcorr,
)


def detect_timestamp_jumps(
    timestamps_s: np.ndarray,
    expected_dt_s: float | None = None,
    gap_threshold_s: float = 2.0,
) -> list[tuple[int, int]]:
    """Split a recording timestamp array at backward resets or packet-drop gaps.

    Parameters
    ----------
    timestamps_s : np.ndarray
        Raw device timestamps in seconds.
    expected_dt_s : float or None, default=None
        Nominal inter-sample period in seconds; inferred from median positive
        difference when omitted.
    gap_threshold_s : float, default=2.0
        Forward gap duration (seconds) triggering a segment boundary.

    Returns
    -------
    list[tuple[int, int]]
        Half-open index slices ``[(start_idx, end_idx), ...]`` covering each
        contiguous monotonic segment.
    """
    ts = np.asarray(timestamps_s, dtype=np.float64).ravel()
    if ts.size == 0:
        raise ValueError("timestamps_s must be non-empty")
    if not np.all(np.isfinite(ts)):
        raise ValueError("timestamps_s must contain finite values")
    if ts.size == 1:
        return [(0, 1)]

    diffs = np.diff(ts)
    if expected_dt_s is None:
        pos_diffs = diffs[diffs > 0.0]
        if pos_diffs.size == 0:
            raise ValueError("Timestamps contain no positive increments")
        expected_dt_s = float(np.median(pos_diffs))

    jump_indices = np.where((diffs <= 0.0) | (diffs > max(gap_threshold_s, 3.0 * expected_dt_s)))[0] + 1
    bounds = [0] + [int(idx) for idx in jump_indices] + [int(ts.size)]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1) if bounds[k + 1] > bounds[k]]


def synchronize_segmented_streams(
    t_ecg_s: np.ndarray,
    ecg: np.ndarray,
    t_ppg_s: np.ndarray,
    ppg: np.ndarray,
    config: SyncConfig | None = None,
    gap_threshold_s: float = 2.0,
    min_segment_s: float = 60.0,
) -> SegmentSyncResult:
    """Detect clock resets or dropout gaps and synchronize each contiguous segment.

    Parameters
    ----------
    t_ecg_s : np.ndarray
        Reference ECG timestamps (seconds).
    ecg : np.ndarray
        Reference ECG samples matching ``t_ecg_s``.
    t_ppg_s : np.ndarray
        Target PPG timestamps (seconds).
    ppg : np.ndarray
        Target PPG samples matching ``t_ppg_s``.
    config : SyncConfig or None, default=None
        Synchronization parameters.
    gap_threshold_s : float, default=2.0
        Minimum timestamp gap (seconds) used to split contiguous segments.
    min_segment_s : float, default=60.0
        Minimum duration (seconds) required to synchronize an isolated segment.

    Returns
    -------
    SegmentSyncResult
        Chronologically ordered per-segment synchronization models.
    """
    cfg = config or SyncConfig()
    t_ref = np.asarray(t_ecg_s, dtype=np.float64).ravel()
    y_ref = np.asarray(ecg, dtype=np.float64).ravel()
    t_tgt = np.asarray(t_ppg_s, dtype=np.float64).ravel()
    y_tgt = np.asarray(ppg, dtype=np.float64).ravel()
    if t_ref.size != y_ref.size or t_tgt.size != y_tgt.size:
        raise ValueError("Timestamp and waveform arrays must share matching lengths")

    ref_slices = detect_timestamp_jumps(t_ref, gap_threshold_s=gap_threshold_s)
    tgt_slices = detect_timestamp_jumps(t_tgt, gap_threshold_s=gap_threshold_s)
    n_pairs = min(len(ref_slices), len(tgt_slices))

    segments: list[SegmentBoundary] = []
    for seg_i in range(n_pairs):
        r0, r1 = ref_slices[seg_i]
        p0, p1 = tgt_slices[seg_i]
        tr_seg = t_ref[r0:r1]
        tp_seg = t_tgt[p0:p1]
        if tr_seg.size < 32 or tp_seg.size < 32:
            continue
        dur_ref = float(tr_seg[-1] - tr_seg[0])
        dur_tgt = float(tp_seg[-1] - tp_seg[0])
        if dur_ref < min_segment_s or dur_tgt < min_segment_s:
            continue

        fs_ref = float(tr_seg.size - 1) / dur_ref
        fs_tgt = float(tp_seg.size - 1) / dur_tgt
        local_sync = synchronize_clocks(
            ecg=y_ref[r0:r1],
            ppg=y_tgt[p0:p1],
            fs_ecg_hz=fs_ref,
            fs_ppg_hz=fs_tgt,
            config=cfg,
            subject_id=f"seg_{seg_i}",
        )
        # Re-anchor local segment intercept onto global segment start timestamps
        delta_start = float(tr_seg[0] - tp_seg[0])
        b = local_sync.drift_fit.slope
        a_global = delta_start + local_sync.drift_fit.intercept_s - b * float(tr_seg[0])
        shifted_fit = DriftFitResult(
            intercept_s=a_global,
            slope=b,
            drift_ppm=local_sync.drift_fit.drift_ppm,
            ci_low_s=a_global + (local_sync.drift_fit.ci_low_s - local_sync.drift_fit.intercept_s),
            ci_high_s=a_global + (local_sync.drift_fit.ci_high_s - local_sync.drift_fit.intercept_s),
            slope_ci_low_ppm=local_sync.drift_fit.slope_ci_low_ppm,
            slope_ci_high_ppm=local_sync.drift_fit.slope_ci_high_ppm,
            n_input=local_sync.drift_fit.n_input,
            n_inliers=local_sync.drift_fit.n_inliers,
            mad_residual_s=local_sync.drift_fit.mad_residual_s,
            method=local_sync.drift_fit.method,
        )
        global_sync = SyncResult(
            coarse_offset_s=delta_start + local_sync.coarse_offset_s,
            coarse_lag_s=local_sync.coarse_lag_s,
            coarse_prominence=local_sync.coarse_prominence,
            drift_fit=shifted_fit,
            windows=local_sync.windows,
            n_windows_total=local_sync.n_windows_total,
            n_windows_retained=local_sync.n_windows_retained,
            n_windows_rescued=local_sync.n_windows_rescued,
            used_global_consensus=local_sync.used_global_consensus,
            top1_inlier_ratio_200ms=local_sync.top1_inlier_ratio_200ms,
            ptt_prior_s=local_sync.ptt_prior_s,
            config=cfg,
        )
        segments.append(
            SegmentBoundary(
                segment_index=seg_i,
                ref_start_s=float(tr_seg[0]),
                ref_end_s=float(tr_seg[-1]),
                target_start_s=float(tp_seg[0]),
                target_end_s=float(tp_seg[-1]),
                sync_result=global_sync,
            )
        )
    if not segments:
        raise ValueError("No valid recording segments met minimum duration criteria")
    return SegmentSyncResult(
        segments=tuple(segments),
        n_ref_jumps=max(0, len(ref_slices) - 1),
        n_target_jumps=max(0, len(tgt_slices) - 1),
    )


def evaluate_alignment_accuracy(
    result: SyncResult,
    eval_times_s: np.ndarray,
    true_offset_s: float,
    true_drift_ppm: float,
    baseline_result: SyncResult | None = None,
    heldout_anchor_offset_s: float | None = None,
) -> dict[str, float]:
    """Compute relative or held-out clock alignment error metrics across a grid.

    Parameters
    ----------
    result : SyncResult
        Synchronization estimate on the target or distorted stream.
    eval_times_s : np.ndarray
        Reference evaluation timestamps (seconds).
    true_offset_s : float
        Injected startup offset (seconds) at ``t_ref = 0``.
    true_drift_ppm : float
        Injected relative crystal drift in parts per million.
    baseline_result : SyncResult or None, default=None
        Optional zero-injection baseline fit on the native recording to subtract
        intrinsic sensor latency when computing relative injection error.
    heldout_anchor_offset_s : float or None, default=None
        Optional independent out-of-segment physiological anchor offset (seconds).

    Returns
    -------
    dict[str, float]
        Dictionary containing ``median_err_s``, ``max_err_s``, ``p95_err_s``,
        and ``drift_err_ppm``.
    """
    t_arr = np.asarray(eval_times_s, dtype=np.float64).ravel()
    pred = np.asarray(result.predict_offset(t_arr), dtype=np.float64)
    truth = true_offset_s + (true_drift_ppm * 1e-6) * t_arr
    if baseline_result is not None:
        truth = truth + np.asarray(baseline_result.predict_offset(t_arr), dtype=np.float64)
    elif heldout_anchor_offset_s is not None:
        truth = truth + float(heldout_anchor_offset_s)
    errs = np.abs(pred - truth)
    base_ppm = baseline_result.drift_fit.drift_ppm if baseline_result is not None else 0.0
    return {
        "median_err_s": float(np.median(errs)),
        "max_err_s": float(np.max(errs)),
        "p95_err_s": float(np.percentile(errs, 95.0)),
        "drift_err_ppm": float(abs(result.drift_fit.drift_ppm - base_ppm - true_drift_ppm)),
    }


def _fit_ols_clipped(
    times_s: np.ndarray,
    offsets_s: np.ndarray,
    max_drift_ppm: float,
    method_name: str,
) -> DriftFitResult:
    """Compute ordinary least-squares affine fit bounded by physical crystal limits."""
    t_arr = np.asarray(times_s, dtype=np.float64).ravel()
    y_arr = np.asarray(offsets_s, dtype=np.float64).ravel()
    if t_arr.size < 2:
        y0 = float(y_arr[0]) if y_arr.size else 0.0
        return DriftFitResult(y0, 0.0, 0.0, y0, y0, 0.0, 0.0, t_arr.size, t_arr.size, 0.0, method_name)
    slope, intercept = np.polyfit(t_arr, y_arr, 1)
    bound = max_drift_ppm * 1e-6
    slope_c = float(np.clip(slope, -bound, bound))
    inter_c = float(np.mean(y_arr - slope_c * t_arr))
    mad = float(np.median(np.abs(y_arr - (inter_c + slope_c * t_arr))))
    return DriftFitResult(
        intercept_s=inter_c,
        slope=slope_c,
        drift_ppm=slope_c * 1e6,
        ci_low_s=inter_c,
        ci_high_s=inter_c,
        slope_ci_low_ppm=slope_c * 1e6,
        slope_ci_high_ppm=slope_c * 1e6,
        n_input=int(t_arr.size),
        n_inliers=int(t_arr.size),
        mad_residual_s=mad,
        method=method_name,
    )


def fit_ibi_ols_baseline(
    events: CardiacEvents,
    config: SyncConfig | None = None,
) -> DriftFitResult:
    """Baseline: wide coarse lock followed by nearest-beat OLS regression."""
    cfg = config or SyncConfig()
    fs = cfg.grid_fs_hz
    hb_ref = encode_gaussian_impulses(events.ecg_rpeaks_s, events.ref_duration_s, fs, cfg.coarse_sigma_s)
    hb_tgt = encode_gaussian_impulses(events.ppg_peaks_s, events.target_duration_s, fs, cfg.coarse_sigma_s)
    bound = min(cfg.max_coarse_lag_s, min(events.ref_duration_s, events.target_duration_s) * 0.65)
    lags_c, corr_c = fft_xcorr(hb_ref, hb_tgt, -bound, bound, grid_fs_hz=fs)
    d0 = float(-(lags_c[int(np.argmax(corr_c))] - cfg.ptt_prior_s))

    p_shifted = events.ppg_peaks_s + d0
    idx_p = np.searchsorted(p_shifted, events.ecg_rpeaks_s)
    pair_t: list[float] = []
    pair_y: list[float] = []
    for r_time, pos in zip(events.ecg_rpeaks_s, idx_p):
        best_err = 0.40
        best_dp = None
        for jp in (pos - 1, pos, pos + 1):
            if 0 <= jp < events.n_ppg:
                err = abs((events.ppg_peaks_s[jp] + d0) - r_time)
                if err < best_err:
                    best_err = err
                    best_dp = (float(r_time) - cfg.ptt_prior_s) - float(events.ppg_peaks_s[jp])
        if best_dp is not None:
            pair_t.append(float(r_time))
            pair_y.append(float(best_dp))
    if len(pair_t) < 10:
        return DriftFitResult(d0, 0.0, 0.0, d0, d0, 0.0, 0.0, len(pair_t), len(pair_t), 0.0, "coarse_ibi_ols")
    return _fit_ols_clipped(np.asarray(pair_t), np.asarray(pair_y), cfg.max_drift_ppm, "coarse_ibi_ols")


def fit_single_peak_baseline(
    events: CardiacEvents,
    config: SyncConfig | None = None,
) -> DriftFitResult:
    """Ablation baseline: dual-scale coarse shift with top-1 Theil-Sen only (no rescue)."""
    cfg = config or SyncConfig()
    _, _, _, windows = dual_scale_xcorr(events, cfg)
    if not windows:
        return DriftFitResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "single_peak_theil_sen")
    qs = np.array([w.quality for w in windows], dtype=np.float64)
    q_cut = float(np.quantile(qs, cfg.quality_quantile))
    kept = [w for w in windows if w.quality >= q_cut] or windows
    times = np.array([w.center_time_s for w in kept], dtype=np.float64)
    offs = np.array([w.top1_offset() for w in kept], dtype=np.float64)
    fit = fit_theil_sen_drift(
        times,
        offs,
        max_drift_ppm=cfg.max_drift_ppm,
        min_slope_span_s=cfg.min_slope_span_s,
        mad_iterations=cfg.mad_iterations,
        mad_scale_k=cfg.mad_scale_k,
        bootstrap_draws=0,
    )
    return DriftFitResult(**{**fit.to_dict(), "method": "single_peak_theil_sen"})


def fit_coarse_ols_baseline(
    events: CardiacEvents,
    config: SyncConfig | None = None,
) -> DriftFitResult:
    """Ablation baseline: dual-scale coarse shift with top-1 OLS regression."""
    cfg = config or SyncConfig()
    _, _, _, windows = dual_scale_xcorr(events, cfg)
    if not windows:
        return DriftFitResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "coarse_top1_ols")
    qs = np.array([w.quality for w in windows], dtype=np.float64)
    q_cut = float(np.quantile(qs, cfg.quality_quantile))
    kept = [w for w in windows if w.quality >= q_cut] or windows
    times = np.array([w.center_time_s for w in kept], dtype=np.float64)
    offs = np.array([w.top1_offset() for w in kept], dtype=np.float64)
    return _fit_ols_clipped(times, offs, cfg.max_drift_ppm, "coarse_top1_ols")


def fit_motion_xcorr_baseline(
    chest_motion: np.ndarray,
    wrist_motion: np.ndarray,
    config: SyncConfig | None = None,
) -> DriftFitResult:
    """Baseline: sliding-window acceleration-envelope cross-correlation + Theil-Sen."""
    cfg = config or SyncConfig()
    fs = cfg.grid_fs_hz
    win_n = int(round(cfg.window_s * fs))
    step_n = max(1, int(round(cfg.step_s * fs)))
    n_avail = min(len(chest_motion), len(wrist_motion))
    max_lag = min(cfg.max_coarse_lag_s, 30.0)

    times: list[float] = []
    offs: list[float] = []
    quals: list[float] = []
    for st in range(0, max(1, n_avail - win_n + 1), step_n):
        en = st + win_n
        if en > n_avail:
            break
        seg_a = chest_motion[st:en]
        seg_b = wrist_motion[st:en]
        if float(np.std(seg_a)) < 1e-8 or float(np.std(seg_b)) < 1e-8:
            continue
        lags, corr = fft_xcorr(seg_a, seg_b, -max_lag, max_lag, grid_fs_hz=fs)
        top = extract_topk_peaks(lags, corr, grid_fs_hz=fs, ptt_prior_s=0.0, top_k=1)
        times.append((st + 0.5 * win_n) / float(fs))
        offs.append(top[0].offset_s)
        quals.append(compute_prominence_quality(corr))

    if len(times) < 2:
        return DriftFitResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, len(times), len(times), 0.0, "motion_theil_sen")
    q_cut = float(np.quantile(np.asarray(quals), cfg.quality_quantile))
    keep_mask = np.asarray(quals) >= q_cut
    t_arr = np.asarray(times)[keep_mask]
    y_arr = np.asarray(offs)[keep_mask]
    fit = fit_theil_sen_drift(
        t_arr,
        y_arr,
        max_drift_ppm=cfg.max_drift_ppm,
        min_slope_span_s=cfg.min_slope_span_s,
        mad_iterations=cfg.mad_iterations,
        mad_scale_k=cfg.mad_scale_k,
        bootstrap_draws=0,
    )
    return DriftFitResult(**{**fit.to_dict(), "method": "motion_theil_sen"})


def verify_scipy_equivalence(
    events: CardiacEvents | None = None,
    seed: int = 2026,
    tol: float = 1e-11,
) -> EquivalenceReport:
    """Verify exact numerical equivalence against SciPy reference implementations.

    Compares:
    1. ``fft_xcorr`` against ``scipy.signal.correlate(..., method="direct")``.
    2. ``fit_theil_sen_drift(..., min_slope_span_s=0.0, mad_iterations=0,
       intercept_mode="scipy_separate")`` against ``scipy.stats.theilslopes``.
    3. ``bandpass_zero_phase`` against explicit ``scipy.signal.butter`` + ``filtfilt``.

    Parameters
    ----------
    events : CardiacEvents or None, default=None
        Optional real cardiac event pair; when omitted, generates a deterministic
        180-second pulse pair at 64 Hz for direct comparison.
    seed : int, default=2026
        Deterministic random seed for test vector construction.
    tol : float, default=1e-11
        Strict upper tolerance for machine-precision numerical agreement.

    Returns
    -------
    EquivalenceReport
        Numerical error bounds, runtime speedup ratio, and pass/fail status.
    """
    rng = np.random.default_rng(seed)
    fs = 64.0
    if events is not None:
        dur = min(180.0, events.ref_duration_s, events.target_duration_s)
        r_sub = events.ecg_rpeaks_s[events.ecg_rpeaks_s <= dur]
        p_sub = events.ppg_peaks_s[events.ppg_peaks_s <= dur]
    else:
        dur = 180.0
        ibis = 0.80 + 0.05 * np.sin(np.linspace(0.0, 12.0 * np.pi, 220))
        r_sub = np.cumsum(ibis)
        r_sub = r_sub[r_sub < dur - 2.0]
        p_sub = r_sub + 0.25 + rng.normal(0.0, 0.005, size=r_sub.size)

    a = encode_gaussian_impulses(r_sub, dur, grid_fs_hz=fs, sigma_s=0.035)
    b = encode_gaussian_impulses(p_sub, dur, grid_fs_hz=fs, sigma_s=0.035)

    t0 = time.perf_counter()
    lags_fft, corr_fft = fft_xcorr(a, b, -15.0, 15.0, grid_fs_hz=fs)
    fft_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    lags_dir, corr_dir = direct_xcorr_reference(a, b, -15.0, 15.0, grid_fs_hz=fs)
    direct_ms = (time.perf_counter() - t1) * 1000.0

    xcorr_err = float(np.max(np.abs(corr_fft - corr_dir)) + np.max(np.abs(lags_fft - lags_dir)))
    speedup = direct_ms / max(fft_ms, 1e-6)

    x_pts = np.linspace(22.5, 877.5, 40)
    y_pts = -14.25 + 2.75e-4 * x_pts + rng.normal(0.0, 0.006, size=40)
    ts_ref = theilslopes(y_pts, x_pts)
    ts_ours = fit_theil_sen_drift(
        x_pts,
        y_pts,
        max_drift_ppm=5000.0,
        min_slope_span_s=0.0,
        mad_iterations=0,
        intercept_mode="scipy_separate",
        bootstrap_draws=0,
    )
    slope_err = float(abs(ts_ours.slope - float(ts_ref.slope)))
    intercept_err = float(abs(ts_ours.intercept_s - float(ts_ref.intercept)))

    sig = np.sin(2.0 * np.pi * 1.3 * np.arange(4096) / 250.0) + 0.25 * rng.normal(size=4096)
    nyq = 125.0
    b_coef, a_coef = butter(2, [5.0 / nyq, 25.0 / nyq], btype="band")
    bp_ref = filtfilt(b_coef, a_coef, sig)
    bp_ours = bandpass_zero_phase(sig, 250.0, 5.0, 25.0, order=2)
    bp_err = float(np.max(np.abs(bp_ours - bp_ref)))

    passed = bool(
        xcorr_err <= tol
        and slope_err <= tol
        and intercept_err <= tol
        and bp_err <= tol
    )
    return EquivalenceReport(
        xcorr_max_abs_err=xcorr_err,
        xcorr_fft_ms=fft_ms,
        xcorr_direct_ms=direct_ms,
        xcorr_speedup=speedup,
        theil_sen_slope_abs_err=slope_err,
        theil_sen_intercept_abs_err=intercept_err,
        bandpass_max_abs_err=bp_err,
        passed=passed,
    )
