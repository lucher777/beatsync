"""Stage 2b & Stage 3: Sub-period trajectory consensus, cycle-slip rescue, and Theil-Sen drift estimation."""
from __future__ import annotations

import numpy as np

from .events import detect_cardiac_events
from .types import (
    CardiacEvents,
    DriftFitResult,
    SyncConfig,
    SyncResult,
    WindowEstimate,
)
from .xcorr import dual_scale_xcorr


def _pairwise_theil_sen_core(
    times_s: np.ndarray,
    offsets_s: np.ndarray,
    max_drift_ppm: float,
    min_slope_span_s: float,
    intercept_mode: str,
) -> tuple[float, float]:
    """Compute a single Theil-Sen slope and intercept pass on 1D arrays."""
    n = times_s.size
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(offsets_s[0]), 0.0

    i_idx, j_idx = np.triu_indices(n, k=1)
    dt = times_s[j_idx] - times_s[i_idx]
    valid = dt > min_slope_span_s if min_slope_span_s > 0.0 else np.abs(dt) > 1e-12
    slope_bound = float(max_drift_ppm) * 1e-6
    if np.any(valid):
        slopes = (offsets_s[j_idx[valid]] - offsets_s[i_idx[valid]]) / dt[valid]
        slope = float(np.clip(np.median(slopes), -slope_bound, slope_bound))
    else:
        slope = 0.0

    if intercept_mode == "scipy_separate":
        intercept = float(np.median(offsets_s) - slope * np.median(times_s))
    elif intercept_mode == "residual_median":
        intercept = float(np.median(offsets_s - slope * times_s))
    else:
        raise ValueError(f"Unsupported intercept_mode: {intercept_mode!r}")
    return intercept, slope


def fit_theil_sen_drift(
    times_s: np.ndarray,
    offsets_s: np.ndarray,
    max_drift_ppm: float = 1000.0,
    min_slope_span_s: float = 60.0,
    mad_iterations: int = 2,
    mad_scale_k: float = 4.0,
    intercept_mode: str = "residual_median",
    bootstrap_draws: int = 200,
    seed: int = 2026,
) -> DriftFitResult:
    """Fit an affine clock-drift trajectory using iterative-MAD Theil-Sen regression.

    Parameters
    ----------
    times_s : np.ndarray
        Window center timestamps (seconds) on the reference clock.
    offsets_s : np.ndarray
        Estimated clock offsets ``t_ref - t_target`` (seconds) per window.
    max_drift_ppm : float, default=1000.0
        Physical oscillator drift magnitude cap in parts per million.
    min_slope_span_s : float, default=60.0
        Minimum baseline separation (seconds) between pairs used for slopes.
        Set to ``0.0`` for unconstrained all-pair slopes matching SciPy.
    mad_iterations : int, default=2
        Number of iterative Median Absolute Deviation outlier rejection passes.
    mad_scale_k : float, default=4.0
        Inlier rejection cutoff as a multiple of ``MAD + 1 ms``.
    intercept_mode : {"residual_median", "scipy_separate"}, default="residual_median"
        Intercept definition: ``median(y - b * x)`` or ``median(y) - b * median(x)``.
    bootstrap_draws : int, default=200
        Number of bootstrap resamples on final inliers for 95 percent CIs.
    seed : int, default=2026
        Random seed for deterministic bootstrap resampling.

    Returns
    -------
    DriftFitResult
        Estimated intercept, slope (ppm), MAD residual, and bootstrap CIs.
    """
    t_arr = np.asarray(times_s, dtype=np.float64).ravel()
    y_arr = np.asarray(offsets_s, dtype=np.float64).ravel()
    if t_arr.size != y_arr.size:
        raise ValueError("times_s and offsets_s must have identical lengths")
    if not (np.all(np.isfinite(t_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("times_s and offsets_s must be finite")

    n_input = int(t_arr.size)
    if n_input == 0:
        return DriftFitResult(
            intercept_s=0.0,
            slope=0.0,
            drift_ppm=0.0,
            ci_low_s=0.0,
            ci_high_s=0.0,
            slope_ci_low_ppm=0.0,
            slope_ci_high_ppm=0.0,
            n_input=0,
            n_inliers=0,
            mad_residual_s=0.0,
        )
    if n_input == 1:
        y0 = float(y_arr[0])
        return DriftFitResult(
            intercept_s=y0,
            slope=0.0,
            drift_ppm=0.0,
            ci_low_s=y0,
            ci_high_s=y0,
            slope_ci_low_ppm=0.0,
            slope_ci_high_ppm=0.0,
            n_input=1,
            n_inliers=1,
            mad_residual_s=0.0,
        )

    cur_t = t_arr.copy()
    cur_y = y_arr.copy()
    for _ in range(mad_iterations):
        a0, b0 = _pairwise_theil_sen_core(
            cur_t,
            cur_y,
            max_drift_ppm=max_drift_ppm,
            min_slope_span_s=min_slope_span_s,
            intercept_mode=intercept_mode,
        )
        res = np.abs(cur_y - (a0 + b0 * cur_t))
        mad = float(np.median(res)) + 1e-3
        inliers = res <= mad_scale_k * mad
        if int(np.sum(inliers)) >= 3:
            cur_t = cur_t[inliers]
            cur_y = cur_y[inliers]

    a_final, b_final = _pairwise_theil_sen_core(
        cur_t,
        cur_y,
        max_drift_ppm=max_drift_ppm,
        min_slope_span_s=min_slope_span_s,
        intercept_mode=intercept_mode,
    )
    final_res = np.abs(cur_y - (a_final + b_final * cur_t))
    mad_final = float(np.median(final_res))

    n_inl = int(cur_t.size)
    if bootstrap_draws > 0 and n_inl >= 4:
        rng = np.random.default_rng(seed)
        boot_a = np.empty(bootstrap_draws, dtype=np.float64)
        boot_b = np.empty(bootstrap_draws, dtype=np.float64)
        for draw in range(bootstrap_draws):
            idx = rng.integers(0, n_inl, size=n_inl)
            ba, bb = _pairwise_theil_sen_core(
                cur_t[idx],
                cur_y[idx],
                max_drift_ppm=max_drift_ppm,
                min_slope_span_s=min_slope_span_s,
                intercept_mode=intercept_mode,
            )
            boot_a[draw] = ba
            boot_b[draw] = bb
        ci_low_s, ci_high_s = (float(v) for v in np.percentile(boot_a, [2.5, 97.5]))
        s_lo, s_hi = (float(v * 1e6) for v in np.percentile(boot_b, [2.5, 97.5]))
    else:
        ci_low_s = ci_high_s = a_final
        s_lo = s_hi = b_final * 1e6

    return DriftFitResult(
        intercept_s=a_final,
        slope=b_final,
        drift_ppm=b_final * 1e6,
        ci_low_s=ci_low_s,
        ci_high_s=ci_high_s,
        slope_ci_low_ppm=s_lo,
        slope_ci_high_ppm=s_hi,
        n_input=n_input,
        n_inliers=n_inl,
        mad_residual_s=mad_final,
    )


def score_subperiod_trajectory(
    intercept_s: float,
    slope: float,
    times_s: np.ndarray,
    offset_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    quality_weights: np.ndarray,
    kernel_width_s: float = 0.075,
) -> float:
    """Evaluate weighted truncated-triangle consensus score over multi-peak candidates.

    Parameters
    ----------
    intercept_s : float
        Candidate affine trajectory intercept (seconds).
    slope : float
        Candidate affine trajectory slope (dimensionless).
    times_s : np.ndarray
        Window center times of shape ``(W,)``.
    offset_matrix : np.ndarray
        Candidate offset matrix of shape ``(W, K)`` padded with ``np.nan``.
    weight_matrix : np.ndarray
        Relative peak height weights of shape ``(W, K)``.
    quality_weights : np.ndarray
        Per-window prominence weights ``q_h^1.5 + 0.05`` of shape ``(W,)``.
    kernel_width_s : float, default=0.075
        Half-width (seconds) of the sub-period truncated linear kernel.

    Returns
    -------
    float
        Scalar consensus objective summed across quality-gated windows.
    """
    pred = (intercept_s + slope * times_s)[:, None]
    dist = np.abs(offset_matrix - pred)
    with np.errstate(invalid="ignore"):
        kernel = np.where(
            np.isnan(offset_matrix),
            0.0,
            np.maximum(0.0, 1.0 - dist / kernel_width_s) * weight_matrix,
        )
    return float(np.sum(quality_weights * np.max(kernel, axis=1)))


def rescue_subperiod_peaks(
    raw_windows: list[WindowEstimate],
    config: SyncConfig | None = None,
) -> tuple[DriftFitResult, tuple[WindowEstimate, ...], int, bool, float]:
    """Perform cross-half trajectory consensus and secondary-peak cycle-slip rescue.

    Parameters
    ----------
    raw_windows : list[WindowEstimate]
        Sliding-window top-K candidate lists from ``dual_scale_xcorr``.
    config : SyncConfig or None, default=None
        Synchronization parameters (uses ``SyncConfig()`` defaults if None).

    Returns
    -------
    tuple
        ``(drift_fit, updated_windows, n_rescued, used_global_consensus, top1_inlier_ratio_200ms)``.
    """
    cfg = config or SyncConfig()
    if len(raw_windows) < 3:
        times = np.array([w.center_time_s for w in raw_windows], dtype=np.float64)
        offs = np.array([w.top1_offset() for w in raw_windows], dtype=np.float64)
        fit = fit_theil_sen_drift(
            times,
            offs,
            max_drift_ppm=cfg.max_drift_ppm,
            min_slope_span_s=0.0,
            mad_iterations=0,
            bootstrap_draws=0,
        )
        return fit, tuple(raw_windows), 0, False, 1.0

    qualities = np.array([w.quality for w in raw_windows], dtype=np.float64)
    q_cutoff = float(np.quantile(qualities, cfg.quality_quantile))
    keep_indices = [idx for idx, w in enumerate(raw_windows) if w.quality >= q_cutoff]
    if len(keep_indices) < 3:
        keep_indices = list(range(len(raw_windows)))
    keep_set = set(keep_indices)
    kept = [raw_windows[i] for i in keep_indices]

    t_arr = np.array([w.center_time_s for w in kept], dtype=np.float64)
    top1_arr = np.array([w.top1_offset() for w in kept], dtype=np.float64)
    ts_fit = fit_theil_sen_drift(
        t_arr,
        top1_arr,
        max_drift_ppm=cfg.max_drift_ppm,
        min_slope_span_s=cfg.min_slope_span_s,
        mad_iterations=cfg.mad_iterations,
        mad_scale_k=cfg.mad_scale_k,
        bootstrap_draws=0,
    )
    a0, b0 = ts_fit.intercept_s, ts_fit.slope

    q_weights = np.array([w.quality ** 1.5 + 0.05 for w in kept], dtype=np.float64)
    max_k = max(len(w.candidates) for w in kept)
    off_mat = np.full((len(kept), max_k), np.nan, dtype=np.float64)
    wt_mat = np.zeros((len(kept), max_k), dtype=np.float64)
    for row_idx, win in enumerate(kept):
        h0 = max(win.candidates[0].correlation, 1e-6)
        for col_idx, cand in enumerate(win.candidates):
            off_mat[row_idx, col_idx] = cand.offset_s
            wt_mat[row_idx, col_idx] = (
                1.0 if col_idx == 0 else 0.65 * ((cand.correlation / h0) ** 2)
            )

    sc_ts = score_subperiod_trajectory(a0, b0, t_arr, off_mat, wt_mat, q_weights)
    res_top1 = np.abs(top1_arr - (a0 + b0 * t_arr))
    inlier_010 = float(np.mean(res_top1 <= 0.10))
    inlier_020 = float(np.mean(res_top1 <= 0.20))

    used_global = False
    if inlier_020 < cfg.min_inlier_ratio_for_global:
        mid_t = 0.5 * (float(t_arr.min()) + float(t_arr.max()))
        left_anc = sorted(
            [w for w in kept if w.center_time_s <= mid_t],
            key=lambda w: w.quality,
            reverse=True,
        )[:10]
        right_anc = sorted(
            [w for w in kept if w.center_time_s > mid_t],
            key=lambda w: w.quality,
            reverse=True,
        )[:10]
        min_dt = max(120.0, (float(t_arr.max()) - float(t_arr.min())) * 0.25)
        slope_bound = cfg.max_drift_ppm * 1e-6
        best_sc = sc_ts * 1.10
        for wl in left_anc:
            for wr in right_anc:
                dt = wr.center_time_s - wl.center_time_s
                if dt < min_dt:
                    continue
                for cl in wl.candidates[:3]:
                    for cr in wr.candidates[:3]:
                        b_cand = (cr.offset_s - cl.offset_s) / dt
                        if abs(b_cand) <= slope_bound:
                            a_cand = cl.offset_s - b_cand * wl.center_time_s
                            sc = score_subperiod_trajectory(
                                a_cand, b_cand, t_arr, off_mat, wt_mat, q_weights
                            )
                            if sc > best_sc:
                                best_sc = sc
                                a0, b0 = a_cand, b_cand
                                used_global = True

    if inlier_010 >= cfg.fast_path_inlier_ratio and not used_global:
        final_fit = fit_theil_sen_drift(
            t_arr,
            top1_arr,
            max_drift_ppm=cfg.max_drift_ppm,
            min_slope_span_s=cfg.min_slope_span_s,
            mad_iterations=cfg.mad_iterations,
            mad_scale_k=cfg.mad_scale_k,
            bootstrap_draws=cfg.bootstrap_draws,
            seed=cfg.random_seed,
        )
        updated = tuple(
            WindowEstimate(
                center_time_s=w.center_time_s,
                quality=w.quality,
                candidates=w.candidates,
                selected_offset_s=w.top1_offset(),
                selected_rank=1,
                was_rescued=False,
                retained=(idx in keep_set),
            )
            for idx, w in enumerate(raw_windows)
        )
        return final_fit, updated, 0, False, inlier_020

    selected_t: list[float] = []
    selected_y: list[float] = []
    rescued_cnt = 0
    updated_windows: list[WindowEstimate] = []

    for idx, win in enumerate(raw_windows):
        if idx not in keep_set:
            updated_windows.append(
                WindowEstimate(
                    center_time_s=win.center_time_s,
                    quality=win.quality,
                    candidates=win.candidates,
                    selected_offset_s=win.top1_offset(),
                    selected_rank=1,
                    was_rescued=False,
                    retained=False,
                )
            )
            continue

        pred = a0 + b0 * win.center_time_s
        c0 = win.candidates[0]
        if abs(c0.offset_s - pred) <= cfg.slip_threshold_s:
            chosen = c0
            rescued = False
            use_in_fit = True
        else:
            best_cand = None
            best_dist = cfg.subperiod_tol_s
            for cand in win.candidates[1:]:
                if (
                    cand.correlation >= cfg.secondary_height_ratio * c0.correlation
                    and abs(cand.offset_s - pred) <= best_dist
                ):
                    best_dist = abs(cand.offset_s - pred)
                    best_cand = cand
            if best_cand is not None:
                chosen = best_cand
                rescued = True
                rescued_cnt += 1
                use_in_fit = True
            else:
                chosen = c0
                rescued = False
                use_in_fit = not used_global

        if use_in_fit:
            selected_t.append(win.center_time_s)
            selected_y.append(chosen.offset_s)

        updated_windows.append(
            WindowEstimate(
                center_time_s=win.center_time_s,
                quality=win.quality,
                candidates=win.candidates,
                selected_offset_s=chosen.offset_s,
                selected_rank=chosen.rank,
                was_rescued=rescued,
                retained=use_in_fit,
            )
        )

    if len(selected_t) < 4:
        final_fit = fit_theil_sen_drift(
            t_arr,
            top1_arr,
            max_drift_ppm=cfg.max_drift_ppm,
            min_slope_span_s=cfg.min_slope_span_s,
            mad_iterations=cfg.mad_iterations,
            mad_scale_k=cfg.mad_scale_k,
            bootstrap_draws=cfg.bootstrap_draws,
            seed=cfg.random_seed,
        )
    else:
        final_fit = fit_theil_sen_drift(
            np.asarray(selected_t, dtype=np.float64),
            np.asarray(selected_y, dtype=np.float64),
            max_drift_ppm=cfg.max_drift_ppm,
            min_slope_span_s=cfg.min_slope_span_s,
            mad_iterations=cfg.mad_iterations,
            mad_scale_k=cfg.mad_scale_k,
            bootstrap_draws=cfg.bootstrap_draws,
            seed=cfg.random_seed,
        )
    return final_fit, tuple(updated_windows), rescued_cnt, used_global, inlier_020


def synchronize_clocks(
    ecg: np.ndarray | None = None,
    ppg: np.ndarray | None = None,
    fs_ecg_hz: float = 700.0,
    fs_ppg_hz: float = 64.0,
    events: CardiacEvents | None = None,
    config: SyncConfig | None = None,
    subject_id: str = "recording",
) -> SyncResult:
    """Run end-to-end dual-scale cardiac clock synchronization.

    Accepts either raw waveform arrays ``(ecg, ppg)`` or pre-extracted
    ``CardiacEvents``.

    Parameters
    ----------
    ecg : np.ndarray or None, default=None
        Reference chest ECG waveform (required if ``events`` is None).
    ppg : np.ndarray or None, default=None
        Target peripheral PPG/BVP waveform (required if ``events`` is None).
    fs_ecg_hz : float, default=700.0
        Sampling frequency of ``ecg`` in Hz.
    fs_ppg_hz : float, default=64.0
        Sampling frequency of ``ppg`` in Hz.
    events : CardiacEvents or None, default=None
        Pre-extracted cardiac event container; skips Stage 1 detection when provided.
    config : SyncConfig or None, default=None
        Synchronization parameters (uses ``SyncConfig()`` defaults if None).
    subject_id : str, default="recording"
        De-identified session label attached when detecting events from waveforms.

    Returns
    -------
    SyncResult
        Complete coarse lock, affine drift fit, and per-window rescue diagnostics.
    """
    cfg = config or SyncConfig()
    if events is None:
        if ecg is None or ppg is None:
            raise ValueError("Provide either (ecg, ppg) waveforms or a CardiacEvents object")
        events_obj = detect_cardiac_events(
            ecg,
            ppg,
            fs_ecg_hz=fs_ecg_hz,
            fs_ppg_hz=fs_ppg_hz,
            subject_id=subject_id,
        )
    else:
        events_obj = events

    coarse_offset_s, coarse_lag_s, coarse_prom, raw_windows = dual_scale_xcorr(events_obj, cfg)
    if not raw_windows:
        fallback_fit = DriftFitResult(
            intercept_s=coarse_offset_s,
            slope=0.0,
            drift_ppm=0.0,
            ci_low_s=coarse_offset_s,
            ci_high_s=coarse_offset_s,
            slope_ci_low_ppm=0.0,
            slope_ci_high_ppm=0.0,
            n_input=0,
            n_inliers=0,
            mad_residual_s=0.0,
            method="coarse_fallback",
        )
        return SyncResult(
            coarse_offset_s=coarse_offset_s,
            coarse_lag_s=coarse_lag_s,
            coarse_prominence=coarse_prom,
            drift_fit=fallback_fit,
            windows=(),
            n_windows_total=0,
            n_windows_retained=0,
            n_windows_rescued=0,
            used_global_consensus=False,
            top1_inlier_ratio_200ms=1.0,
            ptt_prior_s=cfg.ptt_prior_s,
            config=cfg,
        )

    drift_fit, updated_windows, n_rescued, used_global, inlier_020 = rescue_subperiod_peaks(
        raw_windows, cfg
    )
    retained_count = sum(1 for w in updated_windows if w.retained)
    return SyncResult(
        coarse_offset_s=coarse_offset_s,
        coarse_lag_s=coarse_lag_s,
        coarse_prominence=coarse_prom,
        drift_fit=drift_fit,
        windows=updated_windows,
        n_windows_total=len(updated_windows),
        n_windows_retained=retained_count,
        n_windows_rescued=n_rescued,
        used_global_consensus=used_global,
        top1_inlier_ratio_200ms=inlier_020,
        ptt_prior_s=cfg.ptt_prior_s,
        config=cfg,
    )
