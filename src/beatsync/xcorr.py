"""Stage 2a: Dual-scale FFT cross-correlation and multi-peak sub-sample extraction."""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate, correlation_lags, find_peaks

from .events import encode_gaussian_impulses
from .types import CandidatePeak, CardiacEvents, SyncConfig, WindowEstimate


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Standardize a 1D array to zero mean and unit standard deviation safely."""
    x = np.asarray(arr, dtype=np.float64).ravel()
    return (x - float(np.mean(x))) / (float(np.std(x)) + 1e-9)


def _overlap_counts(n_ref: int, n_target: int, lags_idx: np.ndarray) -> np.ndarray:
    """Compute sample overlap counts and floor-clipped denominators across lags."""
    overlap = np.minimum(n_ref, n_target - lags_idx) - np.maximum(0, -lags_idx)
    floor_overlap = max(1, int(0.35 * min(n_ref, n_target)))
    return np.maximum(overlap, floor_overlap).astype(np.float64)


def fft_xcorr(
    ref_series: np.ndarray,
    target_series: np.ndarray,
    min_lag_s: float,
    max_lag_s: float,
    grid_fs_hz: float = 64.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute overlap-normalized cross-correlation via power-of-two real FFT.

    Parameters
    ----------
    ref_series : np.ndarray
        Reference impulse or envelope train sampled at ``grid_fs_hz``.
    target_series : np.ndarray
        Target impulse or envelope train sampled at ``grid_fs_hz``.
    min_lag_s : float
        Minimum lag (seconds) to include in output.
    max_lag_s : float
        Maximum lag (seconds) to include in output.
    grid_fs_hz : float, default=64.0
        Uniform grid sampling frequency in Hz.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(lags_s, corr)`` arrays sorted in increasing lag order, where
        positive lag indicates ``target_series`` occurs later than
        ``ref_series`` (``t_target - t_ref``).
    """
    if grid_fs_hz <= 0.0:
        raise ValueError("grid_fs_hz must be positive")
    if min_lag_s >= max_lag_s:
        raise ValueError("min_lag_s must be strictly less than max_lag_s")
    ref_arr = np.asarray(ref_series, dtype=np.float64).ravel()
    tgt_arr = np.asarray(target_series, dtype=np.float64).ravel()
    n_ref, n_tgt = ref_arr.size, tgt_arr.size
    if n_ref < 2 or n_tgt < 2:
        raise ValueError("Input series must contain at least 2 samples")

    a_z = _zscore(ref_arr)
    b_z = _zscore(tgt_arr)
    nfft = 1 << (n_ref + n_tgt - 1).bit_length()
    raw = np.fft.irfft(
        np.fft.rfft(b_z, nfft) * np.conj(np.fft.rfft(a_z, nfft)),
        n=nfft,
    )
    corr_full = np.concatenate([raw[-(n_ref - 1):], raw[:n_tgt]])
    lags_idx = np.arange(-(n_ref - 1), n_tgt, dtype=int)
    corr_norm = corr_full / _overlap_counts(n_ref, n_tgt, lags_idx)
    lags_s = lags_idx.astype(np.float64) / float(grid_fs_hz)
    mask = (lags_s >= min_lag_s) & (lags_s <= max_lag_s)
    if not np.any(mask):
        raise ValueError("Requested lag interval yields zero overlapping samples")
    return lags_s[mask], corr_norm[mask]


def direct_xcorr_reference(
    ref_series: np.ndarray,
    target_series: np.ndarray,
    min_lag_s: float,
    max_lag_s: float,
    grid_fs_hz: float = 64.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact time-domain cross-correlation using ``scipy.signal.correlate``.

    Provided primarily as an independent O(N^2) numerical reference against
    which ``fft_xcorr`` is verified to machine precision.
    """
    ref_arr = np.asarray(ref_series, dtype=np.float64).ravel()
    tgt_arr = np.asarray(target_series, dtype=np.float64).ravel()
    n_ref, n_tgt = ref_arr.size, tgt_arr.size
    a_z = _zscore(ref_arr)
    b_z = _zscore(tgt_arr)
    corr_full = correlate(b_z, a_z, mode="full", method="direct")
    lags_idx = correlation_lags(n_tgt, n_ref, mode="full")
    corr_norm = corr_full / _overlap_counts(n_ref, n_tgt, lags_idx)
    lags_s = lags_idx.astype(np.float64) / float(grid_fs_hz)
    mask = (lags_s >= min_lag_s) & (lags_s <= max_lag_s)
    if not np.any(mask):
        raise ValueError("Requested lag interval yields zero overlapping samples")
    return lags_s[mask], corr_norm[mask]


def compute_prominence_quality(corr: np.ndarray, exclusion_bins: int = 16) -> float:
    """Compute cross-correlation peak prominence z-score ``q_h``.

    Measures how strongly the primary correlation peak stands out above the
    strongest secondary peak outside an exclusion collar of ``+/- exclusion_bins``.
    """
    c = np.asarray(corr, dtype=np.float64).ravel()
    if c.size <= 2 * exclusion_bins + 2:
        return 0.0
    best_idx = int(np.argmax(c))
    masked = c.copy()
    lo = max(0, best_idx - exclusion_bins)
    hi = min(c.size, best_idx + exclusion_bins + 1)
    masked[lo:hi] = -np.inf
    second_max = float(np.max(masked))
    if not np.isfinite(second_max):
        return 0.0
    score = (float(c[best_idx]) - second_max) / (float(np.std(c)) + 1e-9)
    return float(max(score, 0.0))


def extract_topk_peaks(
    lags_s: np.ndarray,
    corr: np.ndarray,
    grid_fs_hz: float = 64.0,
    ptt_prior_s: float = 0.250,
    top_k: int = 5,
    min_peak_sep_s: float = 0.32,
) -> tuple[CandidatePeak, ...]:
    """Extract top-K local maxima with parabolic sub-sample lag refinement.

    Parameters
    ----------
    lags_s : np.ndarray
        Monotonic lag grid (seconds).
    corr : np.ndarray
        Normalized cross-correlation curve matching ``lags_s``.
    grid_fs_hz : float, default=64.0
        Grid frequency in Hz used for fractional bin conversion.
    ptt_prior_s : float, default=0.250
        Physiological pulse transit time prior (seconds) subtracted from raw
        cardiac lag prior to sign inversion into ``t_ref - t_target``.
    top_k : int, default=5
        Maximum number of candidate peaks to return.
    min_peak_sep_s : float, default=0.32
        Minimum spacing (seconds) enforced between distinct local peaks.

    Returns
    -------
    tuple[CandidatePeak, ...]
        Candidate offsets ranked in descending order of correlation height.
    """
    lags = np.asarray(lags_s, dtype=np.float64).ravel()
    c = np.asarray(corr, dtype=np.float64).ravel()
    if lags.size != c.size or c.size == 0:
        raise ValueError("lags_s and corr must be non-empty arrays of equal length")

    min_dist = max(1, int(min_peak_sep_s * grid_fs_hz))
    pks, _ = find_peaks(c, distance=min_dist)
    if pks.size == 0:
        idx = int(np.argmax(c))
        return (
            CandidatePeak(
                offset_s=float(-(lags[idx] - ptt_prior_s)),
                correlation=float(c[idx]),
                rank=1,
            ),
        )

    order = pks[np.argsort(c[pks])[::-1][:top_k]]
    candidates: list[CandidatePeak] = []
    for rank, idx in enumerate(order, start=1):
        lag_val = float(lags[idx])
        if 0 < idx < c.size - 1:
            y0, y1, y2 = float(c[idx - 1]), float(c[idx]), float(c[idx + 1])
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta_bin = 0.5 * (y0 - y2) / denom
                if abs(delta_bin) <= 1.0:
                    lag_val += delta_bin / float(grid_fs_hz)
        candidates.append(
            CandidatePeak(
                offset_s=float(-(lag_val - ptt_prior_s)),
                correlation=float(c[idx]),
                rank=rank,
            )
        )
    return tuple(candidates)


def dual_scale_xcorr(
    events: CardiacEvents,
    config: SyncConfig | None = None,
) -> tuple[float, float, float, list[WindowEstimate]]:
    """Run wide-kernel global lock followed by coarse-shifted narrow-kernel windows.

    Parameters
    ----------
    events : CardiacEvents
        Reference ECG R-peak and target PPG pulse event trains.
    config : SyncConfig or None, default=None
        Synchronization parameters (uses ``SyncConfig()`` defaults when omitted).

    Returns
    -------
    tuple[float, float, float, list[WindowEstimate]]
        ``(coarse_offset_s, coarse_lag_s, coarse_prominence, raw_windows)``.
    """
    cfg = config or SyncConfig()
    if events.n_ecg < 5 or events.n_ppg < 5:
        raise ValueError("At least 5 ECG and 5 PPG cardiac events are required")

    fs = cfg.grid_fs_hz
    hb_ref_coarse = encode_gaussian_impulses(
        events.ecg_rpeaks_s,
        events.ref_duration_s,
        grid_fs_hz=fs,
        sigma_s=cfg.coarse_sigma_s,
    )
    hb_tgt_coarse = encode_gaussian_impulses(
        events.ppg_peaks_s,
        events.target_duration_s,
        grid_fs_hz=fs,
        sigma_s=cfg.coarse_sigma_s,
    )
    max_feasible = min(events.ref_duration_s, events.target_duration_s) * 0.65
    search_bound = min(cfg.max_coarse_lag_s, max_feasible)
    if search_bound <= 1.0:
        search_bound = min(cfg.max_coarse_lag_s, max(events.ref_duration_s, events.target_duration_s) * 0.90)

    lags_c, corr_c = fft_xcorr(
        hb_ref_coarse,
        hb_tgt_coarse,
        min_lag_s=-search_bound,
        max_lag_s=search_bound,
        grid_fs_hz=fs,
    )
    coarse_idx = int(np.argmax(corr_c))
    coarse_lag_s = float(lags_c[coarse_idx])
    coarse_offset_s = float(-(coarse_lag_s - cfg.ptt_prior_s))
    coarse_prom = compute_prominence_quality(
        corr_c, exclusion_bins=max(16, int(3.5 * cfg.coarse_sigma_s * fs))
    )

    hb_ref_fine = encode_gaussian_impulses(
        events.ecg_rpeaks_s,
        events.ref_duration_s,
        grid_fs_hz=fs,
        sigma_s=cfg.fine_sigma_s,
    )
    hb_tgt_fine = encode_gaussian_impulses(
        events.ppg_peaks_s,
        events.target_duration_s,
        grid_fs_hz=fs,
        sigma_s=cfg.fine_sigma_s,
    )

    win_samples = int(round(cfg.window_s * fs))
    step_samples = max(1, int(round(cfg.step_s * fs)))
    shift_samples = int(round(coarse_lag_s * fs))
    n_ref = hb_ref_fine.size
    n_tgt = hb_tgt_fine.size

    raw_windows: list[WindowEstimate] = []
    for st_ref in range(0, max(1, n_ref - win_samples + 1), step_samples):
        en_ref = st_ref + win_samples
        if en_ref > n_ref:
            break
        st_tgt = st_ref + shift_samples
        en_tgt = st_tgt + win_samples
        if st_tgt < 0 or en_tgt > n_tgt:
            continue
        ref_win = hb_ref_fine[st_ref:en_ref]
        tgt_win = hb_tgt_fine[st_tgt:en_tgt]
        if float(np.std(ref_win)) < 1e-8 or float(np.std(tgt_win)) < 1e-8:
            continue

        lags_w, corr_w = fft_xcorr(
            ref_win,
            tgt_win,
            min_lag_s=-cfg.fine_search_lag_s,
            max_lag_s=cfg.fine_search_lag_s,
            grid_fs_hz=fs,
        )
        quality = compute_prominence_quality(corr_w, exclusion_bins=max(6, int(0.25 * fs)))
        cands = extract_topk_peaks(
            lags_w + coarse_lag_s,
            corr_w,
            grid_fs_hz=fs,
            ptt_prior_s=cfg.ptt_prior_s,
            top_k=cfg.top_k_peaks,
            min_peak_sep_s=cfg.min_peak_sep_s,
        )
        center_t = (st_ref + 0.5 * win_samples) / float(fs)
        raw_windows.append(
            WindowEstimate(
                center_time_s=float(center_t),
                quality=quality,
                candidates=cands,
                selected_offset_s=cands[0].offset_s,
                selected_rank=1,
                was_rescued=False,
                retained=True,
            )
        )
    return coarse_offset_s, coarse_lag_s, coarse_prom, raw_windows
