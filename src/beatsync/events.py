"""Stage 1: Dual-channel cardiac event extraction and Gaussian impulse encoding."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from .types import CardiacEvents


def _validate_1d_signal(signal: np.ndarray, fs_hz: float, name: str) -> np.ndarray:
    """Validate that a 1D biosignal array is finite, non-empty, and sampled > 0 Hz."""
    arr = np.asarray(signal, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if fs_hz <= 0.0:
        raise ValueError(f"Sampling rate for {name} must be positive")
    return arr


def _parabolic_refine(signal: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Refine integer peak indices using local three-point quadratic interpolation."""
    refined = indices.astype(np.float64)
    n = signal.size
    for pos, idx in enumerate(indices):
        if 0 < idx < n - 1:
            y0 = float(signal[idx - 1])
            y1 = float(signal[idx])
            y2 = float(signal[idx + 1])
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom
                if abs(delta) <= 1.0:
                    refined[pos] += delta
    return refined


def bandpass_zero_phase(
    signal: np.ndarray,
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 2,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to a 1D physiological signal.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D waveform array.
    fs_hz : float
        Sampling rate in Hz.
    low_hz : float
        Lower cutoff frequency in Hz.
    high_hz : float
        Upper cutoff frequency in Hz (must satisfy ``0 < low_hz < high_hz < fs_hz / 2``).
    order : int, default=2
        Butterworth filter order prior to forward-backward ``filtfilt`` application.

    Returns
    -------
    np.ndarray
        Zero-phase filtered signal of matching length.
    """
    arr = _validate_1d_signal(signal, fs_hz, "signal")
    nyquist = 0.5 * fs_hz
    if not (0.0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Bandpass edges ({low_hz}, {high_hz}) invalid for Nyquist={nyquist:.2f} Hz"
        )
    if order < 1:
        raise ValueError("Filter order must be >= 1")
    min_len = max(12, 3 * (2 * order + 1))
    if arr.size <= min_len:
        raise ValueError(f"Signal length ({arr.size}) too short for order-{order} filtfilt")
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return filtfilt(b, a, arr)


def detect_ecg_rpeaks(
    ecg: np.ndarray,
    fs_ecg_hz: float = 700.0,
    min_rr_s: float = 0.33,
    percentile_q: float = 97.0,
    height_scale: float = 0.40,
    refine_subsample: bool = False,
) -> np.ndarray:
    """Extract QRS R-peak timestamps (seconds) from chest ECG.

    Parameters
    ----------
    ecg : np.ndarray
        1D ECG waveform.
    fs_ecg_hz : float, default=700.0
        ECG sampling frequency in Hz.
    min_rr_s : float, default=0.33
        Refractory minimum inter-beat interval in seconds (corresponds to 181.8 bpm).
    percentile_q : float, default=97.0
        High amplitude percentile used to anchor adaptive threshold scaling.
    height_scale : float, default=0.40
        Fraction of ``percentile_q`` amplitude required to accept an R-peak.
    refine_subsample : bool, default=False
        Whether to apply three-point parabolic sub-sample peak timing refinement.

    Returns
    -------
    np.ndarray
        Strictly increasing R-peak timestamps in seconds.
    """
    arr = _validate_1d_signal(ecg, fs_ecg_hz, "ecg")
    high_hz = min(25.0, 0.45 * fs_ecg_hz)
    low_hz = min(5.0, 0.25 * high_hz)
    filtered = bandpass_zero_phase(arr, fs_ecg_hz, low_hz, high_hz, order=2)
    threshold = float(np.percentile(filtered, percentile_q)) * height_scale
    min_dist = max(1, int(min_rr_s * fs_ecg_hz))
    peaks, _ = find_peaks(filtered, distance=min_dist, height=threshold)
    if refine_subsample and peaks.size > 0:
        return _parabolic_refine(filtered, peaks) / float(fs_ecg_hz)
    return peaks.astype(np.float64) / float(fs_ecg_hz)


def detect_ppg_peaks(
    ppg: np.ndarray,
    fs_ppg_hz: float = 64.0,
    min_ibi_s: float = 0.33,
    ectopic_ratio: float = 0.50,
    filter_short_ibi: bool = True,
    refine_subsample: bool = False,
) -> np.ndarray:
    """Extract peripheral PPG systolic pulse timestamps (seconds) without interpolation.

    Parameters
    ----------
    ppg : np.ndarray
        1D photoplethysmography (BVP) waveform.
    fs_ppg_hz : float, default=64.0
        PPG sampling frequency in Hz.
    min_ibi_s : float, default=0.33
        Minimum refractory peak separation in seconds.
    ectopic_ratio : float, default=0.50
        Ratio of median IBI below which premature artifact peaks are pruned
        rather than interpolated.
    filter_short_ibi : bool, default=True
        Whether to prune short-IBI motion/dicrotic artifact double peaks while
        leaving genuine missed-beat dropouts unfilled.
    refine_subsample : bool, default=False
        Whether to apply parabolic sub-sample interpolation around peak maxima.

    Returns
    -------
    np.ndarray
        Sorted systolic pulse peak timestamps in seconds.
    """
    arr = _validate_1d_signal(ppg, fs_ppg_hz, "ppg")
    high_hz = min(3.5, 0.45 * fs_ppg_hz)
    low_hz = min(0.7, 0.25 * high_hz)
    filtered = bandpass_zero_phase(arr, fs_ppg_hz, low_hz, high_hz, order=2)
    min_dist = max(1, int(min_ibi_s * fs_ppg_hz))
    sig_std = float(np.std(filtered))
    peaks, _ = find_peaks(
        filtered,
        distance=min_dist,
        height=0.05 * sig_std,
        prominence=0.20 * sig_std,
    )
    if peaks.size == 0:
        return np.empty(0, dtype=np.float64)

    times = (
        _parabolic_refine(filtered, peaks) / float(fs_ppg_hz)
        if refine_subsample
        else peaks.astype(np.float64) / float(fs_ppg_hz)
    )
    if not filter_short_ibi or times.size < 4:
        return times

    med_ibi = float(np.median(np.diff(times)))
    min_valid_ibi = ectopic_ratio * med_ibi
    kept_idx = [0]
    for idx in range(1, times.size):
        if times[idx] - times[kept_idx[-1]] >= min_valid_ibi:
            kept_idx.append(idx)
        else:
            # Keep the taller filtered systolic peak within the collided double-peak pair
            if filtered[peaks[idx]] > filtered[peaks[kept_idx[-1]]]:
                kept_idx[-1] = idx
    return times[np.asarray(kept_idx, dtype=int)]


def detect_cardiac_events(
    ecg: np.ndarray,
    ppg: np.ndarray,
    fs_ecg_hz: float = 700.0,
    fs_ppg_hz: float = 64.0,
    subject_id: str = "recording",
    filter_short_ibi: bool = True,
    refine_subsample: bool = False,
) -> CardiacEvents:
    """Detect asynchronous reference ECG and target PPG cardiac event trains.

    Parameters
    ----------
    ecg : np.ndarray
        Reference chest ECG signal.
    ppg : np.ndarray
        Target peripheral PPG/BVP signal.
    fs_ecg_hz : float, default=700.0
        ECG sampling rate in Hz.
    fs_ppg_hz : float, default=64.0
        PPG sampling rate in Hz.
    subject_id : str, default="recording"
        De-identified subject or session label.
    filter_short_ibi : bool, default=True
        Prune spurious short-IBI PPG detections without imputing missing beats.
    refine_subsample : bool, default=False
        Apply sub-sample quadratic peak timing refinement.

    Returns
    -------
    CardiacEvents
        Detected R-peak and pulse-peak event container with exact durations.
    """
    ecg_arr = _validate_1d_signal(ecg, fs_ecg_hz, "ecg")
    ppg_arr = _validate_1d_signal(ppg, fs_ppg_hz, "ppg")
    r_peaks = detect_ecg_rpeaks(
        ecg_arr,
        fs_ecg_hz=fs_ecg_hz,
        refine_subsample=refine_subsample,
    )
    p_peaks = detect_ppg_peaks(
        ppg_arr,
        fs_ppg_hz=fs_ppg_hz,
        filter_short_ibi=filter_short_ibi,
        refine_subsample=refine_subsample,
    )
    return CardiacEvents(
        ecg_rpeaks_s=r_peaks,
        ppg_peaks_s=p_peaks,
        ref_duration_s=float(ecg_arr.size) / float(fs_ecg_hz),
        target_duration_s=float(ppg_arr.size) / float(fs_ppg_hz),
        subject_id=subject_id,
    )


def encode_gaussian_impulses(
    event_times_s: np.ndarray,
    duration_s: float,
    grid_fs_hz: float = 64.0,
    sigma_s: float = 0.035,
) -> np.ndarray:
    """Project discrete cardiac beat timestamps onto a Gaussian-smoothed impulse train.

    Parameters
    ----------
    event_times_s : np.ndarray
        Cardiac beat timestamps (seconds) relative to segment start.
    duration_s : float
        Total timeline duration in seconds.
    grid_fs_hz : float, default=64.0
        Uniform grid sampling frequency in Hz.
    sigma_s : float, default=0.035
        Gaussian kernel standard deviation in seconds (e.g., 0.120 s for
        coarse drift-tolerant lock, 0.035 s for fine sub-sample alignment).

    Returns
    -------
    np.ndarray
        Unit-impulse train convolved with the truncated Gaussian kernel.
    """
    if duration_s <= 0.0 or grid_fs_hz <= 0.0 or sigma_s <= 0.0:
        raise ValueError("duration_s, grid_fs_hz, and sigma_s must be positive")
    n_samples = int(duration_s * grid_fs_hz) + 1
    grid = np.zeros(n_samples, dtype=np.float64)
    times = np.asarray(event_times_s, dtype=np.float64).ravel()
    if times.size > 0:
        idx = np.round(times * grid_fs_hz).astype(int)
        valid = idx[(idx >= 0) & (idx < n_samples)]
        if valid.size > 0:
            np.add.at(grid, valid, 1.0)
    half_width = max(1, int(4.0 * sigma_s * grid_fs_hz))
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / (sigma_s * grid_fs_hz)) ** 2)
    return np.convolve(grid, kernel, mode="same")


def extract_motion_envelope(
    accel: np.ndarray,
    fs_in_hz: float,
    grid_fs_hz: float = 64.0,
) -> np.ndarray:
    """Compute a resampled motion-magnitude jerk envelope from 1D or 3D acceleration.

    Parameters
    ----------
    accel : np.ndarray
        1D motion magnitude or 2D ``(N, C)`` multi-axis accelerometer array.
    fs_in_hz : float
        Input accelerometer sampling frequency in Hz.
    grid_fs_hz : float, default=64.0
        Target output grid frequency in Hz.

    Returns
    -------
    np.ndarray
        First-difference acceleration norm envelope interpolated onto ``grid_fs_hz``.
    """
    if fs_in_hz <= 0.0 or grid_fs_hz <= 0.0:
        raise ValueError("Sampling frequencies must be positive")
    arr = np.asarray(accel, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("accel must be non-empty and finite")
    if arr.ndim == 1:
        mag = np.abs(arr)
    elif arr.ndim == 2:
        mag = np.linalg.norm(arr, axis=1)
    else:
        raise ValueError("accel must be 1D or 2D")
    jerk = np.abs(np.diff(mag, prepend=mag[0]))
    n_out = max(1, int(len(jerk) / float(fs_in_hz) * float(grid_fs_hz)))
    t_out = np.arange(n_out, dtype=np.float64) / float(grid_fs_hz)
    t_in = np.arange(len(jerk), dtype=np.float64) / float(fs_in_hz)
    return np.interp(t_out, t_in, jerk)
