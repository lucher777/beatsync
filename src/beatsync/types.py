"""Typed configuration and structured output containers for beatsync."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyncConfig:
    """Configuration parameters for dual-scale cardiac clock synchronization.

    Attributes
    ----------
    grid_fs_hz : float
        Uniform impulse-train resampling frequency in Hz.
    coarse_sigma_s : float
        Standard deviation (seconds) of the wide Gaussian kernel used for
        global coarse cross-correlation under unknown clock drift.
    fine_sigma_s : float
        Standard deviation (seconds) of the narrow Gaussian kernel used for
        sliding-window sub-sample alignment.
    ptt_prior_s : float
        Physiological pulse transit time offset prior (seconds) from ECG R-peak
        to peripheral PPG systolic peak.
    max_coarse_lag_s : float
        Symmetric maximum lag search bound (seconds) for global coarse lock.
    window_s : float
        Sliding-window duration (seconds) for local fine cross-correlation.
    step_s : float
        Sliding-window stride (seconds).
    fine_search_lag_s : float
        Residual symmetric lag search bound (seconds) around coarse alignment.
    top_k_peaks : int
        Number of top local cross-correlation peaks retained per window.
    min_peak_sep_s : float
        Minimum separation (seconds) between candidate peaks to isolate adjacent
        cardiac cycles.
    quality_quantile : float
        Lower quantile of window prominence quality filtered prior to trajectory
        estimation.
    slip_threshold_s : float
        Deviation threshold (seconds) triggering secondary-peak cycle-slip rescue.
    subperiod_tol_s : float
        Tolerance radius (seconds) around the consensus trajectory required to
        accept a secondary candidate peak.
    secondary_height_ratio : float
        Minimum peak correlation relative to the top-1 peak required to accept
        a secondary candidate peak during cycle-slip rescue.
    min_inlier_ratio_for_global : float
        Minimum top-1 inlier fraction within 0.20 s before activating cross-half
        anchor candidate-line consensus search.
    fast_path_inlier_ratio : float
        Minimum top-1 inlier fraction within 0.10 s to skip secondary rescue when
        no global consensus override occurred.
    max_drift_ppm : float
        Physical upper bound on oscillator clock drift magnitude in parts per
        million (ppm).
    mad_iterations : int
        Number of iterative Median Absolute Deviation outlier pruning passes.
    mad_scale_k : float
        MAD multiplier threshold for outlier rejection.
    min_slope_span_s : float
        Minimum temporal separation (seconds) between window pairs included in
        Theil-Sen pairwise slope computation.
    bootstrap_draws : int
        Number of window-level bootstrap replicates used for 95 percent
        confidence intervals (0 disables bootstrap CIs).
    random_seed : int
        Random seed controlling bootstrap resampling reproducibility.
    """

    grid_fs_hz: float = 64.0
    coarse_sigma_s: float = 0.120
    fine_sigma_s: float = 0.035
    ptt_prior_s: float = 0.250
    max_coarse_lag_s: float = 1800.0
    window_s: float = 45.0
    step_s: float = 15.0
    fine_search_lag_s: float = 5.0
    top_k_peaks: int = 5
    min_peak_sep_s: float = 0.32
    quality_quantile: float = 0.25
    slip_threshold_s: float = 0.220
    subperiod_tol_s: float = 0.060
    secondary_height_ratio: float = 0.60
    min_inlier_ratio_for_global: float = 0.35
    fast_path_inlier_ratio: float = 0.80
    max_drift_ppm: float = 1000.0
    mad_iterations: int = 2
    mad_scale_k: float = 4.0
    min_slope_span_s: float = 60.0
    bootstrap_draws: int = 200
    random_seed: int = 2026

    def __post_init__(self) -> None:
        if self.grid_fs_hz <= 0.0:
            raise ValueError("grid_fs_hz must be positive")
        if self.coarse_sigma_s <= 0.0 or self.fine_sigma_s <= 0.0:
            raise ValueError("Gaussian kernel widths must be positive")
        if self.coarse_sigma_s < self.fine_sigma_s:
            raise ValueError("coarse_sigma_s should be >= fine_sigma_s")
        if self.max_coarse_lag_s <= 0.0 or self.fine_search_lag_s <= 0.0:
            raise ValueError("Search lag bounds must be positive")
        if self.window_s <= 0.0 or self.step_s <= 0.0:
            raise ValueError("window_s and step_s must be positive")
        if self.top_k_peaks < 1:
            raise ValueError("top_k_peaks must be >= 1")
        if not (0.0 <= self.quality_quantile < 0.9):
            raise ValueError("quality_quantile must lie in [0.0, 0.9)")
        if not (0.0 <= self.secondary_height_ratio <= 1.0):
            raise ValueError("secondary_height_ratio must lie in [0.0, 1.0]")
        if self.max_drift_ppm <= 0.0:
            raise ValueError("max_drift_ppm must be positive")
        if self.mad_iterations < 0 or self.bootstrap_draws < 0:
            raise ValueError("mad_iterations and bootstrap_draws must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration fields into a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class CardiacEvents:
    """Detected cardiac event timestamps for reference ECG and target PPG devices.

    Attributes
    ----------
    ecg_rpeaks_s : np.ndarray
        Strictly increasing R-peak timestamps (seconds) on the reference clock.
    ppg_peaks_s : np.ndarray
        Strictly increasing systolic pulse timestamps (seconds) on the target
        clock.
    ref_duration_s : float
        Total duration (seconds) of the reference recording.
    target_duration_s : float
        Total duration (seconds) of the target recording.
    subject_id : str
        Optional de-identified recording identifier.
    metadata : dict[str, Any]
        Optional metadata dictionary attached to the recording.
    """

    ecg_rpeaks_s: np.ndarray
    ppg_peaks_s: np.ndarray
    ref_duration_s: float
    target_duration_s: float
    subject_id: str = "recording"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        r_arr = np.asarray(self.ecg_rpeaks_s, dtype=np.float64).ravel()
        p_arr = np.asarray(self.ppg_peaks_s, dtype=np.float64).ravel()
        if not np.all(np.isfinite(r_arr)) or not np.all(np.isfinite(p_arr)):
            raise ValueError("Event timestamps must be finite")
        if self.ref_duration_s <= 0.0 or self.target_duration_s <= 0.0:
            raise ValueError("Recording durations must be positive")
        object.__setattr__(self, "ecg_rpeaks_s", np.sort(r_arr))
        object.__setattr__(self, "ppg_peaks_s", np.sort(p_arr))

    @property
    def n_ecg(self) -> int:
        """Return the number of detected reference ECG R-peaks."""
        return int(self.ecg_rpeaks_s.size)

    @property
    def n_ppg(self) -> int:
        """Return the number of detected target PPG systolic peaks."""
        return int(self.ppg_peaks_s.size)

    def mean_hr_bpm(self) -> tuple[float, float]:
        """Compute median-IBI-derived heart rate (bpm) for ECG and PPG streams."""
        ecg_hr = (
            60.0 / float(np.median(np.diff(self.ecg_rpeaks_s)))
            if self.ecg_rpeaks_s.size >= 2
            else float("nan")
        )
        ppg_hr = (
            60.0 / float(np.median(np.diff(self.ppg_peaks_s)))
            if self.ppg_peaks_s.size >= 2
            else float("nan")
        )
        return ecg_hr, ppg_hr

    def summary(self) -> str:
        """Return a concise single-line summary of event counts and heart rates."""
        ecg_hr, ppg_hr = self.mean_hr_bpm()
        return (
            f"CardiacEvents(subject={self.subject_id}, n_ecg={self.n_ecg} "
            f"[{ecg_hr:.1f} bpm], n_ppg={self.n_ppg} [{ppg_hr:.1f} bpm], "
            f"durations=({self.ref_duration_s:.1f}s, {self.target_duration_s:.1f}s))"
        )


@dataclass(frozen=True)
class CandidatePeak:
    """Single cross-correlation candidate peak extracted from a sliding window.

    Attributes
    ----------
    offset_s : float
        Candidate clock offset estimate ``t_ref - t_target`` (seconds) after
        compensating for the physiological PTT prior.
    correlation : float
        Normalized cross-correlation peak height.
    rank : int
        One-based height rank within the parent window (1 = dominant peak).
    """

    offset_s: float
    correlation: float
    rank: int


@dataclass(frozen=True)
class WindowEstimate:
    """Sliding-window alignment candidates and final consensus selection.

    Attributes
    ----------
    center_time_s : float
        Window midpoint timestamp (seconds) on the reference clock.
    quality : float
        Cross-correlation prominence quality score ``q_h``.
    candidates : tuple[CandidatePeak, ...]
        Ranked top-K candidate offset peaks in the window.
    selected_offset_s : float
        Offset selected after trajectory consensus and cycle-slip rescue.
    selected_rank : int
        Rank of the chosen candidate peak (1 for primary, >=2 for rescued).
    was_rescued : bool
        Whether a secondary candidate replaced a cycle-slipped primary peak.
    retained : bool
        Whether the window survived quality gating and MAD outlier pruning.
    """

    center_time_s: float
    quality: float
    candidates: tuple[CandidatePeak, ...]
    selected_offset_s: float
    selected_rank: int
    was_rescued: bool = False
    retained: bool = True

    def top1_offset(self) -> float:
        """Return the highest-correlation primary candidate offset."""
        if not self.candidates:
            return float("nan")
        return float(self.candidates[0].offset_s)


@dataclass(frozen=True)
class DriftFitResult:
    """Affine clock-drift trajectory fit ``delta(t) = intercept_s + slope * t``.

    Attributes
    ----------
    intercept_s : float
        Estimated clock offset (seconds) at ``t_ref = 0``.
    slope : float
        Dimensionless relative clock drift ``b`` (seconds per second).
    drift_ppm : float
        Estimated oscillator clock drift in parts per million (``slope * 1e6``).
    ci_low_s : float
        Lower 2.5th-percentile bootstrap confidence bound on intercept (seconds).
    ci_high_s : float
        Upper 97.5th-percentile bootstrap confidence bound on intercept (seconds).
    slope_ci_low_ppm : float
        Lower 2.5th-percentile bootstrap confidence bound on drift (ppm).
    slope_ci_high_ppm : float
        Upper 97.5th-percentile bootstrap confidence bound on drift (ppm).
    n_input : int
        Number of window estimates supplied to the fitter.
    n_inliers : int
        Number of inlier windows retained after MAD outlier rejection.
    mad_residual_s : float
        Median absolute residual (seconds) across final inliers.
    method : str
        Identifier string for the estimation method.
    """

    intercept_s: float
    slope: float
    drift_ppm: float
    ci_low_s: float
    ci_high_s: float
    slope_ci_low_ppm: float
    slope_ci_high_ppm: float
    n_input: int
    n_inliers: int
    mad_residual_s: float
    method: str = "theil_sen_iterative_mad"

    def predict(self, t_s: np.ndarray | float) -> np.ndarray | float:
        """Evaluate predicted clock offset ``t_ref - t_target`` at ``t_s``."""
        if np.ndim(t_s) == 0:
            return float(self.intercept_s + self.slope * float(t_s))
        t_arr = np.asarray(t_s, dtype=np.float64)
        return self.intercept_s + self.slope * t_arr

    def to_dict(self) -> dict[str, Any]:
        """Serialize drift fit metrics into a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class SyncResult:
    """Complete synchronization result produced by ``synchronize_clocks``.

    Attributes
    ----------
    coarse_offset_s : float
        Global wide-kernel coarse offset estimate ``t_ref - t_target`` (seconds).
    coarse_lag_s : float
        Raw global lag index (seconds) before PTT sign adjustment.
    coarse_prominence : float
        Prominence quality score of the global coarse cross-correlation peak.
    drift_fit : DriftFitResult
        Final robust affine drift model ``delta(t_ref) = a + b * t_ref``.
    windows : tuple[WindowEstimate, ...]
        Per-window candidate and rescue diagnostics.
    n_windows_total : int
        Total number of evaluated sliding windows.
    n_windows_retained : int
        Number of windows passing initial quality gating.
    n_windows_rescued : int
        Count of windows where a secondary peak rescued a cycle slip.
    used_global_consensus : bool
        Whether cross-half anchor consensus overrode a collapsed top-1 line.
    top1_inlier_ratio_200ms : float
        Fraction of quality-gated windows whose top-1 peak lies within 0.20 s
        of the initial Theil-Sen trajectory.
    ptt_prior_s : float
        Physiological pulse transit time prior (seconds) applied to offsets.
    config : SyncConfig
        Configuration used for synchronization.
    """

    coarse_offset_s: float
    coarse_lag_s: float
    coarse_prominence: float
    drift_fit: DriftFitResult
    windows: tuple[WindowEstimate, ...]
    n_windows_total: int
    n_windows_retained: int
    n_windows_rescued: int
    used_global_consensus: bool
    top1_inlier_ratio_200ms: float
    ptt_prior_s: float
    config: SyncConfig

    def predict_offset(self, t_ref_s: np.ndarray | float) -> np.ndarray | float:
        """Predict clock offset ``delta(t_ref) = t_ref - t_target`` (seconds)."""
        return self.drift_fit.predict(t_ref_s)

    def map_target_to_ref(self, t_target_s: np.ndarray | float) -> np.ndarray | float:
        """Map target device timestamps onto the reference timeline.

        Uses ``t_ref - t_target = a + b * t_ref``, yielding
        ``t_ref = (t_target + a) / (1.0 - b)``.
        """
        denom = 1.0 - self.drift_fit.slope
        if abs(denom) < 1e-9:
            raise ZeroDivisionError("Degenerate drift slope equals 1.0")
        if np.ndim(t_target_s) == 0:
            return float((float(t_target_s) + self.drift_fit.intercept_s) / denom)
        t_arr = np.asarray(t_target_s, dtype=np.float64)
        return (t_arr + self.drift_fit.intercept_s) / denom

    def map_ref_to_target(self, t_ref_s: np.ndarray | float) -> np.ndarray | float:
        """Map reference device timestamps onto the target timeline."""
        if np.ndim(t_ref_s) == 0:
            val = float(t_ref_s)
            return float((1.0 - self.drift_fit.slope) * val - self.drift_fit.intercept_s)
        t_arr = np.asarray(t_ref_s, dtype=np.float64)
        return (1.0 - self.drift_fit.slope) * t_arr - self.drift_fit.intercept_s

    def align_series(
        self,
        t_target_s: np.ndarray,
        values: np.ndarray,
        t_ref_grid_s: np.ndarray,
        fill_value: float = float("nan"),
    ) -> np.ndarray:
        """Resample a target time series onto a reference time grid."""
        t_tgt = np.asarray(t_target_s, dtype=np.float64)
        vals = np.asarray(values, dtype=np.float64)
        t_grid = np.asarray(t_ref_grid_s, dtype=np.float64)
        if t_tgt.shape[0] != vals.shape[0]:
            raise ValueError("t_target_s and values must share leading length")
        mapped_ref = np.asarray(self.map_target_to_ref(t_tgt), dtype=np.float64)
        order = np.argsort(mapped_ref)
        return np.interp(
            t_grid,
            mapped_ref[order],
            vals[order],
            left=fill_value,
            right=fill_value,
        )

    def window_dataframe(self) -> pd.DataFrame:
        """Export per-window estimates and rescue flags as a pandas DataFrame."""
        rows = [
            {
                "center_time_s": w.center_time_s,
                "quality": w.quality,
                "top1_offset_s": w.top1_offset(),
                "selected_offset_s": w.selected_offset_s,
                "selected_rank": w.selected_rank,
                "was_rescued": w.was_rescued,
                "retained": w.retained,
            }
            for w in self.windows
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a human-readable synchronization summary string."""
        df = self.drift_fit
        return (
            f"SyncResult(coarse_offset={self.coarse_offset_s:.3f}s, "
            f"intercept={df.intercept_s:.4f}s [{df.ci_low_s:.4f}, {df.ci_high_s:.4f}], "
            f"drift={df.drift_ppm:+.2f} ppm [{df.slope_ci_low_ppm:+.2f}, {df.slope_ci_high_ppm:+.2f}], "
            f"windows={self.n_windows_retained}/{self.n_windows_total}, "
            f"rescued={self.n_windows_rescued}, global_consensus={self.used_global_consensus})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar metrics into a JSON-compatible dictionary."""
        return {
            "coarse_offset_s": self.coarse_offset_s,
            "coarse_lag_s": self.coarse_lag_s,
            "coarse_prominence": self.coarse_prominence,
            "n_windows_total": self.n_windows_total,
            "n_windows_retained": self.n_windows_retained,
            "n_windows_rescued": self.n_windows_rescued,
            "used_global_consensus": self.used_global_consensus,
            "top1_inlier_ratio_200ms": self.top1_inlier_ratio_200ms,
            "ptt_prior_s": self.ptt_prior_s,
            "drift_fit": self.drift_fit.to_dict(),
        }


@dataclass(frozen=True)
class SegmentBoundary:
    """Synchronization model bound to one contiguous recording segment."""

    segment_index: int
    ref_start_s: float
    ref_end_s: float
    target_start_s: float
    target_end_s: float
    sync_result: SyncResult


@dataclass(frozen=True)
class SegmentSyncResult:
    """Piecewise clock synchronization across timestamp jumps or discontinuities.

    Attributes
    ----------
    segments : tuple[SegmentBoundary, ...]
        Synchronized contiguous segments ordered chronologically.
    n_ref_jumps : int
        Number of timestamp discontinuities detected on the reference stream.
    n_target_jumps : int
        Number of timestamp discontinuities detected on the target stream.
    """

    segments: tuple[SegmentBoundary, ...]
    n_ref_jumps: int
    n_target_jumps: int

    def map_piecewise(self, t_target_s: np.ndarray | float) -> np.ndarray | float:
        """Map target timestamps onto the reference timeline using segment models."""
        if not self.segments:
            raise ValueError("SegmentSyncResult contains no synchronized segments")
        scalar = np.ndim(t_target_s) == 0
        t_arr = np.atleast_1d(np.asarray(t_target_s, dtype=np.float64))
        out = np.empty_like(t_arr)
        bounds = np.array([seg.target_start_s for seg in self.segments], dtype=np.float64)
        for i, t_val in enumerate(t_arr):
            seg_idx = int(np.searchsorted(bounds, t_val, side="right") - 1)
            seg_idx = max(0, min(seg_idx, len(self.segments) - 1))
            out[i] = float(self.segments[seg_idx].sync_result.map_target_to_ref(float(t_val)))
        return float(out[0]) if scalar else out

    def to_dataframe(self) -> pd.DataFrame:
        """Return a summary DataFrame with one row per synchronized segment."""
        rows = [
            {
                "segment_index": seg.segment_index,
                "ref_start_s": seg.ref_start_s,
                "ref_end_s": seg.ref_end_s,
                "target_start_s": seg.target_start_s,
                "target_end_s": seg.target_end_s,
                "intercept_s": seg.sync_result.drift_fit.intercept_s,
                "drift_ppm": seg.sync_result.drift_fit.drift_ppm,
                "n_rescued": seg.sync_result.n_windows_rescued,
            }
            for seg in self.segments
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a concise description of segmented synchronization."""
        return (
            f"SegmentSyncResult(n_segments={len(self.segments)}, "
            f"n_ref_jumps={self.n_ref_jumps}, n_target_jumps={self.n_target_jumps})"
        )


@dataclass(frozen=True)
class EquivalenceReport:
    """Cross-package numerical parity report against SciPy reference routines.

    Attributes
    ----------
    xcorr_max_abs_err : float
        Maximum absolute discrepancy between FFT and direct SciPy correlation.
    xcorr_fft_ms : float
        Elapsed runtime (ms) of beatsync FFT cross-correlation.
    xcorr_direct_ms : float
        Elapsed runtime (ms) of direct time-domain correlation.
    xcorr_speedup : float
        Ratio ``xcorr_direct_ms / xcorr_fft_ms``.
    theil_sen_slope_abs_err : float
        Absolute slope error relative to ``scipy.stats.theilslopes``.
    theil_sen_intercept_abs_err : float
        Absolute intercept error relative to ``scipy.stats.theilslopes``.
    bandpass_max_abs_err : float
        Maximum absolute error of zero-phase bandpass vs explicit SciPy calls.
    passed : bool
        True when all numerical tolerances are strictly satisfied.
    """

    xcorr_max_abs_err: float
    xcorr_fft_ms: float
    xcorr_direct_ms: float
    xcorr_speedup: float
    theil_sen_slope_abs_err: float
    theil_sen_intercept_abs_err: float
    bandpass_max_abs_err: float
    passed: bool

    def summary(self) -> str:
        """Return a concise formatted parity summary."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"EquivalenceReport[{status}]: "
            f"xcorr_max_err={self.xcorr_max_abs_err:.2e} ({self.xcorr_speedup:.1f}x speedup), "
            f"theil_sen_slope_err={self.theil_sen_slope_abs_err:.2e}, "
            f"theil_sen_intercept_err={self.theil_sen_intercept_abs_err:.2e}, "
            f"bandpass_err={self.bandpass_max_abs_err:.2e}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize numerical parity fields to a dictionary."""
        return asdict(self)
