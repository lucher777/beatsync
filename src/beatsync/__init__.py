"""beatsync: Dual-scale cardiac event cross-correlation and clock drift synchronization."""
from __future__ import annotations

__version__ = "0.1.0"

from .datasets import (
    load_wesad_benchmark_tables,
    load_wesad_event_cohort,
    load_wesad_waveform_slice,
)
from .events import (
    bandpass_zero_phase,
    detect_cardiac_events,
    detect_ecg_rpeaks,
    detect_ppg_peaks,
    encode_gaussian_impulses,
    extract_motion_envelope,
)
from .segments import (
    detect_timestamp_jumps,
    evaluate_alignment_accuracy,
    fit_coarse_ols_baseline,
    fit_ibi_ols_baseline,
    fit_motion_xcorr_baseline,
    fit_single_peak_baseline,
    synchronize_segmented_streams,
    verify_scipy_equivalence,
)
from .synthetic import (
    DualDeviceSimulator,
    inject_clock_distortion,
    inject_waveform_distortion,
)
from .trajectory import (
    fit_theil_sen_drift,
    rescue_subperiod_peaks,
    score_subperiod_trajectory,
    synchronize_clocks,
)
from .types import (
    CandidatePeak,
    CardiacEvents,
    DriftFitResult,
    EquivalenceReport,
    SegmentBoundary,
    SegmentSyncResult,
    SyncConfig,
    SyncResult,
    WindowEstimate,
)
from .xcorr import (
    compute_prominence_quality,
    direct_xcorr_reference,
    dual_scale_xcorr,
    extract_topk_peaks,
    fft_xcorr,
)

__all__ = [
    "__version__",
    "CandidatePeak",
    "CardiacEvents",
    "DriftFitResult",
    "DualDeviceSimulator",
    "EquivalenceReport",
    "SegmentBoundary",
    "SegmentSyncResult",
    "SyncConfig",
    "SyncResult",
    "WindowEstimate",
    "bandpass_zero_phase",
    "compute_prominence_quality",
    "detect_cardiac_events",
    "detect_ecg_rpeaks",
    "detect_ppg_peaks",
    "detect_timestamp_jumps",
    "direct_xcorr_reference",
    "dual_scale_xcorr",
    "encode_gaussian_impulses",
    "evaluate_alignment_accuracy",
    "extract_motion_envelope",
    "extract_topk_peaks",
    "fft_xcorr",
    "fit_coarse_ols_baseline",
    "fit_ibi_ols_baseline",
    "fit_motion_xcorr_baseline",
    "fit_single_peak_baseline",
    "fit_theil_sen_drift",
    "inject_clock_distortion",
    "inject_waveform_distortion",
    "load_wesad_benchmark_tables",
    "load_wesad_event_cohort",
    "load_wesad_waveform_slice",
    "rescue_subperiod_peaks",
    "score_subperiod_trajectory",
    "synchronize_clocks",
    "synchronize_segmented_streams",
    "verify_scipy_equivalence",
]
