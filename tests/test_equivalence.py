"""Cross-package numerical equivalence and packaged dataset integrity tests."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import theilslopes

from beatsync import (
    bandpass_zero_phase,
    direct_xcorr_reference,
    encode_gaussian_impulses,
    fft_xcorr,
    fit_theil_sen_drift,
    load_wesad_benchmark_tables,
    verify_scipy_equivalence,
)


def test_fft_xcorr_matches_scipy_direct_to_machine_precision(wesad_cohort: list[dict]) -> None:
    """beatsync FFT cross-correlation matches scipy.signal.correlate(method='direct') < 1e-12."""
    ev = wesad_cohort[1]["events"]
    dur = 120.0
    a = encode_gaussian_impulses(ev.ecg_rpeaks_s[ev.ecg_rpeaks_s <= dur], dur, grid_fs_hz=64.0, sigma_s=0.035)
    b = encode_gaussian_impulses(ev.ppg_peaks_s[ev.ppg_peaks_s <= dur], dur, grid_fs_hz=64.0, sigma_s=0.035)
    lags_fft, corr_fft = fft_xcorr(a, b, -15.0, 15.0, grid_fs_hz=64.0)
    lags_dir, corr_dir = direct_xcorr_reference(a, b, -15.0, 15.0, grid_fs_hz=64.0)
    assert np.allclose(lags_fft, lags_dir, atol=1e-14)
    assert np.max(np.abs(corr_fft - corr_dir)) < 1e-12


def test_fft_xcorr_matches_direct_for_unequal_lengths() -> None:
    """Exact FFT/direct numerical parity holds even when input signals differ in length."""
    a = encode_gaussian_impulses(np.arange(2.0, 88.0, 0.82), 90.0, grid_fs_hz=64.0, sigma_s=0.05)
    b = encode_gaussian_impulses(np.arange(4.5, 112.0, 0.82), 115.0, grid_fs_hz=64.0, sigma_s=0.05)
    lags_fft, corr_fft = fft_xcorr(a, b, -20.0, 20.0, grid_fs_hz=64.0)
    lags_dir, corr_dir = direct_xcorr_reference(a, b, -20.0, 20.0, grid_fs_hz=64.0)
    assert np.max(np.abs(lags_fft - lags_dir)) < 1e-14
    assert np.max(np.abs(corr_fft - corr_dir)) < 1e-12


def test_theil_sen_matches_scipy_theilslopes_to_machine_precision() -> None:
    """fit_theil_sen_drift matches scipy.stats.theilslopes within < 1e-14."""
    rng = np.random.default_rng(123)
    for _ in range(5):
        x = np.sort(rng.uniform(0.0, 800.0, size=32))
        y = -18.3 + 310e-6 * x + rng.normal(0.0, 0.015, size=32)
        sp = theilslopes(y, x)
        bs = fit_theil_sen_drift(
            x,
            y,
            max_drift_ppm=5000.0,
            min_slope_span_s=0.0,
            mad_iterations=0,
            intercept_mode="scipy_separate",
            bootstrap_draws=0,
        )
        assert abs(bs.slope - float(sp.slope)) < 1e-14
        assert abs(bs.intercept_s - float(sp.intercept)) < 1e-14


def test_bandpass_matches_scipy_butter_filtfilt(wesad_s10_slice: dict) -> None:
    """Zero-phase bandpass filter matches explicit SciPy butter + filtfilt within < 1e-12."""
    ecg = wesad_s10_slice["ecg"][:5000]
    fs = wesad_s10_slice["fs_ecg_hz"]
    b, a = butter(2, [5.0 / (0.5 * fs), 25.0 / (0.5 * fs)], btype="band")
    ref = filtfilt(b, a, ecg)
    ours = bandpass_zero_phase(ecg, fs, 5.0, 25.0, order=2)
    assert np.max(np.abs(ours - ref)) < 1e-12


def test_verify_scipy_equivalence_report_passes_and_shows_speedup(wesad_cohort: list[dict]) -> None:
    """Full equivalence suite passes all checks and confirms substantial FFT speedup."""
    report = verify_scipy_equivalence(events=wesad_cohort[1]["events"])
    assert report.passed is True
    assert report.xcorr_max_abs_err < 1e-12
    assert report.theil_sen_slope_abs_err < 1e-14
    assert report.theil_sen_intercept_abs_err < 1e-14
    assert report.bandpass_max_abs_err < 1e-12
    assert report.xcorr_speedup > 5.0
    assert "PASS" in report.summary()


def test_packaged_75seg_summary_table_integrity() -> None:
    """Bundled 75-segment WESAD comparison table contains all 8 methods and valid bounds."""
    df_sum, _ = load_wesad_benchmark_tables()
    assert len(df_sum) == 8
    assert set(df_sum["n_segments"]) == {75}
    row_full = df_sum.loc[df_sum["method_id"] == "beatsync_full"].iloc[0]
    assert row_full["rate_lt_100ms"] == 1.0
    assert row_full["rate_lt_50ms"] >= 0.90
    assert row_full["median_error_s"] < 0.015


def test_packaged_15subj_raw_hardware_table_integrity() -> None:
    """Bundled 15-subject raw WESAD hardware table reports 15/15 physically plausible drifts."""
    _, df_hw = load_wesad_benchmark_tables()
    assert len(df_hw) == 15
    assert np.all(np.abs(df_hw["drift_ppm"]) <= 600.0)
    assert np.all(df_hw["e4_channel_lag_s"] > 0.5)
    assert np.isclose(df_hw["e4_channel_lag_s"].median(), 1.639, atol=0.01)
