"""Tests for Stage 2b cycle-slip rescue, Stage 3 Theil-Sen fitting, and WESAD cohort recovery."""
from __future__ import annotations

import numpy as np
import pytest

from beatsync import (
    CardiacEvents,
    DualDeviceSimulator,
    SyncConfig,
    evaluate_alignment_accuracy,
    fit_coarse_ols_baseline,
    fit_ibi_ols_baseline,
    fit_single_peak_baseline,
    fit_theil_sen_drift,
    inject_clock_distortion,
    inject_waveform_distortion,
    rescue_subperiod_peaks,
    synchronize_clocks,
)


@pytest.mark.parametrize(
    "offset_s,drift_ppm",
    [
        (-85.0, -280.0),
        (-24.0, 290.0),
        (18.5, -195.0),
        (82.0, 265.0),
    ],
)
def test_wide_offset_and_drift_grid_recovery(
    wesad_cohort: list[dict],
    default_config: SyncConfig,
    offset_s: float,
    drift_ppm: float,
) -> None:
    """Recover injected startup offsets in [-90, 90]s and drift in [-300, 300]ppm within < 50 ms."""
    rec = wesad_cohort[0]
    nat_res = synchronize_clocks(events=rec["events"], config=default_config)
    dist_ev = inject_clock_distortion(rec["events"], offset_s=offset_s, drift_ppm=drift_ppm)
    res = synchronize_clocks(events=dist_ev, config=default_config)
    t_eval = np.linspace(60.0, 840.0, 40)
    metrics = evaluate_alignment_accuracy(
        res,
        t_eval,
        true_offset_s=offset_s,
        true_drift_ppm=drift_ppm,
        baseline_result=nat_res,
    )
    assert metrics["max_err_s"] < 0.050
    assert metrics["drift_err_ppm"] < 35.0


def test_severe_cycle_slip_rescue_beats_single_peak_baseline(
    simulator: DualDeviceSimulator,
    default_config: SyncConfig,
) -> None:
    """With 35 percent integer-IBI cycle slips, secondary-peak rescue crushes top-1 baselines."""
    windows, true_offsets = simulator.simulate_cycle_slip_windows(
        n_windows=40,
        duration_s=600.0,
        offset_s=-12.4,
        drift_ppm=280.0,
        slip_fraction=0.35,
        ibi_s=0.80,
    )
    times = np.array([w.center_time_s for w in windows])
    top1 = np.array([w.top1_offset() for w in windows])

    # Top-1 only Theil-Sen suffers bias from asymmetric/heavy cycle-slip contamination
    fit_rescued, updated, n_rescued, _, _ = rescue_subperiod_peaks(windows, default_config)
    # Test extreme 75 percent multi-harmonic contamination where single-peak collapses
    windows_heavy, true_heavy = simulator.simulate_cycle_slip_windows(
        n_windows=40,
        duration_s=600.0,
        offset_s=-12.4,
        drift_ppm=280.0,
        slip_fraction=0.75,
        ibi_s=0.80,
    )
    times_h = np.array([w.center_time_s for w in windows_heavy])
    top1_h = np.array([w.top1_offset() for w in windows_heavy])
    fit_single = fit_theil_sen_drift(times_h, top1_h, bootstrap_draws=0)
    fit_heavy, _, n_resc_h, used_global, _ = rescue_subperiod_peaks(windows_heavy, default_config)

    err_35_rescued = float(np.max(np.abs(fit_rescued.predict(times) - true_offsets)))
    err_heavy_single = float(np.median(np.abs(fit_single.predict(times_h) - true_heavy)))
    err_heavy_rescued = float(np.max(np.abs(fit_heavy.predict(times_h) - true_heavy)))

    assert n_rescued >= 6
    assert err_35_rescued < 0.015
    assert used_global is True
    assert n_resc_h >= 18
    assert err_heavy_rescued < 0.015
    assert err_heavy_single > 0.35
    assert err_heavy_single / max(err_heavy_rescued, 1e-6) > 20.0
    assert sum(1 for w in updated if w.was_rescued) == n_rescued


def test_real_wesad_s10_waveform_synchronization_and_series_alignment(
    wesad_s10_slice: dict,
    default_config: SyncConfig,
) -> None:
    """Synchronize distorted real WESAD S10 waveforms and align PPG onto ECG timeline."""
    nat = synchronize_clocks(
        ecg=wesad_s10_slice["ecg"],
        ppg=wesad_s10_slice["bvp"],
        fs_ecg_hz=wesad_s10_slice["fs_ecg_hz"],
        fs_ppg_hz=wesad_s10_slice["fs_bvp_hz"],
        config=default_config,
    )
    ecg_in, ppg_dist, t_eval = inject_waveform_distortion(
        wesad_s10_slice["ecg"],
        wesad_s10_slice["bvp"],
        fs_ecg_hz=wesad_s10_slice["fs_ecg_hz"],
        fs_ppg_hz=wesad_s10_slice["fs_bvp_hz"],
        offset_s=-38.4,
        drift_ppm=-210.0,
    )
    res = synchronize_clocks(
        ecg=ecg_in,
        ppg=ppg_dist,
        fs_ecg_hz=wesad_s10_slice["fs_ecg_hz"],
        fs_ppg_hz=wesad_s10_slice["fs_bvp_hz"],
        config=default_config,
    )
    metrics = evaluate_alignment_accuracy(
        res,
        t_eval,
        true_offset_s=-38.4,
        true_drift_ppm=-210.0,
        baseline_result=nat,
    )
    assert metrics["median_err_s"] < 0.025
    assert metrics["max_err_s"] < 0.045

    t_tgt = np.arange(ppg_dist.size, dtype=np.float64) / wesad_s10_slice["fs_bvp_hz"]
    t_grid = np.linspace(60.0, 260.0, 500)
    aligned = res.align_series(t_tgt, ppg_dist, t_grid)
    assert aligned.shape == t_grid.shape
    assert np.all(np.isfinite(aligned))


def test_five_subject_wesad_cohort_beats_ibi_ols(
    wesad_cohort: list[dict],
    default_config: SyncConfig,
) -> None:
    """Across all 5 real WESAD subjects, beatsync outperforms nearest-beat IBI OLS."""
    t_grid = np.linspace(60.0, 840.0, 40)
    beatsync_errs = []
    ibi_errs = []
    for rec in wesad_cohort:
        nat = synchronize_clocks(events=rec["events"], config=default_config)
        dist = inject_clock_distortion(rec["events"], offset_s=32.0, drift_ppm=-220.0)
        res = synchronize_clocks(events=dist, config=default_config)
        ibi = fit_ibi_ols_baseline(dist, config=default_config)
        m = evaluate_alignment_accuracy(res, t_grid, 32.0, -220.0, baseline_result=nat)
        truth = nat.predict_offset(t_grid) + 32.0 - 220.0e-6 * t_grid
        beatsync_errs.append(m["median_err_s"])
        ibi_errs.append(float(np.median(np.abs(ibi.predict(t_grid) - truth))))
    assert max(beatsync_errs) < 0.025
    assert float(np.median(beatsync_errs)) < float(np.median(ibi_errs)) * 0.15


def test_theil_sen_mad_pruning_and_physical_ppm_clamp() -> None:
    """Iterative MAD removes outliers and physical oscillator cap bounds slope."""
    times = np.linspace(0.0, 900.0, 25)
    offsets = 14.0 + 200e-6 * times
    offsets_noisy = offsets.copy()
    offsets_noisy[[2, 9, 17]] += np.array([1.2, -0.9, 1.5])
    fit = fit_theil_sen_drift(times, offsets_noisy, max_drift_ppm=1000.0, bootstrap_draws=50)
    assert abs(fit.intercept_s - 14.0) < 0.01
    assert abs(fit.drift_ppm - 200.0) < 5.0
    assert fit.n_inliers == 22

    # Extreme synthetic slope must clamp cleanly at +/- max_drift_ppm
    fit_clamped = fit_theil_sen_drift(times, 5.0 + 3500e-6 * times, max_drift_ppm=800.0, bootstrap_draws=0)
    assert np.isclose(fit_clamped.drift_ppm, 800.0)


def test_bootstrap_confidence_intervals_bracket_estimate() -> None:
    """Bootstrap 95 percent intervals properly bracket point intercept and drift."""
    rng = np.random.default_rng(19)
    times = np.linspace(30.0, 870.0, 35)
    offsets = -8.5 + 150e-6 * times + rng.normal(0.0, 0.008, size=35)
    fit = fit_theil_sen_drift(times, offsets, bootstrap_draws=200, seed=19)
    assert fit.ci_low_s < fit.intercept_s < fit.ci_high_s
    assert fit.slope_ci_low_ppm < fit.drift_ppm < fit.slope_ci_high_ppm


def test_synthetic_end_to_end_and_inverse_mapping(
    simulator: DualDeviceSimulator,
    default_config: SyncConfig,
) -> None:
    """Full waveform simulation recovers offset/drift and round-trips timestamps."""
    sim = simulator.simulate(duration_s=300.0, offset_s=-27.2, drift_ppm=260.0)
    res = synchronize_clocks(
        ecg=sim["ecg"],
        ppg=sim["ppg"],
        fs_ecg_hz=sim["fs_ecg_hz"],
        fs_ppg_hz=sim["fs_ppg_hz"],
        config=default_config,
    )
    assert abs(res.drift_fit.intercept_s - (-27.2)) < 0.03
    assert abs(res.drift_fit.drift_ppm - 260.0) < 40.0
    t_ref = np.linspace(10.0, 290.0, 20)
    round_trip = res.map_target_to_ref(res.map_ref_to_target(t_ref))
    assert np.allclose(round_trip, t_ref, atol=1e-11)


def test_ablation_baselines_return_valid_fits(wesad_cohort: list[dict], default_config: SyncConfig) -> None:
    """Single-peak and coarse-OLS ablation baselines execute cleanly on real events."""
    ev = wesad_cohort[1]["events"]
    sp_fit = fit_single_peak_baseline(ev, config=default_config)
    ols_fit = fit_coarse_ols_baseline(ev, config=default_config)
    assert sp_fit.method == "single_peak_theil_sen"
    assert ols_fit.method == "coarse_top1_ols"
    assert np.isfinite(sp_fit.drift_ppm) and np.isfinite(ols_fit.drift_ppm)


def test_short_recording_coarse_fallback() -> None:
    """Short recordings with fewer than one sliding window return a coarse fallback."""
    beats = np.arange(1.0, 29.0, 0.80)
    ev = CardiacEvents(
        ecg_rpeaks_s=beats,
        ppg_peaks_s=beats + 0.25 - 4.0,
        ref_duration_s=32.0,
        target_duration_s=32.0,
    )
    res = synchronize_clocks(events=ev, config=SyncConfig(window_s=45.0))
    assert res.drift_fit.method == "coarse_fallback"
    assert abs(res.coarse_offset_s - 4.0) < 0.20
