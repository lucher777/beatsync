"""Tests for Stage 1 event extraction and Stage 2a dual-scale cross-correlation."""
from __future__ import annotations

import numpy as np
import pytest

from beatsync import (
    CardiacEvents,
    DualDeviceSimulator,
    SyncConfig,
    bandpass_zero_phase,
    compute_prominence_quality,
    detect_cardiac_events,
    detect_ecg_rpeaks,
    detect_ppg_peaks,
    dual_scale_xcorr,
    encode_gaussian_impulses,
    extract_motion_envelope,
    extract_topk_peaks,
    fft_xcorr,
)


def test_detect_events_on_synthetic_waveforms(simulator: DualDeviceSimulator) -> None:
    """Detected R-peaks and PPG pulses recover true beat counts and heart rate."""
    sim = simulator.simulate(duration_s=120.0, offset_s=0.0, drift_ppm=0.0)
    events = detect_cardiac_events(
        sim["ecg"],
        sim["ppg"],
        fs_ecg_hz=sim["fs_ecg_hz"],
        fs_ppg_hz=sim["fs_ppg_hz"],
    )
    assert abs(events.n_ecg - sim["true_rpeaks_s"].size) <= 4
    assert events.n_ppg >= int(0.85 * sim["true_ppg_peaks_s"].size)
    ecg_hr, ppg_hr = events.mean_hr_bpm()
    assert 60.0 <= ecg_hr <= 85.0
    assert 60.0 <= ppg_hr <= 85.0


def test_detect_events_on_real_wesad_s10(wesad_s10_slice: dict) -> None:
    """Real WESAD S10 slice yields physiological beat counts and clean summary."""
    events = detect_cardiac_events(
        wesad_s10_slice["ecg"],
        wesad_s10_slice["bvp"],
        fs_ecg_hz=wesad_s10_slice["fs_ecg_hz"],
        fs_ppg_hz=wesad_s10_slice["fs_bvp_hz"],
        subject_id=wesad_s10_slice["subject_id"],
        refine_subsample=True,
    )
    assert 500 <= events.n_ecg <= 700
    assert 500 <= events.n_ppg <= 700
    assert "subject=S10" in events.summary()


def test_ppg_short_ibi_artifact_pruning_without_interpolation() -> None:
    """Short premature artifact peaks are pruned without filling dropped beats."""
    fs = 64.0
    dur = 30.0
    t = np.arange(int(dur * fs)) / fs
    beat_times = np.arange(1.0, 29.0, 0.85)
    # Drop two adjacent beats around t=12s to confirm large genuine gaps remain unfilled
    beat_times = beat_times[(beat_times < 11.8) | (beat_times > 13.1)]
    ppg = np.zeros_like(t)
    for bt in beat_times:
        ppg += np.exp(-0.5 * ((t - bt) / 0.05) ** 2)
    artifact_t = float(beat_times[6] + 0.36)
    ppg += 0.70 * np.exp(-0.5 * ((t - artifact_t) / 0.045) ** 2)

    raw_peaks = detect_ppg_peaks(ppg, fs_ppg_hz=fs, filter_short_ibi=False)
    cleaned_peaks = detect_ppg_peaks(ppg, fs_ppg_hz=fs, filter_short_ibi=True)
    assert np.min(np.abs(raw_peaks - artifact_t)) < 0.05
    assert np.min(np.abs(cleaned_peaks - artifact_t)) > 0.25
    assert cleaned_peaks.size < raw_peaks.size
    # Verify genuine gap around t=12.5s remains unfilled without artificial interpolation
    assert np.max(np.diff(cleaned_peaks)) > 1.3


def test_encode_gaussian_impulses_energy_and_widths() -> None:
    """Wide coarse kernel produces broader autocorrelation support than narrow kernel."""
    beats = np.arange(5.0, 115.0, 0.80)
    wide = encode_gaussian_impulses(beats, duration_s=120.0, grid_fs_hz=64.0, sigma_s=0.120)
    narrow = encode_gaussian_impulses(beats, duration_s=120.0, grid_fs_hz=64.0, sigma_s=0.035)
    assert wide.shape == narrow.shape
    assert np.sum(wide > 0.01) > np.sum(narrow > 0.01)
    assert np.isclose(np.sum(wide) / np.sum(narrow), 0.120 / 0.035, rtol=0.10)


def test_fft_xcorr_recovers_exact_lag_and_prominence() -> None:
    """FFT cross-correlation locates known shift with high prominence quality."""
    rng = np.random.default_rng(7)
    beats_ref = np.cumsum(rng.uniform(0.70, 0.95, size=140))
    beats_ref = beats_ref[beats_ref < 110.0]
    shift = 6.25
    beats_tgt = beats_ref + shift
    a = encode_gaussian_impulses(beats_ref, 125.0, grid_fs_hz=64.0, sigma_s=0.035)
    b = encode_gaussian_impulses(beats_tgt, 125.0, grid_fs_hz=64.0, sigma_s=0.035)

    lags, corr = fft_xcorr(a, b, min_lag_s=-15.0, max_lag_s=15.0, grid_fs_hz=64.0)
    est_lag = float(lags[int(np.argmax(corr))])
    assert abs(est_lag - shift) < 0.02
    assert compute_prominence_quality(corr) > 3.0


def test_extract_topk_peaks_parabolic_subsample_accuracy() -> None:
    """Three-point parabolic interpolation resolves sub-sample fractional lags."""
    fs = 64.0
    lags = np.arange(-128, 129, dtype=np.float64) / fs
    true_lag = 0.25 + 0.0041  # off-grid fraction of 1/64 s
    corr = np.exp(-0.5 * ((lags - true_lag) / 0.045) ** 2)
    corr += 0.55 * np.exp(-0.5 * ((lags - (true_lag + 0.85)) / 0.045) ** 2)
    cands = extract_topk_peaks(lags, corr, grid_fs_hz=fs, ptt_prior_s=0.25, top_k=3)
    assert len(cands) == 2
    assert cands[0].rank == 1
    assert abs(cands[0].offset_s - (-0.0041)) < 0.002


def test_dual_scale_xcorr_handles_asymmetric_durations() -> None:
    """Coarse and windowed cross-correlation handle unequal ECG and PPG lengths."""
    rng = np.random.default_rng(42)
    beats = np.cumsum(rng.uniform(0.60, 1.10, size=230))
    ev = CardiacEvents(
        ecg_rpeaks_s=beats[beats < 180.0],
        ppg_peaks_s=(beats + 0.25 - 12.5)[(beats + 0.25 - 12.5) > 0.0],
        ref_duration_s=180.0,
        target_duration_s=210.0,
    )
    coarse_off, _, prom, wins = dual_scale_xcorr(ev, SyncConfig(ptt_prior_s=0.25))
    assert abs(coarse_off - 12.5) < 0.25
    assert prom > 0.1
    assert len(wins) >= 8
    assert np.median([w.quality for w in wins]) > 1.5


def test_extract_motion_envelope_1d_and_3d() -> None:
    """Motion envelope works on both 1D magnitude and 3D triaxial accelerometer inputs."""
    t = np.arange(0, 10.0, 1.0 / 32.0)
    acc_3d = np.column_stack([np.sin(2.0 * np.pi * t), np.cos(2.0 * np.pi * t), np.ones_like(t)])
    env_3d = extract_motion_envelope(acc_3d, fs_in_hz=32.0, grid_fs_hz=64.0)
    env_1d = extract_motion_envelope(np.linalg.norm(acc_3d, axis=1), fs_in_hz=32.0, grid_fs_hz=64.0)
    assert env_3d.shape == (640,)
    assert np.allclose(env_3d, env_1d)


def test_stage1_and_xcorr_validation_errors() -> None:
    """Invalid sampling rates, empty inputs, and bad filter bands raise ValueError."""
    sig = np.ones(100)
    with pytest.raises(ValueError, match="Bandpass"):
        bandpass_zero_phase(sig, fs_hz=20.0, low_hz=5.0, high_hz=12.0)
    with pytest.raises(ValueError, match="non-empty"):
        detect_ecg_rpeaks(np.array([]), fs_ecg_hz=250.0)
    with pytest.raises(ValueError, match="min_lag_s"):
        fft_xcorr(sig, sig, min_lag_s=5.0, max_lag_s=2.0)
