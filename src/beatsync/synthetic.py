"""Deterministic synthetic dual-device wearable simulator and distortion injector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import CandidatePeak, CardiacEvents, WindowEstimate


def inject_clock_distortion(
    events: CardiacEvents,
    offset_s: float,
    drift_ppm: float,
    keep_within_duration: bool = False,
) -> CardiacEvents:
    """Apply an exact affine clock offset and crystal drift to cardiac events.

    Applies the mapping ``t_ref - t_target = offset_s + (drift_ppm * 1e-6) * t_ref``,
    yielding ``t_target = (1 - drift_ppm * 1e-6) * t_ref - offset_s``.

    Parameters
    ----------
    events : CardiacEvents
        Reference-aligned cardiac event trains.
    offset_s : float
        Injected startup offset ``t_ref - t_target`` (seconds) at ``t_ref = 0``.
    drift_ppm : float
        Injected oscillator clock drift in parts per million.
    keep_within_duration : bool, default=False
        If True, clip shifted PPG events to ``[0, target_duration_s]``; if False,
        shift timeline origin so negative target times are preserved without loss.

    Returns
    -------
    CardiacEvents
        Distorted event container with updated target duration.
    """
    slope = float(drift_ppm) * 1e-6
    warped_ppg = (1.0 - slope) * events.ppg_peaks_s - float(offset_s)
    if keep_within_duration:
        mask = (warped_ppg >= 0.0) & (warped_ppg <= events.target_duration_s)
        warped_ppg = warped_ppg[mask]
        new_tgt_dur = events.target_duration_s
    else:
        max_t = float(np.max(warped_ppg)) + 1.0 if warped_ppg.size else events.target_duration_s
        new_tgt_dur = max(events.target_duration_s, max_t)
    return CardiacEvents(
        ecg_rpeaks_s=events.ecg_rpeaks_s.copy(),
        ppg_peaks_s=warped_ppg,
        ref_duration_s=events.ref_duration_s,
        target_duration_s=new_tgt_dur,
        subject_id=events.subject_id,
        metadata={
            **events.metadata,
            "injected_offset_s": float(offset_s),
            "injected_drift_ppm": float(drift_ppm),
        },
    )


def inject_waveform_distortion(
    ecg: np.ndarray,
    ppg: np.ndarray,
    fs_ecg_hz: float,
    fs_ppg_hz: float,
    offset_s: float,
    drift_ppm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample an aligned target PPG waveform under startup offset and crystal drift.

    Parameters
    ----------
    ecg : np.ndarray
        Reference ECG signal at ``fs_ecg_hz``.
    ppg : np.ndarray
        Aligned target PPG signal at ``fs_ppg_hz``.
    fs_ecg_hz : float
        ECG sampling rate in Hz.
    fs_ppg_hz : float
        PPG sampling rate in Hz.
    offset_s : float
        Injected startup offset ``t_ref - t_target`` (seconds).
    drift_ppm : float
        Injected relative crystal drift in parts per million.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(ecg_out, ppg_distorted, t_ref_eval_s)`` sampled on the target grid.
    """
    n_ppg = ppg.size
    t_target = np.arange(n_ppg, dtype=np.float64) / float(fs_ppg_hz)
    slope = float(drift_ppm) * 1e-6
    t_ref_lookup = (t_target + float(offset_s)) / (1.0 - slope)
    t_orig = np.arange(n_ppg, dtype=np.float64) / float(fs_ppg_hz)
    ppg_distorted = np.interp(t_ref_lookup, t_orig, ppg.astype(np.float64), left=0.0, right=0.0)
    t_eval = np.linspace(30.0, max(31.0, float(ecg.size) / float(fs_ecg_hz) - 30.0), 40)
    return ecg.copy(), ppg_distorted, t_eval


@dataclass
class DualDeviceSimulator:
    """Parametric generator for asynchronous dual-device ECG and PPG recordings.

    Synthesizes non-stationary heart rate variability (RSA + Mayer waves),
    morphological ECG QRS complexes, peripheral dicrotic PPG pulses delayed by
    pulse transit time, missed-beat dropout, motion bursts, startup offset,
    and crystal oscillator drift.

    Attributes
    ----------
    base_hr_bpm : float
        Mean resting heart rate in beats per minute.
    ptt_s : float
        Physiological pulse transit time (seconds) between ECG R-peak and PPG peak.
    ecg_fs_hz : float
        Simulated chest ECG sampling rate in Hz.
    ppg_fs_hz : float
        Simulated wrist PPG sampling rate in Hz.
    dropout_prob : float
        Probability of dropping an individual peripheral PPG beat.
    noise_std : float
        Additive Gaussian measurement noise standard deviation.
    seed : int
        Deterministic random generator seed.
    """

    base_hr_bpm: float = 72.0
    ptt_s: float = 0.250
    ecg_fs_hz: float = 250.0
    ppg_fs_hz: float = 64.0
    dropout_prob: float = 0.04
    noise_std: float = 0.04
    seed: int = 2026

    def _generate_beat_schedule(
        self,
        total_span_s: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate physiological R-peak timestamps with RSA and Mayer variability."""
        base_ibi = 60.0 / self.base_hr_bpm
        beats: list[float] = []
        t = 0.8
        while t < total_span_s:
            rsa = 0.065 * np.sin(2.0 * np.pi * 0.23 * t)
            mayer = 0.045 * np.sin(2.0 * np.pi * 0.08 * t + 0.9)
            jitter = float(rng.normal(0.0, 0.014))
            ibi = float(np.clip(base_ibi + rsa + mayer + jitter, 0.45, 1.35))
            beats.append(t)
            t += ibi
        return np.asarray(beats, dtype=np.float64)

    def simulate(
        self,
        duration_s: float = 360.0,
        offset_s: float = 18.5,
        drift_ppm: float = 220.0,
        subject_id: str = "synthetic_01",
    ) -> dict[str, Any]:
        """Simulate paired asynchronous ECG and PPG waveforms with ground truth.

        Parameters
        ----------
        duration_s : float, default=360.0
            Target recording duration (seconds) on both clocks.
        offset_s : float, default=18.5
            Startup clock offset ``t_ref - t_target`` (seconds) at ``t_ref = 0``.
        drift_ppm : float, default=220.0
            Linear relative oscillator drift in parts per million.
        subject_id : str, default="synthetic_01"
            Identifier stored in the returned payload.

        Returns
        -------
        dict[str, Any]
            Dictionary containing ``ecg``, ``ppg``, ``fs_ecg_hz``, ``fs_ppg_hz``,
            ``true_rpeaks_s``, ``true_ppg_peaks_s``, ``true_offset_s``,
            ``true_drift_ppm``, ``t_ecg_s``, and ``t_ppg_s``.
        """
        if duration_s < 30.0:
            raise ValueError("duration_s must be >= 30.0 s")
        rng = np.random.default_rng(self.seed)
        pad = abs(offset_s) + 30.0
        raw_beats = self._generate_beat_schedule(duration_s + 2.0 * pad, rng) - pad

        # Reference ECG timestamps and waveform
        n_ecg = int(round(duration_s * self.ecg_fs_hz))
        t_ecg = np.arange(n_ecg, dtype=np.float64) / self.ecg_fs_hz
        ecg = rng.normal(0.0, self.noise_std * 0.5, size=n_ecg)
        r_valid = raw_beats[(raw_beats >= 0.5) & (raw_beats <= duration_s - 0.5)]
        for r_t in raw_beats:
            if -0.2 <= r_t <= duration_s + 0.2:
                dt = t_ecg - r_t
                mask = np.abs(dt) <= 0.12
                dtm = dt[mask]
                qrs = np.exp(-0.5 * (dtm / 0.011) ** 2) - 0.25 * np.exp(-0.5 * ((dtm - 0.025) / 0.013) ** 2)
                t_wave = 0.22 * np.exp(-0.5 * ((dtm - 0.16) / 0.035) ** 2)
                ecg[mask] += qrs + t_wave

        # Target PPG beats: physical pulse arrival at t_ref = r_t + ptt_s
        slope = drift_ppm * 1e-6
        ppg_ref_times = raw_beats + self.ptt_s + rng.normal(0.0, 0.004, size=raw_beats.size)
        keep_mask = rng.random(ppg_ref_times.size) >= self.dropout_prob
        ppg_ref_times = ppg_ref_times[keep_mask]
        ppg_target_times = (1.0 - slope) * ppg_ref_times - offset_s

        n_ppg = int(round(duration_s * self.ppg_fs_hz))
        t_ppg = np.arange(n_ppg, dtype=np.float64) / self.ppg_fs_hz
        ppg = rng.normal(0.0, self.noise_std, size=n_ppg)
        ppg += 0.12 * np.sin(2.0 * np.pi * 0.20 * t_ppg)
        for p_t in ppg_target_times:
            if -0.4 <= p_t <= duration_s + 0.4:
                dt = t_ppg - p_t
                mask = np.abs(dt) <= 0.30
                dtm = dt[mask]
                systolic = np.exp(-0.5 * (dtm / 0.055) ** 2)
                dicrotic = 0.35 * np.exp(-0.5 * ((dtm - 0.17) / 0.060) ** 2)
                ppg[mask] += systolic + dicrotic

        p_valid = ppg_target_times[(ppg_target_times >= 0.5) & (ppg_target_times <= duration_s - 0.5)]
        return {
            "subject_id": subject_id,
            "ecg": ecg,
            "ppg": ppg,
            "fs_ecg_hz": self.ecg_fs_hz,
            "fs_ppg_hz": self.ppg_fs_hz,
            "t_ecg_s": t_ecg,
            "t_ppg_s": t_ppg,
            "true_rpeaks_s": r_valid,
            "true_ppg_peaks_s": p_valid,
            "true_offset_s": float(offset_s),
            "true_drift_ppm": float(drift_ppm),
            "duration_s": float(duration_s),
        }

    def simulate_cycle_slip_windows(
        self,
        n_windows: int = 40,
        duration_s: float = 600.0,
        offset_s: float = -12.4,
        drift_ppm: float = 280.0,
        slip_fraction: float = 0.35,
        ibi_s: float = 0.80,
    ) -> tuple[list[WindowEstimate], np.ndarray]:
        """Generate synthetic multi-peak sliding windows with controlled cycle slips.

        Useful for controlled ablation testing of ``rescue_subperiod_peaks``
        where ``slip_fraction`` (e.g. 0.35) of windows experience dominant
        top-1 correlation peaks shifted by +/- 1 integer cardiac period while
        the true physiological offset remains as a secondary candidate peak.
        """
        rng = np.random.default_rng(self.seed + 101)
        times = np.linspace(22.5, duration_s - 22.5, n_windows)
        slope = drift_ppm * 1e-6
        true_offsets = offset_s + slope * times
        n_slips = int(round(slip_fraction * n_windows))
        slip_indices = set(rng.choice(n_windows, size=n_slips, replace=False).tolist())

        windows: list[WindowEstimate] = []
        multipliers = (1.0, 2.0, -1.0)
        for idx, (tc, y_true) in enumerate(zip(times, true_offsets)):
            noise_true = float(rng.normal(0.0, 0.004))
            y_phys = y_true + noise_true
            qual = float(rng.uniform(2.5, 6.5))
            if idx in slip_indices:
                mult = multipliers[idx % 3]
                y_slip = y_true + mult * ibi_s + float(rng.normal(0.0, 0.008))
                cands = (
                    CandidatePeak(offset_s=y_slip, correlation=0.68, rank=1),
                    CandidatePeak(offset_s=y_phys, correlation=0.62, rank=2),
                    CandidatePeak(offset_s=y_true - mult * ibi_s, correlation=0.36, rank=3),
                )
            else:
                cands = (
                    CandidatePeak(offset_s=y_phys, correlation=0.72, rank=1),
                    CandidatePeak(offset_s=y_true + ibi_s, correlation=0.45, rank=2),
                    CandidatePeak(offset_s=y_true - ibi_s, correlation=0.42, rank=3),
                )
            windows.append(
                WindowEstimate(
                    center_time_s=float(tc),
                    quality=qual,
                    candidates=cands,
                    selected_offset_s=cands[0].offset_s,
                    selected_rank=1,
                    was_rescued=False,
                    retained=True,
                )
            )
        return windows, true_offsets

    def simulate_segmented(
        self,
        segment_duration_s: float = 240.0,
        offset_seg1_s: float = 12.0,
        offset_seg2_s: float = 28.5,
        drift_ppm: float = 180.0,
        gap_s: float = 40.0,
    ) -> dict[str, Any]:
        """Simulate a two-segment recording interrupted by a mid-session clock jump."""
        sim1 = self.simulate(
            duration_s=segment_duration_s,
            offset_s=offset_seg1_s,
            drift_ppm=drift_ppm,
            subject_id="seg_0",
        )
        sim2 = DualDeviceSimulator(
            base_hr_bpm=self.base_hr_bpm,
            ptt_s=self.ptt_s,
            ecg_fs_hz=self.ecg_fs_hz,
            ppg_fs_hz=self.ppg_fs_hz,
            dropout_prob=self.dropout_prob,
            noise_std=self.noise_std,
            seed=self.seed + 7,
        ).simulate(
            duration_s=segment_duration_s,
            offset_s=offset_seg2_s,
            drift_ppm=drift_ppm,
            subject_id="seg_1",
        )

        ref_shift = segment_duration_s + gap_s
        tgt_shift = segment_duration_s + gap_s
        t_ecg = np.concatenate([sim1["t_ecg_s"], sim2["t_ecg_s"] + ref_shift])
        ecg = np.concatenate([sim1["ecg"], sim2["ecg"]])
        t_ppg = np.concatenate([sim1["t_ppg_s"], sim2["t_ppg_s"] + tgt_shift])
        ppg = np.concatenate([sim1["ppg"], sim2["ppg"]])
        return {
            "t_ecg_s": t_ecg,
            "ecg": ecg,
            "t_ppg_s": t_ppg,
            "ppg": ppg,
            "offset_seg1_s": offset_seg1_s,
            "offset_seg2_s": offset_seg2_s,
            "drift_ppm": drift_ppm,
            "ref_shift_s": ref_shift,
            "tgt_shift_s": tgt_shift,
        }
