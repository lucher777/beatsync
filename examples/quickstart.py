"""End-to-end quickstart on built-in real WESAD data and SciPy parity checks."""
from __future__ import annotations

import numpy as np

from beatsync import (
    SyncConfig,
    evaluate_alignment_accuracy,
    fit_ibi_ols_baseline,
    inject_clock_distortion,
    inject_waveform_distortion,
    load_wesad_event_cohort,
    load_wesad_waveform_slice,
    synchronize_clocks,
    verify_scipy_equivalence,
)


def main() -> None:
    """Demonstrate beatsync synchronization, 5-subject cohort comparison, and SciPy parity."""
    print("1. Real WESAD S10 Dual-Device Waveform Demonstration")
    sl = load_wesad_waveform_slice()
    cfg = SyncConfig(ptt_prior_s=0.250, bootstrap_draws=100)
    native_res = synchronize_clocks(
        ecg=sl["ecg"],
        ppg=sl["bvp"],
        fs_ecg_hz=sl["fs_ecg_hz"],
        fs_ppg_hz=sl["fs_bvp_hz"],
        config=cfg,
        subject_id=sl["subject_id"],
    )
    ecg_in, ppg_dist, t_eval = inject_waveform_distortion(
        sl["ecg"],
        sl["bvp"],
        fs_ecg_hz=sl["fs_ecg_hz"],
        fs_ppg_hz=sl["fs_bvp_hz"],
        offset_s=-48.5,
        drift_ppm=240.0,
    )
    sync_res = synchronize_clocks(
        ecg=ecg_in,
        ppg=ppg_dist,
        fs_ecg_hz=sl["fs_ecg_hz"],
        fs_ppg_hz=sl["fs_bvp_hz"],
        config=cfg,
        subject_id="S10_distorted",
    )
    acc = evaluate_alignment_accuracy(
        sync_res,
        t_eval,
        true_offset_s=-48.5,
        true_drift_ppm=240.0,
        baseline_result=native_res,
    )
    print(f"   {sync_res.summary()}")
    print(
        f"   Recovered accuracy: median={acc['median_err_s'] * 1000.0:.2f} ms, "
        f"max={acc['max_err_s'] * 1000.0:.2f} ms, drift_err={acc['drift_err_ppm']:.2f} ppm"
    )

    print("\n2. Five-Subject Real WESAD 15-Minute Event Cohort (offset=+32.0s, drift=-220 ppm)")
    cohort = load_wesad_event_cohort()
    t_grid = np.linspace(60.0, 840.0, 40)
    for rec in cohort:
        nat_res = synchronize_clocks(events=rec["events"], config=cfg)
        dist_ev = inject_clock_distortion(
            rec["events"], offset_s=32.0, drift_ppm=-220.0, keep_within_duration=False
        )
        res = synchronize_clocks(events=dist_ev, config=cfg)
        ibi_fit = fit_ibi_ols_baseline(dist_ev, config=cfg)
        m = evaluate_alignment_accuracy(
            res,
            t_grid,
            true_offset_s=32.0,
            true_drift_ppm=-220.0,
            baseline_result=nat_res,
        )
        ibi_err = float(
            np.median(
                np.abs(
                    ibi_fit.predict(t_grid)
                    - (nat_res.predict_offset(t_grid) + 32.0 - 220.0e-6 * t_grid)
                )
            )
        )
        print(
            f"   {rec['subject_id']}: beatsync_med={m['median_err_s'] * 1000:.2f} ms "
            f"(drift_err={m['drift_err_ppm']:.1f} ppm, rescued={res.n_windows_rescued}) "
            f"| IBI-OLS_med={ibi_err * 1000:.1f} ms"
        )

    print("\n3. Cross-Package SciPy Numerical Parity & Speedup Check")
    report = verify_scipy_equivalence(events=cohort[1]["events"])
    print(f"   {report.summary()}")


if __name__ == "__main__":
    main()
