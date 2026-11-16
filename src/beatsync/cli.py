"""Command-line interface for beatsync cardiac clock synchronization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

from .datasets import (
    load_wesad_benchmark_tables,
    load_wesad_event_cohort,
    load_wesad_waveform_slice,
)
from .segments import (
    evaluate_alignment_accuracy,
    fit_single_peak_baseline,
    verify_scipy_equivalence,
)
from .synthetic import inject_clock_distortion, inject_waveform_distortion
from .trajectory import synchronize_clocks
from .types import SyncConfig


def _cmd_demo(args: argparse.Namespace) -> int:
    """Execute synchronization demo on the built-in real WESAD S10 slice."""
    sl = load_wesad_waveform_slice()
    cfg = SyncConfig(ptt_prior_s=args.ptt, bootstrap_draws=args.bootstrap)
    base_res = synchronize_clocks(
        ecg=sl["ecg"],
        ppg=sl["bvp"],
        fs_ecg_hz=sl["fs_ecg_hz"],
        fs_ppg_hz=sl["fs_bvp_hz"],
        config=cfg,
        subject_id="S10_native",
    )
    ecg_in, ppg_dist, t_eval = inject_waveform_distortion(
        sl["ecg"],
        sl["bvp"],
        fs_ecg_hz=sl["fs_ecg_hz"],
        fs_ppg_hz=sl["fs_bvp_hz"],
        offset_s=args.offset,
        drift_ppm=args.drift_ppm,
    )
    inj_res = synchronize_clocks(
        ecg=ecg_in,
        ppg=ppg_dist,
        fs_ecg_hz=sl["fs_ecg_hz"],
        fs_ppg_hz=sl["fs_bvp_hz"],
        config=cfg,
        subject_id="S10_injected",
    )
    metrics = evaluate_alignment_accuracy(
        inj_res,
        t_eval,
        true_offset_s=args.offset,
        true_drift_ppm=args.drift_ppm,
        baseline_result=base_res,
    )
    payload = {
        "subject_id": sl["subject_id"],
        "injected_offset_s": args.offset,
        "injected_drift_ppm": args.drift_ppm,
        "sync_result": inj_res.to_dict(),
        "accuracy_metrics": metrics,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Loaded real WESAD {sl['subject_id']} slice ({sl['duration_s']:.0f}s)")
        print(f"Injected: offset={args.offset:+.2f}s, drift={args.drift_ppm:+.1f} ppm")
        print(inj_res.summary())
        print(
            f"Recovered relative error: median={metrics['median_err_s']*1000:.2f} ms, "
            f"max={metrics['max_err_s']*1000:.2f} ms, "
            f"drift_err={metrics['drift_err_ppm']:.2f} ppm"
        )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Run live 5-subject WESAD cohort evaluation and print archived benchmark tables."""
    cohort = load_wesad_event_cohort()
    cfg = SyncConfig(ptt_prior_s=args.ptt, bootstrap_draws=100)
    rows = []
    for rec in cohort:
        ev = rec["events"]
        nat_res = synchronize_clocks(events=ev, config=cfg)
        ev_dist = inject_clock_distortion(
            ev,
            offset_s=args.offset,
            drift_ppm=args.drift_ppm,
            keep_within_duration=False,
        )
        res = synchronize_clocks(events=ev_dist, config=cfg)
        sp_fit = fit_single_peak_baseline(ev_dist, config=cfg)
        t_eval = np.linspace(60.0, 840.0, 40)
        m_full = evaluate_alignment_accuracy(
            res,
            t_eval,
            true_offset_s=args.offset,
            true_drift_ppm=args.drift_ppm,
            baseline_result=nat_res,
        )
        rows.append(
            {
                "subject": rec["subject_id"],
                "beatsync_med_ms": round(m_full["median_err_s"] * 1000.0, 2),
                "drift_err_ppm": round(m_full["drift_err_ppm"], 1),
                "top1_only_drift_ppm": round(sp_fit.drift_ppm, 1),
                "rescued_windows": res.n_windows_rescued,
            }
        )
    df_live = pd.DataFrame(rows)
    df_sum, df_hw = load_wesad_benchmark_tables()
    if args.json:
        print(
            json.dumps(
                {
                    "live_5subj_cohort": rows,
                    "wesad_75seg_summary": df_sum.to_dict(orient="records"),
                    "wesad_15subj_raw_hardware": df_hw.to_dict(orient="records"),
                },
                indent=2,
            )
        )
    else:
        print("=== Live 5-Subject WESAD 15-min Event Cohort ===")
        print(df_live.to_string(index=False))
        print("\n=== Archived 75-Segment Benchmark Summary ===")
        print(df_sum[["method_id", "median_error_s", "rate_lt_100ms", "rate_lt_50ms", "heldout_median_error_s"]].to_string(index=False))
    return 0


def _cmd_equiv(args: argparse.Namespace) -> int:
    """Run machine-precision numerical equivalence verification against SciPy."""
    cohort = load_wesad_event_cohort()
    report = verify_scipy_equivalence(events=cohort[1]["events"], seed=args.seed)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())
        print(
            f"Timing on 180s pulse train at 64 Hz: "
            f"FFT={report.xcorr_fft_ms:.2f} ms vs direct={report.xcorr_direct_ms:.2f} ms "
            f"({report.xcorr_speedup:.1f}x faster)"
        )
    return 0 if report.passed else 1


def _read_signal_column(path: Path, col: str) -> np.ndarray:
    """Read a numeric column or single-column array from CSV/TSV or NPY."""
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float64).ravel()
    df = pd.read_csv(path)
    if col in df.columns:
        return df[col].to_numpy(dtype=np.float64)
    if str(col).isdigit() and int(col) < df.shape[1]:
        return df.iloc[:, int(col)].to_numpy(dtype=np.float64)
    if df.shape[1] == 1:
        return df.iloc[:, 0].to_numpy(dtype=np.float64)
    raise ValueError(f"Column {col!r} not found in {path.name}; available: {list(df.columns)}")


def _cmd_sync(args: argparse.Namespace) -> int:
    """Synchronize two external ECG and PPG waveform CSV or NPY files."""
    ecg = _read_signal_column(args.ecg, args.ecg_col)
    ppg = _read_signal_column(args.ppg, args.ppg_col)
    cfg = SyncConfig(
        ptt_prior_s=args.ptt,
        max_coarse_lag_s=args.max_lag,
        bootstrap_draws=args.bootstrap,
    )
    res = synchronize_clocks(
        ecg=ecg,
        ppg=ppg,
        fs_ecg_hz=args.fs_ecg,
        fs_ppg_hz=args.fs_ppg,
        config=cfg,
        subject_id=args.ecg.stem,
    )
    payload = res.to_dict()
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(res.summary())
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for beatsync."""
    parser = argparse.ArgumentParser(
        prog="beatsync",
        description="Dual-scale cardiac event cross-correlation and clock drift synchronization.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run real WESAD S10 synchronization demonstration")
    p_demo.add_argument("--offset", type=float, default=-48.5, help="Injected startup offset in seconds")
    p_demo.add_argument("--drift-ppm", type=float, default=240.0, help="Injected clock drift in ppm")
    p_demo.add_argument("--ptt", type=float, default=0.25, help="Pulse transit time prior in seconds")
    p_demo.add_argument("--bootstrap", type=int, default=100, help="Bootstrap replicates for 95 percent CIs")
    p_demo.add_argument("--json", action="store_true", help="Emit structured JSON output")
    p_demo.set_defaults(func=_cmd_demo)

    p_bench = sub.add_parser("benchmark", help="Run 5-subject WESAD cohort and print benchmark summary")
    p_bench.add_argument("--offset", type=float, default=-36.0, help="Injected startup offset in seconds")
    p_bench.add_argument("--drift-ppm", type=float, default=210.0, help="Injected clock drift in ppm")
    p_bench.add_argument("--ptt", type=float, default=0.25, help="Pulse transit time prior in seconds")
    p_bench.add_argument("--json", action="store_true", help="Emit structured JSON output")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_eq = sub.add_parser("equiv", help="Verify numerical parity against SciPy reference routines")
    p_eq.add_argument("--seed", type=int, default=2026, help="Deterministic random seed")
    p_eq.add_argument("--json", action="store_true", help="Emit structured JSON output")
    p_eq.set_defaults(func=_cmd_equiv)

    p_sync = sub.add_parser("sync", help="Synchronize external ECG and PPG waveform files")
    p_sync.add_argument("--ecg", type=Path, required=True, help="Path to reference ECG CSV or NPY file")
    p_sync.add_argument("--ppg", type=Path, required=True, help="Path to target PPG CSV or NPY file")
    p_sync.add_argument("--fs-ecg", type=float, default=250.0, help="ECG sampling frequency in Hz")
    p_sync.add_argument("--fs-ppg", type=float, default=64.0, help="PPG sampling frequency in Hz")
    p_sync.add_argument("--ecg-col", type=str, default="ecg", help="ECG CSV column name or 0-based index")
    p_sync.add_argument("--ppg-col", type=str, default="ppg", help="PPG CSV column name or 0-based index")
    p_sync.add_argument("--ptt", type=float, default=0.25, help="Pulse transit time prior in seconds")
    p_sync.add_argument("--max-lag", type=float, default=1800.0, help="Max coarse lag search bound in seconds")
    p_sync.add_argument("--bootstrap", type=int, default=200, help="Bootstrap replicates for 95 percent CIs")
    p_sync.add_argument("--out-json", type=Path, default=None, help="Optional path to save JSON sync result")
    p_sync.add_argument("--json", action="store_true", help="Emit structured JSON output")
    p_sync.set_defaults(func=_cmd_sync)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``beatsync`` command-line interface."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
