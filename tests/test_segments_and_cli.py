"""Tests for timestamp-jump segmentation, motion baseline, error guards, and CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from beatsync import (
    DualDeviceSimulator,
    SyncConfig,
    detect_timestamp_jumps,
    fit_motion_xcorr_baseline,
    synchronize_clocks,
    synchronize_segmented_streams,
)
from beatsync.cli import main as cli_main


def test_detect_timestamp_jumps_splits_gaps_and_backward_resets() -> None:
    """Forward packet drops and backward clock resets split timestamp streams cleanly."""
    seg1 = np.arange(0.0, 20.0, 0.1)
    seg2 = np.arange(35.0, 55.0, 0.1)  # 15s forward gap
    seg3 = np.arange(10.0, 30.0, 0.1)  # backward clock reset
    ts = np.concatenate([seg1, seg2, seg3])
    slices = detect_timestamp_jumps(ts, gap_threshold_s=2.0)
    assert len(slices) == 3
    assert slices[0] == (0, seg1.size)
    assert slices[1] == (seg1.size, seg1.size + seg2.size)
    assert slices[2] == (seg1.size + seg2.size, ts.size)


def test_synchronize_segmented_streams_tracks_jump_offsets(
    simulator: DualDeviceSimulator,
    default_config: SyncConfig,
) -> None:
    """Piecewise synchronization tracks independent segment offsets across clock resets."""
    sim_seg = simulator.simulate_segmented(
        segment_duration_s=210.0,
        offset_seg1_s=12.0,
        offset_seg2_s=28.5,
        drift_ppm=180.0,
        gap_s=45.0,
    )
    seg_res = synchronize_segmented_streams(
        sim_seg["t_ecg_s"],
        sim_seg["ecg"],
        sim_seg["t_ppg_s"],
        sim_seg["ppg"],
        config=default_config,
        gap_threshold_s=5.0,
    )
    assert len(seg_res.segments) == 2
    assert seg_res.n_ref_jumps == 1
    assert seg_res.n_target_jumps == 1
    df = seg_res.to_dataframe()
    assert len(df) == 2

    # Verify piecewise timestamp mapping accuracy in both segments
    mapped_1 = float(seg_res.map_piecewise(50.0))
    expected_1 = (50.0 + 12.0) / (1.0 - 180e-6)
    assert abs(mapped_1 - expected_1) < 0.08

    t2_tgt = sim_seg["tgt_shift_s"] + 50.0
    mapped_2 = float(seg_res.map_piecewise(t2_tgt))
    expected_2 = (t2_tgt + 28.5) / (1.0 - 180e-6)
    assert abs(mapped_2 - expected_2) < 0.08


def test_motion_xcorr_baseline_on_stationary_wesad_slice(wesad_s10_slice: dict) -> None:
    """Motion envelope cross-correlation runs cleanly on real WESAD accelerometer streams."""
    fit = fit_motion_xcorr_baseline(
        wesad_s10_slice["chest_acc"],
        wesad_s10_slice["wrist_acc"],
        config=SyncConfig(bootstrap_draws=0),
    )
    assert fit.method == "motion_theil_sen"
    assert np.isfinite(fit.intercept_s)


def test_seed_reproducibility_of_bootstrap_intervals(wesad_cohort: list[dict]) -> None:
    """Identical random seeds yield bit-identical drift fits and bootstrap bounds."""
    ev = wesad_cohort[0]["events"]
    cfg1 = SyncConfig(bootstrap_draws=80, random_seed=99)
    cfg2 = SyncConfig(bootstrap_draws=80, random_seed=99)
    res1 = synchronize_clocks(events=ev, config=cfg1)
    res2 = synchronize_clocks(events=ev, config=cfg2)
    assert res1.to_dict() == res2.to_dict()
    assert len(res1.window_dataframe()) == res1.n_windows_total


def test_configuration_and_input_validation_errors() -> None:
    """Invalid SyncConfig parameters or missing inputs raise explicit ValueErrors."""
    with pytest.raises(ValueError, match="coarse_sigma_s"):
        SyncConfig(coarse_sigma_s=0.02, fine_sigma_s=0.05)
    with pytest.raises(ValueError, match="quality_quantile"):
        SyncConfig(quality_quantile=0.95)
    with pytest.raises(ValueError, match="Provide either"):
        synchronize_clocks()


def test_cli_help_and_demo_equiv_json_outputs(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI demo and equiv commands execute end-to-end and emit valid JSON."""
    assert cli_main(["demo", "--offset", "-20.0", "--drift-ppm", "150", "--bootstrap", "20", "--json"]) == 0
    out_demo = json.loads(capsys.readouterr().out)
    assert out_demo["subject_id"] == "S10"
    assert out_demo["accuracy_metrics"]["median_err_s"] < 0.03

    assert cli_main(["equiv", "--json"]) == 0
    out_equiv = json.loads(capsys.readouterr().out)
    assert out_equiv["passed"] is True


def test_cli_benchmark_and_sync_csv_roundtrip(
    tmp_path: Path,
    simulator: DualDeviceSimulator,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI benchmark and sync subcommands process external CSV waveforms cleanly."""
    assert cli_main(["benchmark", "--json"]) == 0
    bench_payload = json.loads(capsys.readouterr().out)
    assert len(bench_payload["live_5subj_cohort"]) == 5
    assert len(bench_payload["wesad_75seg_summary"]) == 8

    sim = simulator.simulate(duration_s=150.0, offset_s=14.0, drift_ppm=-120.0)
    ecg_csv = tmp_path / "ecg.csv"
    ppg_csv = tmp_path / "ppg.csv"
    out_json = tmp_path / "sync_out.json"
    pd.DataFrame({"ecg": sim["ecg"]}).to_csv(ecg_csv, index=False)
    pd.DataFrame({"ppg": sim["ppg"]}).to_csv(ppg_csv, index=False)

    rc = cli_main(
        [
            "sync",
            "--ecg",
            str(ecg_csv),
            "--ppg",
            str(ppg_csv),
            "--fs-ecg",
            str(sim["fs_ecg_hz"]),
            "--fs-ppg",
            str(sim["fs_ppg_hz"]),
            "--bootstrap",
            "20",
            "--out-json",
            str(out_json),
            "--json",
        ]
    )
    assert rc == 0
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert abs(saved["drift_fit"]["intercept_s"] - 14.0) < 0.05


def test_cli_subprocess_help_exits_zero() -> None:
    """Running python -m beatsync.cli --help exits cleanly without formatting crashes."""
    pkg_src = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "beatsync.cli", "--help"],
        env={**os.environ, "PYTHONPATH": pkg_src},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "beatsync" in proc.stdout
