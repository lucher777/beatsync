"""Bundled real WESAD benchmark excerpts and summary table loaders."""
from __future__ import annotations

from importlib.resources import files
import json
from typing import Any

import numpy as np
import pandas as pd

from .types import CardiacEvents


def _data_root():
    """Return the package resource traversable for bundled WESAD files."""
    return files("beatsync").joinpath("data")


def load_wesad_waveform_slice() -> dict[str, Any]:
    """Load the built-in 360-second real WESAD subject S10 dual-device slice.

    Returns
    -------
    dict[str, Any]
        Dictionary containing ``subject_id``, ``duration_s``, ``fs_ecg_hz``
        (250 Hz), ``fs_bvp_hz`` (64 Hz), ``fs_acc_hz`` (64 Hz), ``ecg``,
        ``bvp``, ``chest_acc``, and ``wrist_acc`` float64 arrays.
    """
    npz_res = _data_root().joinpath("wesad_s10_slice.npz")
    with npz_res.open("rb") as fh:
        raw = np.load(fh, allow_pickle=False)
        return {
            "subject_id": str(raw["subject_id"].item()),
            "duration_s": float(raw["duration_s"].item()),
            "fs_ecg_hz": float(raw["fs_ecg_hz"].item()),
            "fs_bvp_hz": float(raw["fs_bvp_hz"].item()),
            "fs_acc_hz": float(raw["fs_acc_hz"].item()),
            "ecg": np.asarray(raw["ecg_250hz"], dtype=np.float64),
            "bvp": np.asarray(raw["bvp_64hz"], dtype=np.float64),
            "chest_acc": np.asarray(raw["chest_acc_64hz"], dtype=np.float64),
            "wrist_acc": np.asarray(raw["wrist_acc_64hz"], dtype=np.float64),
        }


def load_wesad_event_cohort() -> list[dict[str, Any]]:
    """Load extracted 900-second ECG/PPG event trains for five WESAD subjects.

    Returns
    -------
    list[dict[str, Any]]
        Records for subjects ``S2``, ``S10``, ``S11``, ``S13``, and ``S16``,
        each containing ``events`` (``CardiacEvents``), ``heldout_anchor_offset_s``,
        ``heldout_anchor_slope``, ``start_s``, and ``duration_s``.
    """
    json_res = _data_root().joinpath("wesad_5subj_events.json")
    payload = json.loads(json_res.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for rec in payload["records"]:
        events = CardiacEvents(
            ecg_rpeaks_s=np.asarray(rec["ecg_rpeaks_s"], dtype=np.float64),
            ppg_peaks_s=np.asarray(rec["ppg_peaks_s"], dtype=np.float64),
            ref_duration_s=float(rec["duration_s"]),
            target_duration_s=float(rec["duration_s"]),
            subject_id=str(rec["subject_id"]),
            metadata={"source": "WESAD", "start_s": float(rec["start_s"])},
        )
        out.append(
            {
                "subject_id": str(rec["subject_id"]),
                "start_s": float(rec["start_s"]),
                "duration_s": float(rec["duration_s"]),
                "heldout_anchor_offset_s": float(rec["heldout_anchor_offset_s"]),
                "heldout_anchor_slope": float(rec["heldout_anchor_slope"]),
                "events": events,
            }
        )
    return out


def load_wesad_benchmark_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the aggregated 75-segment and 15-subject raw WESAD benchmark tables.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(summary_75seg_df, raw_hardware_15subj_df)``.
    """
    sum_res = _data_root().joinpath("wesad_75seg_summary.csv")
    hw_res = _data_root().joinpath("wesad_15subj_raw_hardware.csv")
    with sum_res.open("r", encoding="utf-8") as fh_sum:
        df_sum = pd.read_csv(fh_sum)
    with hw_res.open("r", encoding="utf-8") as fh_hw:
        df_hw = pd.read_csv(fh_hw)
    return df_sum, df_hw
