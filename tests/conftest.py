"""Shared pytest fixtures for beatsync unit, parity, and ablation tests."""
from __future__ import annotations

import pytest

from beatsync import (
    DualDeviceSimulator,
    SyncConfig,
    load_wesad_event_cohort,
    load_wesad_waveform_slice,
)


@pytest.fixture(scope="session")
def default_config() -> SyncConfig:
    """Return a fast deterministic configuration for test assertions."""
    return SyncConfig(ptt_prior_s=0.250, bootstrap_draws=40, random_seed=2026)


@pytest.fixture(scope="session")
def wesad_s10_slice() -> dict:
    """Load the bundled real WESAD S10 dual-device waveform excerpt once."""
    return load_wesad_waveform_slice()


@pytest.fixture(scope="session")
def wesad_cohort() -> list[dict]:
    """Load the bundled 5-subject real WESAD 15-minute event cohort."""
    return load_wesad_event_cohort()


@pytest.fixture(scope="session")
def simulator() -> DualDeviceSimulator:
    """Return a deterministic dual-device cardiac simulator."""
    return DualDeviceSimulator(seed=2026)
