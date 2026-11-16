# Contributing to beatsync

Thank you for your interest in contributing to `beatsync`. Contributions covering bug fixes, sensor-modality presets, cycle-slip rescue diagnostics, documentation, and unit tests are welcome.

## Development Setup

```bash
git clone https://github.com/peacekange/beatsync.git
cd beatsync
pip install -e .[dev]
```

## Running Tests and Numerical Parity Verification

Before submitting a pull request, ensure that the full unit test suite, CLI smoke tests, and SciPy numerical equivalence checks pass locally:

```bash
pytest -v
python examples/quickstart.py
beatsync --help
beatsync equiv
```

## Code Standards

1. Keep runtime dependencies lightweight (`numpy`, `scipy`, `pandas`).
2. Document all public classes and functions using NumPy-style English docstrings and PEP 484 type annotations (`from __future__ import annotations`).
3. Verify that `verify_scipy_equivalence()` passes within machine tolerance (`< 1e-12` for FFT cross-correlation and zero-phase Butterworth filtering; `< 1e-14` for Theil-Sen drift estimation).
4. Add unit tests under `tests/` covering normal waveforms, cycle-slip stress cases, timestamp jump boundaries, and invalid arguments.

## Reporting Issues

Please open an issue on the project repository including a minimal reproducible snippet, `beatsync` version, Python version, and operating system.

## Community Conduct

Contributors are expected to engage constructively and respectfully across issues, discussions, and code reviews.
