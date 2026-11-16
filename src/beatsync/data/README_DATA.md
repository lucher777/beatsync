# Built-In Real Wearable Reference Data

This directory bundles compact, de-identified benchmark excerpts derived from the public **WESAD (Wearable Stress and Affect Detection)** multimodal dataset (Schmidt et al., 2018, *Proceedings of the 20th ACM International Conference on Multimodal Interaction*, `doi:10.1145/3242969.3242985`):

- `wesad_s10_slice.npz`: Contiguous 360-second synchronized recording excerpt from subject `S10` (`ecg_250hz` chest ECG resampled from 700 Hz to 250 Hz, `bvp_64hz` wrist photoplethysmography at 64 Hz, `chest_acc_64hz` and `wrist_acc_64hz` triaxial motion-difference envelopes at 64 Hz). Used for self-contained waveform demonstrations, SciPy parity verification, and offset/drift injection tests.
- `wesad_5subj_events.json`: Extracted 900-second (15-minute) chest ECG R-peak and wrist PPG systolic peak event trains for five subjects (`S2`, `S10`, `S11`, `S13`, `S16`), accompanied by held-out session anchor offsets estimated outside each 900-second window.
- `wesad_75seg_summary.csv`: Aggregated comparison and ablation metrics across 75 real 15-minute WESAD recording segments under wide injected clock offsets (`[-90, +90] s`) and crystal drift (`[-300, +300] ppm`).
- `wesad_15subj_raw_hardware.csv`: Zero-injection blind alignment summary across all 15 unsynchronized raw dual-device WESAD sessions (`S2`–`S17`) benchmarked against physical chest double-tap ground truth.
