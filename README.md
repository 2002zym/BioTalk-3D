# BioTalk-3D

Official repository for **BioTalk-3D: A Synchronized EEG, Audio, and
Blendshape Dataset and Benchmark for 3D Facial Animation**.

BioTalk-3D is a multimodal dataset for physiology-aware speech-driven
3D facial animation. It provides temporally aligned EEG recordings,
speech audio, ARKit blendshape coefficients, and ground-truth facial
mesh renderings.

> **Important:** The released EEG recordings contain mixed scalp
> electrophysiological activity acquired during speech, including
> brain-related, ocular, facial-muscle, and articulation-related
> components. They should not be interpreted as isolated or
> artifact-free neural activity.

---

## News

- **2026-XX-XX:** BioTalk-3D v1.0.0 publicly released.
- **2026-XX-XX:** BioTalk-3D accepted by PRCV 2026.

---

## Dataset Overview

The cleaned BioTalk-3D release contains:

- 5 anonymized participants
- 20 recording sessions
- 2,300 utterance-level samples
- 13,120.016 seconds of cleaned recordings
- 3 h 38 min 40.016 s of total cleaned duration
- 64-channel scalp EEG recorded at 1 kHz
- Chinese and English speech recordings
- 51-dimensional ARKit blendshape coefficients
- Ground-truth facial mesh rendering videos
- Official training, validation, and test splits

The dataset supports research on:

- Speech-driven 3D facial animation
- Multimodal representation learning
- Physiology-aware facial motion reconstruction
- EEG–audio–facial motion alignment
- Audio-only and audio-plus-EEG benchmark evaluation
- Cross-subject, cross-session, and cross-language generalization

---

## Cleaned Dataset Statistics

The following statistics are calculated after manual and automatic
quality control.

| Subject | Samples | Duration (s) | Duration | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 468 | 2810.728 | 00:46:50.728 | 374 | 37 | 57 |
| S2 | 483 | 3045.618 | 00:50:45.618 | 386 | 40 | 57 |
| S3 | 460 | 2595.622 | 00:43:15.622 | 374 | 35 | 51 |
| S4 | 443 | 2358.328 | 00:39:18.328 | 350 | 36 | 57 |
| S5 | 446 | 2309.720 | 00:38:29.720 | 357 | 36 | 53 |
| **Total** | **2300** | **13120.016** | **03:38:40.016** | **1841** | **184** | **275** |

### Quality-control policy

The cleaned release follows the policy below:

- Samples with severe reading mistakes were removed.
- Samples with clearly truncated sentence endings were removed.
- Repeated readings were removed.
- Samples with unusual mouth or facial motion may be retained.
- Samples with mild audio–visual misalignment may be retained and
  marked for robustness or comparison studies.
- Samples with severe sensor failures or unusable tracking were removed.

Detailed quality-control records are provided in the accompanying CSV,
JSON, Markdown, and processing-log files.

---

## Public Release Scope

### Included data

The public release contains the following modalities and artifacts.

#### 1. Raw EEG acquisition recordings

Continuous EEG recordings exported from the original acquisition system,
for example:

```text
*.cnt
