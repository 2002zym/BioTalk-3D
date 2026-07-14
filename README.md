# BioTalk-3D

Official repository for **BioTalk-3D: A Synchronized EEG, Audio, and Blendshape Dataset and Benchmark for 3D Facial Animation**.

BioTalk-3D is a multimodal dataset for physiology-aware speech-driven
3D facial animation. Its public release contains preprocessed session-level
scalp electrophysiological recordings, segmented utterance-level
electrophysiological arrays, speech audio, ARKit blendshape coefficients,
and ground-truth mesh-rendering videos.

> **Signal interpretation.** The released scalp electrophysiological recordings were acquired during speech and may contain mixed brain-related, ocular, facial-muscle, movement, and articulation-related components. They should not be interpreted as isolated or artifact-free neural activity.

## News

- **[RELEASE_DATE]:** BioTalk-3D v1.0.0 released.

## Dataset Overview

The cleaned BioTalk-3D release contains:

- 5 anonymized participants
- 20 recording sessions
- 2,300 utterance-level samples
- 13,120.016 seconds of cleaned recordings
- 3 h 38 min 40.016 s of total cleaned duration
- 64-channel scalp electrophysiological recordings sampled at 1 kHz
- Chinese and English speech recordings
- 51-dimensional ARKit blendshape coefficients
- Ground-truth facial mesh-rendering videos
- Official training, validation, and test splits

BioTalk-3D supports research on:

- Speech-driven 3D facial animation
- Multimodal sequence modeling
- Physiology-aware facial motion reconstruction
- Audio–physiology–facial-motion alignment
- Audio-only and audio-plus-physiology benchmark evaluation
- Cross-subject, cross-session, and cross-language generalization

## Cleaned Dataset Statistics

| Public subject ID | Samples | Duration (s) | Duration | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 468 | 2810.728 | 00:46:50.728 | 374 | 37 | 57 |
| S2 | 483 | 3045.618 | 00:50:45.618 | 386 | 40 | 57 |
| S3 | 460 | 2595.622 | 00:43:15.622 | 374 | 35 | 51 |
| S4 | 443 | 2358.328 | 00:39:18.328 | 350 | 36 | 57 |
| S5 | 446 | 2309.720 | 00:38:29.720 | 357 | 36 | 53 |
| **Total** | **2300** | **13120.016** | **03:38:40.016** | **1841** | **184** | **275** |

## Quality-Control Policy

The cleaned release follows this policy:

- Samples with severe reading mistakes were removed.
- Samples with clearly truncated sentence endings were removed.
- Repeated readings were removed.
- Samples with mild mouth-shape or facial-motion abnormalities may be retained for robustness studies.
- Samples with mild audio–visual misalignment may be retained and marked for later comparison studies.
- Samples with severe sensor failures or unusable facial tracking were removed.

Detailed processing and quality-control records are included in the release package.

## Public Release Scope

> **Release boundary.** Original raw EEG acquisition files (`*.cnt`) are
> not publicly released. The publicly released EEG data consist of
> preprocessed session-level files (`*.set` and paired `.fdt` files when
> present) and segmented utterance-level arrays (`*_eeg.npy`).

### Included

The public release includes the following data and artifacts.


#### 1. Preprocessed session-level EEG

Preprocessed continuous EEG session files:

- `*.set`
- `*.fdt`, when present

For EEGLAB datasets that use external binary storage, the `.set` file and its paired `.fdt` file must remain in the same directory.

The preprocessing pipeline includes:

- 1 Hz high-pass filtering
- 50 Hz notch filtering
- Robust channel-wise normalization where applicable
- Temporal alignment and segmentation preparation

Speech-related ocular, myogenic, movement, and articulation-related components are not necessarily removed.

#### 2. Segmented utterance-level EEG

Utterance-aligned EEG arrays:

- `*_eeg.npy`

Each segmented EEG sample is temporally aligned with the corresponding speech waveform and ARKit blendshape sequence.

#### 3. Segmented speech audio

Utterance-level audio files:

- `*.wav`

#### 4. Ground-truth ARKit blendshape coefficients

Utterance-level ground-truth blendshape arrays:

- `*_bs.npy`

Each array stores the temporally aligned ARKit blendshape sequence for one utterance.

#### 5. Ground-truth rendering videos

Ground-truth mesh-rendering videos are provided under:

- `render_demo/`

These videos are generated from the released ground-truth blendshape coefficients and paired speech audio.

They are:

- not model predictions;
- not original participant face videos;
- not generated from EEG.

#### 6. Metadata, splits, and processing records

The release also includes:

- anonymized participant identifiers;
- session identifiers;
- language labels;
- sample durations;
- quality-control annotations;
- training, validation, and test split files;
- processing and alignment summaries.

### Excluded

The public release does not include:

- raw EEG acquisition files (`*.cnt`);
- original RGB or TrueDepth participant face videos;
- participant names or identity mappings;
- consent forms;
- contact information;
- directly identifying metadata;
- unnecessary absolute recording dates;
- device serial numbers;
- private local paths or user-account information.

## Download

### BioTalk-3D v1.0.0

- **Baidu Netdisk:** [Download the full dataset](https://pan.baidu.com/s/17m1xekiNR08e_CIgu-0Tbg?pwd=em25)
- **Extraction code:** `em25`
- **Version:** v1.0.0
- **Release date:** 2026-07-15
- **Approximate size:** 18.05 GB

The full dataset is hosted on Baidu Netdisk because of its size.

## Repository File Map

| File | Description |
|---|---|
| `README.md` | Main project page and release summary |
| `README_en.md` | English release documentation |
| `README_zh.md` | Chinese release documentation |
| `DATA_CARD_en.md` | English dataset card |
| `DATA_CARD_zh.md` | Chinese dataset card |
| `ETHICS_en.md` | English ethics, consent, and privacy statement |
| `ETHICS_zh.md` | Chinese ethics, consent, and privacy statement |
| `RENDERING_en.md` | English ground-truth rendering protocol |
| `RENDERING_zh.md` | Chinese ground-truth rendering protocol |
| `DATA_LICENSE.md` | Dataset terms of use |
| `CITATION.cff` | Machine-readable citation metadata |
| `CHANGELOG.md` | Dataset version history |


## Subject-Level Data

Each subject directory contains:

```text
S1/
├── preprocessed_eeg/
│   ├── *.set
│   └── *.fdt
├── BioTalk/
│   ├── audio/
│   │   └── *.wav
│   ├── blendshape/
│   │   └── *_bs.npy
│   ├── eeg/
│   │   └── *_eeg.npy
│   └── splits/
│       ├── train.json
│       ├── val.json
│       └── test.json
└── render_demo/
    └── *.mp4
```

The actual organization may differ slightly between releases. The release documentation should be treated as authoritative.

## File Naming and Modality Pairing

Files belonging to the same utterance should share the same sample identifier.

Example:

```text
S1_session01_sentence0001.wav
S1_session01_sentence0001_bs.npy
S1_session01_sentence0001_eeg.npy
```

| File pattern | Description |
|---|---|
| `*.wav` | Segmented speech waveform |
| `*_bs.npy` | Ground-truth ARKit blendshape sequence |
| `*_eeg.npy` | Temporally aligned segmented EEG |
| `*.set` | Preprocessed EEGLAB session file |
| `*.fdt` | External binary data paired with a `.set` file |
| `*.mp4` | Ground-truth mesh-rendering video |

For every `.set` file that references an external `.fdt` file, both files must be kept together.

## Ground-Truth Rendering Protocol

Files under `render_demo/` are ground-truth facial mesh renderings.

Each video is generated from:

- a segmented speech waveform;
- the corresponding ground-truth `*_bs.npy` sequence;
- a neutral template mesh;
- triangle faces;
- a canonical blendshape basis;
- fixed camera, lighting, frame-rate, and rendering settings.

The rendering procedure follows the general ground-truth rendering protocol used by the official UniTalker evaluation pipeline.

### Important Clarification

- `render_demo` contains ground-truth renderings.
- `render_demo` does not contain model predictions.
- EEG is not used to generate these videos.
- EEG is used only as an input modality for model training and evaluation.
- The released videos are mesh renderings, not original participant face videos.

See `RENDERING_en.md` and `RENDERING_zh.md` for the complete rendering specification.

## Benchmark Protocols

### Intra-subject evaluation

Training, validation, and testing use disjoint utterances from the same participant.

### Cross-subject evaluation

Models are trained on a subset of participants and evaluated on held-out participants.

### Cross-session evaluation

Models are trained and evaluated using disjoint recording sessions.

### Cross-language evaluation

Models are trained on one language subset and evaluated on another language subset.

Official split files are included in the release package.

## Processing and Quality-Control Records

The release package may include the following records:

| File | Description |
|---|---|
| `LC_processing_v2.md` | Baseline processing notes for the initial subject pipeline |
| `fixed6_adaptive_strategy.md` | Fixed-6 adaptive tail-padding strategy |
| `secondary_processed_list_all.csv` | Samples modified during the secondary padding stage |
| `secondary_processed_list_all.md` | Human-readable secondary-processing summary |
| `fixed6_adaptive_processed_list_all.csv` | Final sample-level effective-padding records |
| `fixed6_adaptive_summary.json` | Summary of the fixed-6 adaptive run |
| `auto_pad_batch_summary.json` | Per-subject summary of the initial automatic padding pass |
| `processing_validation_summary.json` | Cross-subject alignment-validation summary |
| `manual_filter_drop_list_all.csv` | Quality-issue list before deletion |
| `manual_filter_drop_list_all_summary.md` | Summary of candidate exclusions |
| `manual_filter_delete_applied.csv` | Deletion operations actually applied |
| `manual_filter_delete_applied_summary.md` | Summary of retained and deleted samples |
| `fixed6_adaptive_run.log` | Runtime log, when available |

These files are provided for transparency and reproducibility. They are not required for loading the core training samples.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/2002zym/BioTalk-3D.git
cd BioTalk-3D
```

Install basic dependencies:

```bash
pip install numpy scipy soundfile mne
```

After downloading the full dataset, verify the release:

```bash
python scripts/verify_dataset.py \
  --dataset-root /path/to/BioTalk-3D_v1.0.0
```

Minimal utterance-level loading example:

```python
from pathlib import Path

import numpy as np
import soundfile as sf

sample_root = Path("/path/to/sample_directory")
sample_id = "S1_session01_sentence0001"

audio, sample_rate = sf.read(sample_root / f"{sample_id}.wav")
blendshape = np.load(sample_root / f"{sample_id}_bs.npy")
eeg = np.load(sample_root / f"{sample_id}_eeg.npy")

print("Audio shape:", audio.shape)
print("Audio sample rate:", sample_rate)
print("Blendshape shape:", blendshape.shape)
print("EEG shape:", eeg.shape)
```

Consult the data-format documentation before assuming array-axis order, units, or sampling rates.

## Intended Uses

BioTalk-3D is intended for academic research and education in areas including:

- speech-driven 3D facial animation;
- digital humans and avatars;
- multimodal sequence modeling;
- scalp electrophysiological signal processing;
- audio–physiology fusion;
- facial motion reconstruction;
- cross-subject multimodal generalization;
- multimodal synchronization and alignment.

## Prohibited Uses

Users must not:

- attempt to identify or re-identify any participant;
- use the dataset for biometric identification;
- use the dataset for surveillance;
- use the dataset for medical diagnosis or clinical decision-making;
- redistribute the full dataset without authorization;
- remove attribution, ethics, or license notices;
- present the recordings as purely neural or artifact-free signals;
- use the data in ways that harm participants;
- use the data in violation of applicable law or institutional ethics requirements.

See `DATA_LICENSE.md` for the complete terms.

## Ethics, Consent, and Privacy

All participants provided written informed consent before data collection. The recording protocol was reviewed and approved by the relevant institutional ethics committee.

The public release excludes original participant face videos and directly identifying personal information.

Researchers should recognize that:

- speech recordings may contain person-specific vocal characteristics;
- scalp electrophysiological recordings may contain person-specific physiological patterns;
- multimodal data may carry residual re-identification risks;
- released data should be stored and used responsibly;
- re-identification attempts are strictly prohibited.

Ethics approval information:

- **Approval number:** Available upon reasonable request

See `ETHICS_en.md` and `ETHICS_zh.md` for additional details.

## Dataset Versioning

Current public version:

```text
BioTalk-3D v1.0.0
```

All updates should be documented in `CHANGELOG.md`.

When reporting results, researchers should state the dataset version used in their experiments.

## Citation

If you use BioTalk-3D, please cite:

```bibtex
@inproceedings{biotalk3d2026,
  title     = {BioTalk-3D: A Synchronized EEG, Audio, and Blendshape Dataset and Benchmark for 3D Facial Animation},
  author    = {Yiming Zhao, Weichen Dai, Shenzhou Chen, Li Zhu, Yu xuan Huang, Ruixun Yu, Chen Liang, Jin Peng, Wensheng Liu, Beilin Li, Wanzeng Kong},
  booktitle = {Proceedings of the Chinese Conference on Pattern Recognition and Computer Vision},
  year      = {2026}
}
```

The citation will be updated after the final proceedings and DOI become available.

## License

The repository code and the dataset may use different terms:

- Repository code: see `LICENSE`
- BioTalk-3D dataset: see `DATA_LICENSE.md`

Downloading or using the dataset indicates acceptance of the applicable dataset terms.

## Acknowledgements

The ground-truth mesh-rendering procedure follows the general protocol used by UniTalker.

Please cite relevant upstream projects when using their code, mesh assets, rendering resources, or evaluation implementations.

## Contact

For questions, bug reports, access problems, or broken download links, please open an issue in this repository.

- **Project page:** https://github.com/2002zym/BioTalk-3D
- **Maintainer:** Alan 
- **Institution:** Hangzhou Dianzi university
- **Email:** weichendai@hdu.edu.cn
