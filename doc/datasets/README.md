# Dataset Specification

The table below reflects `configs/tasks.yaml` and `load_datasets.py`, which together define the current instruction-tuning and standard-evaluation suite.

## Registered datasets

| Family | Dataset key | Classes in stored-label order | Subjects/split |
|---|---|---|---|
| Motor imagery | `MI_OpenBMI` | Right, Left | 42 train/validation subjects, 12 test subjects |
| Motor imagery | `MI_BCIC_IV2a` | Left, Right, Foot, Tongue | 7 train/validation, 2 test |
| Motor imagery | `MI_ShanghaiU` | Right, Left | 20 train/validation, 5 test |
| Motor imagery | `MI_BCIC_Upperlimb` | Cylin, Sphe, Lumbrical | 11 train/validation, 4 test |
| Motor imagery | `MI_HighGamma` | Left, Right | 10 train/validation, 4 test; feet removed at load time |
| Motor imagery | `MI_Cho2017` | Left, Right | 40 train/validation, 9 test |
| Motor imagery | `MI_Shin2017A` | Left, Right | 22 train/validation, 6 test |
| Motor imagery | `MI_PhysioNet` | Left, Right | 80 train/validation, 29 test |
| Emotion | `EMO_FACED` | 9 emotions | 100 train/validation, 23 test |
| Emotion | `EMO_SEED_3_seg4` | Negative, Neutral, Positive | predefined trial split |
| Emotion | `EMO_SEED_4_seg4` | Neutral, Sad, Fear, Happy | predefined trial split |
| Emotion | `EMO_SEED_5_seg4` | Disgust, Fear, Sad, Neutral, Happy | predefined trial split |
| Emotion | `EMO_SEED_7_seg4` | Happy, Surprise, Neutral, Sad, Disgust, Fear, Anger | predefined trial split |
| SSVEP | `SSVEP_OpenBMI` | 12.0, 8.6, 6.6, 5.4 Hz | 42 train/validation, 12 test |
| Covert speech | `CS_BCIC_Speech` | hello, help-me, stop, thank-you, yes | first 250/next 50/remaining trials per subject |
| Workload | `Workload` | Resting, Workload | 32 train/validation, 4 test |
| Healthcare | `ADHD_AliMotie` | Healthy, ADHD | predefined subject split |

For subject-based train/validation groups, `load_dataset` concatenates the selected subjects and sends every fifth sample to validation. Training and validation therefore share subjects, while the test subjects are held out.

## Held-out datasets not registered for instruction tuning

- `MI_Weibo2014.py`: held-out binary motor-imagery evaluation.
- `MI_Dreyer2023.py`: validator/loader for an externally preprocessed held-out file.

See [held-out zero-shot datasets](held-out.md) for details.

## File naming requirements

`load_datasets.py` resolves exactly `paths.downstream/<dataset-key>.h5`. Several builders currently produce a different filename:

| Builder output | Loader expects |
|---|---|
| `EMO_FACED_seg10.h5` | `EMO_FACED.h5` |
| `EMO_SEED_3_seg4_v2.h5` | `EMO_SEED_3_seg4.h5` |
| `EMO_SEED_4_seg4_v2.h5` | `EMO_SEED_4_seg4.h5` |
| `EMO_SEED_5_seg4_v2.h5` | `EMO_SEED_5_seg4.h5` |
| `EMO_SEED_7_seg4_v2.h5` | `EMO_SEED_7_seg4.h5` |

Rename or copy these outputs to the expected names before training. This is a description of the current code, not an additional preprocessing step from the method.

## Detailed preprocessing

- [Instruction tuning datasets](instruction-tuning-datasets.md)
- [Held-out datasets](held-out.md)

## Common data format and preprocessing

Signals use `(trials, channels, time)`, are normally stored as `float32`, resampled to 200 Hz, and mapped to the 65-channel LEAF montage. Most decoding trials are 4 seconds/800 samples; FACED and ADHD use 10 seconds/2,000 samples.

After dataset-specific filtering and epoching, `leaf_datasets/shared.py::pipeline`:

1. clips values to the global 0.1/99.9 percentiles of the provided array;
2. applies global robust scaling `(X - median) / (q75 - q25)`;
3. maps channel aliases to the 65-channel template;
4. averages multiple source channels mapped to one template electrode;
5. interpolates missing template electrodes with MNE and warns when more than half are missing.

The statistics are calculated over the full array passed to `pipeline`, not per trial or channel. Subject-keyed builders normally call it once per subject; predefined-split builders call it separately per subject and split.

General EEG builders use a 0.1-70 Hz band, while motor-imagery builders use 0.3-40 Hz. Notch filtering is dataset-specific. The shared pipeline itself does not filter or resample.

### HDF5 layouts

Most datasets are subject-keyed:

```text
<subject>/X    (trials, 65, time)
<subject>/Y    (trials,)
```

SEED variants and ADHD use flat predefined arrays: `trainX/trainY`, `validX/validY`, and `testX/testY`. Subject-index splitting depends on top-level HDF5 key order.

Pretraining is separate: `a_pretrain.py` scans `paths.pretrain/*.npy`, where each file must be shaped `(trials, 65, time)`.

During pretraining and instruction tuning, samples are normalized to 2,000 time points. Training uses random zero-padding placement or random cropping; validation uses right padding or the leading crop. Direct inference and downstream tuning use native trial lengths.
