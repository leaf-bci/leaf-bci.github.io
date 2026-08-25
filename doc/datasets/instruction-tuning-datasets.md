# Instruction tuning datasets

This chapter covers all 17 datasets registered in `configs/tasks.yaml` and used during instruction tuning. They span motor imagery, emotion recognition, SSVEP, covert speech, workload, and ADHD.

## Motor imagery

All registered motor-imagery datasets are converted to 200 Hz, mapped to the 65-channel montage, and stored as subject-keyed HDF5 files. Unless stated otherwise, training/validation is an 80/20 sample split within the training subjects and testing uses held-out subjects.

### MI_OpenBMI

- **Source:** MATLAB files under `MI_OpenBMI/DB_mat/session1|session2/<subject>/EEG_MI.mat`.
- **Subjects:** 54 (`s01`-`s54`); first 42 train/validation, last 12 test.
- **Labels:** stored `0=Right`, `1=Left` according to `configs/tasks.yaml`; source labels are reduced by one.
- **Trials:** concatenate train and test arrays from both acquisition sessions.
- **Signal processing:** construct epochs at 1,000 Hz, resample to 200 Hz, band-pass 0.3-40 Hz, then apply the shared clipping/scaling/montage pipeline.
- **Output:** `MI_OpenBMI.h5`, grouped by subject.

### MI_BCIC_IV2a

- **Source:** each subject has evaluation and training MATLAB files, `<subject>E.mat` and `<subject>T.mat`, under `MI_BCIC_IV2a_mat/`.
- **Subjects:** 9 (`A01`-`A09`); first 7 train/validation, last 2 test.
- **Labels:** Left, Right, Foot, Tongue; MATLAB labels are reduced by one.
- **Trial extraction:** skip the first three MATLAB data blocks. Use event boundaries to extract 22 EEG channels, subtract the across-channel mean at each time point, and initially retain 0-6 seconds at 250 Hz.
- **Signal processing:** band-pass 0.3-40 Hz, resample to 200 Hz, retain seconds 2-6 (800 samples), then apply the shared pipeline.
- **Output:** `MI_BCIC_IV2a.h5`.

### MI_BCIC_Upperlimb

- **Source:** `Training set/sampleXX.mat` and `Validation set/sampleXX.mat`.
- **Subjects:** 15; first 11 train/validation, last 4 test.
- **Labels:** Cylin, Sphe, Lumbrical from the one-hot source labels.
- **Trial extraction:** concatenate the published training and validation arrays. The code removes the final time sample before creating MNE epochs.
- **Signal processing:** resample from the file's `fs` value to 200 Hz; band-pass 0.1-70 Hz; retain the final motor-imagery stage beginning at second 6, which is documented in code as 4 seconds; apply the shared pipeline.
- **Output:** `MI_BCIC_Upperlimb.h5`.

### MI_ShanghaiU

- **Source:** five files `d1.mat`-`d5.mat` per subject.
- **Subjects:** 25 (`S01`-`S25`); first 20 train/validation, last 5 test.
- **Labels:** `configs/tasks.yaml` interprets `0=Right`, `1=Left` after subtracting one from source labels. The unused `TEXT_LABELS` constant in the builder lists the opposite order, so the task configuration is the effective mapping used during training.
- **Trial extraction:** concatenate all five sessions.
- **Signal processing:** create 250 Hz epochs, drop `A1` and `A2`, resample to 200 Hz, band-pass 0.3-40 Hz, and apply the shared pipeline.
- **Output:** `MI_ShanghaiU.h5`.

### MI_HighGamma

- **Source:** one exported `.npz` per subject containing `x_data`, `y_data`, sampling rate, and metadata.
- **Subjects:** 14; first 10 train/validation, last 4 test.
- **Builder labels:** keep source classes 1-3 and convert to `0=Left`, `1=Right`, `2=Feet`; the fourth source class is removed.
- **Training labels:** `load_dataset` removes label 2 again, so the registered task is binary Left/Right.
- **Signal processing:** drop mastoid channels `M1` and `M2`, resample to 200 Hz, band-pass 0.3-40 Hz, and apply the shared pipeline.
- **Output:** `MI_HighGamma.h5`.

### MI_Cho2017

- **Source:** exported `.npz` files for 49 available subjects; `s32`, `s46`, and `s49` are excluded.
- **Split:** first 40 subjects train/validation, final 9 test.
- **Labels:** Left and Right; labels are used without an offset.
- **Signal processing:** divide source values by 100, resample to 200 Hz, band-pass 0.3-40 Hz, append 200 samples by edge padding, and apply the shared pipeline.
- **Output:** `MI_Cho2017.h5`.

### MI_Shin2017A

- **Source:** `cnt.mat` and `mrk.mat` from the `with occular artifact` directory for each subject.
- **Subjects:** 28; `S05` is unavailable. The first 22 available subjects are train/validation and the last 6 are test.
- **Sessions:** process source blocks with indices 0, 2, and 4.
- **Signal processing:** load at 200 Hz; remove `VEOG` and `HEOG`; band-pass 0.3-40 Hz; notch at 50 Hz; epoch 0-4 seconds; then discard the first sample so the inclusive MNE epoch becomes 800 samples. Apply the shared pipeline.
- **Labels:** marker codes are reduced by one to Left/Right.
- **Output:** `MI_Shin2017A.h5`.

### MI_PhysioNet

- **Source:** EDF runs `R03` through `R14` for 109 PhysioNet EEG Motor Movement/Imagery subjects.
- **Split:** first 80 subjects train/validation, last 29 test.
- **Signal processing:** interpolate channels marked bad by the EDF reader, resample to 200 Hz, notch at 60 Hz, band-pass 0.3-40 Hz, strip trailing periods from channel names, and epoch annotations from 0 to 4 seconds. The code removes the trailing inclusive sample to produce 800 samples, converts signals to microvolts, and applies the shared pipeline.
- **Label filtering:** keep annotation IDs 2 and 3 and convert them to labels 0 and 1.
- **Implementation note:** although a `tasks = ['04', '08', '12']` variable says “motor imagery only,” it is not used. The current loop processes every run from 3 to 14, including motor execution and bilateral-hands/feet protocols whose T1/T2 semantics differ. Reproducible use should follow the current loop or fix it deliberately and record the change.
- **Output:** `MI_PhysioNet.h5`.

### Shared split behavior

For all eight registered datasets above, `load_dataset` concatenates the declared training subjects and assigns sample indices `0, 5, 10, ...` to validation. All other samples become training data. Subject order is obtained from the top-level HDF5 key order.

## Emotion

The registered emotion suite contains FACED and four SEED variants. FACED uses 10-second windows and a held-out-subject test split. SEED datasets use 4-second windows and predefined trial splits.

### EMO_FACED

- **Source:** `Processed_data/subXXX.pkl` for subjects `000`-`122`.
- **Channels:** the two reference channels `A1` and `A2` are dropped from the 32-channel source list.
- **Signal processing:** create 250 Hz epochs, band-pass 0.1-70 Hz, resample to 200 Hz, and divide each recording into non-overlapping 10-second windows. Incomplete final windows are discarded.
- **Labels:** the 28 source trials are assigned to Anger, Fear, Disgust, Sad, Amusement, Inspiration, Joy, Tenderness, and Neutral using the fixed label array in the builder. Each trial label is repeated for all of its windows.
- **Normalization:** apply the shared pipeline per subject after segmentation.
- **Split:** subjects 000-099 form the train/validation pool; subjects 100-122 are test. Within the first group, every fifth concatenated sample is validation.
- **Builder output:** `EMO_FACED_seg10.h5`; the loader expects `EMO_FACED.h5`, so the file must currently be renamed or copied.

### EMO_SEED_3_seg4

- **Source:** three MATLAB sessions for each of 15 subjects under `EMO_SEED_3/eeg_raw_data/`.
- **Channels/rate:** 62 channels at 200 Hz.
- **Trial construction:** for each of 15 trial IDs, concatenate that trial from all three sessions along time before filtering and segmentation.
- **Signal processing:** band-pass 0.1-70 Hz, notch at 50 Hz with width 2 Hz, read in microvolts, and divide into non-overlapping 4-second windows.
- **Split by trial ID:** trials 1-9 train, 10-12 validation, 13-15 test.
- **Normalization:** shared pipeline is run separately for each subject and split. After subjects are concatenated, values are clipped again to `[-10, 10]`.
- **Builder output:** `EMO_SEED_3_seg4_v2.h5`; loader expects `EMO_SEED_3_seg4.h5`.
- **Labels:** the builder converts raw `-1, 0, +1` to stored `0, 1, 2`, giving Negative, Neutral, Positive. This order now matches both `EMO_SEED_3.py::TEXT_LABELS` and `configs/tasks.yaml`.

### EMO_SEED_4_seg4

- **Source:** three session directories, each containing one MATLAB file per each of 15 subjects.
- **Channels/rate:** 62 channels at 200 Hz.
- **Signal processing:** band-pass 0.1-70 Hz, notch at 50 Hz with width 2 Hz, and split each trial into non-overlapping 4-second windows.
- **Split within each session:** trials 1-16 train, 17-20 validation, 21-24 test.
- **Labels:** `0=Neutral`, `1=Sad`, `2=Fear`, `3=Happy`, matching `configs/tasks.yaml`.
- **Normalization:** shared pipeline per subject and split; final arrays clipped to `[-10, 10]`.
- **Builder output:** `EMO_SEED_4_seg4_v2.h5`; loader expects `EMO_SEED_4_seg4.h5`.

### EMO_SEED_5_seg4

- **Source:** three CNT sessions for each of 16 subjects under `EMO_SEED_5/EEG_raw/`.
- **Channels:** drop `VEO`, `HEO`, `M1`, and `M2`, leaving the declared 62 channels.
- **Trial extraction:** fixed start/end timestamps define 15 video trials in each session.
- **Signal processing:** resample to 200 Hz, band-pass 0.1-70 Hz, notch at 50 Hz with width 2 Hz, read microvolts, and split into non-overlapping 4-second windows.
- **Split within each session:** trials 1-5 train, 6-10 validation, 11-15 test.
- **Labels:** `0=Disgust`, `1=Fear`, `2=Sad`, `3=Neutral`, `4=Happy`.
- **Normalization:** shared pipeline per subject and split; final arrays clipped to `[-10, 10]`.
- **Builder output:** `EMO_SEED_5_seg4_v2.h5`; loader expects `EMO_SEED_5_seg4.h5`.

### EMO_SEED_7_seg4

- **Source:** four CNT sessions for each of 20 subjects.
- **Channels:** drop `M1`, `M2`, `ECG`, `HEO`, and `VEO`, leaving 62 channels.
- **Events:** use annotation pairs as trial start/end positions. Two known recordings (`14_20221015_1.cnt` and `9_20221111_3.cnt`) reconstruct event times from CSV trigger logs.
- **Signal processing:** resample to 200 Hz, band-pass 0.1-70 Hz, notch at 50 Hz with width 2 Hz, read microvolts, and split each of 20 trials per session into non-overlapping 4-second windows.
- **Split within each session:** trials 1-10 train, 11-15 validation, 16-20 test.
- **Labels:** `0=Happy`, `1=Surprise`, `2=Neutral`, `3=Sad`, `4=Disgust`, `5=Fear`, `6=Anger` from the fixed video order.
- **Normalization:** shared pipeline per subject and split; final arrays clipped to `[-10, 10]`.
- **Builder output:** `EMO_SEED_7_seg4_v2.h5`; loader expects `EMO_SEED_7_seg4.h5`.

### Predefined HDF5 split

After file naming is aligned, all four SEED task files are loaded directly from `trainX/trainY`, `validX/validY`, and `testX/testY`. `load_dataset` does not re-split or reshuffle them.

## SSVEP, covert speech, workload, and ADHD

These four datasets are also registered in `configs/tasks.yaml` and included in the standard instruction-tuning loop.

### SSVEP_OpenBMI

- **Source:** the OpenBMI MATLAB structure under `MI_OpenBMI/DB_mat/session1|session2/<subject>/EEG_SSVEP.mat`.
- **Subjects:** 54; first 42 train/validation, last 12 test.
- **Trials:** concatenate the published train/test arrays from both sessions.
- **Labels:** source labels are reduced by one and interpreted as `0=12.0`, `1=8.6`, `2=6.6`, `3=5.4` Hz.
- **Signal processing:** create epochs at 1,000 Hz, resample to 200 Hz, band-pass 0.3-40 Hz, and apply the shared pipeline. The general 0.1-70 Hz filter is commented out; the current implementation uses the motor-imagery band.
- **Output:** `SSVEP_OpenBMI.h5`.
- **Split:** within the first 42 subjects, every fifth concatenated sample is validation; the last 12 subjects are test.

### CS_BCIC_Speech

- **Source:** `Training set/Data_SampleXX.mat` and `Validation set/Data_SampleXX.mat` for 15 subjects.
- **Labels:** hello, help-me, stop, thank-you, yes, obtained by `argmax` over source one-hot labels.
- **Trial construction:** transpose both source arrays to `(trial, channel, time)`, concatenate them, and append five time samples using edge padding.
- **Signal processing:** the builder does not explicitly filter or resample. It assumes the published epochs already have the desired rate and length, then applies only the shared clipping/scaling/montage pipeline.
- **Output:** `CS_BCIC_Speech.h5`, grouped by subject.
- **Split:** for every subject, trials 0-249 are training, 250-299 validation, and 300 onward test; the subject and trial axes are then flattened.

### Workload

- **Source:** two EDF files per subject, `SubjectXX_1.edf` and `SubjectXX_2.edf`, for 36 subjects.
- **Labels:** recording 1 becomes `0=Resting`; recording 2 becomes `1=Workload`.
- **Channels:** remove `ECG ECG` and `EEG A2-A1`; remove the first four characters from all remaining source channel names.
- **Signal processing:** resample to 200 Hz, band-pass 0.1-70 Hz, divide each continuous recording into non-overlapping 4-second windows, concatenate both conditions, and apply the shared pipeline.
- **Output:** `Workload.h5`.
- **Split:** subjects 00-31 train/validation, subjects 32-35 test; validation is every fifth sample within the first group.

### ADHD_AliMotie

- **Source:** MATLAB recordings under separate `ADHD_Diagnosed/` and `ADHD_Control/` directories.
- **Subjects:** 61 diagnosed and 60 controls.
- **Labels:** `0=Healthy/Control`, `1=ADHD/Diagnosed`.
- **Signal processing:** divide source values by 1,000, transpose to channel-first format, create a 128 Hz continuous recording, resample to 200 Hz, band-pass 0.1-70 Hz, divide into non-overlapping 10-second windows, and apply the shared pipeline per subject.
- **Split by subject list position:** first 35 ADHD plus first 35 controls train; next 5 plus next 5 validate; remaining 21 ADHD plus 20 controls test.
- **Output:** `ADHD_AliMotie.h5` with flat `trainX/trainY`, `validX/validY`, and `testX/testY` datasets.
