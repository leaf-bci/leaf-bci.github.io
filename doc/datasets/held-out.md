# Held-out datasets

These motor-imagery datasets are excluded from `configs/tasks.yaml` and never appear in instruction tuning. They are reserved for zero-shot evaluation with the frozen instruction-tuned LEAF model.

## MI_Weibo2014

- **Role:** held-out binary motor-imagery evaluation with 10 subjects.
- **Preprocessing:** read exported `.npz` files, keep only event codes 1/2 for left/right hand, convert them to labels 0/1, band-pass filter at 0.3-40 Hz, resample to 200 Hz when necessary, and apply the shared pipeline.
- **Paths:** the builder uses hard-coded `/media/datasets/EEG_Dataset` and `/media/public/LEAF` paths rather than `configs/env.yaml`.
- **Evaluation loading:** `zero_shot_heldout.py` concatenates all subjects unless exclusions are supplied and treats `0=Left`, `1=Right` as verified.

## MI_Dreyer2023

- **Role:** held-out binary motor-imagery evaluation from an externally preprocessed file; no raw-data preprocessing is included.
- **Expected input:** `/media/public/LEAF/MI_Dreyer2023.h5` with nested `train/X,Y`, `val/X,Y`, and `test/X,Y` groups.
- **Expected shapes:** 10,000/2,152/2,160 trials with shape `(N, 65, 801)`; evaluation keeps the first 800 samples.
- **Label mapping:** the delivered file has no label legend, so evaluation assumes `0=Left`, `1=Right`.
- **Compatibility:** it is not compatible with the standard loader without changes to path, layout, and task registration.

## Zero-shot evaluation

`zero_shot_heldout.py` evaluates Dreyer2023 and Weibo2014 with frozen MPNet or Qwen3 LEAF checkpoints at instruction levels 0, 1, and 2. It uses only the Left and Right text prototypes and reports balanced accuracy, Cohen's kappa, weighted F1, AUROC, confusion counts, and label-mapping provenance.
