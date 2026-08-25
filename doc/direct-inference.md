# Direct inference

Direct inference evaluates an instruction-tuned LEAF checkpoint without training a dataset-specific classifier head. Predictions are made by comparing the normalized EEG embedding with frozen text-label prototypes.

## Standard direct inference

`c_inference.py` evaluates every dataset registered in `configs/tasks.yaml`.

### Instruction levels

| Level | Instruction embedding |
|---:|---|
| 0 | random Default string; no label blending |
| 1 | random task-family instruction; no label blending |
| 2 | task-family instruction averaged with the normalized mean of candidate-label embeddings, then normalized |

The random instruction choice is seeded by `--seed`. Each dataset is evaluated against only its own ordered label prototypes.

### Metrics and outputs

The standard script reports pooled-test balanced accuracy and Cohen's kappa. It creates a `direct_inference/` directory beside the checkpoint supplied through `--ckpt` and writes one CSV per level. For the public v1.0 checkpoint:

```text
checkpoints/direct_inference/leaf-v1.0-instruct-mpnet-base_level0.csv
checkpoints/direct_inference/leaf-v1.0-instruct-mpnet-base_level1.csv
checkpoints/direct_inference/leaf-v1.0-instruct-mpnet-base_level2.csv
```

Each CSV includes dataset, balanced accuracy, kappa, sample count, and an unweighted mean across datasets.

### Command

```bash
python c_inference.py \
  --config configs/LEAF_mpnet.yaml \
  --gpu 0 \
  --bs 256 \
  --ckpt checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt \
  --level 0,1,2 \
  --seed 42
```

Use `--ckpt` to select the checkpoint, `--gpu` to select the evaluation device, and `--bs` to control the evaluation batch size.

## Global embedding export

`export_test_embeddings.py` exports normalized EEG embeddings, true labels, predicted labels, label names, instruction text, checkpoint path, and instruction level for each test dataset. It writes one `.npz` per dataset plus `summary.csv` under:

```text
ckpt/<architecture>/<tuning-run>/global_inference/level<level>/
```

It supports a comma-separated dataset subset and resumable output. For inputs longer than 1,000 samples, such as FACED, it caps effective batch size at 64 to reduce CUDA activation pressure.

## Held-out zero-shot inference

`zero_shot_heldout.py` evaluates the frozen model on `MI_Dreyer2023` and `MI_Weibo2014`, neither of which appears in instruction tuning. It uses only the Left and Right prototypes and reports balanced accuracy, kappa, weighted F1, AUROC, and confusion counts. See [held-out datasets](datasets/held-out.md).
