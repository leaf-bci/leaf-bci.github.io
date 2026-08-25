# Dataset-specific fine-tuning with a classification head

`d_downstream.py` adapts an instruction-tuned LEAF checkpoint to one selected dataset by adding a task-specific classification head and optimizing it together with the EEG model.

## Default hyperparameters

| Item | Value |
|---|---:|
| Batch size | 128 |
| Epochs | 200 |
| Base learning rate | `1e-3` |
| Fast LR scale | 1.0 |
| Slow LR scale | 0.1 |
| Fast parameters | Tower tokenizer and classifier |
| Slow parameters | remaining model parameters |
| AdamW weight decay | `1e-3` |
| Warmup | 1 epoch |
| Schedule | linear warmup, cosine multiplier 1.0 to 0.1 |
| Precision in loop | CUDA bfloat16 autocast |
| Default seed | 42 |

## Metrics

Evaluation metrics include balanced accuracy, AUROC, weighted F1, and Cohen's kappa. Multiclass AUROC uses macro one-vs-rest averaging.

## Command

```bash
python d_downstream.py \
  --config configs/LEAF_mpnet.yaml \
  --gpu 0 \
  --ds MI_BCIC_IV2a \
  --itEpo 100 \
  --instrct None \
  --seed 42
```

Results are saved to:

```text
ckpt/<architecture>/<tuning-run>/<itEpo>-<seed>/<dataset>.npz
```
