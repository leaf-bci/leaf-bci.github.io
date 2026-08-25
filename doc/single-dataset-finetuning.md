# Dataset-specific fine-tuning with a classification head

`d_downstream.py` adapts an instruction-tuned LEAF checkpoint to one selected dataset. It adds a `FlattenClassifier` over Tower tokens and trains both the Tower and classifier end to end.

## Model behavior

The classifier uses concatenated Tower tokens, not the aligned Q-Former embedding. Although an instruction embedding is passed through `LEAF.forward`, the returned Q-Former embedding is ignored by the classifier; the classification logits are therefore independent of the selected instruction text in the current implementation.

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

The YAML field `downstream.check_val_every_n_epoch=5` is not used by this manual loop. Validation and test metrics are computed after every epoch. No early stopping or best-checkpoint selection is implemented.

## Metrics

Each epoch records validation and test balanced accuracy, AUROC, weighted F1, and Cohen's kappa. Multiclass AUROC uses macro one-vs-rest averaging.

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

The implementation is CUDA-oriented: the training loop requests CUDA autocast explicitly even when `--gpu -1` is supplied.
