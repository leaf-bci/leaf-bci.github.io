# EEG pretraining

`a_pretrain.py` trains the two-branch EEG Tower with no labels.

## Input corpus

The pretraining root is `paths.pretrain` in `configs/env.yaml`. The loader sorts all `*.npy` files and treats every first-axis item as a trial:

```text
file.npy            shape: (number_of_trials, 65, time)
file.npy[i]         shape: (65, time)
```

Trials may have variable time length across files. The dataset wrapper converts them to 2,000 samples:

- training: random left/right zero padding or random crop;
- validation: right zero padding or leading crop.

## Objective

For every batch:

1. Remove one randomly selected 6 Hz frequency band whose lower edge is sampled from 1-50 Hz.
2. Tokenize the corrupted EEG into non-overlapping 100-sample windows.
3. Mask 50% of tokens in the contextual branch and reconstruct the original waveform with MSE.
4. In the causal branch, let state `i` attend to corrupted windows `0..i` and predict the clean signal in window `i+1`. The next-window MSE is computed over the resulting `N-1` predictions.
5. Optimize `masked_reconstruction_loss + next_window_reconstruction_loss`.

The causal branch uses a consistent positional-indexing convention during training and inference.

## Default hyperparameters

| Item | Value |
|---|---:|
| Batch size | 512 |
| Epochs | 10 (`--epochs`) |
| Base learning rate | `1e-6` |
| Tokenizer learning rate | `1e-6` |
| Remaining Tower learning rate | `1e-7` |
| Warmup | 1 epoch |
| Schedule | linear warmup, cosine multiplier 1.0 to 0.1 |
| Optimizer | AdamW |
| AdamW weight decay | framework default (`0.01` in current PyTorch) |
| Precision | `bf16-mixed` |
| Seed | 42 |
| Data workers | 16 |
| Training shuffle | yes |
| `drop_last` | yes for train and validation |

No gradient clipping or gradient accumulation is configured. Lightning checkpointing and logging are disabled.

## Multi-GPU behavior

Pass a comma-separated GPU list through `--gpu`. Multiple GPUs use DDP with `find_unused_parameters=False`. The per-process step count is approximated as `len(trainLoader) // number_of_gpus` for the learning-rate schedule.

## Command

```bash
python a_pretrain.py \
  --config configs/LEAF_mpnet.yaml \
  --gpu 0 \
  --bs 512 \
  --epochs 10 \
  --seed 42
```

## Outputs

Pretraining writes selected per-epoch Tower state dictionaries and one validation-loss array:

```text
checkpoints/
  leaf-pretrain-epoch-05.ckpt
  leaf-pretrain-epoch-10.ckpt
  leaf-pretrain-loss.npy
```

The checkpoints are raw Tower state dictionaries. `leaf-pretrain-loss.npy` stores one `[masked_reconstruction_loss, next_window_reconstruction_loss]` pair per completed validation epoch.
