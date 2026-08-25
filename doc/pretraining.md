# Self-Supervised Pretraining

`a_pretrain.py` trains the two-branch EEG Tower without task labels. The resulting Tower initializes the signal encoder used during instruction tuning.

## Input and tokenization

The pretraining root is `paths.pretrain` in `configs/env.yaml`. The loader sorts all `*.npy` files and treats every first-axis item as a trial:

```text
file.npy            shape: (number_of_trials, 65, time)
file.npy[i]         shape: (65, time)
```

Trials may have variable time length across files. The dataset wrapper converts them to 2,000 samples:

- training: random left/right zero padding or random crop;
- validation: right zero padding or leading crop.

At 200 Hz, 2,000 samples correspond to 10 seconds. The tokenizer divides each trial into non-overlapping 100-sample windows. A temporal convolution operates within each window, a spatial convolution integrates the 65 channels, and each Tower branch projects the pooled features to 256-dimensional tokens. The default input therefore produces 20 tokens before positional embeddings are added.

## Spectral perturbation

Before tokenization, LEAF transforms the EEG into the frequency domain and sets to zero one randomly selected 6 Hz band whose lower edge is sampled from 1-50 Hz. Both reconstruction branches receive the perturbed signal and predict the original clean waveform. This discourages reliance on a single narrow frequency band.

## Reconstruction objectives

The two branches share the tokenizer and positional embeddings but use independent projection layers and Transformers.

### Bidirectional masked reconstruction

The contextual branch replaces 50% of its input tokens with a learned mask token. A bidirectional Transformer processes the sequence and decodes it to the complete clean waveform. Mean-squared error is computed against the original trial.

### Causal next-window reconstruction

The temporal branch uses a causal attention mask. At position `i`, it can attend only to windows `0..i`, and its representation predicts the clean waveform in window `i+1`. Mean-squared error is computed over the resulting `N-1` predictions.

### Combined representation

The pretraining loss is the unweighted sum of masked reconstruction and next-window reconstruction. During instruction tuning and inference, the reconstruction heads are omitted; clean EEG is encoded by both Transformer branches and their outputs are concatenated. A default 20-window trial produces 40 Tower tokens for the Q-Former.

## Default hyperparameters

| Item | Value |
|---|---:|
| Batch size | 512 |
| Epochs | 10 (`--epochs`) |
| Tokenizer learning rate | `1e-6` |
| Remaining Tower learning rate | `1e-7` |
| Warmup | 1 epoch |
| Schedule | linear warmup, cosine multiplier 1.0 to 0.1 |
| Optimizer | AdamW |
| AdamW weight decay | framework default (`0.01` in current PyTorch) |
| Precision | `bf16-mixed` |
| Seed | 42 |

## Practical considerations

- Input arrays should follow the 65-channel LEAF montage at 200 Hz, with each trial stored as `(65, time)`.
- `--bs` is the batch size per GPU. When comparing runs with different GPU counts, account for the resulting effective batch size.
- Pretraining checkpoints contain only the Tower state and are used to initialize `b_instruct_tuning.py`; direct inference uses a full instruction-tuned LEAF checkpoint.

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
