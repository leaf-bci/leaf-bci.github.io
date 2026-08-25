# Project overview

LEAF is a language-aligned EEG foundation model. It first learns EEG structure without labels, then aligns EEG representations with natural-language task descriptions and label prototypes. The aligned model can make predictions by comparing an EEG embedding with frozen text embeddings instead of using a separate classifier head for every task.

![LEAF architecture](../leaf-architecture.png)

## End-to-end workflow

1. **Prepare each downstream dataset.** Dataset-specific scripts read the original files, filter and resample the signals, divide recordings into trials, and write HDF5 files.
2. **Normalize the electrode space.** The shared preprocessing pipeline clips extreme values, applies robust scaling, maps channel names to the 65-channel LEAF montage, and interpolates missing template channels.
3. **Prepare the pretraining corpus.** Pretraining expects one or more `.npy` arrays containing trials shaped `(trials, 65, time)`.
4. **Pretrain the EEG Tower.** Spectral perturbation is applied to the input. A bidirectional masked branch and a causal next-window branch reconstruct the original waveform.
5. **Cache language embeddings.** All target labels and instruction strings are encoded by the selected text model and stored in `text_embeddings/`.
6. **Instruction-tune LEAF.** The pretrained Tower and Instruction-conditioned Q-Former are trained jointly. The output EEG embedding is classified against frozen text prototypes using cross-entropy.
7. **Evaluate or fine-tune.** Direct inference compares EEG and text embeddings without additional training. Dataset-specific fine-tuning adds a task-specific classification head and adapts the model to one selected dataset.

## Main entry points

| Stage | Entry point | Main input | Main output |
|---|---|---|---|
| Text cache | `init_text_embeddings.py` | `configs/tasks.yaml`, text encoder | `text_embeddings/<model>.txt/.npy` |
| Pretraining | `a_pretrain.py` | Pretraining `.npy` files | epoch-5/10 Tower checkpoints, `leaf-pretrain-loss.npy` |
| Instruction tuning | `b_instruct_tuning.py` | Downstream HDF5 files, text cache | periodic `itNNN.ckpt`, `loss_it.npy` |
| Direct evaluation | `c_inference.py` | tuned checkpoint, test splits | per-level CSV reports |
| Full single-dataset fine-tuning | `d_downstream.py` | tuned checkpoint, one dataset | per-epoch validation/test metrics in `.npz` |
| Held-out evaluation | `zero_shot_heldout.py` | Dreyer2023 and Weibo2014 HDF5 | combined zero-shot CSV |
| Embedding export | `export_test_embeddings.py` | tuned checkpoint, test splits | per-dataset `.npz` embeddings and summary CSV |

## Current dataset scope

Instruction tuning uses eight motor-imagery datasets, five emotion datasets, one SSVEP dataset, one covert-speech dataset, one workload dataset, and one ADHD dataset. See the [dataset index](datasets/README.md) for the exact labels and split rules.

## Important distinction

The downstream HDF5 files and the pretraining `.npy` corpus are separate products:

- `paths.downstream` contains labeled HDF5 datasets used by instruction tuning, inference, and downstream fine-tuning.
- `paths.pretrain` contains unlabeled `.npy` trial arrays used only by `a_pretrain.py`.

## Architecture

The default input convention is 65 channels at 200 Hz, with a maximum instruction-tuning length of 2,000 samples. The tokenizer divides EEG into non-overlapping 100-sample windows and applies a temporal convolution, a spatial convolution across all channels, synchronized batch normalization, GELU, pooling, and dropout.

The Spectral-Temporal Reconstruction Tower has two independent 12-layer Transformer branches sharing the tokenizer and positional embedding:

- a bidirectional branch masks 50% of window tokens and reconstructs the original waveform;
- a causal branch uses corrupted windows through position `i` to predict the clean waveform in the next window, `i+1`.

A random 6 Hz band with a lower edge sampled from 1-50 Hz is removed before both reconstruction losses. At inference, the two branch outputs are concatenated, producing 40 tokens for a 10-second trial or 16 tokens for a 4-second trial.

The Instruction-conditioned Q-Former uses eight learned 256-dimensional queries and four layers. The text instruction modulates Tower tokens through FiLM, after which queries cross-attend to the tokens. Flattened query outputs are projected into the selected text space and L2-normalized. Classification uses cosine-similarity logits against frozen normalized label embeddings.

### Default model dimensions

| Parameter | Value |
|---|---:|
| Sampling rate / channels | 200 Hz / 65 |
| Maximum length | 2,000 samples |
| Window length | 100 samples |
| CNN width / token dimension | 64 / 256 |
| Tower layers per branch | 12 |
| Attention heads / FF expansion | 8 / 4x |
| Transformer dropout | 0.1 |
| Q-Former queries / layers | 8 / 4 |

## Configuration files

| File | Purpose |
|---|---|
| `configs/LEAF_mpnet.yaml` | Model plus pretraining, instruction-tuning, and downstream defaults using MPNet |
| `configs/LEAF_qwen3-4b.yaml` | Same defaults using Qwen3-4B text embeddings |
| `configs/env.yaml` | Raw, pretraining, and downstream data roots plus text-encoder registry |
| `configs/tasks.yaml` | Ordered labels and instruction pools |
| `configs/LEAF-ch65/` | Electrode aliases, montage coordinates, and reference PDF |

The model YAMLs currently differ only in `text_emb_model_name`. `--dim` can override architecture using `window-cnn-token-tower_layers[-queries-qformer_layers]`; the default full value is `100-64-256-12-8-4`. `--emb` overrides the configured text encoder.

Label order in `configs/tasks.yaml` must exactly match integer labels stored in HDF5. Changing any label or instruction requires updating the corresponding text cache.

Default configuration/task/environment paths are anchored to `load_config.py`, but CLI defaults and `ckpt/` outputs are generally relative to the working directory. Run commands from the repository root.
