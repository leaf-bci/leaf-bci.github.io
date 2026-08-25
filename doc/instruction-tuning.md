# Instruction tuning

`b_instruct_tuning.py` aligns EEG with instructions and label text embeddings across every dataset in `configs/tasks.yaml`.

## Inputs

- A pretrained Tower checkpoint, unless `--no_warmup` is passed.
- All 17 processed downstream HDF5 files.
- A cached text-embedding pair for the configured model.
- Dataset labels and instruction pools from `configs/tasks.yaml`.

Test splits are loaded only to be discarded during tuning. Training uses the training split, and validation uses the validation split.

## Text prototypes

All unique label strings across all datasets are sorted alphabetically and assigned a global class index. Their cached normalized embeddings are frozen and stacked into one prototype matrix. Cross-entropy therefore operates over the global unique-label space, not only the current dataset's labels.

## Sample construction

Every trial is padded/cropped to 2,000 samples. Training uses random padding/cropping; validation uses deterministic right padding/leading crop.

For each sample:

1. With 10% probability, select a `Default` instruction.
2. Otherwise select a random instruction from the dataset's task family.
3. Independently, with 90% probability, compute the normalized mean of the dataset's candidate-label embeddings, average it with the instruction embedding, and normalize again.
4. Convert the stored integer label to its ordered label string in `configs/tasks.yaml`, then map that string to the global prototype index.

## Optimization

The pretrained Tower is not frozen. The full LEAF model is trained end to end.

| Parameter group | Components | Default LR |
|---|---|---:|
| Fast | Tower tokenizer | `1e-3` |
| Slow | all other Tower and Q-Former parameters | `2e-4` |

The loss is cross-entropy over cosine-similarity logits between normalized EEG embeddings and frozen normalized text prototypes.

## Default hyperparameters

| Item | Value |
|---|---:|
| Batch size | 256 |
| Epochs | 100 |
| Base learning rate | `1e-3` |
| Fast LR scale | 1.0 |
| Slow LR scale | 0.2 |
| AdamW weight decay | `1e-3` |
| Warmup | 2 epochs |
| Schedule | linear warmup, cosine multiplier 1.0 to 0.1 |
| Precision | `bf16-mixed` |
| Validation interval | every 10 epochs |
| Periodic checkpoint interval | every 10 epochs |
| Default seed | 42 |
| Data workers | 4 |
| Training shuffle | yes |

No gradient clipping or gradient accumulation is configured. Multiple GPUs use DDP with `find_unused_parameters=True`.

## Commands

Warm-start from the pretrained Tower:

```bash
python b_instruct_tuning.py \
  --config configs/LEAF_mpnet.yaml \
  --gpu 0,1,2,3 \
  --bs 256 \
  --seed 42 \
  --pretrain_ckpt checkpoints/leaf-pretrain-epoch-10.ckpt
```

Train the full model without the pretrained Tower:

```bash
python b_instruct_tuning.py --no_warmup --gpu 0 --seed 42
```

Use a different registered text encoder with `--emb`, or select `configs/LEAF_qwen3-4b.yaml`.

## Checkpoint paths

Run directories encode the config name, instruction-tuning seed, and selected pretraining checkpoint. For example, MPNet with seed 0 initialized from epoch 5 writes:

```text
checkpoints/LEAF_mpnet-0-leaf-pretrain-epoch-05/
  it010.ckpt
  it020.ckpt
  ...
  it100.ckpt
  loss_it.npy
```

Periodic `itNNN.ckpt` files contain full LEAF state dictionaries. The save interval comes from `instruct.save_ckpt_every_n_epoch`; setting it to `null` or `none` disables checkpoint saving. The current script does not write a separate final checkpoint. `loss_it.npy` stores validation `[cross_entropy_loss, accuracy]` pairs. If this loss file already exists, tuning exits to avoid overwriting the run.

## Text embeddings

`init_text_embeddings.py` caches every unique label and instruction from `configs/tasks.yaml`. Dataset-script constants such as `TEXT_LABELS` are not used by the current cache; the strings in the task YAML are authoritative.

| Short name | Dimension | Source |
|---|---:|---|
| `bert-base` | 768 | `bert-base-uncased` CLS embedding |
| `mpnet-base` | 768 | `all-mpnet-base-v2` |
| `qwen3-4b` | 2560 | `Qwen/Qwen3-Embedding-4B` |

Use the short name when generating a cache:

```bash
python init_text_embeddings.py mpnet-base
python init_text_embeddings.py qwen3-4b
```

Each cache is a paired text file and NumPy array:

```text
text_embeddings/<model>.txt
text_embeddings/<model>.npy    # (number_of_strings, embedding_dimension)
```

Rows correspond by position. Embeddings are stored as normalized float32 vectors, and tuning checks unit norm before building prototypes. Existing cache entries retain their positions; only unseen strings are appended. Delete both cache files and regenerate when strict YAML order/identity is required.
