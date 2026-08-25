# Instruction tuning

`b_instruct_tuning.py` aligns EEG with instructions and label text embeddings across every dataset in `configs/tasks.yaml`.

## Why align EEG with language?

Different BCI datasets describe different prediction problems and use different label spaces. A conventional classifier normally assigns a separate output head to each dataset. LEAF instead represents task instructions, candidate labels, and EEG trials in a shared semantic space.

This design provides a common interface across heterogeneous tasks. The instruction tells the model what kind of information should be extracted from an EEG trial, while the label embeddings define the possible semantic outputs. After alignment, prediction can be performed by comparing an EEG embedding directly with text-label prototypes.

Pretraining and instruction tuning are therefore complementary: the pretrained Tower supplies signal structure, and instruction tuning organizes that structure according to task and label meaning.

## Inputs

- A pretrained Tower checkpoint, unless `--no_warmup` is passed.
- All 17 processed downstream HDF5 files.
- A cached text-embedding pair for the configured model.
- Dataset labels and instruction pools from `configs/tasks.yaml`.

Training uses the training split, validation uses the validation split, and test data remains separate from optimization.

## Text prototypes

All unique label strings across all datasets are sorted alphabetically and assigned a global class index. Their cached normalized embeddings are frozen and stacked into one prototype matrix. Cross-entropy therefore operates over the global unique-label space, not only the current dataset's labels.

The text encoder is used before training to compute embeddings for every instruction and label. Caching these embeddings avoids repeatedly running the text model and makes the semantic targets consistent throughout instruction tuning. Freezing the prototype matrix gives the EEG model a stable language space to align with.

A shared label string corresponds to one shared prototype even when it appears in multiple datasets. This encourages compatible semantic concepts to occupy the same location rather than creating a separate class representation for every dataset.

## Sample construction

Every trial is padded/cropped to 2,000 samples. Training uses random padding/cropping; validation uses deterministic right padding/leading crop.

For each sample:

1. With 10% probability, select a `Default` instruction.
2. Otherwise select a random instruction from the dataset's task family.
3. Independently, with 90% probability, compute the normalized mean of the dataset's candidate-label embeddings, average it with the instruction embedding, and normalize again.
4. Convert the stored integer label to its ordered label string in `configs/tasks.yaml`, then map that string to the global prototype index.

Sampling different instruction phrasings exposes the model to multiple natural-language descriptions of the same task and reduces dependence on one fixed prompt. The occasional `Default` instruction also teaches the model to operate with a generic task description.

The mean of all candidate-label embeddings summarizes the answer space of a dataset without identifying the correct label for the individual trial. Blending this summary with the instruction gives the conditioning vector both task context and label-space context. Applying the blend probabilistically keeps the model effective with either an instruction alone or the richer combined description.

## How an EEG trial becomes a semantic embedding

The forward path can be summarized as:

```text
EEG trial ──> pretrained two-branch Tower ──> EEG tokens ──> FiLM + Q-Former ──> normalized EEG embedding
                                                    ^                                  |
                                                    |                                  v
                                           instruction embedding          cosine similarity
                                                                                       |
label text ──> frozen text encoder ──> normalized label prototypes <───────────────────┘
```

The process has four steps:

1. **Encode the EEG.** The pretrained Tower processes the clean trial with its bidirectional and causal branches, then concatenates their token sequences.
2. **Condition the tokens.** A linear layer maps the instruction embedding to FiLM scale and shift parameters. These parameters modulate every Tower token, allowing the same EEG features to be viewed differently for motor imagery, emotion, speech, SSVEP, workload, or healthcare tasks.
3. **Summarize task-relevant information.** Eight learned Q-Former queries attend to the instruction-conditioned EEG tokens through four layers of self-attention, cross-attention, and feed-forward processing. The queries provide a fixed-size semantic summary even when the number of EEG windows varies.
4. **Project into text space.** The final query states are flattened, projected to the dimension of the selected text encoder, and L2-normalized. The result can then be compared directly with normalized label embeddings.

FiLM provides early task conditioning at the token level, while the Q-Former acts as a compact information bottleneck that selects and aggregates EEG evidence relevant to the instruction.

## Optimization

The pretrained Tower is not frozen. The full LEAF model is trained end to end.

| Parameter group | Components | Default LR |
|---|---|---:|
| Fast | Tower tokenizer | `1e-3` |
| Slow | all other Tower and Q-Former parameters | `2e-4` |

The loss is cross-entropy over cosine-similarity logits between normalized EEG embeddings and frozen normalized text prototypes. Because both sides are normalized, their dot product measures angular similarity rather than vector magnitude. Cross-entropy increases similarity to the correct label prototype while decreasing relative similarity to competing prototypes.

The Tower remains trainable so that its EEG features can adapt to the multi-dataset semantic objective. A larger learning rate is assigned to the tokenizer, while the pretrained Transformer branches and Q-Former use a smaller rate. This lets the signal front end adapt efficiently to the combined downstream corpus while updating the higher-level representation more gradually.

At the end of instruction tuning, LEAF has learned a mapping conditioned on both the EEG trial and the instruction. Direct inference can then replace a dataset-specific classifier with a set of label text prototypes and select the label with the highest similarity.

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

## Practical considerations

- **Match the text cache to the model configuration.** `text_emb_model_name` determines both the cached embedding file and its dimensionality. A checkpoint trained with MPNet should be used with the corresponding MPNet configuration and text cache; the same rule applies to other text encoders.
- **Keep label order synchronized.** Integer labels stored in each HDF5 file are interpreted according to the ordered label list in `configs/tasks.yaml`. Update and regenerate the text cache whenever labels or instruction strings change.
- **Use the correct checkpoint type.** EEG pretraining produces a Tower-only checkpoint for `--pretrain_ckpt`, whereas instruction tuning produces a full LEAF checkpoint for direct inference and downstream evaluation.
- **Prefer a pretrained initialization for the standard workflow.** `--no_warmup` is useful for controlled comparisons, while the released training recipe initializes the Tower from EEG pretraining.
- **Interpret batch size per device.** With multiple entries in `--gpu`, `--bs` applies to each GPU. Keep the effective batch size in mind when changing the number of devices or comparing experiments.
- **Preserve the input convention.** Instruction tuning expects the same 65-channel montage and 200 Hz sampling rate used during pretraining. Trials are standardized to the configured maximum length before entering the model.
- **Record the semantic setup.** For a reproducible run, keep the task YAML, text-embedding model, pretrained checkpoint, seed, and instruction-tuning configuration together.

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

Periodic `itNNN.ckpt` files contain full LEAF state dictionaries. The save interval is configured through `instruct.save_ckpt_every_n_epoch`. `loss_it.npy` stores validation `[cross_entropy_loss, accuracy]` pairs.

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
