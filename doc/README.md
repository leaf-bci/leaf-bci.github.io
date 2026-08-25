# LEAF documentation

This directory documents the public LEAF code release as it is implemented in the repository. Configuration files and source code remain the source of truth; paths, datasets, and hyperparameters may change after the documentation is written.

## Start here

- [Project overview, architecture, and configuration](overview.md)
- [Dataset Specification](datasets/README.md)

## Training and evaluation

- [EEG pretraining](pretraining.md)
- [Instruction tuning and text embeddings](instruction-tuning.md)
- [Direct inference](direct-inference.md)
- [Dataset-specific fine-tuning with a classification head](single-dataset-finetuning.md)

## Dataset preprocessing

- [Instruction tuning datasets](datasets/instruction-tuning-datasets.md)
- [Held-out datasets](datasets/held-out.md)

## Scope

The current `configs/tasks.yaml` registers 17 datasets for instruction tuning and standard evaluation. SSVEP, covert speech, workload, and ADHD are part of this suite. Weibo2014 and Dreyer2023 are excluded from instruction tuning and reserved for held-out zero-shot evaluation.
