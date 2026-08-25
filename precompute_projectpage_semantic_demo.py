"""Precompute the ProjectPage semantic-guidance interaction.

This script is the only component that reads EEG, text embeddings, or a LEAF
checkpoint.  It exports a compact JSON file containing fixed two-dimensional
coordinates for the same trials under instruction levels L0/L1/L2.  By default,
the complete official test splits are exported.  The browser only renders those
coordinates.

The three levels use the exact inference recipe in ``c_inference.py``:

* L0: the cached MPNet embedding of the literal string ``None``;
* L1: a task-family instruction embedding;
* L2: the normalized mean of the task embedding and the normalized mean of the
  candidate-label embeddings.

Run from any directory, for example::

    python ProjectPage/precompute_projectpage_semantic_demo.py --gpu 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.metrics import balanced_accuracy_score


PROJECT_PAGE = Path(__file__).resolve().parent
WORKSPACE = PROJECT_PAGE.parent
LEAF_ROOT = WORKSPACE / "LEAF"

# Local imports use paths anchored by their modules, except the cached text
# embeddings, whose historical loader resolves from the working directory.
import sys

sys.path.insert(0, str(LEAF_ROOT))
from LEAF import LEAF  # noqa: E402
from init_text_embeddings import load_embeddings  # noqa: E402
from load_config import build_model_config, load_yaml  # noqa: E402


DEFAULT_CHECKPOINT = (
    LEAF_ROOT
    / "ckpt"
    / "w100-c64-t256-lay12"
    / "q8-qlay4-mpnet-base-0"
    / "LEAF_Instruct.ckpt"
)
DEFAULT_CONFIG = LEAF_ROOT / "Configs" / "LEAF_mpnet.yaml"
DEFAULT_DATA_ROOT = Path("/media/public/LEAF/Downstream")
DEFAULT_JSON = PROJECT_PAGE / "assets" / "data" / "semantic-guidance-demo.json"
DEFAULT_META = (
    PROJECT_PAGE / "assets" / "data" / "semantic-guidance-demo.meta.json"
)
DEFAULT_FALLBACK = (
    PROJECT_PAGE
    / "assets"
    / "figures"
    / "semantic-guidance-demo-fallback.png"
)

LEVELS = [
    {
        "id": 0,
        "short": "L0",
        "label": "No task context",
        "summary": "A generic placeholder is encoded without task or candidate-label semantics.",
    },
    {
        "id": 1,
        "short": "L1",
        "label": "Task instruction",
        "summary": "A natural-language task description conditions the EEG representation.",
    },
    {
        "id": 2,
        "short": "L2",
        "label": "Task + targets",
        "summary": "The task embedding is blended with the mean candidate-label embedding.",
    },
]

DATASETS = {
    "MI_OpenBMI": {
        "title": "OpenBMI-MI",
        "labels": ["Right", "Left"],
        "colors": ["#315a80", "#d27a37"],
        "task_prompt": "Decode motor imagery",
        "full_test_count": 4800,
    },
    "EMO_SEED_3_seg4": {
        "title": "SEED",
        "labels": ["Positive", "Neutral", "Negative"],
        "colors": ["#147d70", "#7a8582", "#6f62a6"],
        "task_prompt": "Decode emotional states",
        "full_test_count": 7620,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector, ord=2)


def load_openbmi_subset(path: Path, per_subject_class: int, seed: int):
    """Read every held-out trial, or a balanced per-subject/class subset."""
    rng = np.random.default_rng(seed)
    batches, labels = [], []
    with h5py.File(path, "r", locking=False) as file:
        subjects = [f"s{index:02d}" for index in range(43, 55)]
        for subject in subjects:
            y = file[subject]["Y"][:].astype(np.int64, copy=False)
            for class_id in range(len(DATASETS["MI_OpenBMI"]["labels"])):
                candidates = np.flatnonzero(y == class_id)
                if per_subject_class <= 0:
                    indices = candidates
                elif len(candidates) < per_subject_class:
                    raise ValueError(
                        f"{subject} class {class_id} has only {len(candidates)} trials"
                    )
                else:
                    indices = np.sort(
                        rng.choice(candidates, size=per_subject_class, replace=False)
                    )
                batches.append(file[subject]["X"][indices].astype(np.float32, copy=False))
                labels.append(np.full(len(indices), class_id, dtype=np.int64))
    x = np.concatenate(batches)
    y = np.concatenate(labels)
    order = rng.permutation(len(y))
    return x[order], y[order]


def load_seed_subset(path: Path, per_class: int, seed: int):
    """Read the full predefined SEED test split, or a balanced subset."""
    rng = np.random.default_rng(seed)
    batches, labels = [], []
    with h5py.File(path, "r", locking=False) as file:
        y_all = file["testY"][:].astype(np.int64, copy=False)
        for class_id in range(len(DATASETS["EMO_SEED_3_seg4"]["labels"])):
            candidates = np.flatnonzero(y_all == class_id)
            if per_class <= 0:
                indices = candidates
            elif len(candidates) < per_class:
                raise ValueError(
                    f"SEED class {class_id} has only {len(candidates)} trials"
                )
            else:
                indices = np.sort(rng.choice(candidates, size=per_class, replace=False))
            batches.append(file["testX"][indices].astype(np.float32, copy=False))
            labels.append(np.full(len(indices), class_id, dtype=np.int64))
    x = np.concatenate(batches)
    y = np.concatenate(labels)
    order = rng.permutation(len(y))
    return x[order], y[order]


def instruction_vectors(spec: dict, text_to_embedding: dict[str, np.ndarray]):
    generic = text_to_embedding["None"].astype(np.float32, copy=True)
    task = text_to_embedding[spec["task_prompt"]].astype(np.float32, copy=True)
    target_mean = normalize(
        np.mean([text_to_embedding[label] for label in spec["labels"]], axis=0)
    ).astype(np.float32, copy=False)
    task_targets = normalize((task + target_mean) / 2.0).astype(np.float32, copy=False)
    return [generic, task, task_targets]


def prompt_records(spec: dict):
    labels = spec["labels"]
    return [
        {
            "prompt": "None",
            "candidate_labels": [],
            "model_input": 'MPNet embedding of the literal string “None”',
        },
        {
            "prompt": spec["task_prompt"],
            "candidate_labels": [],
            "model_input": "MPNet task-prompt embedding",
        },
        {
            "prompt": spec["task_prompt"],
            "candidate_labels": labels,
            "model_input": "normalize(0.5 × task + 0.5 × mean(targets))",
        },
    ]


def infer_levels(
    model: LEAF,
    x: np.ndarray,
    instructions: list[np.ndarray],
    prototypes: np.ndarray,
    device: torch.device,
    batch_size: int,
):
    outputs, accuracies = [], []
    for level, instruction in enumerate(instructions):
        instruction_tensor = torch.from_numpy(instruction).to(device)
        batches = []
        for start in range(0, len(x), batch_size):
            x_batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            instruction_batch = instruction_tensor.unsqueeze(0).expand(len(x_batch), -1)
            with torch.inference_mode():
                embedding, _ = model(x_batch, instruction_batch)
            batches.append(embedding.float().cpu().numpy())
        embedding = np.concatenate(batches).astype(np.float32, copy=False)
        prediction = (embedding @ prototypes.T).argmax(axis=1)
        outputs.append(embedding)
        accuracies.append(prediction)
        print(f"    L{level}: embeddings {embedding.shape}", flush=True)
        del instruction_tensor, batches
    return np.stack(outputs), np.stack(accuracies)


def shared_umap(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
):
    """Fit all three levels and fixed prototypes in one coordinate system."""
    n_levels, n_samples, _ = embeddings.shape
    combined = np.concatenate(
        [embeddings.reshape(n_levels * n_samples, -1), prototypes], axis=0
    )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        low_memory=True,
    )
    xy = reducer.fit_transform(combined).astype(np.float32, copy=False)
    point_xy = xy[: n_levels * n_samples].reshape(n_levels, n_samples, 2)
    prototype_xy = xy[n_levels * n_samples :]

    # One fixed extent across every level prevents visual rescaling during a
    # transition.  A small padding keeps points clear of the canvas boundary.
    low = xy.min(axis=0)
    high = xy.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    low -= 0.07 * span
    high += 0.07 * span
    span = high - low
    point_xy = (point_xy - low) / span
    prototype_xy = (prototype_xy - low) / span
    return point_xy, prototype_xy


def rounded_list(array: np.ndarray, decimals: int = 5):
    return np.round(array, decimals=decimals).tolist()


def save_fallback(payload: dict, output: Path):
    """Render a no-JavaScript snapshot of L2 using the exported coordinates."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    for ax, dataset in zip(axes, payload["datasets"]):
        coordinates = np.asarray(dataset["coordinates"][2])
        labels = np.asarray(dataset["point_labels"])
        prototypes = np.asarray(dataset["prototype_coordinates"])
        for class_id, (name, color) in enumerate(
            zip(dataset["label_names"], dataset["colors"])
        ):
            mask = labels == class_id
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=12,
                alpha=0.7,
                color=color,
                linewidths=0,
                label=name,
            )
        ax.scatter(
            prototypes[:, 0],
            prototypes[:, 1],
            s=90,
            marker="*",
            color="#14201f",
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        ax.set_title(dataset["title"], fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=False, loc="lower center", ncol=len(dataset["label_names"]), fontsize=8)
        for spine in ax.spines.values():
            spine.set_color("#d7dfdd")
    fig.suptitle("Task + target guidance", fontsize=11, color="#667370", y=0.99)
    fig.tight_layout(pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0", help="CUDA index, or -1 for CPU")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--fallback", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--openbmi-per-subject-class",
        type=int,
        default=0,
        help="0 exports every held-out OpenBMI trial; positive values sample each subject/class",
    )
    parser.add_argument(
        "--seed-per-class",
        type=int,
        default=0,
        help="0 exports the full SEED test split; positive values sample each class",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-neighbors", type=int, default=25)
    parser.add_argument("--min-dist", type=float, default=0.12)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if args.gpu != "-1" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --gpu -1")
    device = torch.device("cpu" if args.gpu == "-1" else f"cuda:{args.gpu}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("medium")

    print(">> Reading balanced test subsets", flush=True)
    subsets = {
        "MI_OpenBMI": load_openbmi_subset(
            args.data_root / "MI_OpenBMI.h5",
            args.openbmi_per_subject_class,
            args.seed,
        ),
        "EMO_SEED_3_seg4": load_seed_subset(
            args.data_root / "EMO_SEED_3_seg4.h5",
            args.seed_per_class,
            args.seed + 1,
        ),
    }
    for dataset, (_, y) in subsets.items():
        print(f"  {dataset}: {len(y)} trials, class counts {np.bincount(y).tolist()}")

    print(f">> Loading {checkpoint}", flush=True)
    cfg = load_yaml(config_path)
    config = build_model_config(cfg)
    if config.text_emb_model_name != "mpnet-base":
        raise ValueError(f"Expected mpnet-base config, got {config.text_emb_model_name}")
    model = LEAF(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    # Keep compatibility with the historical text-embedding cache loader.
    previous_cwd = Path.cwd()
    try:
        import os

        os.chdir(LEAF_ROOT)
        text_to_embedding = load_embeddings(config.text_emb_model_name)
    finally:
        os.chdir(previous_cwd)

    payload = {
        "schema_version": 1,
        "levels": LEVELS,
        "projection_note": (
            "Each panel uses one UMAP fitted jointly to the same trials at L0, L1, "
            "and L2 plus its fixed text prototypes. Coordinates are comparable across "
            "levels within a panel, but not between datasets."
        ),
        "datasets": [],
    }
    meta_datasets = {}

    for dataset, spec in DATASETS.items():
        print(f">> {spec['title']}", flush=True)
        x, y = subsets[dataset]
        instructions = instruction_vectors(spec, text_to_embedding)
        prototypes = np.stack(
            [text_to_embedding[label] for label in spec["labels"]]
        ).astype(np.float32, copy=False)
        embeddings, predictions = infer_levels(
            model, x, instructions, prototypes, device, args.batch_size
        )
        point_xy, prototype_xy = shared_umap(
            embeddings,
            prototypes,
            args.seed,
            args.n_neighbors,
            args.min_dist,
        )
        selected_bacc = [
            float(balanced_accuracy_score(y, predictions[level])) for level in range(3)
        ]
        print(
            "    selected-subset bAcc: "
            + ", ".join(f"L{i}={value:.4f}" for i, value in enumerate(selected_bacc)),
            flush=True,
        )
        payload["datasets"].append(
            {
                "id": dataset,
                "title": spec["title"],
                "label_names": spec["labels"],
                "colors": spec["colors"],
                "prompts": prompt_records(spec),
                "point_labels": y.tolist(),
                "coordinates": rounded_list(point_xy),
                "prototype_coordinates": rounded_list(prototype_xy),
                "sample_count": int(len(y)),
                "full_test_count": spec["full_test_count"],
            }
        )
        meta_datasets[dataset] = {
            "display_name": spec["title"],
            "split": "test",
            "full_test_trials": spec["full_test_count"],
            "selected_trials": int(len(y)),
            "class_counts": np.bincount(y).tolist(),
            "selected_subset_balanced_accuracy": selected_bacc,
        }
        del x, y, embeddings, predictions

    generated_at = datetime.now(timezone.utc).isoformat()
    checkpoint_hash = sha256(checkpoint)
    config_hash = sha256(config_path)
    meta = {
        "schema_version": 1,
        "generated_at": generated_at,
        "generator": "ProjectPage/precompute_projectpage_semantic_demo.py",
        "checkpoint": {
            "name": checkpoint.name,
            "run": checkpoint.parent.name,
            "sha256": checkpoint_hash,
        },
        "config": {"name": config_path.name, "sha256": config_hash},
        "text_embedding_model": config.text_emb_model_name,
        "seed": args.seed,
        "sampling": {
            "MI_OpenBMI": (
                "all trials from held-out subjects s43-s54"
                if args.openbmi_per_subject_class <= 0
                else f"{args.openbmi_per_subject_class} trials per class from each held-out subject s43-s54"
            ),
            "EMO_SEED_3_seg4": (
                "all trials from predefined test split"
                if args.seed_per_class <= 0
                else f"{args.seed_per_class} trials per class from predefined test split"
            ),
        },
        "projection": {
            "algorithm": "UMAP",
            "fit": "joint over L0/L1/L2 embeddings and fixed label prototypes, separately per dataset",
            "metric": "cosine",
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "seed": args.seed,
        },
        "datasets": meta_datasets,
        "privacy": "No raw EEG, subject identifiers, trial identifiers, or 768-D embeddings are exported.",
    }
    payload["provenance"] = {
        "generated_at": generated_at,
        "checkpoint_run": checkpoint.parent.name,
        "checkpoint_sha256": checkpoint_hash,
        "seed": args.seed,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    args.meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_fallback(payload, args.fallback)
    print(f">> Wrote {args.json} ({args.json.stat().st_size / 1024:.1f} KiB)")
    print(f">> Wrote {args.meta}")
    print(f">> Wrote {args.fallback}")


if __name__ == "__main__":
    main()
