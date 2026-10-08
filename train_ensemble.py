"""Train the Dutta et al. (2020) stacking ensemble for IoT-23 intrusion detection.

Final stage of the NIDS pipeline: trains the heterogeneous Level-0 deep
learners and the Level-1 stacking meta-learner, then evaluates the complete
ensemble on the held-out test split.

Pipeline position
-----------------
::

    src/dataset.py              train_dsae.py               train_ensemble.py (this script)
    ─────────────              ────────────                ──────────────────────────────
    data/processed/             data/processed/             models/saved/dnn_level0.pt
      X_train/y_train.npy  ──►  z_train.npy  ────────────►  models/saved/lstm_level0.pt
      X_test/y_test.npy    ──►  z_test.npy   ────────────►  models/saved/meta_learner.joblib
                                  (DSAE latents)             + console evaluation report

Stacking workflow (leakage-safe)
--------------------------------
1. Alignment — ``TemporalLSTM`` needs sliding windows (``seq_len=5``)
   produced by ``src.classifiers.create_lstm_sequences``; the first
   ``seq_len - 1`` flows serve only as window context. SpatialDNN inputs
   and labels are sliced from index ``seq_len - 1`` so both learners
   evaluate the exact same network events, row-for-row.
2. Out-of-fold (OOF) meta-features — ``StratifiedKFold(n_splits=5,
   shuffle=True, random_state=42)`` partitions the aligned training events.
   Per fold, FRESH ``SpatialDNN`` and ``TemporalLSTM`` instances are
   trained on the fold's training events only; their softmax posteriors on
   the untouched validation events fill ``[p_dnn || p_lstm]`` (10-dim) into
   the OOF matrix. No model ever predicts an event it trained on, so the
   Level-1 learner only ever sees honest generalisation behaviour.
3. Final Level-0 models — fresh instances trained on the FULL aligned
   training set; state_dicts persisted for inference.
4. Level-1 meta-learner — multinomial ``LogisticRegression`` fit on the
   OOF matrix + aligned labels, persisted via joblib.
5. Evaluation — the final Level-0 models are RELOADED from disk, produce
   test posteriors, the meta-learner stacks them, and the console receives
   the classification report, accuracy, macro-F1 and a formatted confusion
   matrix.

Caveat (sliding-window adjacency): consecutive windows share ``seq_len - 1``
context flows, so LSTM fold-validation borders carry mild input overlap
across the fold boundary; the predicted event's LABEL, however, is strictly
out-of-fold. (Upstream SMOTE-ENN resampling already interleaves synthetic
flows, so strictly chronological purity is unattainable by construction.)

Usage (from the repository root)
--------------------------------
.. code-block:: console

    python train_ensemble.py                                 # 15 epochs, batch 128, 5 folds
    python train_ensemble.py --epochs 25 --batch-size 256 --k-folds 5
    python train_ensemble.py --device cpu -v

Dependencies: ``torch``, ``numpy``, ``scikit-learn``, ``joblib``; imports the
Level-0 learners from ``src/classifiers.py``.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from torch.utils.data import DataLoader, TensorDataset

from src.classifiers import SpatialDNN, TemporalLSTM, create_lstm_sequences

# --------------------------------------------------------------------------- #
# Public configuration                                                         #
# --------------------------------------------------------------------------- #

LOGGER: Final[logging.Logger] = logging.getLogger("iot23.train_ensemble")

#: Training hyper-parameters (Dutta et al., 2020 baseline).
DEFAULT_EPOCHS: Final[int] = 15
DEFAULT_BATCH_SIZE: Final[int] = 128
DEFAULT_K_FOLDS: Final[int] = 5
DEFAULT_LEARNING_RATE: Final[float] = 1e-3
WEIGHT_DECAY: Final[float] = 1e-5
DEFAULT_SEQ_LEN: Final[int] = 5
SEED: Final[int] = 42

#: Ensemble geometry.
NUM_CLASSES: Final[int] = 5
META_MAX_ITER: Final[int] = 1000

#: Rows per forward block during chunked probability generation.
PREDICTION_CHUNK_SIZE: Final[int] = 4096

#: Console class names for the report / confusion matrix (spec §4).
REPORT_CLASS_NAMES: Final[list[str]] = ["Benign", "Recon", "DoS", "Malware", "Botnet"]

#: Artifacts consumed (from train_dsae.py) and produced by this script.
Z_TRAIN_PATH: Final[Path] = Path("data/processed/z_train.npy")
Y_TRAIN_PATH: Final[Path] = Path("data/processed/y_train.npy")
Z_TEST_PATH: Final[Path] = Path("data/processed/z_test.npy")
Y_TEST_PATH: Final[Path] = Path("data/processed/y_test.npy")
DNN_MODEL_PATH: Final[Path] = Path("models/saved/dnn_level0.pt")
LSTM_MODEL_PATH: Final[Path] = Path("models/saved/lstm_level0.pt")
META_LEARNER_PATH: Final[Path] = Path("models/saved/meta_learner.joblib")

__all__: Final[list[str]] = ["main"]


# --------------------------------------------------------------------------- #
# Internal utilities                                                           #
# --------------------------------------------------------------------------- #


@contextmanager
def _stage_timer(description: str) -> Iterator[None]:
    """Context manager that logs the wall-clock duration of a pipeline stage.

    Args:
        description: Human-readable stage name used in the log line.
    """
    start = time.perf_counter()
    try:
        yield
    except Exception:
        LOGGER.exception("%s — FAILED after %.2f s", description, time.perf_counter() - start)
        raise
    LOGGER.info("%s — done in %.2f s", description, time.perf_counter() - start)


def _resolve_device(requested: str) -> torch.device:
    """Resolve the target device (CPU, or CUDA when available).

    Args:
        requested: One of ``"cpu"``, ``"cuda"``, ``"auto"``. ``"auto"`` picks
            CUDA when available; an unavailable ``"cuda"`` request falls
            back to CPU with a warning instead of crashing.

    Returns:
        The resolved ``torch.device``.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        selected = torch.device("cuda")
        LOGGER.info("Using device: %s (%s).", selected, torch.cuda.get_device_name(0))
        return selected
    LOGGER.warning("Device '%s' requested but CUDA is unavailable — falling back to CPU.", requested)
    return torch.device("cpu")


def _load_latent_matrix(path: Path, label: str) -> np.ndarray:
    """Load and validate a DSAE latent matrix (.npy).

    Args:
        path: Artifact path (e.g. ``data/processed/z_train.npy``).
        label: Human-readable name used in logs and error messages.

    Returns:
        Validated matrix, shape ``(n_samples, latent_dim)``, float32,
        C-contiguous, values in [0, 1] (Sigmoid-bounded DSAE codes).

    Raises:
        FileNotFoundError: If the artifact is missing (hint: run
            ``python train_dsae.py`` first).
        ValueError: If the matrix is not 2-D, is empty, contains non-finite
            values, or escapes [0, 1].
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Required artifact '{path}' was not found — generate it first with "
            f"`python train_dsae.py`."
        )
    matrix = np.load(path)
    LOGGER.info("Loaded %s ← %s (shape=%s, dtype=%s).", label, path, matrix.shape, matrix.dtype)
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be 2-D (n_samples, latent_dim), got shape {matrix.shape}.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{label} is empty: shape {matrix.shape}.")
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains non-finite (NaN/Inf) values.")
    if matrix.min() < 0.0 or matrix.max() > 1.0:
        raise ValueError(f"{label} contains values outside [0, 1] — not Sigmoid-bounded DSAE latents.")
    return matrix


def _load_label_vector(path: Path, label: str) -> np.ndarray:
    """Load and validate an integer target vector (.npy).

    Args:
        path: Artifact path (e.g. ``data/processed/y_train.npy``).
        label: Human-readable name used in logs and error messages.

    Returns:
        Validated labels, shape ``(n_samples,)``, int64, values in
        ``[0, NUM_CLASSES)``.

    Raises:
        FileNotFoundError: If the artifact is missing (hint: run
            ``python src/dataset.py`` first).
        ValueError: If the vector is not 1-D, is empty, or contains targets
            outside the 5-class taxonomy.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Required artifact '{path}' was not found — generate it first with "
            f"`python src/dataset.py`."
        )
    labels = np.load(path)
    LOGGER.info("Loaded %s ← %s (shape=%s, dtype=%s).", label, path, labels.shape, labels.dtype)
    if labels.ndim != 1:
        raise ValueError(f"{label} must be 1-D (n_samples,), got shape {labels.shape}.")
    if labels.shape[0] < 1:
        raise ValueError(f"{label} is empty: shape {labels.shape}.")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError(
            f"{label} contains targets outside [0, {NUM_CLASSES}): "
            f"[{labels.min()}, {labels.max()}]."
        )
    return labels


def _load_state_dict_file(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a state-dict checkpoint from disk, CPU-mapped and safely.

    ``weights_only=True`` (PyTorch >= 1.13) prevents arbitrary unpickling;
    older PyTorch versions fall back to the legacy loader.

    Raises:
        FileNotFoundError: If ``path`` does not point to an existing file.
    """
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint!s}")
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 — the `weights_only` kwarg does not exist.
        return torch.load(checkpoint, map_location="cpu")


def _save_state_dict(model: nn.Module, path: Path, label: str) -> None:
    """Persist a Level-0 model's state_dict (parent directories created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    torch.save(state, path)
    LOGGER.info("Saved %s state_dict (%d tensors) → %s", label, len(state), path)


def _log_class_distribution(y: np.ndarray, title: str) -> None:
    """Log per-class sample counts with the console class names."""
    counts = np.bincount(y, minlength=NUM_CLASSES)
    rendered = ", ".join(
        f"{REPORT_CLASS_NAMES[index]}[{index}]={int(counts[index])}" for index in range(NUM_CLASSES)
    )
    LOGGER.info("%s — total=%d | %s", title, len(y), rendered)


def _render_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str]) -> str:
    """Render a confusion matrix as an aligned, monospaced console table.

    Args:
        matrix: Square confusion matrix — rows are ground truth, columns
            are predictions.
        class_names: One name per class index, aligned with the matrix.

    Returns:
        Multi-line string table including row/column totals and a legend.
    """
    total = int(matrix.sum())
    cell_width = max(
        max(len(str(int(value))) for value in matrix.ravel()),
        max(len(name) for name in class_names),
    ) + 2
    row_label_width = max(len(name) for name in class_names) + 3
    total_width = max(len("Total"), len(str(total))) + 2

    header = (
        "true\\pred".ljust(row_label_width)
        + "".join(name.rjust(cell_width) for name in class_names)
        + "Total".rjust(total_width)
    )
    lines = [header, "-" * len(header)]
    for index, name in enumerate(class_names):
        lines.append(
            name.ljust(row_label_width)
            + "".join(str(int(value)).rjust(cell_width) for value in matrix[index])
            + str(int(matrix[index].sum())).rjust(total_width)
        )
    lines.append("-" * len(header))
    lines.append(
        "Total".ljust(row_label_width)
        + "".join(str(int(value)).rjust(cell_width) for value in matrix.sum(axis=0))
        + str(total).rjust(total_width)
    )
    lines.append("(rows = ground truth, columns = predictions)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Level-0 training primitives                                                  #
# --------------------------------------------------------------------------- #


def _train_level0_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: np.ndarray,
    *,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> None:
    """Train a Level-0 classifier in place (Adam + CrossEntropyLoss).

    Accepts point vectors ``(N, latent_dim)`` for :class:`SpatialDNN` or
    sliding windows ``(N, seq_len, latent_dim)`` for :class:`TemporalLSTM`;
    both emit raw logits, so ``nn.CrossEntropyLoss`` is the correct objective.

    Args:
        model: Classifier already moved to ``device``; trained in place.
        x_train: Input tensor, float32.
        y_train: Integer labels, shape ``(N,)``.
        model_name: Label used in log lines (e.g. ``"fold 3/5 SpatialDNN"``).
        epochs: Number of optimization epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam L2 weight decay.
        device: Target compute device.

    Raises:
        ValueError: If inputs mismatch or fewer than 2 samples are given.
        RuntimeError: On a non-finite batch loss (numerical divergence).
    """
    if len(x_train) != len(y_train):
        raise ValueError(f"{model_name}: x/y length mismatch ({len(x_train)} != {len(y_train)}).")
    if len(x_train) < 2:
        raise ValueError(f"{model_name}: need >= 2 training samples (BatchNorm1d limitation).")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    targets = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.int64))
    loader = DataLoader(
        TensorDataset(x_train, targets),
        batch_size=batch_size,
        shuffle=True,
        # Drop the trailing batch ONLY when it would contain a single
        # sample, which BatchNorm1d (inside SpatialDNN) cannot train on.
        drop_last=len(x_train) % batch_size == 1,
    )

    model.train()
    started = time.perf_counter()
    final_loss = math.nan
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        samples = 0
        correct = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)                     # (B, num_classes) raw logits
            loss = criterion(logits, y_batch)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"{model_name}: non-finite loss ({loss_value}) at epoch {epoch}."
                )
            loss.backward()
            optimizer.step()
            loss_sum += loss_value * y_batch.size(0)
            samples += y_batch.size(0)
            correct += int((logits.argmax(dim=-1) == y_batch).sum().item())
        final_loss = loss_sum / samples
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "%s | epoch %02d/%02d | loss %.6f | train acc %.4f",
                model_name, epoch, epochs, final_loss, correct / samples,
            )
    LOGGER.info(
        "%s trained — %d epochs over %d samples in %.1fs | final train loss %.6f.",
        model_name, epochs, samples, time.perf_counter() - started, final_loss,
    )


def _predict_proba_chunked(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    chunk_size: int = PREDICTION_CHUNK_SIZE,
) -> np.ndarray:
    """Eval-mode class probabilities in memory-bounded chunks.

    Args:
        model: Level-0 classifier (switched to ``eval()``; its
            ``predict_proba`` runs under ``torch.no_grad()`` internally).
        x: Inputs — ``(N, latent_dim)`` or ``(N, seq_len, latent_dim)``.
        device: Compute device (chunks are moved per block; results are
            gathered back on the CPU).
        chunk_size: Rows per forward block.

    Returns:
        Softmax posteriors, shape ``(N, num_classes)``, float32, rows
        summing to 1.0.
    """
    model.eval()
    blocks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_size):
            block = x[start : start + chunk_size].to(device)
            probabilities = model.predict_proba(block)  # (b, num_classes)
            blocks.append(probabilities.cpu().numpy())
    return np.concatenate(blocks, axis=0).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Stacking: Level-1 meta-learner and out-of-fold Level-0 training              #
# --------------------------------------------------------------------------- #


def _build_meta_learner() -> LogisticRegression:
    """Build the multinomial Level-1 LogisticRegression across sklearn versions.

    ``multi_class='multinomial'`` is required by the reference architecture,
    but the kwarg was deprecated in scikit-learn 1.5 (multinomial became the
    default behaviour) and removed in later releases. This shim passes it
    when accepted and falls back to the (equivalent) default otherwise.

    Returns:
        An unfitted ``LogisticRegression(max_iter=1000)``.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return LogisticRegression(max_iter=META_MAX_ITER, multi_class="multinomial")
    except (TypeError, ValueError):  # scikit-learn >= 1.7: kwarg removed.
        return LogisticRegression(max_iter=META_MAX_ITER)


def _run_oof_stacking(
    z_dnn: np.ndarray,
    x_seq: np.ndarray,
    y_aligned: np.ndarray,
    *,
    k_folds: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Out-of-fold training of both Level-0 learners → OOF meta-features.

    For every ``StratifiedKFold`` split, FRESH ``SpatialDNN`` and
    ``TemporalLSTM`` instances are trained on the fold's training events
    only; their softmax posteriors on the untouched validation events fill
    the ``(M, 2 * num_classes)`` OOF matrix ``[p_dnn || p_lstm]``. Because
    no model ever predicts an event it saw during training, the Level-1
    logistic regression can be fit on the OOF matrix without leakage.

    Args:
        z_dnn: Aligned SpatialDNN inputs, shape ``(M, latent_dim)``.
        x_seq: Aligned TemporalLSTM windows, shape ``(M, seq_len, latent_dim)``.
        y_aligned: Aligned labels, shape ``(M,)``.
        k_folds: Number of stratified folds.
        epochs: Training epochs per fold model.
        batch_size: Training mini-batch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam L2 weight decay.
        device: Target compute device.
        seed: Seed for the stratified splitter.

    Returns:
        ``(oof_matrix, fold_stats)`` — the ``(M, 10)`` OOF meta-feature
        matrix and one dict of validation accuracies per fold.

    Raises:
        ValueError: If ``k_folds < 2``, fewer than 2 classes are present,
            or the smallest class has fewer than ``k_folds`` samples.
        RuntimeError: If the OOF matrix is not fully populated.
    """
    if k_folds < 2:
        raise ValueError(f"k_folds must be >= 2, got {k_folds}.")
    class_counts = np.bincount(y_aligned, minlength=NUM_CLASSES)
    present = class_counts[class_counts > 0]
    if present.size < 2:
        raise ValueError("Aligned training set has fewer than 2 classes — cannot train the ensemble.")
    if int(present.min()) < k_folds:
        raise ValueError(
            f"Smallest class in the aligned training set has {int(present.min())} sample(s) "
            f"< k_folds={k_folds}; lower --k-folds or increase the training sample size."
        )

    n_events = len(y_aligned)
    latent_dim = z_dnn.shape[1]
    oof_matrix = np.zeros((n_events, 2 * NUM_CLASSES), dtype=np.float32)
    fold_stats: list[dict[str, float]] = []
    splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(z_dnn, y_aligned), start=1):
        LOGGER.info(
            "── Fold %d/%d — %d train events / %d validation events ──",
            fold, k_folds, len(train_idx), len(val_idx),
        )
        fold_prefix = f"fold {fold}/{k_folds}"

        # Fresh Level-0 instances per fold — no cross-fold weight reuse.
        dnn = SpatialDNN(latent_dim=latent_dim, num_classes=NUM_CLASSES).to(device)
        _train_level0_model(
            dnn,
            torch.from_numpy(z_dnn[train_idx]),
            y_aligned[train_idx],
            model_name=f"{fold_prefix} SpatialDNN",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )
        lstm = TemporalLSTM(latent_dim=latent_dim, num_classes=NUM_CLASSES).to(device)
        _train_level0_model(
            lstm,
            torch.from_numpy(x_seq[train_idx]),
            y_aligned[train_idx],
            model_name=f"{fold_prefix} TemporalLSTM",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )

        # OOF posteriors on the untouched validation events.
        p_dnn_val = _predict_proba_chunked(dnn, torch.from_numpy(z_dnn[val_idx]), device)
        p_lstm_val = _predict_proba_chunked(lstm, torch.from_numpy(x_seq[val_idx]), device)
        oof_matrix[val_idx, :NUM_CLASSES] = p_dnn_val
        oof_matrix[val_idx, NUM_CLASSES:] = p_lstm_val

        y_val = y_aligned[val_idx]
        stats = {
            "dnn": float(accuracy_score(y_val, p_dnn_val.argmax(axis=1))),
            "lstm": float(accuracy_score(y_val, p_lstm_val.argmax(axis=1))),
            "mean_prob": float(
                accuracy_score(y_val, (0.5 * (p_dnn_val + p_lstm_val)).argmax(axis=1))
            ),
        }
        fold_stats.append(stats)
        LOGGER.info(
            "Fold %d/%d OOF accuracies — SpatialDNN %.4f | TemporalLSTM %.4f | mean-prob blend %.4f.",
            fold, k_folds, stats["dnn"], stats["lstm"], stats["mean_prob"],
        )

    # Every event must have received exactly one OOF prediction pair —
    # each 5-column softmax block must therefore sum to 1 per row.
    if not (
        np.allclose(oof_matrix[:, :NUM_CLASSES].sum(axis=1), 1.0, atol=1e-4)
        and np.allclose(oof_matrix[:, NUM_CLASSES:].sum(axis=1), 1.0, atol=1e-4)
    ):
        raise RuntimeError("OOF matrix incomplete — every event must receive exactly one out-of-fold prediction pair.")
    return oof_matrix, fold_stats


# --------------------------------------------------------------------------- #
# Console entry point                                                          #
# --------------------------------------------------------------------------- #


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the ensemble training script."""
    parser = argparse.ArgumentParser(
        prog="python train_ensemble.py",
        description=(
            "Train the Level-0 learners (SpatialDNN, TemporalLSTM) and the Level-1 "
            "stacking meta-learner (multinomial LogisticRegression) of Dutta et al. "
            "(2020) on the DSAE latents, using out-of-fold cross-validation to keep "
            "the meta-classifier leakage-free, then evaluate on the held-out test split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Training epochs for every Level-0 model (per fold and final).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Training mini-batch size for the Level-0 models.")
    parser.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS,
                        help="Number of StratifiedKFold splits for OOF meta-feature generation.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE,
                        help="Adam learning rate for the Level-0 models.")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help="Sliding-window length for the TemporalLSTM (alignment offset = seq_len - 1).")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto",
                        help="Target device — CPU, or CUDA when available.")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Seed for the stratified splitter, initialisation and shuffling.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging (per-epoch metrics).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full stacking-ensemble training pipeline.

    Stages: load artifacts → align DNN/LSTM inputs → OOF Level-0 training →
    final Level-0 training + persistence → Level-1 meta-learner fit +
    persistence → end-to-end evaluation on the aligned test split.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be >= 1.")
    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2 (BatchNorm1d requires > 1 sample per training batch).")
    if args.k_folds < 2:
        parser.error("--k-folds must be >= 2.")
    if args.lr <= 0.0:
        parser.error("--lr must be positive.")
    if args.seq_len < 1:
        parser.error("--seq-len must be >= 1.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        device = _resolve_device(args.device)
        LOGGER.info("=" * 78)
        LOGGER.info("Ensemble training — Dutta et al. (2020) deep-learning stacking ensemble")
        LOGGER.info(
            "epochs=%d | batch_size=%d | k_folds=%d | lr=%.2e | weight_decay=%.1e | "
            "seq_len=%d | device=%s | seed=%d",
            args.epochs, args.batch_size, args.k_folds, args.lr, WEIGHT_DECAY,
            args.seq_len, device, args.seed,
        )
        LOGGER.info("=" * 78)

        for artifact in (DNN_MODEL_PATH, LSTM_MODEL_PATH, META_LEARNER_PATH):
            if artifact.exists():
                LOGGER.warning("Overwriting existing artifact: %s", artifact)

        # -- Stage 1: load artifacts & align DNN / LSTM inputs ------------- #
        with _stage_timer("Artifact loading & sequence alignment"):
            z_train = _load_latent_matrix(Z_TRAIN_PATH, "z_train")
            y_train = _load_label_vector(Y_TRAIN_PATH, "y_train")
            z_test = _load_latent_matrix(Z_TEST_PATH, "z_test")
            y_test = _load_label_vector(Y_TEST_PATH, "y_test")
            if len(z_train) != len(y_train):
                raise ValueError(f"z_train/y_train row mismatch: {len(z_train)} != {len(y_train)}.")
            if len(z_test) != len(y_test):
                raise ValueError(f"z_test/y_test row mismatch: {len(z_test)} != {len(y_test)}.")
            if z_train.shape[1] != z_test.shape[1]:
                raise ValueError(
                    f"Latent-dimension mismatch: z_train={z_train.shape[1]}, z_test={z_test.shape[1]}."
                )
            latent_dim = z_train.shape[1]

            # LSTM windows (labels aligned to each window's FINAL step) …
            x_seq_train, y_seq_train = create_lstm_sequences(z_train, y_train, seq_len=args.seq_len)
            x_seq_test, y_seq_test = create_lstm_sequences(z_test, y_test, seq_len=args.seq_len)
            # … and DNN inputs/labels sliced from index (seq_len - 1) so both
            # learners evaluate the exact same network events.
            z_dnn_train = z_train[args.seq_len - 1 :]
            y_train_aligned = y_train[args.seq_len - 1 :]
            z_dnn_test = z_test[args.seq_len - 1 :]
            y_test_aligned = y_test[args.seq_len - 1 :]
            # Invariant: the utility's final-step labels must equal the slice.
            if not np.array_equal(y_seq_train, y_train_aligned):
                raise RuntimeError("Train alignment invariant violated: LSTM labels ≠ y_train[seq_len-1:].")
            if not np.array_equal(y_seq_test, y_test_aligned):
                raise RuntimeError("Test alignment invariant violated: LSTM labels ≠ y_test[seq_len-1:].")
            if y_train_aligned.shape[0] < 1 or y_test_aligned.shape[0] < 1:
                raise ValueError("Alignment produced zero events — training set too small for seq_len.")

        LOGGER.info("Auto-detected latent_dim=%d from z_train.", latent_dim)
        LOGGER.info(
            "Sequence alignment (seq_len=%d): train %d flows → %d events | test %d flows → %d events "
            "(first %d flows of each split serve as window context only).",
            args.seq_len, len(z_train), len(y_train_aligned), len(z_test), len(y_test_aligned),
            args.seq_len - 1,
        )
        _log_class_distribution(y_train_aligned, "Aligned training events")
        _log_class_distribution(y_test_aligned, "Aligned test events (held out, imbalanced by design)")

        # -- Stage 2: out-of-fold Level-0 stacking -------------------------- #
        with _stage_timer("Out-of-fold Level-0 stacking"):
            oof_matrix, fold_stats = _run_oof_stacking(
                z_dnn=z_dnn_train,
                x_seq=x_seq_train,
                y_aligned=y_train_aligned,
                k_folds=args.k_folds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=WEIGHT_DECAY,
                device=device,
                seed=args.seed,
            )
        dnn_accs = np.array([entry["dnn"] for entry in fold_stats])
        lstm_accs = np.array([entry["lstm"] for entry in fold_stats])
        blend_accs = np.array([entry["mean_prob"] for entry in fold_stats])
        LOGGER.info(
            "OOF summary (%d folds) — SpatialDNN %.4f ± %.4f | TemporalLSTM %.4f ± %.4f | "
            "mean-prob blend %.4f ± %.4f.",
            args.k_folds, dnn_accs.mean(), dnn_accs.std(),
            lstm_accs.mean(), lstm_accs.std(), blend_accs.mean(), blend_accs.std(),
        )

        # -- Stage 3: final Level-0 models on the full aligned training set - #
        with _stage_timer("Final Level-0 training (full aligned set)"):
            dnn_final = SpatialDNN(latent_dim=latent_dim, num_classes=NUM_CLASSES).to(device)
            _train_level0_model(
                dnn_final,
                torch.from_numpy(z_dnn_train),
                y_train_aligned,
                model_name="SpatialDNN (final)",
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=WEIGHT_DECAY,
                device=device,
            )
            lstm_final = TemporalLSTM(latent_dim=latent_dim, num_classes=NUM_CLASSES).to(device)
            _train_level0_model(
                lstm_final,
                torch.from_numpy(x_seq_train),
                y_train_aligned,
                model_name="TemporalLSTM (final)",
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=WEIGHT_DECAY,
                device=device,
            )
            _save_state_dict(dnn_final, DNN_MODEL_PATH, "SpatialDNN")
            _save_state_dict(lstm_final, LSTM_MODEL_PATH, "TemporalLSTM")

        # -- Stage 4: Level-1 meta-learner on the OOF matrix ---------------- #
        with _stage_timer("Level-1 meta-learner fitting"):
            meta_learner = _build_meta_learner()
            meta_learner.fit(oof_matrix, y_train_aligned)
            oof_fit_accuracy = float(accuracy_score(y_train_aligned, meta_learner.predict(oof_matrix)))
            cv_scores = cross_val_score(
                _build_meta_learner(),
                oof_matrix,
                y_train_aligned,
                cv=StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed),
                scoring="accuracy",
            )
            LOGGER.info(
                "Meta-learner fit on %s OOF features | classes=%s | OOF-fit accuracy %.4f (in-sample) | "
                "%d-fold CV on OOF features %.4f ± %.4f (honest stacked-generalisation estimate).",
                oof_matrix.shape, meta_learner.classes_.tolist(), oof_fit_accuracy, args.k_folds,
                float(cv_scores.mean()), float(cv_scores.std()),
            )
            META_LEARNER_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(meta_learner, META_LEARNER_PATH)
            LOGGER.info("Saved fitted meta-learner → %s", META_LEARNER_PATH)

        # -- Stage 5: end-to-end evaluation on the aligned test split ------- #
        with _stage_timer("End-to-end test evaluation"):
            # Evaluate the PERSISTED artifacts (also verifies the checkpoints).
            dnn_eval = SpatialDNN(latent_dim=latent_dim, num_classes=NUM_CLASSES).to(device)
            dnn_eval.load_state_dict(_load_state_dict_file(DNN_MODEL_PATH))
            lstm_eval = TemporalLSTM(latent_dim=latent_dim, num_classes=NUM_CLASSES).to(device)
            lstm_eval.load_state_dict(_load_state_dict_file(LSTM_MODEL_PATH))
            LOGGER.info("Reloaded persisted Level-0 artifacts for evaluation (round-trip verification).")

            p_dnn_test = _predict_proba_chunked(dnn_eval, torch.from_numpy(z_dnn_test), device)
            p_lstm_test = _predict_proba_chunked(lstm_eval, torch.from_numpy(x_seq_test), device)
            meta_test = np.concatenate([p_dnn_test, p_lstm_test], axis=1)
            if meta_test.shape != (len(y_test_aligned), 2 * NUM_CLASSES):
                raise ValueError(f"Test meta-features have shape {meta_test.shape}, "
                                 f"expected {(len(y_test_aligned), 2 * NUM_CLASSES)}.")
            LOGGER.info(
                "Test posteriors — p_dnn_test%s, p_lstm_test%s, meta-features%s.",
                p_dnn_test.shape, p_lstm_test.shape, meta_test.shape,
            )

            y_pred = meta_learner.predict(meta_test)
            acc_dnn = float(accuracy_score(y_test_aligned, p_dnn_test.argmax(axis=1)))
            acc_lstm = float(accuracy_score(y_test_aligned, p_lstm_test.argmax(axis=1)))
            ensemble_accuracy = float(accuracy_score(y_test_aligned, y_pred))
            ensemble_macro_f1 = float(
                f1_score(y_test_aligned, y_pred, labels=list(range(NUM_CLASSES)),
                         average="macro", zero_division=0)
            )
            report = classification_report(
                y_test_aligned, y_pred,
                labels=list(range(NUM_CLASSES)),
                target_names=REPORT_CLASS_NAMES,
                digits=4, zero_division=0,
            )
            matrix = confusion_matrix(y_test_aligned, y_pred, labels=list(range(NUM_CLASSES)))

        LOGGER.info("=" * 78)
        LOGGER.info("Ensemble evaluation on held-out test split (%d aligned events):", len(y_test_aligned))
        LOGGER.info("Level-0 references — SpatialDNN accuracy %.4f | TemporalLSTM accuracy %.4f.",
                    acc_dnn, acc_lstm)
        LOGGER.info("Level-1 stacked ensemble — accuracy %.4f | macro-F1 %.4f.",
                    ensemble_accuracy, ensemble_macro_f1)
        LOGGER.info("Per-class classification report:\n%s", report)
        LOGGER.info("Confusion matrix:\n%s", _render_confusion_matrix(matrix, REPORT_CLASS_NAMES))
        LOGGER.info("=" * 78)
        LOGGER.info("Artifacts: %s | %s | %s", DNN_MODEL_PATH, LSTM_MODEL_PATH, META_LEARNER_PATH)
        LOGGER.info("Pipeline complete — inference reloads the Level-0 state_dicts plus the joblib meta-learner.")
        return 0
    except Exception:
        LOGGER.exception("Ensemble training failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())