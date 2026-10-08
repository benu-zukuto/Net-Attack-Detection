"""Train the Deep Sparse Autoencoder (DSAE) on preprocessed IoT-23 flows.

Stage 2 of the NIDS pipeline of *Dutta et al. (2020)* — "A Deep Learning
Ensemble for Network Anomaly and Cyber-Attack Detection". This script
consumes the leakage-safe feature matrices produced by ``src/dataset.py``,
trains the DSAE from ``src/autoencoder.py`` with the sparse-reconstruction
objective, checkpoints the best model, and exports the compressed latent
representations consumed by the Level-0 learners of ``src/classifiers.py``.

Pipeline position
-----------------
::

    src/dataset.py                     train_dsae.py                  src/classifiers.py
    ─────────────                      ────────────                   ─────────────────
    data/processed/X_train.npy   ──►   DSAE training        ──►   data/processed/z_train.npy
    data/processed/X_test.npy    ──►   (Adam +              ──►   data/processed/z_test.npy
                                        ReduceLROnPlateau)        models/saved/dsae_full.pt
                                                                  models/saved/dsae_encoder.pt

Training configuration (Dutta et al., 2020 baseline)
----------------------------------------------------
========================  ==================================================
Setting                   Value
========================  ==================================================
Model                     ``DeepSparseAutoencoder(latent_dim=32)``
Loss                      ``SparseLoss(rho=0.05, beta=2.0)``
Optimizer                 Adam (lr=1e-3, weight_decay=1e-5)
LR scheduler              ReduceLROnPlateau(mode='min', factor=0.5, patience=3)
Batching                  batch_size=256, shuffle=True, drop_last=True
Epochs                    25 (``--epochs``)
Device                    cpu by default (``--device {cpu,cuda,auto}``)
========================  ==================================================

Checkpointing & selection
-------------------------
Validation runs on the untouched ``X_test`` fold after every epoch.
Whenever the validation loss reaches a new minimum, BOTH artifacts are
rewritten: the best full autoencoder (``models/saved/dsae_full.pt``) and
the encoder-only weights (``models/saved/dsae_encoder.pt``). After
training, the best checkpoint is reloaded from disk before latent
extraction, so ``z_train.npy`` / ``z_test.npy`` are always produced by the
best-performing weights.

Latent export
-------------
``z_train.npy`` aligns row-for-row with ``y_train.npy`` (including the
SMOTE-ENN synthetic rows) and ``z_test.npy`` aligns row-for-row with
``y_test.npy`` — the downstream classifiers pair them directly.

Note (evaluation protocol): per the reference methodology and this task's
specification, per-epoch model selection uses the held-out ``X_test``
fold. For a stricter protocol, carve a validation split out of ``X_train``
instead — the orchestration in :func:`main` makes that a one-line change.

Usage (from the repository root)
--------------------------------
.. code-block:: console

    python train_dsae.py                                    # defaults: 25 epochs, CPU
    python train_dsae.py --epochs 40 --batch-size 512 --lr 5e-4
    python train_dsae.py --device auto -v                   # GPU when available, DEBUG logs

Dependencies: ``torch``, ``numpy``; imports the model from ``src/autoencoder``.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.autoencoder import DeepSparseAutoencoder, SparseLoss

# --------------------------------------------------------------------------- #
# Public configuration                                                         #
# --------------------------------------------------------------------------- #

LOGGER: Final[logging.Logger] = logging.getLogger("iot23.train_dsae")

#: Training hyper-parameters (Dutta et al., 2020 baseline).
DEFAULT_EPOCHS: Final[int] = 25
DEFAULT_BATCH_SIZE: Final[int] = 256
DEFAULT_LEARNING_RATE: Final[float] = 1e-3
WEIGHT_DECAY: Final[float] = 1e-5
LATENT_DIM: Final[int] = 32
RHO: Final[float] = 0.05
BETA: Final[float] = 2.0
SCHEDULER_FACTOR: Final[float] = 0.5
SCHEDULER_PATIENCE: Final[int] = 3
SEED: Final[int] = 42

#: Batch size for the validation pass.
EVAL_BATCH_SIZE: Final[int] = 1024
#: Rows encoded per forward chunk during latent extraction (bounds peak memory).
EXTRACTION_CHUNK_SIZE: Final[int] = 4096
#: DEBUG-mode interval between per-batch log lines.
_DEBUG_LOG_INTERVAL: Final[int] = 50

#: Artifacts consumed (from src/dataset.py) and produced by this script.
X_TRAIN_PATH: Final[Path] = Path("data/processed/X_train.npy")
X_TEST_PATH: Final[Path] = Path("data/processed/X_test.npy")
Z_TRAIN_PATH: Final[Path] = Path("data/processed/z_train.npy")
Z_TEST_PATH: Final[Path] = Path("data/processed/z_test.npy")
FULL_MODEL_PATH: Final[Path] = Path("models/saved/dsae_full.pt")
ENCODER_PATH: Final[Path] = Path("models/saved/dsae_encoder.pt")

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
    """Resolve the target device with a safe fallback.

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


def _load_feature_matrix(path: Path, label: str) -> np.ndarray:
    """Load and validate a preprocessed feature matrix (.npy).

    Args:
        path: Artifact path (e.g. ``data/processed/X_train.npy``).
        label: Human-readable name used in logs and error messages.

    Returns:
        Validated matrix, shape ``(n_samples, n_features)``, float32,
        C-contiguous.

    Raises:
        FileNotFoundError: If the artifact is missing (hint: run
            ``python src/dataset.py`` first).
        ValueError: If the matrix is not 2-D, is empty, or contains
            non-finite values.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Required artifact '{path}' was not found — generate it first with "
            f"`python src/dataset.py`."
        )
    matrix = np.load(path)
    LOGGER.info("Loaded %s ← %s (shape=%s, dtype=%s).", label, path, matrix.shape, matrix.dtype)
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be 2-D (n_samples, n_features), got shape {matrix.shape}.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{label} is empty: shape {matrix.shape}.")
    if not np.issubdtype(matrix.dtype, np.floating):
        LOGGER.warning("%s has non-float dtype %s — casting to float32.", label, matrix.dtype)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        n_bad = int(np.count_nonzero(~np.isfinite(matrix)))
        raise ValueError(
            f"{label} contains {n_bad} non-finite (NaN/Inf) entries — regenerate "
            f"the artifacts with `python src/dataset.py`."
        )
    LOGGER.debug("%s value range: [%.4f, %.4f].", label, float(matrix.min()), float(matrix.max()))
    return matrix


# --------------------------------------------------------------------------- #
# Training primitives                                                          #
# --------------------------------------------------------------------------- #


def _run_epoch(
    model: DeepSparseAutoencoder,
    criterion: SparseLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    """Run one training or evaluation epoch over ``loader``.

    Args:
        model: The DSAE (switched to train/eval mode internally).
        criterion: Sparse loss; its ``reconstruction_loss`` and
            ``sparsity_penalty`` helpers provide the logged components.
        loader: Batched feature matrix (batches arrive as plain tensors).
        device: Target compute device.
        optimizer: When provided the epoch *trains* (gradient + update);
            when ``None`` the epoch *evaluates* under ``torch.no_grad()``.

    Returns:
        ``(mean_total, mean_reconstruction, mean_sparsity)`` averaged per
        sample, where ``reconstruction + sparsity`` sums back to ``total``
        (the sparsity component is already beta-weighted).

    Raises:
        RuntimeError: On a non-finite batch loss (numerical divergence) or
            if the loader yields no batches.
    """
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_sum = 0.0
    reconstruction_sum = 0.0
    sparsity_sum = 0.0
    samples_seen = 0

    gradient_context = torch.enable_grad() if training else torch.no_grad()
    with gradient_context:
        for batch_index, x in enumerate(loader):
            x = x.to(device, non_blocking=True)   # (B, input_dim)
            x_hat, z = model(x)                   # (B, input_dim), (B, latent_dim)
            loss = criterion(x, x_hat, z)         # scalar

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"Numerical divergence: non-finite loss ({loss_value}) at "
                    f"batch {batch_index} (training={training})."
                )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if LOGGER.isEnabledFor(logging.DEBUG) and batch_index % _DEBUG_LOG_INTERVAL == 0:
                    LOGGER.debug("batch %04d | batch loss %.6f", batch_index, loss_value)

            batch_size = x.size(0)
            total_sum += loss_value * batch_size
            reconstruction_sum += criterion.reconstruction_loss(x, x_hat).item() * batch_size
            sparsity_sum += criterion.beta * criterion.sparsity_penalty(z).item() * batch_size
            samples_seen += batch_size

    if samples_seen == 0:
        raise RuntimeError(
            "DataLoader yielded no batches — check dataset size vs. batch_size/drop_last."
        )
    return (
        total_sum / samples_seen,
        reconstruction_sum / samples_seen,
        sparsity_sum / samples_seen,
    )


def _fit(
    model: DeepSparseAutoencoder,
    criterion: SparseLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    full_model_path: Path,
    encoder_path: Path,
) -> tuple[float, int]:
    """Run the training loop with plateau LR scheduling and best-model saving.

    After every epoch the model is validated on ``val_loader``; the
    ``ReduceLROnPlateau`` scheduler steps on the validation loss, and
    whenever the loss reaches a new minimum BOTH checkpoints are rewritten:
    the full autoencoder (``full_model_path``) and the encoder-only weights
    (``encoder_path``).

    Args:
        model: DSAE to train (mutated in place).
        criterion: Sparse reconstruction loss.
        optimizer: Adam optimizer.
        scheduler: ReduceLROnPlateau scheduler (``mode='min'``).
        train_loader: Training batches (shuffle=True, drop_last=True).
        val_loader: Validation batches (evaluation on ``X_test``).
        device: Target compute device.
        epochs: Total number of epochs.
        full_model_path: Destination for the best full-model state_dict.
        encoder_path: Destination for the best encoder-only state_dict.

    Returns:
        ``(best_val_loss, best_epoch)`` — the lowest validation loss
        observed and the epoch at which it occurred.
    """
    best_val_loss = math.inf
    best_epoch = 0
    fit_start = time.perf_counter()
    LOGGER.info(
        "Starting training: %d epochs | %d train batches/epoch | %d val batches/epoch.",
        epochs, len(train_loader), len(val_loader),
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_total, train_recon, train_sparsity = _run_epoch(
            model, criterion, train_loader, device, optimizer=optimizer
        )
        val_total, val_recon, val_sparsity = _run_epoch(model, criterion, val_loader, device)
        elapsed = time.perf_counter() - epoch_start

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_total)
        lr_after = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            model.save_full(full_model_path)
            model.save_encoder(encoder_path)
            marker = " | * new best → checkpoints saved"

        LOGGER.info(
            "Epoch %02d/%02d | train %.6f (recon %.6f, sparsity %.6f) | "
            "val %.6f (recon %.6f, sparsity %.6f) | lr %.2e | %6.1fs%s",
            epoch, epochs,
            train_total, train_recon, train_sparsity,
            val_total, val_recon, val_sparsity,
            lr_after, elapsed, marker,
        )
        if lr_after < lr_before:
            LOGGER.info(
                "ReduceLROnPlateau: learning rate reduced %.2e → %.2e (factor=%.2f, patience=%d).",
                lr_before, lr_after, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
            )

    LOGGER.info(
        "Training finished in %.1fs — best val loss %.6f at epoch %d.",
        time.perf_counter() - fit_start, best_val_loss, best_epoch,
    )
    return best_val_loss, best_epoch


# --------------------------------------------------------------------------- #
# Feature extraction & export                                                  #
# --------------------------------------------------------------------------- #


def _extract_latents(
    model: DeepSparseAutoencoder,
    matrix: np.ndarray,
    device: torch.device,
    *,
    label: str,
    chunk_size: int = EXTRACTION_CHUNK_SIZE,
) -> np.ndarray:
    """Encode a feature matrix into latent codes, deterministically.

    The matrix is processed in ``chunk_size`` row blocks so peak memory
    stays bounded regardless of dataset size. The model is switched to
    ``eval()`` first (BatchNorm running statistics, dropout disabled) and
    ``model.encode(..., detach=True)`` runs under ``torch.no_grad()``
    internally — deterministic, inference-grade features.

    Args:
        model: Trained DSAE.
        matrix: Feature matrix, shape ``(N, input_dim)``, float32.
        device: Compute device (chunks are moved per block; results are
            gathered back on the CPU).
        label: Human-readable name for logs.
        chunk_size: Rows per forward block.

    Returns:
        Latent codes, shape ``(N, latent_dim)``, float32, values in (0, 1).
    """
    model.eval()
    blocks: list[np.ndarray] = []
    for start in range(0, matrix.shape[0], chunk_size):
        block = torch.from_numpy(matrix[start : start + chunk_size]).to(device)
        with torch.no_grad():
            z = model.encode(block, detach=True)  # (b, latent_dim)
        blocks.append(z.cpu().numpy())
    latents = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    LOGGER.info(
        "Encoded %s: %s → %s | mean activation %.4f (target rho=%.2f) | value range [%.4f, %.4f].",
        label, matrix.shape, latents.shape, float(latents.mean()), RHO,
        float(latents.min()), float(latents.max()),
    )
    return latents


def _persist_latents(
    latents: np.ndarray,
    path: Path,
    expected_rows: int,
    expected_dim: int,
    label: str,
    aligns_with: str,
) -> None:
    """Validate and persist a latent matrix to disk.

    Args:
        latents: Latent codes, shape ``(N, latent_dim)``.
        path: Destination ``.npy`` file.
        expected_rows: Row count that must match the source feature matrix.
        expected_dim: Latent dimensionality that must match.
        label: Human-readable name for logs.
        aligns_with: Companion artifact the rows align with (for logs).

    Raises:
        ValueError: If shape, finiteness or Sigmoid-bound checks fail.
    """
    if latents.shape != (expected_rows, expected_dim):
        raise ValueError(
            f"{label} has shape {latents.shape}, expected {(expected_rows, expected_dim)}."
        )
    if not np.isfinite(latents).all():
        raise ValueError(f"{label} contains non-finite values.")
    if latents.min() < 0.0 or latents.max() > 1.0:
        raise ValueError(f"{label} activations escaped [0, 1] — Sigmoid bottleneck violated.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, latents)
    LOGGER.info(
        "Saved %s → %s (shape=%s, dtype=%s) — rows align with %s.",
        label, path, latents.shape, latents.dtype, aligns_with,
    )


# --------------------------------------------------------------------------- #
# Console entry point                                                          #
# --------------------------------------------------------------------------- #


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the DSAE training script."""
    parser = argparse.ArgumentParser(
        prog="python train_dsae.py",
        description=(
            "Train the Deep Sparse Autoencoder (DSAE) of Dutta et al. (2020) on the "
            "artifacts produced by src/dataset.py, checkpoint the best model, and "
            "export 32-d sparse latent vectors (z_train.npy / z_test.npy) for the "
            "Level-0 classifiers in src/classifiers.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Training mini-batch size (DataLoader: shuffle=True, drop_last=True).")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE,
                        help="Adam learning rate.")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM,
                        help="DSAE bottleneck dimensionality.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                        help="Target device; 'cuda' falls back to CPU with a warning "
                             "if unavailable, 'auto' selects CUDA when present.")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Torch seed for reproducible init and shuffling.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging (per-batch losses).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full DSAE training pipeline.

    Stages: load artifacts → train (Adam + ReduceLROnPlateau, best-checkpoint
    selection on ``X_test``) → restore best weights → extract and persist
    latent vectors.

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
    if args.lr <= 0.0:
        parser.error("--lr must be positive.")
    if args.latent_dim < 1:
        parser.error("--latent-dim must be >= 1.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.manual_seed(args.seed)

    try:
        device = _resolve_device(args.device)
        LOGGER.info("=" * 78)
        LOGGER.info("DSAE training — Dutta et al. (2020) deep sparse autoencoder")
        LOGGER.info(
            "epochs=%d | batch_size=%d | lr=%.2e | weight_decay=%.1e | "
            "latent_dim=%d | rho=%.2f | beta=%.1f | device=%s | seed=%d",
            args.epochs, args.batch_size, args.lr, WEIGHT_DECAY,
            args.latent_dim, RHO, BETA, device, args.seed,
        )
        LOGGER.info("=" * 78)

        for artifact in (FULL_MODEL_PATH, ENCODER_PATH, Z_TRAIN_PATH, Z_TEST_PATH):
            if artifact.exists():
                LOGGER.warning("Overwriting existing artifact: %s", artifact)

        # -- Stage 1: data loading (auto-detect input dimension) ------------ #
        with _stage_timer("Feature-matrix loading"):
            x_train = _load_feature_matrix(X_TRAIN_PATH, "X_train")
            x_test = _load_feature_matrix(X_TEST_PATH, "X_test")
            if x_train.shape[1] != x_test.shape[1]:
                raise ValueError(
                    f"Feature-dimension mismatch: X_train has {x_train.shape[1]} "
                    f"features, X_test has {x_test.shape[1]}."
                )
            if x_test.shape[0] < 1:
                raise ValueError("X_test is empty — nothing to validate or encode.")
            if x_train.shape[0] < args.batch_size:
                raise ValueError(
                    f"X_train has {x_train.shape[0]} rows < batch_size={args.batch_size}; "
                    f"drop_last=True would yield zero training batches — lower --batch-size."
                )
        input_dim = x_train.shape[1]
        LOGGER.info("Auto-detected input_dim=%d from X_train (shape=%s).", input_dim, x_train.shape)

        # A torch.Tensor is a duck-typed Dataset (implements __len__/__getitem__),
        # so it can be handed to DataLoader directly; batches arrive as tensors.
        train_tensor = torch.from_numpy(x_train)
        val_tensor = torch.from_numpy(x_test)
        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_tensor,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_tensor,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            pin_memory=pin_memory,
        )
        LOGGER.info(
            "DataLoaders ready: train=%d batches × %d (shuffle=True, drop_last=True) | "
            "val=%d batches × ≤%d.",
            len(train_loader), args.batch_size, len(val_loader), EVAL_BATCH_SIZE,
        )

        # -- Stage 2: model, loss, optimizer, scheduler ---------------------- #
        model = DeepSparseAutoencoder(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
        criterion = SparseLoss(rho=RHO, beta=BETA).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
        )
        LOGGER.info("Model: %s | parameters=%d.", model, sum(p.numel() for p in model.parameters()))
        LOGGER.info(
            "Criterion: %s | Adam(lr=%.2e, weight_decay=%.1e) | "
            "ReduceLROnPlateau(mode=min, factor=%.1f, patience=%d).",
            criterion, args.lr, WEIGHT_DECAY, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
        )

        # -- Stage 3: training loop with best-checkpoint selection ----------- #
        with _stage_timer("DSAE training"):
            best_val_loss, best_epoch = _fit(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                full_model_path=FULL_MODEL_PATH,
                encoder_path=ENCODER_PATH,
            )

        # -- Stage 4: restore best weights & extract latent codes ------------ #
        model.load_full(FULL_MODEL_PATH)
        LOGGER.info(
            "Restored best checkpoint (epoch %d, val loss %.6f) for latent extraction.",
            best_epoch, best_val_loss,
        )
        with _stage_timer("Latent extraction"):
            z_train = _extract_latents(model, x_train, device, label="X_train")
            z_test = _extract_latents(model, x_test, device, label="X_test")

        neuron_means = z_train.mean(axis=0)
        n_active = int((neuron_means > 2.0 * RHO).sum())
        LOGGER.info(
            "Sparsity diagnostics: %d/%d latent neurons exceed 2·rho mean activation "
            "(max neuron mean %.4f).",
            n_active, z_train.shape[1], float(neuron_means.max()),
        )

        # -- Stage 5: persist latent artifacts ------------------------------- #
        _persist_latents(
            z_train, Z_TRAIN_PATH, x_train.shape[0], args.latent_dim, "z_train", "y_train.npy"
        )
        _persist_latents(
            z_test, Z_TEST_PATH, x_test.shape[0], args.latent_dim, "z_test", "y_test.npy"
        )

        LOGGER.info("=" * 78)
        LOGGER.info("DSAE pipeline complete — artifacts written:")
        for artifact in (FULL_MODEL_PATH, ENCODER_PATH, Z_TRAIN_PATH, Z_TEST_PATH):
            LOGGER.info("  %s (%.1f KiB)", artifact, artifact.stat().st_size / 1024.0)
        LOGGER.info(
            "Next stage: train the Level-0 learners of src/classifiers.py on "
            "(z_train.npy, y_train.npy) and evaluate on (z_test.npy, y_test.npy)."
        )
        return 0
    except Exception:
        LOGGER.exception("DSAE training failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())