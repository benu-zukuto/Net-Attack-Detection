"""Deep Sparse Autoencoder (DSAE) for IoT-23 network-flow compression.

This module implements the unsupervised representation-learning stage of the
deep-learning NIDS ensemble of *Dutta et al. (2020)* — "A Deep Learning
Ensemble for Network Anomaly and Cyber-Attack Detection". The autoencoder
consumes the preprocessed feature matrix emitted by ``src/dataset.py``
(RobustScaled numeric block + one-hot categorical block) and compresses each
network-flow vector into a 32-dimensional, Sigmoid-bounded, sparsity-
regularised latent code ``z``. Downstream DNN and LSTM classifiers operate
in this latent space.

Architecture (symmetric hourglass)
----------------------------------
::

    Encoder                              Decoder
    ────────────────────────────────     ────────────────────────────────
    x  (B, input_dim)
    ├─ Linear(input_dim → 128)           z  (B, latent_dim)
    ├─ BatchNorm1d(128)                  ├─ Linear(latent_dim → 64)
    ├─ ReLU                              ├─ BatchNorm1d(64)
    ├─ Dropout(0.1)                      ├─ ReLU
    ├─ Linear(128 → 64)                  ├─ Linear(64 → 128)
    ├─ BatchNorm1d(64)                   ├─ BatchNorm1d(128)
    ├─ ReLU                              ├─ ReLU
    ├─ Linear(64 → latent_dim)           ├─ Dropout(0.1)
    └─ Sigmoid  →  z ∈ (0, 1)            └─ Linear(128 → input_dim)
                                        └─ identity output head

The bottleneck ``Sigmoid`` is CRITICAL: the KL sparsity term is only defined
on activations in (0, 1) (each latent neuron is modelled as a Bernoulli
variable with mean ``rho``). The decoder's identity head matches the
RobustScaled — i.e. unbounded, roughly centred — input distribution.

Loss
----
::

    L(x, x_hat, z) = MSE(x, x_hat) + beta * Σ_j KL(rho ‖ rho_hat_j)

    KL(rho ‖ rho_hat_j) = rho·log(rho / rho_hat_j)
                          + (1 − rho)·log((1 − rho) / (1 − rho_hat_j))

with ``rho_hat_j`` the mean activation of latent neuron ``j`` over the
mini-batch (``dim=0``), ``rho = 0.05`` and ``beta = 2.0``. ``rho_hat`` is
clamped to ``[1e-6, 1 − 1e-6]`` before the logarithms for numerical
stability.

Usage
-----
.. code-block:: python

    model = DeepSparseAutoencoder(input_dim=15, latent_dim=32)
    criterion = SparseLoss(rho=0.05, beta=2.0)
    x_hat, z = model(x)          # x: (B, 15) → x_hat: (B, 15), z: (B, 32)
    loss = criterion(x, x_hat, z)
    loss.backward()
    features = model.encode(x)   # (B, 32) detached latent codes

Dependencies: ``torch`` (Python >= 3.10).
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Public configuration                                                         #
# --------------------------------------------------------------------------- #

LOGGER: Final[logging.Logger] = logging.getLogger("iot23.autoencoder")

#: Width of the first (widest) hidden layer of the symmetric hourglass.
ENCODER_HIDDEN_DIM_1: Final[int] = 128
#: Width of the second hidden layer, immediately before the bottleneck.
ENCODER_HIDDEN_DIM_2: Final[int] = 64
#: Latent bottleneck dimensionality (Dutta et al., 2020 baseline).
DEFAULT_LATENT_DIM: Final[int] = 32
#: Dropout probability applied to the widest hidden layer of each stack.
DROPOUT_PROBABILITY: Final[float] = 0.1

#: Target sparsity parameter rho — desired mean activation per latent neuron.
DEFAULT_RHO: Final[float] = 0.05
#: Sparsity penalty weight beta.
DEFAULT_BETA: Final[float] = 2.0
#: Epsilon used to clamp rho_hat into [eps, 1 - eps] before the KL logarithms.
RHO_HAT_EPSILON: Final[float] = 1e-6

__all__: Final[list[str]] = [
    "DeepSparseAutoencoder",
    "SparseLoss",
    "DEFAULT_LATENT_DIM",
]


# --------------------------------------------------------------------------- #
# Internal utilities                                                           #
# --------------------------------------------------------------------------- #


def _load_state_dict_file(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a state-dict checkpoint from disk, CPU-mapped and safely.

    ``weights_only=True`` (PyTorch >= 1.13) prevents arbitrary unpickling;
    older PyTorch versions fall back to the legacy loader. Mapping to CPU
    first lets ``load_state_dict`` copy tensors onto whatever device the
    target module already lives on.

    Args:
        path: Checkpoint file produced by ``torch.save(state_dict, …)``.

    Returns:
        The loaded state dict.

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


# --------------------------------------------------------------------------- #
# Custom sparsity loss (spec §1)                                               #
# --------------------------------------------------------------------------- #


class SparseLoss(nn.Module):
    """MSE reconstruction loss + KL sparsity penalty (Dutta et al., 2020).

    Total objective::

        L(x, x_hat, z) = MSE(x, x_hat) + beta * Σ_j KL(rho ‖ rho_hat_j)

        KL(rho ‖ rho_hat_j) = rho·log(rho / rho_hat_j)
                              + (1 − rho)·log((1 − rho) / (1 − rho_hat_j))

    where ``rho_hat_j`` is the mean activation of latent neuron ``j`` across
    the mini-batch (``dim=0``). The KL term drives the average activation of
    every bottleneck neuron towards ``rho`` so that only a small subset of
    latent units fires for any given network flow — the sparse code consumed
    by the downstream DNN/LSTM classifiers.

    Reduction semantics (reference formulation): the MSE term is averaged
    over batch and feature dimensions, while the KL term is **summed** over
    the ``latent_dim`` neurons. The constant terms (``rho·log(rho)`` and
    ``(1 − rho)·log(1 − rho)``) do not depend on the data; they are retained
    so the reported loss equals the true objective value.

    Numerical-stability guarantees:
        * ``rho_hat`` is clamped to ``[1e-6, 1 − 1e-6]`` **before** the
          logarithms, so saturated Sigmoid activations (``rho_hat → 0`` or
          ``→ 1``) can never produce ``log(0) → −inf`` or division overflow.
        * Constant terms are evaluated once, in double precision, via
          ``math.log``.
        * ``torch.log1p(-rho_hat)`` is used for the second log term.
        * The formulation remains safe under AMP / half precision.

    Expected tensor shapes:
        * ``x``:      ``(B, input_dim)``  — ground-truth inputs.
        * ``x_hat``:  ``(B, input_dim)``  — reconstructions.
        * ``latent``: ``(B, latent_dim)`` — Sigmoid-bounded codes in (0, 1).
        * Returns:    ``()``              — scalar, differentiable loss.

    Attributes:
        rho: Target sparsity parameter, ``rho ∈ (0, 1)``. Default ``0.05``.
        beta: Sparsity penalty weight, ``beta ≥ 0``. Default ``2.0``.
    """

    def __init__(self, rho: float = DEFAULT_RHO, beta: float = DEFAULT_BETA) -> None:
        """Validate and store the sparsity hyper-parameters.

        Args:
            rho: Target average activation of each latent neuron, in (0, 1).
            beta: Weight of the summed KL penalty, non-negative.

        Raises:
            ValueError: If ``rho ∉ (0, 1)`` or ``beta < 0``.
        """
        super().__init__()
        if not 0.0 < rho < 1.0:
            raise ValueError(f"rho must lie in (0, 1), got {rho}.")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}.")
        self.rho = rho
        self.beta = beta

    def extra_repr(self) -> str:
        """Expose the sparsity hyper-parameters in ``repr()``."""
        return f"rho={self.rho}, beta={self.beta}"

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Mean-squared reconstruction error: ``MSE(x, x_hat)``.

        Args:
            x: Ground truth, shape ``(B, input_dim)``.
            x_hat: Reconstruction, shape ``(B, input_dim)``.

        Returns:
            Scalar tensor — mean over batch and feature dimensions.
        """
        return F.mse_loss(x_hat, x, reduction="mean")

    def sparsity_penalty(self, latent: torch.Tensor) -> torch.Tensor:
        """Summed KL divergence ``Σ_j KL(rho ‖ rho_hat_j)`` over latent units.

        Args:
            latent: Sigmoid-bounded codes, shape ``(B, latent_dim)``.

        Returns:
            Scalar tensor — the unweighted sparsity penalty (multiply by
            ``self.beta`` for its contribution to the total loss).
        """
        # Mean activation of every latent neuron across the mini-batch:
        # (B, latent_dim) → (latent_dim,)
        rho_hat = latent.mean(dim=0)
        # Clamp BEFORE the logarithms: prevents log(0) → −inf and division
        # overflow when Sigmoid activations saturate at 0 or 1 (spec §1).
        rho_hat = rho_hat.clamp(min=RHO_HAT_EPSILON, max=1.0 - RHO_HAT_EPSILON)
        kl_divergence = self.rho * (math.log(self.rho) - torch.log(rho_hat)) + (
            1.0 - self.rho
        ) * (math.log(1.0 - self.rho) - torch.log1p(-rho_hat))
        return kl_divergence.sum()

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined sparse-reconstruction loss.

        Args:
            x: Ground-truth inputs, shape ``(B, input_dim)``.
            x_hat: Reconstructions from ``DeepSparseAutoencoder.forward``.
            latent: Sigmoid-bounded latent codes, shape ``(B, latent_dim)``.

        Returns:
            Scalar tensor ``MSE(x, x_hat) + beta · Σ_j KL(rho ‖ rho_hat_j)``.
        """
        reconstruction = self.reconstruction_loss(x, x_hat)
        sparsity = self.sparsity_penalty(latent)
        return reconstruction + self.beta * sparsity


# --------------------------------------------------------------------------- #
# Model (spec §2 / §3)                                                         #
# --------------------------------------------------------------------------- #


class DeepSparseAutoencoder(nn.Module):
    """Deep Sparse Autoencoder (DSAE) — Dutta et al. (2020) baseline.

    Compresses preprocessed network-flow feature vectors into a
    ``latent_dim``-dimensional sparse representation ``z`` that serves as
    the input space for the downstream DNN and LSTM classifiers.

    Design notes:
        * CRITICAL — the bottleneck ``Sigmoid`` keeps every latent activation
          strictly inside (0, 1), the support the KL sparsity term is defined
          on. Without it, ``rho_hat`` could leave [0, 1] and the penalty
          would be mathematically undefined.
        * The decoder ends with a plain ``Linear`` (identity head): the
          preprocessed inputs are RobustScaled numerics (unbounded, roughly
          centred) plus binary one-hot dummies, so a squashing output head
          would systematically bias the reconstruction.
        * ``BatchNorm1d`` keeps activations well-scaled through the deep
          stack. During training keep the mini-batch size > 1 (use
          ``drop_last=True`` for the final partial batch); call ``eval()``
          before feature extraction so running statistics are used and
          dropout is disabled.

    Args:
        input_dim: Dimensionality of the preprocessed feature vector,
            inferred from the dataset artifacts (e.g. the length of
            ``feature_names`` saved by ``src/dataset.py``).
        latent_dim: Bottleneck dimensionality. Defaults to ``32``.

    Raises:
        ValueError: If ``input_dim`` or ``latent_dim`` is not positive.
    """

    def __init__(self, input_dim: int, latent_dim: int = DEFAULT_LATENT_DIM) -> None:
        """Build the symmetric encoder/decoder stacks."""
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: (B, input_dim) → (B, latent_dim), values in (0, 1).
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, ENCODER_HIDDEN_DIM_1),            # (B, 128)
            nn.BatchNorm1d(ENCODER_HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_PROBABILITY),
            nn.Linear(ENCODER_HIDDEN_DIM_1, ENCODER_HIDDEN_DIM_2),  # (B, 64)
            nn.BatchNorm1d(ENCODER_HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(ENCODER_HIDDEN_DIM_2, latent_dim),           # (B, latent_dim)
            nn.Sigmoid(),  # CRITICAL: bounds z in (0, 1) for the KL term.
        )

        # Decoder: mirror of the encoder, (B, latent_dim) → (B, input_dim).
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, ENCODER_HIDDEN_DIM_2),           # (B, 64)
            nn.BatchNorm1d(ENCODER_HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(ENCODER_HIDDEN_DIM_2, ENCODER_HIDDEN_DIM_1),  # (B, 128)
            nn.BatchNorm1d(ENCODER_HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_PROBABILITY),
            nn.Linear(ENCODER_HIDDEN_DIM_1, input_dim),            # (B, input_dim)
        )

    def extra_repr(self) -> str:
        """Summarise the geometry in ``repr()``."""
        return (
            f"input_dim={self.input_dim}, latent_dim={self.latent_dim}, "
            f"hidden_dims=({ENCODER_HIDDEN_DIM_1}, {ENCODER_HIDDEN_DIM_2})"
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full autoencoder pass.

        Shape transformations (``D = input_dim``, ``L = latent_dim``)::

            x: (B, D) → encoder → z: (B, L), z ∈ (0, 1)
            z: (B, L) → decoder → x_hat: (B, D)

        Args:
            x: Input mini-batch, shape ``(B, input_dim)``. In training mode
                ``B`` must be greater than 1 (``BatchNorm1d`` limitation).

        Returns:
            ``(x_hat, z)`` — the reconstruction ``(B, input_dim)`` and the
            Sigmoid-bounded latent code ``(B, latent_dim)``.
        """
        z = self.encoder(x)      # (B, latent_dim), values in (0, 1)
        x_hat = self.decoder(z)  # (B, input_dim)
        return x_hat, z

    def encode(self, x: torch.Tensor, *, detach: bool = True) -> torch.Tensor:
        """Project inputs into the sparse latent space (feature extraction).

        Shape transformation: ``(B, input_dim) → (B, latent_dim)``.

        This is the API the downstream DNN/LSTM classifiers consume. Call
        ``self.eval()`` first so that BatchNorm uses running statistics and
        dropout is disabled, i.e. extracted features are deterministic.

        Args:
            x: Input mini-batch, shape ``(B, input_dim)``.
            detach: When ``True`` (default) the codes are computed under
                ``torch.no_grad()`` and returned graph-free — the safe,
                memory-friendly path for frozen-encoder feature extraction.
                Set ``False`` to keep the autograd graph for end-to-end
                fine-tuning through the encoder.

        Returns:
            Latent codes, shape ``(B, latent_dim)``, values in (0, 1).
        """
        if detach:
            with torch.no_grad():
                return self.encoder(x)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input-space features from latent codes.

        Shape transformation: ``(B, latent_dim) → (B, input_dim)``.

        Args:
            z: Latent codes, shape ``(B, latent_dim)``.

        Returns:
            Reconstructions, shape ``(B, input_dim)``.
        """
        return self.decoder(z)

    # -- Persistence -------------------------------------------------------- #

    def save_encoder(self, path: str | Path) -> None:
        """Persist ONLY the encoder stack — the downstream-facing artifact.

        Saves ``self.encoder.state_dict()``: the Linear/BatchNorm parameters
        and running statistics of the mapping
        ``(B, input_dim) → (B, latent_dim)``.

        Args:
            path: Destination file (parent directories are created).
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = self.encoder.state_dict()
        torch.save(state, destination)
        LOGGER.info("Saved encoder state_dict (%d tensors) → %s", len(state), destination)

    def load_encoder(self, path: str | Path) -> None:
        """Load encoder weights from a checkpoint made by ``save_encoder``.

        Tensors are CPU-mapped first and copied into the existing
        parameters, so this works regardless of the module's device.

        Args:
            path: Checkpoint file produced by :meth:`save_encoder`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            RuntimeError: If the checkpoint does not match the encoder
                architecture (strict loading).
        """
        self.encoder.load_state_dict(_load_state_dict_file(path), strict=True)
        LOGGER.info("Loaded encoder weights ← %s", path)

    def save_full(self, path: str | Path) -> None:
        """Persist the complete autoencoder (encoder + decoder).

        Convenience for resuming AE training; the downstream classifiers
        only need :meth:`save_encoder`.

        Args:
            path: Destination file (parent directories are created).
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination)
        LOGGER.info("Saved full autoencoder state_dict → %s", destination)

    def load_full(self, path: str | Path) -> None:
        """Restore the complete autoencoder from ``save_full`` output.

        Args:
            path: Checkpoint file.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            RuntimeError: On architecture mismatch (strict loading).
        """
        self.load_state_dict(_load_state_dict_file(path), strict=True)
        LOGGER.info("Loaded full autoencoder weights ← %s", path)


# --------------------------------------------------------------------------- #
# Smoke test (spec §4)                                                         #
# --------------------------------------------------------------------------- #


def _run_smoke_test() -> int:
    """Verify the DSAE and ``SparseLoss`` end-to-end on synthetic data.

    Checks:
        1. Forward-pass shapes and Sigmoid bounds of the latent code.
        2. ``SparseLoss`` yields a finite scalar; ``loss.backward()``
           produces finite (non-NaN, non-Inf) gradients on every parameter.
        3. Numerical-stability probes: fully saturated latent codes
           (all-zero / all-one) still yield finite penalties and gradients.
        4. Deterministic, detached feature extraction in eval mode.
        5. ``save_encoder`` / ``load_encoder`` round-trip fidelity.
        6. Constructor argument validation.

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    input_dim, latent_dim, batch_size = 15, DEFAULT_LATENT_DIM, 64
    numeric_block = 7  # mirrors the 7 RobustScaled numeric features of dataset.py

    LOGGER.info("=" * 78)
    LOGGER.info("DSAE smoke test — device=%s | torch=%s", device, torch.__version__)
    LOGGER.info("=" * 78)

    try:
        model = DeepSparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
        criterion = SparseLoss().to(device)
        LOGGER.info("Model: %s", model)
        LOGGER.info(
            "Parameters: total=%d (encoder=%d, decoder=%d)",
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.encoder.parameters()),
            sum(p.numel() for p in model.decoder.parameters()),
        )
        LOGGER.info("Criterion: %s", criterion)

        # Synthetic mini-batch mimicking the dataset.py feature layout:
        # [RobustScaled numerics ~ N(0, 1) | one-hot dummies ∈ {0, 1}].
        x_numeric = torch.randn(batch_size, numeric_block)
        x_categorical = torch.bernoulli(
            torch.full((batch_size, input_dim - numeric_block), 0.3)
        )
        x = torch.cat([x_numeric, x_categorical], dim=1).to(device)
        LOGGER.info("Synthetic batch: shape=%s, dtype=%s", tuple(x.shape), x.dtype)

        # -- 1) Forward pass: shapes and Sigmoid bounds ---------------------- #
        model.train()
        x_hat, z = model(x)
        assert tuple(x_hat.shape) == (batch_size, input_dim), f"bad reconstruction shape {tuple(x_hat.shape)}"
        assert tuple(z.shape) == (batch_size, latent_dim), f"bad latent shape {tuple(z.shape)}"
        assert z.min().item() >= 0.0 and z.max().item() <= 1.0, "latent codes must be Sigmoid-bounded"
        LOGGER.info(
            "Forward OK — x_hat%s, z%s | latent stats: mean=%.4f (target rho=%.2f), min=%.4f, max=%.4f",
            tuple(x_hat.shape), tuple(z.shape), z.mean().item(), criterion.rho,
            z.min().item(), z.max().item(),
        )

        # -- 2) Loss + backward: no NaN / Inf anywhere ----------------------- #
        loss = criterion(x, x_hat, z)
        assert loss.dim() == 0 and torch.isfinite(loss).all(), "loss must be a finite scalar"
        model.zero_grad(set_to_none=True)
        loss.backward()
        checked = 0
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"missing gradient for {name}"
            assert torch.isfinite(parameter.grad).all(), f"non-finite gradient in {name}"
            checked += 1
        max_grad = max(p.grad.abs().max().item() for p in model.parameters())  # type: ignore[union-attr]
        LOGGER.info(
            "SparseLoss=%.6f | backward OK | %d parameter tensors verified finite | max|grad|=%.3e",
            loss.item(), checked, max_grad,
        )

        # -- 3) Numerical-stability probes on saturated latents -------------- #
        for label, factory in (("all-zero", torch.zeros), ("all-one", torch.ones)):
            z_sat = factory((batch_size, latent_dim), device=device, requires_grad=True)
            penalty = criterion.sparsity_penalty(z_sat)
            assert torch.isfinite(penalty).all(), f"{label} latent produced a non-finite penalty"
            penalty.backward()
            assert torch.isfinite(z_sat.grad).all(), f"{label} latent produced non-finite gradients"
        LOGGER.info(
            "Stability probes OK — clamping rho_hat to [%.0e, %.6f] keeps the KL finite for saturated codes.",
            RHO_HAT_EPSILON, 1.0 - RHO_HAT_EPSILON,
        )

        # -- 4) Deterministic, detached feature extraction (eval mode) ------- #
        model.eval()
        z_a = model.encode(x)
        z_b = model.encode(x)
        assert z_a.shape == (batch_size, latent_dim), "encode() returned wrong shape"
        assert not z_a.requires_grad, "default encode() must return detached codes"
        assert torch.equal(z_a, z_b), "encode() must be deterministic in eval mode"
        LOGGER.info("Feature extraction OK — encode(x) → %s, deterministic in eval mode.", tuple(z_a.shape))

        # -- 5) Encoder persistence round-trip ------------------------------- #
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "dsae_encoder.pt"
            model.save_encoder(checkpoint)
            replica = DeepSparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
            replica.load_encoder(checkpoint)
            replica.eval()
            z_replica = replica.encode(x)
            assert torch.allclose(z_a, z_replica, atol=1e-6), "round-trip encoder weights diverge"
            LOGGER.info("save_encoder/load_encoder round-trip OK → %s", checkpoint.name)

        # -- 6) Constructor argument validation ------------------------------ #
        for bad_kwargs in ({"rho": 0.0}, {"rho": 1.0}, {"beta": -1.0}):
            try:
                SparseLoss(**bad_kwargs)
            except ValueError:
                continue
            raise AssertionError(f"SparseLoss accepted invalid arguments {bad_kwargs}")
        for bad_kwargs in ({"input_dim": 0}, {"input_dim": 15, "latent_dim": 0}):
            try:
                DeepSparseAutoencoder(**bad_kwargs)
            except ValueError:
                continue
            raise AssertionError(f"DeepSparseAutoencoder accepted invalid arguments {bad_kwargs}")
        LOGGER.info("Argument validation OK — invalid rho/beta/dims rejected.")

        LOGGER.info("=" * 78)
        LOGGER.info("SMOKE TEST PASSED — forward, SparseLoss.backward, stability probes and persistence are NaN/Inf-free.")
        return 0
    except Exception:
        LOGGER.exception("SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_smoke_test())