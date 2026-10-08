"""Level-0 base learners for the IoT-23 intrusion-detection ensemble.

Implements the two deep base classifiers of the ensemble of *Dutta et al.
(2020)* — "A Deep Learning Ensemble for Network Anomaly and Cyber-Attack
Detection" — whose posteriors are later combined by the Level-1 stacking
stage:

* :class:`SpatialDNN` — per-flow, fully-connected learner that scores the
  *spatial* structure of each network flow independently.
* :class:`TemporalLSTM` — stacked-LSTM sequence learner that scores the
  *temporal* structure of sliding windows of consecutive flows.

Both models consume the 32-dimensional sparse latent vectors ``z`` produced
by the Deep Sparse Autoencoder of ``src/autoencoder.py`` and emit raw
(unnormalised) logits over the five IoT-23 threat categories:

=====  =====================
Class  Category
=====  =====================
0      Benign
1      Recon / PortScan
2      DoS / DDoS
3      Malware / Okiru
4      Botnet C&C
=====  =====================

Both models expose ``predict_proba`` (softmax-normalised posteriors along
``dim=-1``) — the interface the Level-1 meta-learner consumes.

Usage
-----
.. code-block:: python

    dnn = SpatialDNN()                                # (B, 32) → (B, 5)
    logits = dnn(z)                                   # raw logits
    probs = dnn.predict_proba(z)                      # rows sum to 1.0

    X_seq, y_seq = create_lstm_sequences(z_np, y_np)  # (N, 32) → (M, 5, 32)
    lstm = TemporalLSTM()                             # (B, 5, 32) → (B, 5)
    seq_logits = lstm(torch.from_numpy(X_seq))

Dependencies: ``torch``, ``numpy`` >= 1.20 (Python >= 3.10).
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Public configuration                                                         #
# --------------------------------------------------------------------------- #

LOGGER: Final[logging.Logger] = logging.getLogger("iot23.classifiers")

#: Dimensionality of the DSAE latent vectors consumed by both learners.
DEFAULT_LATENT_DIM: Final[int] = 32
#: Number of IoT-23 threat categories (classes 0–4).
DEFAULT_NUM_CLASSES: Final[int] = 5
#: Default sliding-window length for the temporal learner.
DEFAULT_SEQ_LEN: Final[int] = 5

#: Threat taxonomy — kept in sync with ``CLASS_NAMES`` in ``src/dataset.py``
#: (duplicated so this module stays decoupled from the data pipeline).
CLASS_NAMES: Final[dict[int, str]] = {
    0: "Benign",
    1: "Recon / PortScan",
    2: "DoS / DDoS",
    3: "Malware / Okiru",
    4: "Botnet C&C",
}

# SpatialDNN geometry.
DNN_HIDDEN_DIM_1: Final[int] = 128
DNN_HIDDEN_DIM_2: Final[int] = 64
DNN_HIDDEN_DIM_3: Final[int] = 32
DNN_DROPOUT_1: Final[float] = 0.3
DNN_DROPOUT_2: Final[float] = 0.2

# TemporalLSTM geometry.
LSTM_HIDDEN_SIZE: Final[int] = 64
LSTM_NUM_LAYERS: Final[int] = 2
LSTM_DROPOUT: Final[float] = 0.2
LSTM_HEAD_HIDDEN_DIM: Final[int] = 32

__all__: Final[list[str]] = [
    "SpatialDNN",
    "TemporalLSTM",
    "create_lstm_sequences",
    "CLASS_NAMES",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_SEQ_LEN",
]


# --------------------------------------------------------------------------- #
# Model 1: SpatialDNN (spec §1)                                                #
# --------------------------------------------------------------------------- #


class SpatialDNN(nn.Module):
    """Spatial (per-flow) Level-0 base learner — Dutta et al. (2020).

    Classifies each network flow *independently* from its 32-dimensional
    DSAE latent vector ``z`` (``src/autoencoder.py``): a fully-connected
    hourglass that reads the sparse code's spatial structure and emits raw
    logits over the five threat categories.

    Architecture (shapes for a mini-batch of ``B`` flows;
    ``D = latent_dim``, ``K = num_classes``)::

        z  (B, 32)
        ├─ Linear(32 → 128)       (B, 128)
        ├─ BatchNorm1d(128)
        ├─ ReLU
        ├─ Dropout(0.3)           ← heavier regularisation on the widest layer
        ├─ Linear(128 → 64)       (B, 64)
        ├─ BatchNorm1d(64)
        ├─ ReLU
        ├─ Dropout(0.2)
        ├─ Linear(64 → 32)        (B, 32)
        ├─ ReLU
        └─ Linear(32 → K)         (B, K)   raw, unnormalised logits

    Training notes:
        * The output is **raw logits** — train with
          ``torch.nn.CrossEntropyLoss`` (log-softmax + NLL are fused
          internally); do not pair with an extra softmax/LogSoftmax.
        * ``BatchNorm1d`` requires ``B > 1`` in training mode (use
          ``drop_last=True`` for the final partial batch).

    Args:
        latent_dim: Dimensionality of the input latent code. Default 32.
        num_classes: Number of output threat categories. Default 5.

    Raises:
        ValueError: If ``latent_dim < 1`` or ``num_classes < 2``.
    """

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ) -> None:
        """Build the fully-connected classifier stack."""
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}.")
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.network = nn.Sequential(
            nn.Linear(latent_dim, DNN_HIDDEN_DIM_1),          # (B, 128)
            nn.BatchNorm1d(DNN_HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(p=DNN_DROPOUT_1),
            nn.Linear(DNN_HIDDEN_DIM_1, DNN_HIDDEN_DIM_2),    # (B, 64)
            nn.BatchNorm1d(DNN_HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(p=DNN_DROPOUT_2),
            nn.Linear(DNN_HIDDEN_DIM_2, DNN_HIDDEN_DIM_3),    # (B, 32)
            nn.ReLU(),
            nn.Linear(DNN_HIDDEN_DIM_3, num_classes),         # (B, K) raw logits
        )

    def extra_repr(self) -> str:
        """Summarise the geometry in ``repr()``."""
        return f"latent_dim={self.latent_dim}, num_classes={self.num_classes}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of latent vectors.

        Shape transformation: ``(B, latent_dim) → (B, num_classes)``.

        Args:
            x: Mini-batch of DSAE latent codes, shape ``(B, latent_dim)``
                with values in (0, 1).

        Returns:
            Raw, unnormalised logits, shape ``(B, num_classes)``.

        Raises:
            ValueError: If ``x`` is not 2-D or its feature dimension does
                not match ``latent_dim``.
        """
        if x.dim() != 2:
            raise ValueError(
                f"SpatialDNN expects a 2-D batch (batch_size, latent_dim="
                f"{self.latent_dim}), got shape {tuple(x.shape)}."
            )
        if x.size(1) != self.latent_dim:
            raise ValueError(
                f"Feature-dimension mismatch: expected latent_dim="
                f"{self.latent_dim}, got {x.size(1)}."
            )
        return self.network(x)  # (B, num_classes)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Class-membership probabilities for the Level-1 stacking stage.

        Shape transformation: ``(B, latent_dim) → (B, num_classes)``;
        rows are non-negative and sum to 1.0 along ``dim=-1``.

        Args:
            x: Mini-batch of DSAE latent codes, shape ``(B, latent_dim)``.

        Returns:
            Softmax-normalised posteriors, shape ``(B, num_classes)``,
            computed under ``torch.no_grad()``. Call ``self.eval()`` first
            for deterministic outputs (BatchNorm running statistics, no
            dropout).
        """
        return F.softmax(self.forward(x), dim=-1)  # (B, num_classes)


# --------------------------------------------------------------------------- #
# Model 2: TemporalLSTM (spec §2)                                              #
# --------------------------------------------------------------------------- #


class TemporalLSTM(nn.Module):
    """Temporal Level-0 base learner — stacked LSTM over flow windows.

    Models sequential structure across *consecutive* network flows. Each
    input is a sliding time-window tensor of DSAE latent vectors (built by
    :func:`create_lstm_sequences`). Windows are treated as independent
    sequences with a zero-initialised hidden state (stateless), and the
    hidden state of the **final timestep** is projected to raw logits — the
    model predicts the threat category of the most recent flow given its
    short-term traffic context.

    Architecture (shapes for a mini-batch of ``B`` windows of length
    ``L = seq_len``; ``D = latent_dim``, ``H = 64``, ``K = num_classes``)::

        x  (B, L, 32)
        ├─ LSTM(input=32, hidden=64, num_layers=2, batch_first=True,
        │        dropout=0.2)             out: (B, L, 64)   [dropout acts
        │                                                     between LSTM layers]
        ├─ out[:, -1, :]                  (B, 64)   final timestep
        ├─ Linear(64 → 32)                (B, 32)
        ├─ ReLU
        └─ Linear(32 → K)                 (B, K)    raw, unnormalised logits

    For full-length (unpacked) sequences, ``out[:, -1, :]`` equals the last
    LSTM layer's final hidden state ``h_n[-1]``. The canonical input is
    ``(B, 5, 32)`` but any sequence length >= 1 is accepted.

    Args:
        latent_dim: Dimensionality of each flow's latent code (the LSTM
            ``input_size``). Default 32.
        num_classes: Number of output threat categories. Default 5.

    Raises:
        ValueError: If ``latent_dim < 1`` or ``num_classes < 2``.
    """

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ) -> None:
        """Build the stacked LSTM and its classification head."""
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}.")
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # dropout is applied between the two LSTM layers (num_layers=2).
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=LSTM_DROPOUT,
        )
        self.head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE, LSTM_HEAD_HIDDEN_DIM),  # (B, 32)
            nn.ReLU(),
            nn.Linear(LSTM_HEAD_HIDDEN_DIM, num_classes),       # (B, K) raw logits
        )

    def extra_repr(self) -> str:
        """Summarise the geometry in ``repr()``."""
        return (
            f"latent_dim={self.latent_dim}, num_classes={self.num_classes}, "
            f"hidden_size={LSTM_HIDDEN_SIZE}, num_layers={LSTM_NUM_LAYERS}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of sliding-window sequences.

        Shape transformation:
        ``(B, seq_len, latent_dim) → (B, num_classes)``.

        Args:
            x: Mini-batch of latent-code windows, shape
                ``(B, seq_len, latent_dim)`` with values in (0, 1);
                ``seq_len`` may be any length >= 1 (canonical: 5).

        Returns:
            Raw, unnormalised logits computed from the final timestep
            ``out[:, -1, :]``, shape ``(B, num_classes)``.

        Raises:
            ValueError: If ``x`` is not 3-D, its feature dimension does not
                match ``latent_dim``, or the sequence dimension is empty.
        """
        if x.dim() != 3:
            raise ValueError(
                f"TemporalLSTM expects a 3-D batch (batch_size, seq_len, "
                f"latent_dim={self.latent_dim}), got shape {tuple(x.shape)}."
            )
        if x.size(2) != self.latent_dim:
            raise ValueError(
                f"Feature-dimension mismatch: expected latent_dim="
                f"{self.latent_dim}, got {x.size(2)}."
            )
        if x.size(1) < 1:
            raise ValueError("Sequence dimension must be >= 1 timestep.")

        out, _ = self.lstm(x)      # out: (B, seq_len, LSTM_HIDDEN_SIZE)
        last_step = out[:, -1, :]  # (B, LSTM_HIDDEN_SIZE) — final timestep
        return self.head(last_step)  # (B, num_classes) raw logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Class-membership probabilities for the Level-1 stacking stage.

        Shape transformation:
        ``(B, seq_len, latent_dim) → (B, num_classes)``; rows are
        non-negative and sum to 1.0 along ``dim=-1``.

        Args:
            x: Mini-batch of latent-code windows, shape
                ``(B, seq_len, latent_dim)``.

        Returns:
            Softmax-normalised posteriors, shape ``(B, num_classes)``,
            computed under ``torch.no_grad()``. Call ``self.eval()`` first
            for deterministic outputs (inter-layer dropout disabled).
        """
        return F.softmax(self.forward(x), dim=-1)  # (B, num_classes)


# --------------------------------------------------------------------------- #
# Utility: sliding-window sequence construction (spec §3)                      #
# --------------------------------------------------------------------------- #


def create_lstm_sequences(
    z_data: np.ndarray,
    y_data: np.ndarray,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-flow latent vectors into labelled sliding windows.

    A 2-D array of flow representations ``(N, D)`` is reshaped into a 3-D
    sliding-window array ``(M, seq_len, D)`` with ``M = N - seq_len + 1``;
    window ``i`` contains the flows ``[i, i + seq_len - 1]`` in order. Each
    window is labelled with the target of its **final step**::

        y_seq[i] = y_data[i + seq_len - 1]

    matching :meth:`TemporalLSTM.forward`, which reads the last timestep.
    The leading ``seq_len - 1`` flows therefore serve only as context.

    CRITICAL ordering requirement: ``z_data`` / ``y_data`` must be in
    chronological order (per capture or per host). Passing shuffled rows
    destroys exactly the temporal signal this learner exploits.

    Implementation note: windows are generated with NumPy stride tricks
    (``numpy >= 1.20``) and returned as a fresh C-contiguous array, so the
    output can be passed to ``torch.from_numpy`` without copies or aliasing
    surprises. Input dtypes are preserved.

    Args:
        z_data: Latent vectors, shape ``(N, D)`` — e.g. ``(N, 32)`` DSAE
            codes from ``src/autoencoder.py``.
        y_data: Integer target labels, shape ``(N,)``.
        seq_len: Sliding-window length. Default 5.

    Returns:
        ``(sequences, labels)`` with shapes
        ``(N - seq_len + 1, seq_len, D)`` and ``(N - seq_len + 1,)``.

    Raises:
        ValueError: If inputs are not 2-D/1-D, their lengths differ,
            ``seq_len < 1``, or ``seq_len`` exceeds the number of flows.
    """
    flows = np.asarray(z_data)
    labels = np.asarray(y_data)

    if flows.ndim != 2:
        raise ValueError(f"z_data must be 2-D (N, latent_dim), got shape {flows.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"y_data must be 1-D (N,), got shape {labels.shape}.")
    if len(flows) != len(labels):
        raise ValueError(
            f"z_data and y_data must describe the same flows: "
            f"{len(flows)} != {len(labels)}."
        )
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}.")
    n_flows = flows.shape[0]
    if seq_len > n_flows:
        raise ValueError(
            f"seq_len={seq_len} exceeds the number of available flows "
            f"N={n_flows}; at least one full window is required."
        )

    # Stride-trick windows: (N, D) → (M, D, seq_len); reorder to
    # (M, seq_len, D) and materialise a C-contiguous copy.
    windows = np.lib.stride_tricks.sliding_window_view(
        flows, window_shape=seq_len, axis=0
    )
    sequences = np.ascontiguousarray(windows.transpose(0, 2, 1))
    # Align each window's label with the target of its final step.
    window_labels = labels[seq_len - 1 :].copy()

    LOGGER.info(
        "Created LSTM sliding windows: z%s + seq_len=%d → X%s, y%s (labels aligned to final step).",
        flows.shape, seq_len, sequences.shape, window_labels.shape,
    )
    return sequences, window_labels


# --------------------------------------------------------------------------- #
# Smoke test (spec §4)                                                         #
# --------------------------------------------------------------------------- #


def _run_smoke_test() -> int:
    """Verify both Level-0 learners and the sequence utility end-to-end.

    Checks:
        1. ``SpatialDNN``: ``(32, 32) → (32, 5)`` logits; CrossEntropy
           backward produces finite gradients; ``predict_proba`` rows sum
           to 1.0 and are deterministic in eval mode.
        2. ``TemporalLSTM``: ``(32, 5, 32) → (32, 5)`` logits; same
           backward/probability checks.
        3. ``create_lstm_sequences``: shapes, final-step label alignment,
           window content, dtype/contiguity, edge windows and rejection of
           invalid inputs; windows feed cleanly into ``TemporalLSTM``.
        4. Constructor argument validation for both models.

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

    batch_size = 32
    seq_len = DEFAULT_SEQ_LEN
    latent_dim = DEFAULT_LATENT_DIM
    num_classes = DEFAULT_NUM_CLASSES

    LOGGER.info("=" * 78)
    LOGGER.info(
        "Level-0 classifier smoke test — device=%s | torch=%s | numpy=%s",
        device, torch.__version__, np.__version__,
    )
    LOGGER.info("Threat taxonomy (%d classes): %s", num_classes, CLASS_NAMES)
    LOGGER.info("=" * 78)

    try:
        # Synthetic DSAE-like latents: Sigmoid-bounded values in (0, 1).
        z_batch = torch.rand(batch_size, latent_dim, device=device)
        z_windows = torch.rand(batch_size, seq_len, latent_dim, device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        # -- 1) SpatialDNN --------------------------------------------------- #
        dnn = SpatialDNN(latent_dim=latent_dim, num_classes=num_classes).to(device)
        LOGGER.info("SpatialDNN: %s | parameters=%d", dnn, sum(p.numel() for p in dnn.parameters()))

        dnn.train()
        logits = dnn(z_batch)
        assert tuple(logits.shape) == (batch_size, num_classes), f"bad logits shape {tuple(logits.shape)}"
        assert torch.isfinite(logits).all(), "SpatialDNN produced non-finite logits"
        LOGGER.info("SpatialDNN forward OK — logits%s (raw, unnormalised).", tuple(logits.shape))

        loss = F.cross_entropy(logits, targets)
        dnn.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in dnn.named_parameters():
            assert parameter.grad is not None, f"missing gradient for {name}"
            assert torch.isfinite(parameter.grad).all(), f"non-finite gradient in {name}"
        LOGGER.info("SpatialDNN backward OK — CrossEntropyLoss=%.6f, all gradients finite.", loss.item())

        dnn.eval()
        probs = dnn.predict_proba(z_batch)
        assert tuple(probs.shape) == (batch_size, num_classes), f"bad probability shape {tuple(probs.shape)}"
        assert torch.isfinite(probs).all() and (probs >= 0).all(), "invalid probability values"
        assert not probs.requires_grad, "predict_proba must not track gradients"
        assert torch.allclose(
            probs.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-6
        ), "predict_proba rows must sum to 1.0 along dim=-1"
        assert torch.equal(probs, dnn.predict_proba(z_batch)), "predict_proba must be deterministic in eval mode"
        LOGGER.info("SpatialDNN predict_proba OK — rows sum to 1.0 along the class dimension.")

        # -- 2) TemporalLSTM ------------------------------------------------- #
        lstm = TemporalLSTM(latent_dim=latent_dim, num_classes=num_classes).to(device)
        LOGGER.info("TemporalLSTM: %s | parameters=%d", lstm, sum(p.numel() for p in lstm.parameters()))

        lstm.train()
        seq_logits = lstm(z_windows)
        assert tuple(seq_logits.shape) == (batch_size, num_classes), f"bad logits shape {tuple(seq_logits.shape)}"
        assert torch.isfinite(seq_logits).all(), "TemporalLSTM produced non-finite logits"
        LOGGER.info("TemporalLSTM forward OK — logits%s from final timestep of %s windows.",
                    tuple(seq_logits.shape), tuple(z_windows.shape))

        seq_loss = F.cross_entropy(seq_logits, targets)
        lstm.zero_grad(set_to_none=True)
        seq_loss.backward()
        for name, parameter in lstm.named_parameters():
            assert parameter.grad is not None, f"missing gradient for {name}"
            assert torch.isfinite(parameter.grad).all(), f"non-finite gradient in {name}"
        LOGGER.info("TemporalLSTM backward OK — CrossEntropyLoss=%.6f, all gradients finite.", seq_loss.item())

        lstm.eval()
        seq_probs = lstm.predict_proba(z_windows)
        assert tuple(seq_probs.shape) == (batch_size, num_classes), f"bad probability shape {tuple(seq_probs.shape)}"
        assert torch.isfinite(seq_probs).all() and (seq_probs >= 0).all(), "invalid probability values"
        assert not seq_probs.requires_grad, "predict_proba must not track gradients"
        assert torch.allclose(
            seq_probs.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-6
        ), "predict_proba rows must sum to 1.0 along dim=-1"
        assert torch.equal(seq_probs, lstm.predict_proba(z_windows)), "predict_proba must be deterministic in eval mode"
        LOGGER.info("TemporalLSTM predict_proba OK — rows sum to 1.0 along the class dimension.")

        # -- 3) create_lstm_sequences ---------------------------------------- #
        n_flows = 64
        rng = np.random.default_rng(42)
        z_flows = rng.random((n_flows, latent_dim), dtype=np.float32)
        y_flows = (np.arange(n_flows) % num_classes).astype(np.int64)  # pattern makes alignment checkable

        X_seq, y_seq = create_lstm_sequences(z_flows, y_flows, seq_len=seq_len)
        n_windows = n_flows - seq_len + 1
        assert X_seq.shape == (n_windows, seq_len, latent_dim), f"bad sequence shape {X_seq.shape}"
        assert y_seq.shape == (n_windows,), f"bad label shape {y_seq.shape}"
        assert np.array_equal(y_seq, y_flows[seq_len - 1 :]), "labels must align to the final step of each window"
        assert np.array_equal(X_seq[0], z_flows[0:seq_len]), "first window must equal the first seq_len flows"
        assert np.array_equal(X_seq[7], z_flows[7 : 7 + seq_len]), "window i must contain flows [i, i+seq_len)"
        assert X_seq.dtype == z_flows.dtype and y_seq.dtype == y_flows.dtype, "dtypes must be preserved"
        assert X_seq.flags["C_CONTIGUOUS"], "sequences must be C-contiguous for torch.from_numpy"
        LOGGER.info("create_lstm_sequences OK — z%s → X%s, y%s (final-step alignment verified).",
                    z_flows.shape, X_seq.shape, y_seq.shape)

        # Edge windows: single-step and full-history.
        X_one, y_one = create_lstm_sequences(z_flows, y_flows, seq_len=1)
        assert X_one.shape == (n_flows, 1, latent_dim) and np.array_equal(y_one, y_flows)
        X_full, y_full = create_lstm_sequences(z_flows, y_flows, seq_len=n_flows)
        assert X_full.shape == (1, n_flows, latent_dim) and y_full.shape == (1,)

        # Invalid inputs must be rejected cleanly.
        for bad_kwargs in ({"seq_len": 0}, {"seq_len": n_flows + 1}):
            try:
                create_lstm_sequences(z_flows, y_flows, **bad_kwargs)
            except ValueError:
                continue
            raise AssertionError(f"create_lstm_sequences accepted invalid {bad_kwargs}")
        try:
            create_lstm_sequences(z_flows[:-1], y_flows)
        except ValueError:
            pass
        else:
            raise AssertionError("length-mismatched inputs were accepted")
        LOGGER.info("create_lstm_sequences validation OK — edge windows and invalid inputs handled.")

        # End-to-end: utility output feeds the temporal learner directly.
        with torch.no_grad():
            window_logits = lstm(torch.from_numpy(X_seq).to(device))
        assert tuple(window_logits.shape) == (n_windows, num_classes), f"bad shape {tuple(window_logits.shape)}"
        LOGGER.info("End-to-end OK — %d sliding windows → logits%s.", n_windows, tuple(window_logits.shape))

        # -- 4) Constructor argument validation ------------------------------ #
        for factory in (SpatialDNN, TemporalLSTM):
            for bad_kwargs in ({"latent_dim": 0}, {"num_classes": 1}):
                try:
                    factory(**bad_kwargs)
                except ValueError:
                    continue
                raise AssertionError(f"{factory.__name__} accepted invalid {bad_kwargs}")
        LOGGER.info("Constructor validation OK — invalid dims/classes rejected.")

        LOGGER.info("=" * 78)
        LOGGER.info("SMOKE TEST PASSED — both Level-0 learners emit (B, %d) logits with well-formed posteriors.",
                    num_classes)
        return 0
    except Exception:
        LOGGER.exception("SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_smoke_test())