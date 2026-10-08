"""Real-time SHAP feature attribution for the IoT-23 intrusion detection system.

Implements the explainability stage of the NIDS pipeline of *Dutta et al.
(2020)*: per-flow Shapley-value attributions computed with
``shap.DeepExplainer`` over the Level-0 ``SpatialDNN``, operating on the
32-dimensional sparse latent vectors ``z`` produced by the Deep Sparse
Autoencoder.

Pipeline position
-----------------
::

    src/dataset.py   train_dsae.py   train_ensemble.py        src/explainer.py (this module)
    ─────────────    ────────────    ────────────────────     ─────────────────────────────
    X/y artifacts    z_*.npy         models/saved/            FlowExplainer
    feature_names    (DSAE           dnn_level0.pt  ───────►  · real-time top-k attributions
    .joblib          latents)        (Level-0 SpatialDNN)     · natural-language rationale
                                                     │        · offline waterfall audits
                                     data/processed/
                                     z_train.npy ───────────► 100-sample SHAP background

Latency engineering (< 20 ms per flow, CPU)
--------------------------------------------
* ``shap.DeepExplainer`` is constructed ONCE and warmed up at init — never
  re-initialised per request; the background tensor is pre-computed and
  pinned to the model's device, and expected-value lookups are cached.
* The (relatively expensive, occasionally flaky) SHAP additivity check is
  disabled on the hot path; the constructor instead performs a soft,
  non-fatal additivity verification (``model_output ≈ base_value + Σ SHAP``).
* Attributions run under an explicit ``torch.enable_grad()`` guard — SHAP
  needs autograd even when called from inference contexts.
* Batches ``(N, 32)`` are attributed in a single ``shap_values`` call,
  amortising fixed overhead across flows.

SHAP version compatibility
--------------------------
``DeepExplainer.shap_values`` has emitted three layouts across releases:
a list of per-class ``(N, F)`` arrays (SHAP < 0.45), per-class arrays with a
spurious leading singleton dim ``(1, N, F)`` (a PyTorch quirk), and a single
consolidated ``(N, F, C)`` array (SHAP >= 0.45). All are normalised to the
canonical ``(N, classes, features)`` layout by :func:`_normalise_shap_output`.

Feature-name semantics
----------------------
The artifact ``feature_names.joblib`` describes the DSAE INPUT space
(``duration``, ``orig_pkts``, ``conn_state_s0``, …), whereas attributions
are computed over the 32-d DSAE LATENT space. When the provided list does
not match ``latent_dim`` the explainer falls back to synthetic latent names
``z00 … z31`` (logged once) and preserves the human-readable vocabulary in
``FlowExplainer.input_feature_names`` for audit reports.

Usage
-----
.. code-block:: python

    explainer = FlowExplainer(dnn_model, background=z_train[:100], feature_names=names)
    report = explainer.explain_flow(z_test[0], predicted_class=2, top_k=5)
    print(report["explanation_text"])
    explainer.save_waterfall_plot(z_test[0], predicted_class=2)

Dependencies: ``torch``, ``numpy``, ``shap`` (>= 0.40), ``joblib``;
``matplotlib`` only for offline waterfall plots.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import shap
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Public configuration                                                         #
# --------------------------------------------------------------------------- #

LOGGER: Final[logging.Logger] = logging.getLogger("iot23.explainer")

#: Console class-name map (spec §2).
CLASS_NAMES: Final[dict[int, str]] = {
    0: "Benign",
    1: "Recon",
    2: "DoS",
    3: "Malware",
    4: "Botnet",
}

NUM_CLASSES: Final[int] = 5
DEFAULT_TOP_K: Final[int] = 5
#: DeepExplainer latency scales with background size — 100–200 is the sweet spot.
RECOMMENDED_BACKGROUND_RANGE: Final[tuple[int, int]] = (100, 200)
#: Hard per-flow latency budget (spec §4).
LATENCY_BUDGET_MILLISECONDS: Final[float] = 20.0

DEFAULT_WATERFALL_PATH: Final[Path] = Path("data/processed/shap_waterfall.png")

#: Artifacts referenced by this module.
DNN_MODEL_PATH: Final[Path] = Path("models/saved/dnn_level0.pt")
Z_TRAIN_PATH: Final[Path] = Path("data/processed/z_train.npy")
Z_TEST_PATH: Final[Path] = Path("data/processed/z_test.npy")
FEATURE_NAMES_PATH: Final[Path] = Path("data/processed/feature_names.joblib")

#: Smoke-test settings.
SMOKE_BACKGROUND_SAMPLES: Final[int] = 100
LATENCY_BENCHMARK_REPEATS: Final[int] = 10
LATENCY_BENCHMARK_BATCH: Final[int] = 8

__all__: Final[list[str]] = ["FlowExplainer", "CLASS_NAMES"]


# --------------------------------------------------------------------------- #
# SHAP compatibility helpers                                                   #
# --------------------------------------------------------------------------- #


def _strip_leading_singleton_dims(array: np.ndarray) -> np.ndarray:
    """Remove spurious leading size-1 dimensions (old PyTorch SHAP quirk).

    E.g. ``(1, N, F) → (N, F)``. Only leading axes of size 1 beyond 2-D are
    stripped, so legitimate batch/feature axes are never collapsed.
    """
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    return array


def _normalise_shap_output(
    raw_values: Any,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """Normalise ``DeepExplainer.shap_values`` output to ``(N, C, F)``.

    Handles the layouts emitted across SHAP releases:
      * ``list``/``tuple`` of per-class arrays, each ``(N, F)`` (SHAP < 0.45);
      * per-class arrays with a leading singleton dim ``(1, N, F)``;
      * a single consolidated ``(N, F, C)`` array (SHAP >= 0.45);
      * squeezed 2D layout ``(F, C)`` or ``(C, F)`` when ``n_samples == 1`` (SHAP >= 0.52).

    Args:
        raw_values: Whatever ``explainer.shap_values(...)`` returned.
        n_samples: Number of explained rows ``N``.
        n_features: Feature dimensionality ``F``.
        n_classes: Number of model outputs ``C``.

    Returns:
        Float32 array of shape ``(n_samples, n_classes, n_features)``.

    Raises:
        RuntimeError: If the layout cannot be recognised.
    """
    if isinstance(raw_values, (list, tuple)):
        if len(raw_values) != n_classes:
            raise RuntimeError(
                f"SHAP returned {len(raw_values)} per-class blocks, expected {n_classes}."
            )
        per_class: list[np.ndarray] = []
        for class_index, item in enumerate(raw_values):
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            array = _strip_leading_singleton_dims(np.asarray(item, dtype=np.float32))
            if array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0]  # trailing singleton class axis
            if array.shape != (n_samples, n_features):
                raise RuntimeError(
                    f"SHAP class block {class_index} has shape {array.shape}, "
                    f"expected {(n_samples, n_features)}."
                )
            per_class.append(array)
        return np.stack(per_class, axis=1)  # (N, C, F)

    if isinstance(raw_values, torch.Tensor):
        raw_values = raw_values.detach().cpu().numpy()
    array = _strip_leading_singleton_dims(np.asarray(raw_values, dtype=np.float32))

    # 3-D layouts
    if array.ndim == 3:
        if array.shape == (n_samples, n_features, n_classes):
            return np.ascontiguousarray(array.transpose(0, 2, 1))  # → (N, C, F)
        if array.shape == (n_samples, n_classes, n_features):
            return np.ascontiguousarray(array)
        raise RuntimeError(
            f"Unrecognised 3-D SHAP layout {array.shape}; expected "
            f"{(n_samples, n_features, n_classes)} or {(n_samples, n_classes, n_features)}."
        )

    # 2-D layouts when n_samples == 1 (SHAP 0.52+ squeezed layout)
    if n_samples == 1 and array.ndim == 2:
        if array.shape == (n_features, n_classes):
            # (F, C) -> transpose to (C, F) -> expand to (1, C, F)
            return np.ascontiguousarray(np.expand_dims(array.T, axis=0))
        if array.shape == (n_classes, n_features):
            # (C, F) -> expand to (1, C, F)
            return np.ascontiguousarray(np.expand_dims(array, axis=0))

    # Single-class output
    if array.ndim == 2 and array.shape == (n_samples, n_features):
        if n_classes != 1:
            raise RuntimeError(f"2-D SHAP output {array.shape} cannot cover {n_classes} classes.")
        return array[:, None, :]

    raise RuntimeError(f"Unrecognised SHAP output layout: shape={array.shape}.")


def _compute_shap_values(
    explainer: shap.DeepExplainer,
    tensor: torch.Tensor,
    *,
    check_additivity: bool,
) -> Any:
    """Invoke ``DeepExplainer.shap_values`` across SHAP versions.

    Newer releases accept ``check_additivity``; older ones raise ``TypeError``
    on the kwarg, in which case the call is retried without it.
    """
    try:
        return explainer.shap_values(tensor, check_additivity=check_additivity)
    except TypeError:
        return explainer.shap_values(tensor)


# --------------------------------------------------------------------------- #
# FlowExplainer                                                                #
# --------------------------------------------------------------------------- #


class FlowExplainer:
    """Real-time SHAP attributor for the Level-0 ``SpatialDNN``.

    Wraps a single, pre-initialised ``shap.DeepExplainer`` around a frozen
    background sample of DSAE latent vectors and explains model predictions
    as Shapley values over the 32-dimensional latent space. Designed for
    sub-20 ms CPU latency per flow.

    Args:
        dnn_model: Trained Level-0 classifier (any ``nn.Module`` returning
            raw logits of shape ``(N, num_classes)``; typically
            ``src.classifiers.SpatialDNN`` loaded from
            ``models/saved/dnn_level0.pt``). Put in ``eval()`` and kept there.
        background_data: Representative background sample, shape
            ``(100–200, latent_dim)`` — ideally BENIGN flows from
            ``data/processed/z_train.npy`` (see note below), float32.
        feature_names: Names for the explained space. If the list length
            equals ``latent_dim`` it is used directly; otherwise synthetic
            latent names ``z00…`` are used and the provided (DSAE
            input-space) vocabulary is preserved in ``input_feature_names``.
        warm_up: Run one attribution at init to fail fast on unsupported
            model ops and to verify the SHAP output layout (default True).

    Note:
        The background defines the SHAP reference distribution: production
        deployments should pass *benign-only* flows so attributions read as
        "why this flow deviates from normal traffic". The smoke test uses
        the first 100 rows of ``z_train`` per the task specification.

    Raises:
        TypeError: If ``dnn_model`` is not an ``nn.Module``.
        ValueError: If the background/feature names are malformed or the
            model's input dimensionality disagrees with the background.
        RuntimeError: If the warm-up attribution fails.
    """

    def __init__(
        self,
        dnn_model: nn.Module,
        background_data: np.ndarray,
        feature_names: list[str],
        *,
        warm_up: bool = True,
    ) -> None:
        """Initialise the DeepExplainer once (never per request)."""
        if not isinstance(dnn_model, nn.Module):
            raise TypeError(f"dnn_model must be an nn.Module, got {type(dnn_model).__name__}.")
        self.model = dnn_model
        self.model.eval()
        self.device = next(dnn_model.parameters()).device

        background = np.asarray(background_data, dtype=np.float32)
        if background.ndim != 2:
            raise ValueError(
                f"background_data must be 2-D (n_samples, latent_dim), got shape {background.shape}."
            )
        if background.shape[0] < 1 or background.shape[1] < 1:
            raise ValueError(f"background_data is empty: shape {background.shape}.")
        if not np.isfinite(background).all():
            raise ValueError("background_data contains non-finite (NaN/Inf) values.")
        if not RECOMMENDED_BACKGROUND_RANGE[0] <= background.shape[0] <= RECOMMENDED_BACKGROUND_RANGE[1]:
            LOGGER.warning(
                "Background size %d is outside the recommended %s range — "
                "DeepExplainer latency scales with background size.",
                background.shape[0], RECOMMENDED_BACKGROUND_RANGE,
            )
        self.latent_dim = int(background.shape[1])
        self._background_size = int(background.shape[0])

        model_latent_dim = getattr(dnn_model, "latent_dim", None)
        if model_latent_dim is not None and int(model_latent_dim) != self.latent_dim:
            raise ValueError(
                f"Model expects latent_dim={int(model_latent_dim)} but the background "
                f"has {self.latent_dim} features."
            )

        if not isinstance(feature_names, (list, tuple)) or len(feature_names) == 0:
            raise ValueError("feature_names must be a non-empty list of strings.")
        self.input_feature_names = [str(name) for name in feature_names]
        if len(self.input_feature_names) == self.latent_dim:
            self.feature_names: list[str] = list(self.input_feature_names)
        else:
            self.feature_names = [f"z{index:02d}" for index in range(self.latent_dim)]
            LOGGER.warning(
                "feature_names has %d entries but the explained space has %d latent dims — "
                "using synthetic latent names z00…z%02d for attributions. The provided names "
                "describe the DSAE INPUT space and remain available via "
                "FlowExplainer.input_feature_names.",
                len(self.input_feature_names), self.latent_dim, self.latent_dim - 1,
            )

        # Pre-computed background tensor, pinned to the model's device.
        background_tensor = torch.from_numpy(np.ascontiguousarray(background)).to(self.device)

        # Probe the model output dimensionality once (model-agnostic).
        with torch.no_grad():
            probe_logits = self.model(background_tensor[:1])  # (1, C)
        if probe_logits.dim() != 2:
            raise ValueError(
                f"Model must return (N, num_classes) logits, got {tuple(probe_logits.shape)}."
            )
        self.num_classes = int(probe_logits.shape[-1])
        model_num_classes = getattr(dnn_model, "num_classes", None)
        if model_num_classes is not None and int(model_num_classes) != self.num_classes:
            raise ValueError(
                f"Model reports num_classes={int(model_num_classes)} but emits "
                f"{self.num_classes} outputs."
            )

        # Single DeepExplainer instance — the expensive initialisation.
        self.explainer = shap.DeepExplainer(self.model, background_tensor)
        self._expected_value_cache: dict[int, float | None] = {}

        if warm_up:
            try:
                with torch.enable_grad():
                    raw = _compute_shap_values(
                        self.explainer, background_tensor[:1], check_additivity=False
                    )
                warm_shap = _normalise_shap_output(
                    raw, n_samples=1, n_features=self.latent_dim, n_classes=self.num_classes
                )
            except Exception as error:
                raise RuntimeError(
                    "FlowExplainer warm-up failed — the model may contain ops unsupported "
                    "by shap.DeepExplainer."
                ) from error
            # Soft, non-fatal additivity verification: logit ≈ base + Σ SHAP.
            base_value = self._expected_value(0)
            if base_value is not None:
                warm_logit = float(probe_logits[0, 0].detach().cpu().item())
                residual = warm_logit - (base_value + float(warm_shap[0, 0].sum()))
                LOGGER.debug("Warm-up additivity residual (class 0): %.6f", residual)
                if abs(residual) > 0.05 * max(1.0, abs(warm_logit)):
                    LOGGER.warning(
                        "Warm-up additivity residual %.4f is large — verify that the "
                        "background matches the explained feature distribution.", residual,
                    )

        LOGGER.info(
            "FlowExplainer ready — model=%s, latent_dim=%d, num_classes=%d, "
            "background=(%d, %d), attribution labels=%s.",
            type(self.model).__name__, self.latent_dim, self.num_classes,
            self._background_size, self.latent_dim,
            "provided" if self.feature_names == self.input_feature_names else "synthetic z00…",
        )

    def __repr__(self) -> str:
        """Compact summary for logs."""
        naming = "provided" if self.feature_names == self.input_feature_names else "synthetic(z)"
        return (
            f"FlowExplainer(model={type(self.model).__name__}, latent_dim={self.latent_dim}, "
            f"num_classes={self.num_classes}, background={self._background_size} samples, "
            f"feature_names={naming})"
        )

    # -- Public API --------------------------------------------------------- #

    def explain_flow(
        self,
        z_vector: np.ndarray,
        predicted_class: int,
        top_k: int = DEFAULT_TOP_K,
        *,
        check_additivity: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Explain one flow (or a batch) for a specific predicted class.

        Shape transformations:
            ``z_vector``: ``(latent_dim,)`` / ``(1, latent_dim)`` → a single
            report ``dict``; ``(N, latent_dim)`` → a ``list`` of ``N`` dicts.

        The SHAP values explain the model's raw logit for
        ``predicted_class`` relative to the background distribution; positive
        attributions push the logit up (towards the class), negative ones
        push it down.

        Args:
            z_vector: Latent flow vector(s) — ``(32,)``, ``(1, 32)`` or
                ``(N, 32)`` float array (a CPU ``torch.Tensor`` is also
                accepted and converted).
            predicted_class: Class index to attribute (``0–4``).
            top_k: Number of top-|SHAP| features to report. Default 5.
            check_additivity: Enable SHAP's (slower, stricter) additivity
                check for this call. Default ``False`` for latency — the
                constructor already performs a soft verification.

        Returns:
            For a single flow, a dict with keys:

            * ``predicted_class`` (int) — the class being explained.
            * ``class_name`` (str) — e.g. ``"DoS"``.
            * ``top_features`` (list[dict]) — ``[{"feature_idx": int,
              "feature_name": str, "shap_value": float, "direction":
              "increases_risk" | "decreases_risk"}]`` sorted by absolute
              impact (descending).
            * ``explanation_text`` (str) — natural-language rationale.
            * ``base_value`` (float | None) — expected class logit over the
              background (SHAP ``expected_value``).
            * ``model_output`` (float) — the class logit for this flow.
            * ``additivity_residual`` (float | None) — ``model_output −
              (base_value + Σ SHAP)``; ≈ 0 confirms a consistent
              decomposition (audit aid).

            For a batch, a list of such dicts (one per row, same order).

        Raises:
            ValueError: On malformed input, wrong dimensionality, an
                out-of-range class, or non-finite values.
        """
        self.model.eval()
        matrix = self._normalise_flow_matrix(z_vector)

        predicted_class = int(predicted_class)
        if not 0 <= predicted_class < self.num_classes:
            raise ValueError(
                f"predicted_class must be in [0, {self.num_classes}), got {predicted_class}."
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")
        top_k = min(top_k, self.latent_dim)

        tensor = torch.from_numpy(matrix).to(self.device)

        # Reference model output (class logit) for bookkeeping — one cheap forward.
        with torch.no_grad():
            logits = self.model(tensor)                       # (N, num_classes)
        class_logits = logits[:, predicted_class].detach().cpu().numpy()

        # SHAP attribution — DeepExplainer needs autograd even in inference code.
        with torch.enable_grad():
            raw = _compute_shap_values(self.explainer, tensor, check_additivity=check_additivity)
        shap_all = _normalise_shap_output(
            raw,
            n_samples=matrix.shape[0],
            n_features=self.latent_dim,
            n_classes=self.num_classes,
        )                                                      # (N, C, F)
        class_shap = shap_all[:, predicted_class, :]           # (N, F)
        base_value = self._expected_value(predicted_class)

        reports = [
            self._build_row_explanation(
                row_shap=class_shap[row_index],
                predicted_class=predicted_class,
                top_k=top_k,
                base_value=base_value,
                model_output=float(class_logits[row_index]),
            )
            for row_index in range(matrix.shape[0])
        ]
        return reports[0] if matrix.shape[0] == 1 else reports

    def save_waterfall_plot(
        self,
        z_sample: np.ndarray,
        predicted_class: int,
        output_path: str | Path = DEFAULT_WATERFALL_PATH,
        *,
        max_display: int = 10,
    ) -> Path:
        """Generate and save a SHAP waterfall plot for a single flow (audit log).

        Offline diagnostic only — matplotlib is imported lazily and the Agg
        backend is forced so the method works on headless servers. The plot
        anchors at the class ``expected_value`` (base) and stacks per-feature
        Shapley contributions up to the model's class logit.

        Args:
            z_sample: A single flow — ``(latent_dim,)`` or ``(1, latent_dim)``.
            predicted_class: Class index whose attribution to plot.
            output_path: Destination PNG (parent directories created).
                Default ``data/processed/shap_waterfall.png``.
            max_display: Maximum features rendered in the waterfall.

        Returns:
            The resolved destination path.

        Raises:
            ValueError: On malformed input or an out-of-range class.
            RuntimeError: If matplotlib or ``shap.plots.waterfall`` is
                unavailable.
        """
        self.model.eval()
        matrix = self._normalise_flow_matrix(z_sample)
        if matrix.shape[0] != 1:
            raise ValueError(
                "save_waterfall_plot explains a single flow — pass a "
                "(latent_dim,) or (1, latent_dim) sample."
            )
        predicted_class = int(predicted_class)
        if not 0 <= predicted_class < self.num_classes:
            raise ValueError(
                f"predicted_class must be in [0, {self.num_classes}), got {predicted_class}."
            )

        tensor = torch.from_numpy(matrix).to(self.device)
        with torch.enable_grad():
            raw = _compute_shap_values(self.explainer, tensor, check_additivity=False)
        shap_all = _normalise_shap_output(
            raw, n_samples=1, n_features=self.latent_dim, n_classes=self.num_classes
        )
        row_values = shap_all[0, predicted_class, :]          # (F,)

        base_value = self._expected_value(predicted_class)
        if base_value is None:
            LOGGER.warning("DeepExplainer expected_value unavailable — anchoring waterfall at 0.0.")
            base_value = 0.0

        try:
            import matplotlib
            matplotlib.use("Agg")  # headless-safe backend
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "matplotlib is required for waterfall plots — `pip install matplotlib`."
            ) from error

        waterfall = getattr(getattr(shap, "plots", None), "waterfall", None)
        if waterfall is None:
            raise RuntimeError("shap.plots.waterfall is unavailable — install shap >= 0.40.")

        explanation = shap.Explanation(
            values=row_values,
            base_values=float(base_value),
            data=matrix[0],
            feature_names=list(self.feature_names),
        )
        waterfall(explanation, max_display=max_display, show=False)

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(destination, dpi=150, bbox_inches="tight")
        plt.close("all")
        LOGGER.info(
            "Saved SHAP waterfall plot (class %d — %s, base=%.4f) → %s",
            predicted_class, CLASS_NAMES.get(predicted_class, "?"), base_value, destination,
        )
        return destination

    # -- Internal helpers --------------------------------------------------- #

    def _normalise_flow_matrix(self, z_vector: np.ndarray | torch.Tensor) -> np.ndarray:
        """Validate flow input(s) into a ``(N, latent_dim)`` float32 matrix.

        Accepts NumPy arrays or CPU ``torch.Tensor`` inputs.

        Raises:
            ValueError: On wrong rank/dimensionality or non-finite values.
        """
        if isinstance(z_vector, torch.Tensor):
            z_vector = z_vector.detach().cpu().numpy()
        values = np.asarray(z_vector, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError(
                f"Expected a (latent_dim,), (1, latent_dim) or (N, latent_dim) input, "
                f"got shape {values.shape}."
            )
        if values.shape[1] != self.latent_dim:
            raise ValueError(
                f"Feature-dimension mismatch: expected latent_dim={self.latent_dim}, "
                f"got {values.shape[1]}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Input contains non-finite (NaN/Inf) values.")
        return np.ascontiguousarray(values)

    def _expected_value(self, class_index: int) -> float | None:
        """Read DeepExplainer's expected class logit, tolerating all layouts.

        Results are cached per class. Returns ``None`` when the attribute is
        unavailable (older releases, or before the first ``shap_values`` call).
        """
        if class_index in self._expected_value_cache:
            return self._expected_value_cache[class_index]
        try:
            raw = self.explainer.expected_value
        except Exception:
            self._expected_value_cache[class_index] = None
            return None
        value: float | None = None
        if raw is not None:
            if isinstance(raw, torch.Tensor):
                raw = raw.detach().cpu().numpy()
            try:
                array = np.asarray(raw, dtype=np.float64).ravel()
                if array.size == 1:
                    value = float(array[0])
                elif array.size == self.num_classes:
                    value = float(array[class_index])
            except (TypeError, ValueError):
                value = None
        self._expected_value_cache[class_index] = value
        return value

    def _build_row_explanation(
        self,
        row_shap: np.ndarray,
        predicted_class: int,
        top_k: int,
        base_value: float | None,
        model_output: float | None,
    ) -> dict[str, Any]:
        """Assemble the structured report dict for one flow.

        Args:
            row_shap: Per-feature Shapley values, shape ``(latent_dim,)``.
            predicted_class: Class being explained.
            top_k: Features to report.
            base_value: Expected class logit (or ``None``).
            model_output: The flow's class logit (or ``None``).

        Returns:
            The report dictionary documented in :meth:`explain_flow`.
        """
        class_name = CLASS_NAMES.get(predicted_class, f"class_{predicted_class}")
        order = np.argsort(-np.abs(row_shap))[:top_k]

        top_features: list[dict[str, Any]] = []
        for feature_idx in order:
            value = float(row_shap[int(feature_idx)])
            top_features.append(
                {
                    "feature_idx": int(feature_idx),
                    "feature_name": self.feature_names[int(feature_idx)],
                    "shap_value": round(value, 6),
                    "direction": "increases_risk" if value > 0.0 else "decreases_risk",
                }
            )

        positive = [entry for entry in top_features if entry["shap_value"] > 0.0]
        negative = [entry for entry in top_features if entry["shap_value"] <= 0.0]
        if positive:
            indices = ", ".join(str(entry["feature_idx"]) for entry in positive)
            drivers = ", ".join(
                f"{entry['feature_name']} {entry['shap_value']:+.4f}" for entry in positive
            )
            text = (
                f"Flow classified as {class_name} (class {predicted_class}) primarily due to "
                f"elevated activations in latent features [{indices}] ({drivers})"
            )
        else:
            text = (
                f"Flow classified as {class_name} (class {predicted_class}); none of the top "
                f"{top_k} features positively support this class"
            )
        if negative:
            mitigators = ", ".join(
                f"{entry['feature_name']} {entry['shap_value']:+.4f}" for entry in negative
            )
            text += f"; mitigating evidence from {mitigators}"
        text += "."

        additivity_residual: float | None = None
        if base_value is not None and model_output is not None:
            additivity_residual = round(
                model_output - (base_value + float(row_shap.sum())), 6
            )

        return {
            "predicted_class": predicted_class,
            "class_name": class_name,
            "top_features": top_features,
            "explanation_text": text,
            "base_value": None if base_value is None else round(float(base_value), 6),
            "model_output": None if model_output is None else round(float(model_output), 6),
            "additivity_residual": additivity_residual,
        }


# --------------------------------------------------------------------------- #
# Smoke-test artifact loaders                                                  #
# --------------------------------------------------------------------------- #


def _load_latent_matrix(path: Path, label: str) -> np.ndarray:
    """Load and validate a DSAE latent artifact (.npy).

    Raises:
        FileNotFoundError: If the artifact is missing (hint: ``train_dsae.py``).
        ValueError: If not 2-D, empty, non-finite, or outside [0, 1].
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


def _load_feature_names(path: Path) -> list[str]:
    """Load feature names from the dataset artifact (list or metadata dict).

    Raises:
        FileNotFoundError: If the artifact is missing (hint: ``src/dataset.py``).
        ValueError: If the payload is unusable.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Required artifact '{path}' was not found — generate it first with "
            f"`python src/dataset.py`."
        )
    payload = joblib.load(path)
    names = payload.get("feature_names") if isinstance(payload, dict) else payload
    if not isinstance(names, (list, tuple)) or len(names) == 0:
        raise ValueError(f"Unusable feature-name payload in '{path}': {type(payload).__name__}.")
    LOGGER.info("Loaded %d feature names ← %s.", len(names), path)
    return [str(name) for name in names]


def _load_state_dict_file(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a state-dict checkpoint from disk, CPU-mapped and safely.

    Raises:
        FileNotFoundError: If ``path`` does not point to an existing file.
    """
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint}' was not found — generate it first with "
            f"`python train_ensemble.py`."
        )
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 — the `weights_only` kwarg does not exist.
        return torch.load(checkpoint, map_location="cpu")


# --------------------------------------------------------------------------- #
# Smoke test (spec §5)                                                         #
# --------------------------------------------------------------------------- #


def _run_smoke_test() -> int:
    """End-to-end verification of the real-time attribution pipeline.

    Steps (spec §5): load ``z_test.npy``, ``dnn_level0.pt`` and
    ``feature_names.joblib``; instantiate ``FlowExplainer`` with the first
    100 samples of ``z_train.npy``; explain sample #0 of ``z_test.npy``;
    print the structured report and the measured latency. Additionally
    benchmarks the < 20 ms budget, demonstrates batch attribution and writes
    the waterfall audit plot.

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.manual_seed(42)

    LOGGER.info("=" * 78)
    LOGGER.info(
        "FlowExplainer smoke test — shap %s | torch %s | numpy %s",
        getattr(shap, "__version__", "unknown"), torch.__version__, np.__version__,
    )
    LOGGER.info("=" * 78)

    try:
        # -- Artifacts -------------------------------------------------------- #
        z_train = _load_latent_matrix(Z_TRAIN_PATH, "z_train")
        z_test = _load_latent_matrix(Z_TEST_PATH, "z_test")
        feature_names = _load_feature_names(FEATURE_NAMES_PATH)

        # -- Level-0 SpatialDNN ------------------------------------------------ #
        from src.classifiers import SpatialDNN  # local import keeps the class model-agnostic

        latent_dim = z_train.shape[1]
        model = SpatialDNN(latent_dim=latent_dim, num_classes=NUM_CLASSES)
        model.load_state_dict(_load_state_dict_file(DNN_MODEL_PATH), strict=True)
        model.eval()
        LOGGER.info(
            "Loaded SpatialDNN ← %s (latent_dim=%d, num_classes=%d, parameters=%d).",
            DNN_MODEL_PATH, latent_dim, NUM_CLASSES, sum(p.numel() for p in model.parameters()),
        )

        # -- Explainer with the first 100 z_train samples as background -------- #
        background = z_train[:SMOKE_BACKGROUND_SAMPLES]
        LOGGER.info(
            "Background: first %d samples of z_train (production tip: prefer benign-only "
            "flows so attributions read as deviation-from-normal).", len(background),
        )
        explainer = FlowExplainer(model, background, feature_names)

        # -- Predict sample #0, then explain it -------------------------------- #
        sample = z_test[0:1]
        with torch.no_grad():
            logits = model(torch.from_numpy(sample))
        predicted_class = int(logits.argmax(dim=-1).item())
        LOGGER.info(
            "Sample #0 of z_test — logits %s → predicted class %d (%s).",
            np.round(logits.numpy(), 4).tolist(), predicted_class,
            CLASS_NAMES.get(predicted_class, "?"),
        )

        started = time.perf_counter()
        report = explainer.explain_flow(sample, predicted_class, top_k=DEFAULT_TOP_K)
        single_ms = (time.perf_counter() - started) * 1e3

        durations_ms: list[float] = []
        for _ in range(LATENCY_BENCHMARK_REPEATS):
            call_started = time.perf_counter()
            explainer.explain_flow(sample, predicted_class, top_k=DEFAULT_TOP_K)
            durations_ms.append((time.perf_counter() - call_started) * 1e3)
        median_ms = float(np.median(durations_ms))
        mean_ms = float(np.mean(durations_ms))
        LOGGER.info(
            "Latency — first call %.2f ms | %d repeats: mean %.2f ms, median %.2f ms "
            "(budget %.0f ms).",
            single_ms, LATENCY_BENCHMARK_REPEATS, mean_ms, median_ms, LATENCY_BUDGET_MILLISECONDS,
        )
        if median_ms > LATENCY_BUDGET_MILLISECONDS:
            LOGGER.warning(
                "Median latency %.2f ms exceeds the %.0f ms budget — reduce the background "
                "size or attribute flows in batches.",
                median_ms, LATENCY_BUDGET_MILLISECONDS,
            )

        LOGGER.info("Explanation report:\n%s", json.dumps(report, indent=2))

        # -- Batch attribution demo -------------------------------------------- #
        if len(z_test) >= LATENCY_BENCHMARK_BATCH + 1:
            batch = z_test[1 : 1 + LATENCY_BENCHMARK_BATCH]
            batch_started = time.perf_counter()
            batch_reports = explainer.explain_flow(batch, predicted_class, top_k=3)
            batch_ms = (time.perf_counter() - batch_started) * 1e3
            LOGGER.info(
                "Batch demo — %d flows explained in %.2f ms (%.2f ms/flow): %s",
                len(batch_reports), batch_ms, batch_ms / len(batch_reports),
                batch_reports[0]["explanation_text"],
            )

        # -- Offline waterfall audit plot -------------------------------------- #
        try:
            explainer.save_waterfall_plot(sample, predicted_class)
        except Exception as error:  # matplotlib missing / plotting quirk — non-fatal.
            LOGGER.warning("Waterfall plot skipped: %s", error)

        LOGGER.info("=" * 78)
        LOGGER.info("SMOKE TEST PASSED — real-time attribution verified against the latency budget.")
        return 0
    except Exception:
        LOGGER.exception("SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_smoke_test())