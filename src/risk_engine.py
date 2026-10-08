"""Deterministic Multi-Factor Risk Scoring Engine for the IoT-23 NIDS.

Translates raw machine-learning predictions (attack class + classifier
confidence) and measurable network telemetry (flow packet rate, target-asset
criticality) into a mathematically defensible, auditable Composite Risk
Score ``R ∈ [0, 100]`` that drives tiered SOC response policy.

Pipeline position
-----------------
::

    src/dataset.py    train_dsae.py   train_ensemble.py   src/explainer.py     src/risk_engine.py (this module)
    ──────────────    ────────────    ─────────────────   ────────────────     ──────────────────────────────
    X/y artifacts     z_*.npy         Level-0/Level-1     SHAP attributions    Composite Risk Score (R)
    feature_names     (DSAE latents)  models + meta-                           → risk_tier + recommended_action
    .joblib                           learner                                 → full audit breakdown

The Composite Risk Score
------------------------
::

    R = min(100.0,  w1 · (S_attack · 20.0)
                  + w2 · (C_model · 100.0)
                  + w3 · A_vol
                  + w4 · I_asset)

Default weights (Σ = 1.0):

=====  ======  ==============================================
Index  Value   Component
=====  ======  ==============================================
w1     0.35    Attack Threat Severity (MITRE ATT&CK taxonomy)
w2     0.25    Classifier Calibrated Confidence
w3     0.20    Traffic Volume Anomaly
w4     0.20    Target Asset Criticality
=====  ======  ==============================================

Component definitions
---------------------

* ``S_attack ∈ [0, 5]`` — base severity from the threat taxonomy:

    =====  =====================  =====
    Class  Category                S
    =====  =====================  =====
    0      Benign                  0.0
    1      Recon / PortScan        1.5
    2      DoS / DDoS              4.0
    3      Malware / Okiru          4.5
    4      Botnet C&C               5.0
    =====  =====================  =====

* ``C_model ∈ [0.0, 1.0]`` — softmax probability or calibrated confidence
  emitted by the Level-1 stacking meta-learner for the predicted class.

* ``A_vol ∈ [0.0, 100.0)`` — flow-rate anomaly metric::

      A_vol = 100.0 · (1.0 − exp(− packet_rate / (baseline_rate + ε)))

  with ``ε = 1e-10`` preventing division by zero. The exponential saturation
  ensures ``packet_rate = 0 → A_vol = 0`` and ``packet_rate → ∞ → A_vol →
  100``.

* ``I_asset ∈ [0.0, 100.0]`` — destination-asset criticality, resolved by
  looking up ``dst_ip`` in an internal asset priority table. Both exact IP
  strings and CIDR ranges are supported. Unrecognised destination IPs
  default to ``30.0``.

Risk tiers → recommended action
-------------------------------

================  =========  ===================================================
Range             Tier       Recommended action
================  =========  ===================================================
[0.0, 30.0)       LOW        ALLOW — log silently, standard telemetry update
[30.0, 60.0)      MEDIUM     LOG_RATE_LIMIT — flag on SOC dashboard,
                             rate-limit source
[60.0, 85.0)       HIGH       QUARANTINE_TEMP — audible alert, 60-second host
                             isolation
[85.0, 100.0]     CRITICAL   QUARANTINE_ISOLATE — 300-second firewall
                             quarantine lease, disconnect stateful sessions
================  =========  ===================================================

The engine is **deterministic and stateless** after construction: the same
inputs always yield the same score, and ``compute_risk`` performs only
read-only operations on the compiled asset table, so the engine is safe for
concurrent use across multiple inference threads.

Usage
-----
.. code-block:: python

    engine = RiskEngine()
    report = engine.compute_risk(
        predicted_class=2,
        confidence=0.98,
        packet_rate=1500.0,
        dst_ip="10.0.1.10",
    )
    print(report["risk_score"], report["risk_tier"], report["recommended_action"])

Dependencies: Python >= 3.10 standard library only (``math``, ``ipaddress``,
``logging``).
"""

from __future__ import annotations

import ipaddress
import logging
import math
from typing import Final

# --------------------------------------------------------------------------- #
# Public configuration                                                         #
# --------------------------------------------------------------------------- #

LOGGER: Final[logging.Logger] = logging.getLogger("iot23.risk_engine")

#: Default weights — Σ = 1.0 (spec §Mathematical Specification).
DEFAULT_W1: Final[float] = 0.35  # Attack Threat Severity
DEFAULT_W2: Final[float] = 0.25  # Classifier Calibrated Confidence
DEFAULT_W3: Final[float] = 0.20  # Traffic Volume Anomaly
DEFAULT_W4: Final[float] = 0.20  # Target Asset Criticality

#: Attack severity table S_attack ∈ [0, 5] (MITRE ATT&CK / Threat Taxonomy).
ATTACK_SEVERITY: Final[dict[int, float]] = {
    0: 0.0,   # Benign
    1: 1.5,   # Recon / PortScan
    2: 4.0,   # DoS / DDoS
    3: 4.5,   # Malware / Okiru
    4: 5.0,   # Botnet C&C
}

#: Attack class names — kept in sync with ``CLASS_NAMES`` in
#: ``src/classifiers.py`` and ``src/dataset.py``.
ATTACK_CLASS_NAMES: Final[dict[int, str]] = {
    0: "Benign",
    1: "Recon / PortScan",
    2: "DoS / DDoS",
    3: "Malware / Okiru",
    4: "Botnet C&C",
}

#: Number of threat classes recognised by the engine.
NUM_CLASSES: Final[int] = 5

#: Default baseline packet rate (pkts/sec) for the volume anomaly metric.
DEFAULT_BASELINE_RATE: Final[float] = 50.0

#: Numerical stabiliser — prevents division by zero when ``baseline_rate = 0``.
VOLUME_EPSILON: Final[float] = 1e-10

#: Default criticality returned for destination IPs absent from the asset table.
DEFAULT_ASSET_CRITICALITY: Final[float] = 30.0

#: Default asset priority table (CIDR ranges cover each criticality tier).
DEFAULT_ASSET_TABLE: Final[dict[str, float]] = {
    # Sandbox / Testing nodes — isolated lab segments.
    "10.10.0.0/16": 20.0,
    "172.16.0.0/24": 20.0,
    # Standard client workstations — corporate endpoint fleet.
    "192.168.1.0/24": 40.0,
    "192.168.10.0/24": 40.0,
    # IoT Sensors / Field Devices — OT/ICS edge devices.
    "192.168.50.0/24": 60.0,
    "10.20.0.0/16": 60.0,
    # Database / Core Infrastructure — crown-jewel servers.
    "10.0.1.0/24": 100.0,
    "10.0.2.0/24": 100.0,
}

#: Risk tier upper bounds (half-open intervals — see class docstring table).
TIER_LOW_UPPER: Final[float] = 30.0
TIER_MEDIUM_UPPER: Final[float] = 60.0
TIER_HIGH_UPPER: Final[float] = 85.0
SCORE_MAX: Final[float] = 100.0

#: Recommended action codes.
ACTION_ALLOW: Final[str] = "ALLOW"
ACTION_LOG_RATE_LIMIT: Final[str] = "LOG_RATE_LIMIT"
ACTION_QUARANTINE_TEMP: Final[str] = "QUARANTINE_TEMP"
ACTION_QUARANTINE_ISOLATE: Final[str] = "QUARANTINE_ISOLATE"

__all__: Final[list[str]] = ["RiskEngine"]


# --------------------------------------------------------------------------- #
# RiskEngine                                                                   #
# --------------------------------------------------------------------------- #


class RiskEngine:
    """Deterministic Multi-Factor Risk Scoring Engine.

    Combines four orthogonal risk signals — attack severity, classifier
    confidence, traffic-volume anomaly, and target-asset criticality — into
    a single Composite Risk Score ``R ∈ [0, 100]`` that drives tiered SOC
    response policy.

    The engine is **deterministic**: identical inputs always yield identical
    outputs, making every risk decision fully reproducible and auditable.
    The asset table is compiled once at ``__init__`` time into exact-IP and
    CIDR-network lookups; :meth:`compute_risk` performs only read-only
    operations, so the engine is safe for concurrent use across inference
    threads.

    Args:
        asset_table: Mapping from IP address (exact string) or CIDR range
            to criticality score ``∈ [0, 100]``. When ``None``, the
            module-level :data:`DEFAULT_ASSET_TABLE` is used. Unrecognised
            destination IPs resolve to :data:`DEFAULT_ASSET_CRITICALITY`
            (``30.0``).
        baseline_rate: Baseline packets-per-second for the volume anomaly
            metric. Must be strictly positive. Default ``50.0``.
        weights: Optional 4-tuple ``(w1, w2, w3, w4)`` overriding the default
            weights. Must be non-negative and sum to ``1.0`` (within
            ``1e-6``). When ``None`` the defaults ``(0.35, 0.25, 0.20,
            0.20)`` are used.

    Raises:
        ValueError: If ``baseline_rate ≤ 0``, the weights are invalid (do not
            sum to 1.0 or contain a negative value), or any criticality in
            ``asset_table`` is outside ``[0, 100]`` (a warning is logged and
            the value is clamped).
    """

    def __init__(
        self,
        asset_table: dict[str, float] | None = None,
        baseline_rate: float = DEFAULT_BASELINE_RATE,
        *,
        weights: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Initialise the risk engine with an asset table and scoring weights."""
        if baseline_rate <= 0.0:
            raise ValueError(f"baseline_rate must be strictly positive, got {baseline_rate}.")
        self.baseline_rate: float = float(baseline_rate)

        if weights is None:
            self.w1, self.w2, self.w3, self.w4 = (
                DEFAULT_W1, DEFAULT_W2, DEFAULT_W3, DEFAULT_W4,
            )
        else:
            self.w1, self.w2, self.w3, self.w4 = (
                float(weights[0]), float(weights[1]),
                float(weights[2]), float(weights[3]),
            )
            total = self.w1 + self.w2 + self.w3 + self.w4
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise ValueError(
                    f"weights must sum to 1.0 (within 1e-6), got {total:.6f}."
                )
            for label, value in (
                ("w1", self.w1), ("w2", self.w2),
                ("w3", self.w3), ("w4", self.w4),
            ):
                if value < 0.0:
                    raise ValueError(f"weight {label} must be non-negative, got {value}.")

        # Compile the asset table — exact-IP dict + CIDR-network list.
        # Both structures are read-only after construction, so
        # ``compute_risk`` is thread-safe.
        self._asset_exact: dict[str, float] = {}
        self._asset_networks: list[
            tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, float]
        ] = []
        self._compile_asset_table(
            asset_table if asset_table is not None else dict(DEFAULT_ASSET_TABLE)
        )

        LOGGER.info(
            "RiskEngine initialised — baseline_rate=%.2f pkts/s | "
            "weights=(w1=%.2f, w2=%.2f, w3=%.2f, w4=%.2f) | "
            "asset_table: %d exact IP(s), %d CIDR range(s).",
            self.baseline_rate, self.w1, self.w2, self.w3, self.w4,
            len(self._asset_exact), len(self._asset_networks),
        )

    def __repr__(self) -> str:
        """Compact summary for logs."""
        return (
            f"RiskEngine(baseline_rate={self.baseline_rate}, "
            f"weights=({self.w1}, {self.w2}, {self.w3}, {self.w4}), "
            f"assets={len(self._asset_exact)}exact+{len(self._asset_networks)}cidr)"
        )

    # -- Asset table compilation ------------------------------------------- #

    def _compile_asset_table(self, table: dict[str, float]) -> None:
        """Split the user / default asset table into exact-IP and CIDR lookups.

        Keys containing ``/`` are parsed as CIDR networks (``strict=False``
        tolerates host bits set); anything else is stored as a literal
        string match. Criticality values are clamped to ``[0, 100]`` with a
        warning.
        """
        for key, criticality in table.items():
            clamped = self._clamp(float(criticality), 0.0, SCORE_MAX)
            if clamped != float(criticality):
                LOGGER.warning(
                    "Asset criticality for '%s' clamped from %.4f to %.4f "
                    "(outside [0, 100]).",
                    key, criticality, clamped,
                )
            if "/" in key:
                try:
                    network = ipaddress.ip_network(key, strict=False)
                except ValueError:
                    LOGGER.warning(
                        "Asset table key '%s' is not a valid CIDR network — "
                        "treating it as an exact IP string match.", key,
                    )
                    self._asset_exact[key] = clamped
                else:
                    self._asset_networks.append((network, clamped))
            else:
                self._asset_exact[key] = clamped

    def _lookup_asset_criticality(self, dst_ip: str) -> float:
        """Resolve the destination IP's criticality from the asset table.

        Lookup order (first hit wins):

        1. Exact string match against literal keys (fast path for
           single-IP entries).
        2. ``ipaddress``-based CIDR membership test against compiled
           networks.
        3. Fallback to :data:`DEFAULT_ASSET_CRITICALITY` (``30.0``).

        A malformed ``dst_ip`` (neither a valid IP nor an exact key) logs a
        warning and returns the default — the engine never raises on
        unresolvable destinations.
        """
        if dst_ip in self._asset_exact:
            return self._asset_exact[dst_ip]
        try:
            ip = ipaddress.ip_address(dst_ip)
        except ValueError:
            LOGGER.warning(
                "Destination IP '%s' is not a valid IP address — "
                "using default criticality %.1f.",
                dst_ip, DEFAULT_ASSET_CRITICALITY,
            )
            return DEFAULT_ASSET_CRITICALITY
        for network, criticality in self._asset_networks:
            if ip in network:
                return criticality
        return DEFAULT_ASSET_CRITICALITY

    # -- Component computations ------------------------------------------- #

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        """Clamp ``value`` into ``[lo, hi]``."""
        return max(lo, min(hi, value))

    def _compute_volume_anomaly(self, packet_rate: float) -> float:
        """Compute ``A_vol ∈ [0.0, 100.0)`` from the observed packet rate.

        ``A_vol = 100.0 · (1.0 − exp(− packet_rate / (baseline_rate + ε)))``

        The exponential saturation ensures: ``packet_rate = 0 → A_vol = 0``
        and ``packet_rate → ∞ → A_vol → 100``.
        """
        exponent = -packet_rate / (self.baseline_rate + VOLUME_EPSILON)
        return 100.0 * (1.0 - math.exp(exponent))

    # -- Public API ------------------------------------------------------- #

    def compute_risk(
        self,
        predicted_class: int,
        confidence: float,
        packet_rate: float,
        dst_ip: str,
    ) -> dict:
        """Compute the Composite Risk Score for a single network flow.

        Args:
            predicted_class: Integer class index predicted by the Level-1
                stacking meta-learner. Must be in ``{0, 1, 2, 3, 4}``.
            confidence: Calibrated softmax probability or confidence for
                the predicted class, ``∈ [0.0, 1.0]``.
            packet_rate: Packets per second observed in the flow. Must be
                non-negative.
            dst_ip: Destination IP address string (IPv4 or IPv6). Used to
                resolve the asset criticality from the internal table.

        Returns:
            A structured dictionary with the following keys:

            * ``risk_score`` (``float``) — Composite Risk Score, rounded to
              1 decimal place, ``∈ [0.0, 100.0]``.
            * ``risk_tier`` (``str``) — one of ``"LOW"``, ``"MEDIUM"``,
              ``"HIGH"``, ``"CRITICAL"``.
            * ``recommended_action`` (``str``) — one of ``"ALLOW"``,
              ``"LOG_RATE_LIMIT"``, ``"QUARANTINE_TEMP"``,
              ``"QUARANTINE_ISOLATE"``.
            * ``breakdown`` (``dict``) — individual component values
              (``S_attack``, ``C_model``, ``A_vol``, ``I_asset``) plus the
              active weights, weighted contributions, the raw composite
              before clamping, the baseline rate, the destination IP and
              the resolved attack class name — everything an auditor needs
              to reproduce the score by hand.

        Raises:
            ValueError: If ``predicted_class`` is outside ``{0, …, 4}``,
                ``confidence`` is outside ``[0, 1]``, ``packet_rate`` is
                negative, or ``dst_ip`` is empty / not a string.
        """
        # --- Input validation & coercion ------------------------------------ #
        try:
            cls = int(predicted_class)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"predicted_class must be an integer in {sorted(ATTACK_SEVERITY)}, "
                f"got {predicted_class!r}."
            ) from error
        if cls not in ATTACK_SEVERITY:
            raise ValueError(
                f"predicted_class must be in {sorted(ATTACK_SEVERITY)}, got {cls}."
            )

        try:
            c_model = float(confidence)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"confidence must be a real number in [0.0, 1.0], got {confidence!r}."
            ) from error
        if not math.isfinite(c_model) or not 0.0 <= c_model <= 1.0:
            raise ValueError(f"confidence must lie in [0.0, 1.0], got {c_model}.")

        try:
            pkt_rate = float(packet_rate)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"packet_rate must be a non-negative real number, got {packet_rate!r}."
            ) from error
        if not math.isfinite(pkt_rate) or pkt_rate < 0.0:
            raise ValueError(f"packet_rate must be non-negative, got {pkt_rate}.")

        if not isinstance(dst_ip, str) or not dst_ip.strip():
            raise ValueError(f"dst_ip must be a non-empty string, got {dst_ip!r}.")
        dst_ip = dst_ip.strip()

        # --- Component values ------------------------------------------------ #
        s_attack = ATTACK_SEVERITY[cls]
        a_vol = self._compute_volume_anomaly(pkt_rate)
        i_asset = self._lookup_asset_criticality(dst_ip)

        # --- Weighted contributions ------------------------------------------- #
        contrib_attack = self.w1 * (s_attack * 20.0)
        contrib_confidence = self.w2 * (c_model * 100.0)
        contrib_volume = self.w3 * a_vol
        contrib_asset = self.w4 * i_asset
        raw_composite = (
            contrib_attack + contrib_confidence
            + contrib_volume + contrib_asset
        )

        # --- Clamp + round + tier -------------------------------------------- #
        risk_score = round(self._clamp(raw_composite, 0.0, SCORE_MAX), 1)
        risk_tier, recommended_action = self._classify_tier(risk_score)

        LOGGER.debug(
            "compute_risk — class=%d (%s) conf=%.4f rate=%.2f dst=%s → "
            "S=%.1f C=%.4f A=%.4f I=%.1f | raw=%.4f score=%.1f tier=%s action=%s",
            cls, ATTACK_CLASS_NAMES[cls], c_model, pkt_rate, dst_ip,
            s_attack, c_model, a_vol, i_asset,
            raw_composite, risk_score, risk_tier, recommended_action,
        )

        return {
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "recommended_action": recommended_action,
            "breakdown": {
                # Spec-required component values
                "S_attack": s_attack,
                "C_model": c_model,
                "A_vol": round(a_vol, 4),
                "I_asset": i_asset,
                # Context for audit trail
                "predicted_class": cls,
                "attack_class_name": ATTACK_CLASS_NAMES[cls],
                "dst_ip": dst_ip,
                "baseline_rate": self.baseline_rate,
                # Active weights
                "weights": {
                    "w1_attack_severity": self.w1,
                    "w2_model_confidence": self.w2,
                    "w3_volume_anomaly": self.w3,
                    "w4_asset_criticality": self.w4,
                },
                # Weighted contributions — sum equals raw_composite
                "weighted_contributions": {
                    "w1_S_attack_x20": round(contrib_attack, 4),
                    "w2_C_model_x100": round(contrib_confidence, 4),
                    "w3_A_vol": round(contrib_volume, 4),
                    "w4_I_asset": round(contrib_asset, 4),
                },
                "raw_composite": round(raw_composite, 4),
            },
        }

    def _classify_tier(self, score: float) -> tuple[str, str]:
        """Map a clamped risk score to ``(risk_tier, recommended_action)``.

        Tier boundaries (half-open intervals):

        * ``[0, 30)``     → LOW / ALLOW
        * ``[30, 60)``    → MEDIUM / LOG_RATE_LIMIT
        * ``[60, 85)``    → HIGH / QUARANTINE_TEMP
        * ``[85, 100]``   → CRITICAL / QUARANTINE_ISOLATE
        """
        if score < TIER_LOW_UPPER:
            return "LOW", ACTION_ALLOW
        if score < TIER_MEDIUM_UPPER:
            return "MEDIUM", ACTION_LOG_RATE_LIMIT
        if score < TIER_HIGH_UPPER:
            return "HIGH", ACTION_QUARANTINE_TEMP
        return "CRITICAL", ACTION_QUARANTINE_ISOLATE


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #


def _run_smoke_test() -> int:
    """End-to-end verification of the Composite Risk Score engine.

    Covers:

    1. A benign low-rate HTTP connection to a sandbox node → **LOW**.
    2. A high-rate DDoS packet flood targeting a core server
       (criticality 100.0) → **CRITICAL**.
    3. A reconnaissance scan against a corporate workstation → **MEDIUM**.
    4. A Botnet C&C beacon to an IoT field device → **HIGH**.
    5. Tier-boundary determinism, argument validation, custom asset tables,
       custom weights, and the full mathematical audit contract (weighted
       contributions must sum to the raw composite).
    """
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.info("=" * 78)
    LOGGER.info("RiskEngine smoke test — Multi-Factor Risk Scoring Engine")
    LOGGER.info("=" * 78)

    try:
        engine = RiskEngine()
        LOGGER.info(
            "Engine: %s | weights: w1=%.2f (attack), w2=%.2f (confidence), "
            "w3=%.2f (volume), w4=%.2f (asset).",
            engine, engine.w1, engine.w2, engine.w3, engine.w4,
        )

        # -- Case 1: benign low-rate HTTP to sandbox → LOW ------------------ #
        case1 = engine.compute_risk(
            predicted_class=0,    # Benign
            confidence=0.92,
            packet_rate=5.0,
            dst_ip="10.10.0.42",  # sandbox → I_asset = 20.0
        )
        LOGGER.info(
            "Case 1 — Benign HTTP to sandbox (10.10.0.42): "
            "score=%.1f, tier=%s, action=%s",
            case1["risk_score"], case1["risk_tier"], case1["recommended_action"],
        )
        LOGGER.info("  Breakdown:\n%s", json.dumps(case1["breakdown"], indent=2))
        assert case1["risk_tier"] == "LOW", (
            f"Expected LOW for benign low-rate flow, got {case1['risk_tier']} "
            f"(score={case1['risk_score']})."
        )
        assert case1["recommended_action"] == "ALLOW"
        assert case1["breakdown"]["I_asset"] == 20.0
        assert case1["breakdown"]["S_attack"] == 0.0

        # -- Case 2: high-rate DDoS flood against core server → CRITICAL ---- #
        case2 = engine.compute_risk(
            predicted_class=2,    # DoS / DDoS
            confidence=0.98,
            packet_rate=2000.0,
            dst_ip="10.0.1.10",   # core infra → I_asset = 100.0
        )
        LOGGER.info(
            "Case 2 — DDoS flood against core server (10.0.1.10): "
            "score=%.1f, tier=%s, action=%s",
            case2["risk_score"], case2["risk_tier"], case2["recommended_action"],
        )
        LOGGER.info("  Breakdown:\n%s", json.dumps(case2["breakdown"], indent=2))
        assert case2["risk_tier"] == "CRITICAL", (
            f"Expected CRITICAL for DDoS flood, got {case2['risk_tier']} "
            f"(score={case2['risk_score']})."
        )
        assert case2["recommended_action"] == "QUARANTINE_ISOLATE"
        assert case2["breakdown"]["I_asset"] == 100.0
        assert case2["breakdown"]["S_attack"] == 4.0
        assert case2["breakdown"]["A_vol"] > 99.0  # exponential saturation

        # -- Case 3: Recon against corporate workstation → MEDIUM ----------- #
        case3 = engine.compute_risk(
            predicted_class=1,    # Recon / PortScan
            confidence=0.70,
            packet_rate=10.0,
            dst_ip="192.168.1.55",  # workstation → I_asset = 40.0
        )
        LOGGER.info(
            "Case 3 — Recon against workstation (192.168.1.55): "
            "score=%.1f, tier=%s, action=%s",
            case3["risk_score"], case3["risk_tier"], case3["recommended_action"],
        )
        assert case3["risk_tier"] == "MEDIUM", (
            f"Expected MEDIUM, got {case3['risk_tier']} "
            f"(score={case3['risk_score']})."
        )
        assert case3["recommended_action"] == "LOG_RATE_LIMIT"

        # -- Case 4: Botnet C&C beacon to IoT field device → HIGH ---------- #
        case4 = engine.compute_risk(
            predicted_class=4,    # Botnet C&C
            confidence=0.65,
            packet_rate=30.0,
            dst_ip="192.168.50.7",  # IoT sensor → I_asset = 60.0
        )
        LOGGER.info(
            "Case 4 — Botnet C&C beacon to IoT sensor (192.168.50.7): "
            "score=%.1f, tier=%s, action=%s",
            case4["risk_score"], case4["risk_tier"], case4["recommended_action"],
        )
        assert case4["risk_tier"] == "HIGH", (
            f"Expected HIGH, got {case4['risk_tier']} "
            f"(score={case4['risk_score']})."
        )
        assert case4["recommended_action"] == "QUARANTINE_TEMP"

        # -- Tier-boundary & determinism checks ----------------------------- #
        # Pure benign to sandbox: only the asset component contributes.
        zero_flow = engine.compute_risk(0, 0.0, 0.0, "10.10.0.1")
        assert zero_flow["risk_score"] == 4.0, (
            f"Pure benign sandbox score should be 4.0 (0.20·20), "
            f"got {zero_flow['risk_score']}."
        )
        assert zero_flow["risk_tier"] == "LOW"

        # Unrecognised IP defaults to 30.0 criticality.
        unknown = engine.compute_risk(0, 0.0, 0.0, "203.0.113.42")
        assert unknown["breakdown"]["I_asset"] == 30.0
        assert unknown["risk_score"] == 6.0  # 0.20 · 30.0
        assert unknown["risk_tier"] == "LOW"

        # Saturation: very high packet rate → A_vol ≈ 100.
        saturated = engine.compute_risk(2, 0.99, 1_000_000.0, "10.0.1.1")
        assert saturated["breakdown"]["A_vol"] > 99.99
        assert saturated["risk_tier"] == "CRITICAL"

        # Determinism: same inputs → identical output.
        repeat = engine.compute_risk(2, 0.99, 1_000_000.0, "10.0.1.1")
        assert repeat == saturated
        LOGGER.info("Tier-boundary + determinism checks OK.")

        # -- Argument validation ------------------------------------------- #
        invalid_inputs = [
            {"predicted_class": 5, "confidence": 0.9, "packet_rate": 10.0, "dst_ip": "10.0.0.1"},
            {"predicted_class": -1, "confidence": 0.9, "packet_rate": 10.0, "dst_ip": "10.0.0.1"},
            {"predicted_class": 0, "confidence": 1.5, "packet_rate": 10.0, "dst_ip": "10.0.0.1"},
            {"predicted_class": 0, "confidence": -0.1, "packet_rate": 10.0, "dst_ip": "10.0.0.1"},
            {"predicted_class": 0, "confidence": 0.9, "packet_rate": -1.0, "dst_ip": "10.0.0.1"},
            {"predicted_class": 0, "confidence": 0.9, "packet_rate": 10.0, "dst_ip": ""},
        ]
        for bad_kwargs in invalid_inputs:
            try:
                engine.compute_risk(**bad_kwargs)
            except ValueError:
                continue
            raise AssertionError(
                f"compute_risk accepted invalid input: {bad_kwargs}"
            )
        LOGGER.info("Argument validation OK — all invalid inputs rejected.")

        # -- Custom asset table + custom baseline --------------------------- #
        custom = RiskEngine(
            asset_table={"203.0.113.5": 100.0, "198.51.100.0/24": 60.0},
            baseline_rate=10.0,
        )
        c_report = custom.compute_risk(3, 0.99, 50.0, "203.0.113.5")
        assert c_report["breakdown"]["I_asset"] == 100.0
        assert c_report["breakdown"]["baseline_rate"] == 10.0
        # CIDR match in the custom table.
        cidr_report = custom.compute_risk(0, 0.0, 0.0, "198.51.100.42")
        assert cidr_report["breakdown"]["I_asset"] == 60.0
        # Unrecognised IP falls back to default.
        fallback = custom.compute_risk(0, 0.0, 0.0, "8.8.8.8")
        assert fallback["breakdown"]["I_asset"] == 30.0
        LOGGER.info("Custom asset table + custom baseline OK.")

        # -- Custom-weight validation -------------------------------------- #
        try:
            RiskEngine(weights=(0.5, 0.5, 0.5, 0.5))  # sum = 2.0
            raise AssertionError("RiskEngine accepted weights not summing to 1.0.")
        except ValueError:
            pass
        try:
            RiskEngine(weights=(-0.1, 0.5, 0.3, 0.3))  # negative
            raise AssertionError("RiskEngine accepted a negative weight.")
        except ValueError:
            pass
        reweighted = RiskEngine(weights=(0.50, 0.20, 0.15, 0.15))
        assert reweighted.w1 == 0.50 and reweighted.w4 == 0.15
        LOGGER.info("Custom-weight validation OK.")

        # -- Auditability: weighted contributions sum to raw composite ------- #
        audit = engine.compute_risk(3, 0.85, 100.0, "10.0.2.5")
        wc = audit["breakdown"]["weighted_contributions"]
        total_wc = (
            wc["w1_S_attack_x20"]
            + wc["w2_C_model_x100"]
            + wc["w3_A_vol"]
            + wc["w4_I_asset"]
        )
        assert math.isclose(total_wc, audit["breakdown"]["raw_composite"], abs_tol=1e-3), (
            f"Weighted contributions {total_wc} ≠ raw composite "
            f"{audit['breakdown']['raw_composite']}."
        )
        LOGGER.info("Auditability check OK — weighted contributions sum to raw composite.")

        # -- Summary table -------------------------------------------------- #
        LOGGER.info("-" * 78)
        LOGGER.info(
            "%-6s %-18s %6s %8s %8s %8s %8s  %-8s %-22s",
            "Case", "Description", "Class", "Conf", "Rate", "Asset", "Score", "Tier", "Action",
        )
        for case, desc in (
            (case1, "Benign HTTP → sandbox"),
            (case2, "DDoS flood → core"),
            (case3, "Recon → workstation"),
            (case4, "C&C beacon → IoT"),
        ):
            b = case["breakdown"]
            LOGGER.info(
                "%-6s %-18s %6d %8.2f %8.1f %8.1f %8.1f  %-8s %-22s",
                "", desc, b["predicted_class"], b["C_model"],
                b["baseline_rate"] if False else case["breakdown"].get("packet_rate", 0.0)
                if "packet_rate" in b else 0.0,
                b["I_asset"], case["risk_score"],
                case["risk_tier"], case["recommended_action"],
            )
        LOGGER.info("-" * 78)

        LOGGER.info("=" * 78)
        LOGGER.info("SMOKE TEST PASSED — Composite Risk Score engine verified end-to-end.")
        return 0
    except Exception:
        LOGGER.exception("SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_smoke_test())