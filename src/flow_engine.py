"""Real-time network flow aggregation and ML inference pipeline.

This module implements the live inference stage of the IoT-23 NIDS. It captures
live network traffic (or parses PCAP files) using ``NFStream``, extracts
bidirectional statistical flow features, aligns and scales them using the
fitted ``scaler.joblib`` and ``feature_names.joblib``, compresses them with
the Deep Sparse Autoencoder (``dsae_encoder.pt``), and executes inference
using the Level-0 ensemble (``SpatialDNN``, ``TemporalLSTM``) and the Level-1
Meta-Learner (``meta_learner.joblib``). Enriched flow detection records are
yielded in real time.

Pipeline position
-----------------
::

    src/dataset.py   train_dsae.py   train_ensemble.py        src/flow_engine.py (this module)
    ──────────────   ────────────    ────────────────────     ─────────────────────────────────
    scaler.joblib    dsae_encoder    dnn_level0.pt            Live / PCAP Traffic
    feature_names    .pt            lstm_level0.pt             │
                                    meta_learner.joblib       ▼
                                                          NFStreamer
                                                          → Feature Alignment
                                                          → DSAE Encoding
                                                          → Sliding Window (LSTM)
                                                          → Level-0 Inference
                                                          → Level-1 Meta-Learner
                                                          → Enriched Flow Record
"""

from __future__ import annotations

import logging
import math
from collections import deque
from pathlib import Path
from typing import Any, Final, Iterator

import joblib
import numpy as np
import torch
import torch.nn as nn
from nfstream import NFStreamer

# Local module imports for model architectures
from src.classifiers import SpatialDNN, TemporalLSTM, DEFAULT_LATENT_DIM, DEFAULT_NUM_CLASSES
from src.autoencoder import DeepSparseAutoencoder

# --------------------------------------------------------------------------- #
# Public configuration                                                         #
# --------------------------------------------------------------------------- #

LOGGER: Final[logging.Logger] = logging.getLogger("iot23.flow_engine")

#: Artifacts consumed (from previous pipeline stages)
SCALER_PATH: Final[Path] = Path("data/processed/scaler.joblib")
FEATURE_NAMES_PATH: Final[Path] = Path("data/processed/feature_names.joblib")
DSAE_ENCODER_PATH: Final[Path] = Path("models/saved/dsae_encoder.pt")
DNN_MODEL_PATH: Final[Path] = Path("models/saved/dnn_level0.pt")
LSTM_MODEL_PATH: Final[Path] = Path("models/saved/lstm_level0.pt")
META_LEARNER_PATH: Final[Path] = Path("models/saved/meta_learner.joblib")

#: Sliding window length for the TemporalLSTM (must match training).
SEQ_LEN: Final[int] = 5

#: Simplified attack type names for the output schema.
ATTACK_TYPES: Final[dict[int, str]] = {
    0: "Benign",
    1: "Recon",
    2: "DoS",
    3: "Malware",
    4: "Botnet",
}

__all__: Final[list[str]] = ["FlowInferenceEngine", "LiveFlowAggregator"]


# --------------------------------------------------------------------------- #
# Internal utilities                                                           #
# --------------------------------------------------------------------------- #


def _load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
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


def _extract_nfstream_features(flow: Any) -> dict[str, Any]:
    """Map NFStream bidirectional flow attributes to Zeek-equivalent features.

    NFStream provides rich bidirectional metrics. This function extracts the
    9 raw features (7 numeric, 2 categorical) expected by our preprocessing
    pipeline, applying the mappings specified in the engineering requirements.
    """
    proto_map = {6: "tcp", 17: "udp", 1: "icmp"}
    proto_int = getattr(flow, "protocol", 0)
    proto = proto_map.get(proto_int, "unknown")

    # NFStream does not natively provide Zeek's `conn_state` string.
    # We safely default to 'unknown' (or 's0') if the attribute is absent.
    conn_state = getattr(flow, "state", None)
    conn_state = str(conn_state).strip().lower() if conn_state else "unknown"

    return {
        "duration": getattr(flow, "bidirectional_duration_ms", 0.0) / 1000.0,
        "orig_bytes": getattr(flow, "src2dst_bytes", 0.0),
        "resp_bytes": getattr(flow, "dst2src_bytes", 0.0),
        "orig_pkts": getattr(flow, "src2dst_packets", 0.0),
        "resp_pkts": getattr(flow, "dst2src_packets", 0.0),
        "orig_ip_bytes": getattr(flow, "src2dst_bytes", 0.0),  # Spec mapping
        "resp_ip_bytes": getattr(flow, "dst2src_bytes", 0.0),  # Spec mapping
        "proto": proto,
        "conn_state": conn_state,
    }


# --------------------------------------------------------------------------- #
# Flow Inference Engine                                                        #
# --------------------------------------------------------------------------- #


class FlowInferenceEngine:
    """Encapsulates the ML inference logic for a single network flow.

    Loads and manages the pre-fitted transformers and pre-trained PyTorch
    and scikit-learn models. It maintains an in-memory FIFO sequence buffer
    of latent vectors for the temporal learner.

    Args:
        device: Target compute device (CPU by default for portability).
    """

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        """Load all required artifacts and initialise the sliding window buffer."""
        self.device = device

        # Verify artifact existence to fail fast cleanly
        required_artifacts = {
            "Scaler": SCALER_PATH,
            "Feature Names": FEATURE_NAMES_PATH,
            "DSAE Encoder": DSAE_ENCODER_PATH,
            "SpatialDNN": DNN_MODEL_PATH,
            "TemporalLSTM": LSTM_MODEL_PATH,
            "Meta-Learner": META_LEARNER_PATH,
        }
        for name, path in required_artifacts.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Required artifact '{name}' not found at '{path}'. "
                    "Please run the training pipeline (dataset.py, train_dsae.py, train_ensemble.py) first."
                )

        # 1. Load preprocessing metadata and scaler
        self.feature_metadata = joblib.load(FEATURE_NAMES_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.numeric_features = self.feature_metadata["numeric_features"]
        self.dummy_columns = self.feature_metadata["dummy_columns"]

        # 2. Load DSAE Encoder
        input_dim = len(self.numeric_features) + len(self.dummy_columns)
        self.dsae = DeepSparseAutoencoder(input_dim=input_dim, latent_dim=DEFAULT_LATENT_DIM).to(self.device)
        self.dsae.encoder.load_state_dict(_load_state_dict(DSAE_ENCODER_PATH))
        self.dsae.eval()

        # 3. Load Level-0 models
        self.dnn = SpatialDNN(latent_dim=DEFAULT_LATENT_DIM, num_classes=DEFAULT_NUM_CLASSES).to(self.device)
        self.dnn.load_state_dict(_load_state_dict(DNN_MODEL_PATH))
        self.dnn.eval()

        self.lstm = TemporalLSTM(latent_dim=DEFAULT_LATENT_DIM, num_classes=DEFAULT_NUM_CLASSES).to(self.device)
        self.lstm.load_state_dict(_load_state_dict(LSTM_MODEL_PATH))
        self.lstm.eval()

        # 4. Load Level-1 meta-learner
        self.meta_learner = joblib.load(META_LEARNER_PATH)

        # 5. Sliding window management for LSTM
        self.seq_len = SEQ_LEN
        self.z_buffer = deque(maxlen=self.seq_len)

        LOGGER.info("FlowInferenceEngine initialised. Models loaded onto %s.", self.device)

    def _preprocess_features(self, raw_features: dict[str, Any]) -> np.ndarray:
        """Align, one-hot encode, and scale raw flow features.

        Args:
            raw_features: Dictionary of extracted NFStream features.

        Returns:
            A 1-D float32 NumPy array of shape ``(input_dim,)`` ready for the DSAE.
        """
        # Numeric block
        numeric_vals = np.array(
            [[raw_features.get(c, 0.0) for c in self.numeric_features]],
            dtype=np.float32
        )
        # Sanitise: NFStream should yield finite values, but guard against NaN/Inf.
        numeric_vals = np.nan_to_num(numeric_vals, nan=0.0, posinf=0.0, neginf=0.0)

        # Categorical one-hot block
        proto = raw_features.get("proto", "unknown")
        conn_state = raw_features.get("conn_state", "unknown")

        dummy_vec = np.zeros((1, len(self.dummy_columns)), dtype=np.float32)
        for i, col in enumerate(self.dummy_columns):
            if col.startswith("proto_"):
                val = col[len("proto_"):]
                if proto == val:
                    dummy_vec[0, i] = 1.0
            elif col.startswith("conn_state_"):
                val = col[len("conn_state_"):]
                if conn_state == val:
                    dummy_vec[0, i] = 1.0

        # Scale numeric columns using the fitted RobustScaler
        scaled_numeric = self.scaler.transform(numeric_vals).astype(np.float32)

        # Concatenate scaled numeric + one-hot categoricals
        x_vector = np.concatenate([scaled_numeric, dummy_vec], axis=1).astype(np.float32)
        return x_vector

    def predict_flow(self, flow: Any) -> dict[str, Any]:
        """Run the complete inference pipeline for a single network flow.

        Args:
            flow: An ``NFFlow`` object (or mock object with equivalent attributes).

        Returns:
            A structured dictionary containing the enriched flow detection record.
        """
        # 1. Feature extraction & preprocessing
        raw_features = _extract_nfstream_features(flow)
        x_vector = self._preprocess_features(raw_features)
        x_tensor = torch.from_numpy(x_vector).to(self.device)

        # 2. Compress with DSAE
        with torch.no_grad():
            z_tensor = self.dsae.encode(x_tensor)  # (1, 32)

        z_np = z_tensor.cpu().numpy().astype(np.float32)

        # 3. Sliding Window Management for LSTM
        self.z_buffer.append(z_np[0])
        current_len = len(self.z_buffer)
        seq_list = list(self.z_buffer)
        if current_len < self.seq_len:
            # Pad the sequence with the current flow's latent representation
            padding = [z_np[0]] * (self.seq_len - current_len)
            seq_list = padding + seq_list

        seq_np = np.stack(seq_list, axis=0)  # (5, 32)
        seq_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(self.device)  # (1, 5, 32)

        # 4. Level-0 Inference (Pass the compressed 32-dim latent vector z_tensor to SpatialDNN)
        with torch.no_grad():
            p_dnn = self.dnn.predict_proba(z_tensor)       # (1, 5)
            p_lstm = self.lstm.predict_proba(seq_tensor)   # (1, 5)

        # 5. Level-1 Meta-Learner Inference
        meta_features = torch.cat([p_dnn, p_lstm], dim=-1).cpu().numpy()  # (1, 10)
        pred_class = int(self.meta_learner.predict(meta_features)[0])
        confidence = float(self.meta_learner.predict_proba(meta_features)[0, pred_class])

        # 6. Flow Metrics Calculation
        total_packets = int(getattr(flow, "src2dst_packets", 0)) + int(getattr(flow, "dst2src_packets", 0))
        duration_s = float(getattr(flow, "bidirectional_duration_ms", 0.0)) / 1000.0
        packet_rate = total_packets / duration_s if duration_s > 0 else 0.0

        # 7. Construct Enriched Flow Record
        report = {
            # 5-Tuple
            "src_ip": getattr(flow, "src_ip", ""),
            "dst_ip": getattr(flow, "dst_ip", ""),
            "src_port": int(getattr(flow, "src_port", 0)),
            "dst_port": int(getattr(flow, "dst_port", 0)),
            "protocol": getattr(flow, "protocol_name", "UNKNOWN"),
            # Flow Metrics
            "duration": duration_s,
            "total_packets": total_packets,
            "packet_rate": packet_rate,
            # ML Predictions
            "predicted_class": pred_class,
            "attack_type": ATTACK_TYPES.get(pred_class, "Unknown"),
            "confidence": confidence,
            # Latent Vector (1x32)
            "z_vector": z_np,
        }
        return report


# --------------------------------------------------------------------------- #
# Live Flow Aggregator                                                         #
# --------------------------------------------------------------------------- #


class LiveFlowAggregator:
    """Real-time network flow aggregation and ML inference pipeline.

    Wraps ``NFStreamer`` to capture live traffic or parse PCAP files, passing
    each expired flow through the :class:`FlowInferenceEngine` to produce
    enriched detection records.

    Args:
        interface: The network interface to capture live traffic from
            (e.g., ``"eth0"``). Ignored if ``pcap_path`` is provided.
        pcap_path: Path to a PCAP file for offline analysis. When provided,
            this takes precedence over ``interface``.
    """

    def __init__(self, interface: str = "eth0", pcap_path: str | None = None) -> None:
        """Initialise the aggregator and the underlying inference engine."""
        self.interface = interface
        self.pcap_path = pcap_path
        self.inference_engine = FlowInferenceEngine()
        LOGGER.info(
            "LiveFlowAggregator initialised. Source: %s",
            f"PCAP ({pcap_path})" if pcap_path else f"Interface ({interface})"
        )

    def stream_flows(self) -> Iterator[dict[str, Any]]:
        """Yield enriched flow detection records as they expire.

        Configures ``NFStreamer`` with statistical analysis disabled (we only
        need the raw bidirectional counters). Each expired flow is passed to
        the inference engine and the resulting report is yielded.

        Yields:
            A structured dictionary containing the enriched flow detection record.
        """
        source = self.pcap_path if self.pcap_path else self.interface
        streamer = NFStreamer(
            source=source,
            statistical_analysis=False  # We only need basic byte/packet counters
        )

        LOGGER.info("Starting NFStream capture on source '%s'...", source)
        try:
            for flow in streamer:
                try:
                    report = self.inference_engine.predict_flow(flow)
                    yield report
                except Exception as flow_error:
                    LOGGER.error(
                        "Failed to process flow %s:%s -> %s:%s: %s",
                        getattr(flow, "src_ip", "?"), getattr(flow, "src_port", "?"),
                        getattr(flow, "dst_ip", "?"), getattr(flow, "dst_port", "?"),
                        flow_error
                    )
        except Exception as stream_error:
            LOGGER.critical("NFStream capture failed on source '%s': %s", source, stream_error)
            raise
        finally:
            LOGGER.info("NFStream capture stopped on source '%s'.", source)


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #


def _run_smoke_test() -> int:
    """Validate the inference pipeline using a mock NFStream generator.

    Verifies:
        1. ``FlowInferenceEngine`` loads models and scalers without error.
        2. The mock flow feature alignment, scaling, and one-hot encoding work.
        3. The sliding window correctly pads and maintains sequences.
        4. The output dictionary schema matches specifications exactly.
        5. Tensor shapes and prediction ranges are valid.

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class MockNFFlow:
        """Minimal mock of an NFFlow object for offline pipeline validation."""
        def __init__(self, i: int) -> None:
            self.id = i
            self.src_ip = f"192.168.1.{i+1}"
            self.dst_ip = "10.0.0.1"
            self.src_port = 40000 + i
            self.dst_port = 80
            self.protocol = 6  # TCP
            self.protocol_name = "TCP"
            self.bidirectional_duration_ms = 1500.0 + i * 100
            self.src2dst_bytes = 1000 * (i + 1)
            self.dst2src_bytes = 500
            self.src2dst_packets = 10 + i
            self.dst2src_packets = 5 + i
            self.state = "s0"

    LOGGER.info("=" * 78)
    LOGGER.info("Flow Engine smoke test — Mock NFStream generator")
    LOGGER.info("=" * 78)

    try:
        # Instantiate the inference engine (loads all artifacts)
        engine = FlowInferenceEngine()

        # Simulate 3 sequential flows
        for i in range(3):
            mock_flow = MockNFFlow(i)
            LOGGER.info("--- Processing mock flow %d ---", i + 1)

            report = engine.predict_flow(mock_flow)

            # --- Output Dictionary Schema Validation ---
            # 5-Tuple
            assert "src_ip" in report and isinstance(report["src_ip"], str)
            assert "dst_ip" in report and isinstance(report["dst_ip"], str)
            assert "src_port" in report and isinstance(report["src_port"], int)
            assert "dst_port" in report and isinstance(report["dst_port"], int)
            assert "protocol" in report and isinstance(report["protocol"], str)

            # Flow Metrics
            assert "duration" in report and isinstance(report["duration"], float)
            assert "total_packets" in report and isinstance(report["total_packets"], int)
            assert "packet_rate" in report and isinstance(report["packet_rate"], float)
            assert report["packet_rate"] >= 0.0

            # ML Predictions
            assert "predicted_class" in report and isinstance(report["predicted_class"], int)
            assert 0 <= report["predicted_class"] <= 4
            assert "attack_type" in report and isinstance(report["attack_type"], str)
            assert "confidence" in report and isinstance(report["confidence"], float)
            assert 0.0 <= report["confidence"] <= 1.0

            # Latent Vector
            assert "z_vector" in report and isinstance(report["z_vector"], np.ndarray)
            assert report["z_vector"].shape == (1, 32)
            assert np.isfinite(report["z_vector"]).all()

            LOGGER.info(
                "Flow %d Result: %s:%s -> %s:%s (%s) | Class: %d (%s) | Conf: %.4f | Rate: %.2f pps",
                i + 1,
                report["src_ip"], report["src_port"],
                report["dst_ip"], report["dst_port"],
                report["protocol"],
                report["predicted_class"], report["attack_type"],
                report["confidence"], report["packet_rate"]
            )

        # Verify the LSTM sliding window properly handled the 3 flows
        assert len(engine.z_buffer) == 3, "Sliding window buffer should contain 3 flows."
        LOGGER.info("Sliding window buffer maintained correctly (length=%d).", len(engine.z_buffer))

        LOGGER.info("=" * 78)
        LOGGER.info("SMOKE TEST PASSED — Real-time inference pipeline verified.")
        return 0
    except Exception:
        LOGGER.exception("SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_smoke_test())