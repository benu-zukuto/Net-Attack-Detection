"""Centralized backend orchestration server for the NIDS.

Provides a high-throughput, asynchronous FastAPI server that coordinates live
packet flow ingestion, risk scoring, explainable AI attributions, and automated
firewall containment, while streaming live telemetry and security alerts over
WebSockets to the SOC dashboard.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Local module imports
from src.risk_engine import RiskEngine
from src.mitigation import MitigationEngine

# Optional heavy dependencies for live inference and XAI
try:
    import numpy as np
    import torch
    import joblib
    from src.flow_engine import FlowInferenceEngine, LiveFlowAggregator
    from src.explainer import FlowExplainer
    from src.classifiers import SpatialDNN, DEFAULT_LATENT_DIM, DEFAULT_NUM_CLASSES
    ML_DEPS_AVAILABLE = True
except ImportError:
    ML_DEPS_AVAILABLE = False
    np = None  # type: ignore

# --------------------------------------------------------------------------- #
# Public configuration                                                         #
# --------------------------------------------------------------------------- #

LOGGER: Final[logging.Logger] = logging.getLogger("iot23.server")
ARGS: argparse.Namespace | None = None

__all__: Final[list[str]] = ["app"]


# --------------------------------------------------------------------------- #
# Pydantic Models for API Serialization                                       #
# --------------------------------------------------------------------------- #

class TopFeature(BaseModel):
    feature_idx: int
    feature_name: str
    shap_value: float
    direction: str

class EnrichedEvent(BaseModel):
    timestamp: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    duration: float
    total_packets: int
    packet_rate: float
    predicted_class: int
    attack_type: str
    confidence: float
    risk_score: float
    risk_tier: str
    action_taken: str
    top_features: list[TopFeature] = []
    explanation_text: str

class StatusResponse(BaseModel):
    status: str
    uptime_seconds: float
    active_interface: str
    total_flows_analyzed: int
    total_threats_detected: int
    active_mitigations: int
    pipeline_paused: bool

class ReleaseResponse(BaseModel):
    success: bool
    ip: str

class ToggleResponse(BaseModel):
    paused: bool


# --------------------------------------------------------------------------- #
# Mock Generator & Pipeline Worker                                            #
# --------------------------------------------------------------------------- #

def generate_mock_flow() -> dict[str, Any]:
    """Generate synthetic network flow dictionaries for safe offline testing."""
    attack_classes = [0, 1, 2, 3, 4]
    cls = random.choices(attack_classes, weights=[70, 8, 8, 7, 7])[0]
    return {
        "src_ip": f"192.168.1.{random.randint(2, 254)}",
        "dst_ip": random.choice(["10.0.0.1", "10.0.1.10", "8.8.8.8", "192.168.50.7"]),
        "src_port": random.randint(1024, 65535),
        "dst_port": random.choice([80, 443, 22, 8080]),
        "protocol": "TCP",
        "duration": round(random.uniform(0.1, 5.0), 2),
        "total_packets": random.randint(5, 500),
        "packet_rate": round(random.uniform(1.0, 1500.0), 2),
        "predicted_class": cls,
        "attack_type": ["Benign", "Recon", "DoS", "Malware", "Botnet"][cls],
        "confidence": round(random.uniform(0.7, 0.99), 4),
        "z_vector": np.random.rand(1, 32).astype(np.float32) if ML_DEPS_AVAILABLE else None
    }

async def pipeline_worker(app: FastAPI) -> None:
    """Background async task to process network flows sequentially."""
    loop = asyncio.get_running_loop()
    flow_queue = asyncio.Queue()

    def ingestion_thread() -> None:
        if app.state.mock:
            while True:
                if not app.state.is_paused:
                    flow = generate_mock_flow()
                    asyncio.run_coroutine_threadsafe(flow_queue.put(flow), loop)
                time.sleep(1.0)
        else:
            try:
                aggregator = LiveFlowAggregator(interface=app.state.interface)
                for flow in aggregator.stream_flows():
                    if not app.state.is_paused:
                        asyncio.run_coroutine_threadsafe(flow_queue.put(flow), loop)
            except Exception as e:
                LOGGER.error("Live ingestion failed: %s. Falling back to mock.", e)
                app.state.mock = True
                while True:
                    if not app.state.is_paused:
                        flow = generate_mock_flow()
                        asyncio.run_coroutine_threadsafe(flow_queue.put(flow), loop)
                    time.sleep(1.0)

    threading.Thread(target=ingestion_thread, daemon=True).start()

    while True:
        flow = await flow_queue.get()

        predicted_class = flow["predicted_class"]
        confidence = flow["confidence"]
        packet_rate = flow["packet_rate"]
        dst_ip = flow["dst_ip"]
        src_ip = flow["src_ip"]

        # 1. Compute Risk
        risk_report = app.state.risk_engine.compute_risk(
            predicted_class=predicted_class,
            confidence=confidence,
            packet_rate=packet_rate,
            dst_ip=dst_ip
        )

        risk_score = risk_report["risk_score"]
        recommended_action = risk_report["recommended_action"]

        top_features = []
        explanation_text = "Benign flow."

        # 2. Explainability & Mitigation (if attack or high risk)
        if predicted_class != 0 or risk_score >= 60.0:
            if hasattr(app.state, "explainer") and flow.get("z_vector") is not None:
                try:
                    z_vector = flow["z_vector"]
                    explainer_report = app.state.explainer.explain_flow(z_vector, predicted_class, top_k=5)
                    if isinstance(explainer_report, list):
                        explainer_report = explainer_report[0]
                    top_features = explainer_report.get("top_features", [])
                    explanation_text = explainer_report.get("explanation_text", "")
                except Exception as xai_err:
                    LOGGER.error("XAI explanation failed: %s", xai_err)
                    explanation_text = f"High risk detected for class {predicted_class}."
            else:
                explanation_text = f"High risk detected for class {predicted_class}. Automated mitigation triggered."

            # 3. Enforce Mitigation
            try:
                mitigation_result = await loop.run_in_executor(
                    None,
                    app.state.mitigation_engine.apply_action,
                    recommended_action,
                    src_ip,
                    None,
                    explanation_text
                )
                if mitigation_result["status"] not in ["ALLOWED", "RATE_LIMITED"]:
                    LOGGER.warning("Mitigation applied: %s", mitigation_result)
            except Exception as mit_err:
                LOGGER.error("Mitigation enforcement failed: %s", mit_err)

        # 4. Build Enriched Event
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": flow["src_port"],
            "dst_port": flow["dst_port"],
            "protocol": flow["protocol"],
            "duration": flow["duration"],
            "total_packets": flow["total_packets"],
            "packet_rate": packet_rate,
            "predicted_class": predicted_class,
            "attack_type": flow["attack_type"],
            "confidence": confidence,
            "risk_score": risk_score,
            "risk_tier": risk_report["risk_tier"],
            "action_taken": recommended_action,
            "top_features": top_features,
            "explanation_text": explanation_text
        }

        app.state.flows_analyzed += 1
        if predicted_class != 0:
            app.state.threats_detected += 1

        app.state.history.append(event)

        # 5. Broadcast to WebSockets
        await broadcast_event(app, event)


async def broadcast_event(app: FastAPI, event: dict[str, Any]) -> None:
    """Asynchronously broadcast events to all connected WebSocket clients."""
    dead_sockets = set()
    for ws in app.state.active_websockets:
        try:
            await ws.send_json(event)
        except Exception:
            dead_sockets.add(ws)
    if dead_sockets:
        app.state.active_websockets.difference_update(dead_sockets)


# --------------------------------------------------------------------------- #
# FastAPI Lifespan & App Definition                                           #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server startup and shutdown events."""
    global ARGS
    if ARGS is None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        app.state.mock = True
        app.state.interface = "lo"
        app.state.port = 8000
        app.state.host = "0.0.0.0"
    else:
        app.state.mock = ARGS.mock or not ML_DEPS_AVAILABLE
        app.state.interface = ARGS.interface
        app.state.port = ARGS.port
        app.state.host = ARGS.host

    app.state.start_time = time.time()
    app.state.flows_analyzed = 0
    app.state.threats_detected = 0
    app.state.is_paused = False
    app.state.active_websockets = set()
    app.state.history = deque(maxlen=500)

    LOGGER.info("Initializing RiskEngine and MitigationEngine...")
    app.state.risk_engine = RiskEngine()
    app.state.mitigation_engine = MitigationEngine(dry_run=True)

    if not app.state.mock:
        try:
            LOGGER.info("Loading ML inference models...")
            inference_engine = FlowInferenceEngine()
            app.state.inference_engine = inference_engine

            dnn = inference_engine.dnn
            z_train = np.load("data/processed/z_train.npy")
            background = z_train[:100]

            # Safely extract feature names from dictionary or list artifact
            feature_payload = joblib.load("data/processed/feature_names.joblib")
            raw_names: list[str] = []
            if isinstance(feature_payload, dict):
                extracted = feature_payload.get("feature_names")
                if not extracted:
                    extracted = (
                        feature_payload.get("numeric_features", [])
                        + feature_payload.get("dummy_columns", [])
                    )
                raw_names = [str(x) for x in extracted]
            elif isinstance(feature_payload, (list, tuple, np.ndarray)):
                raw_names = [str(x) for x in feature_payload]

            # FlowExplainer validates against background latent dimension (32)
            if len(raw_names) == background.shape[1]:
                feature_names = raw_names
            else:
                feature_names = [f"Latent_Dim_{i}" for i in range(background.shape[1])]

            app.state.explainer = FlowExplainer(dnn, background, feature_names)

            LOGGER.info("ML models loaded successfully. Live capture ready.")
        except Exception as e:
            LOGGER.error("Failed to load ML models: %s. Falling back to mock mode.", e)
            app.state.mock = True
    else:
        LOGGER.info("Starting in Mock Mode (no ML models loaded).")

    app.state.worker_task = asyncio.create_task(pipeline_worker(app))

    yield

    LOGGER.info("Shutting down pipeline and flushing mitigations...")
    app.state.worker_task.cancel()
    try:
        await asyncio.wait_for(app.state.worker_task, timeout=5.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    if hasattr(app.state, "mitigation_engine"):
        app.state.mitigation_engine.flush_all()


app = FastAPI(
    title="NIDS Backend Server",
    description="Real-time network intrusion detection and response orchestration.",
    version="1.0.0",
    lifespan=lifespan
)

# Static directory setup and mount
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# WebSocket Streaming Hub                                                     #
# --------------------------------------------------------------------------- #

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming live telemetry to the SOC dashboard."""
    await websocket.accept()
    app.state.active_websockets.add(websocket)
    LOGGER.info("WebSocket client connected. Total active: %d", len(app.state.active_websockets))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        LOGGER.info("WebSocket client disconnected.")
    finally:
        app.state.active_websockets.discard(websocket)


# --------------------------------------------------------------------------- #
# Web UI & REST Management API Endpoints                                      #
# --------------------------------------------------------------------------- #

@app.get("/")
async def serve_dashboard():
    """Serves the SOC analyst web dashboard if static/index.html exists."""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return JSONResponse(
        content={
            "message": "NIDS Backend Server is running.",
            "warning": "static/index.html not found. Place your frontend template in the static/ folder.",
            "docs": "/docs"
        },
        status_code=200
    )

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Returns system health, uptime, and flow statistics."""
    return {
        "status": "online",
        "uptime_seconds": round(time.time() - app.state.start_time, 2),
        "active_interface": app.state.interface if not app.state.mock else "mock",
        "total_flows_analyzed": app.state.flows_analyzed,
        "total_threats_detected": app.state.threats_detected,
        "active_mitigations": len(app.state.mitigation_engine.get_active_quarantine()),
        "pipeline_paused": app.state.is_paused
    }

@app.get("/api/alerts", response_model=list[EnrichedEvent])
async def get_alerts(limit: int = 50):
    """Returns the latest N threat events (predicted_class != 0)."""
    alerts = [e for e in list(app.state.history)[::-1] if e["predicted_class"] != 0]
    return alerts[:limit]

@app.get("/api/quarantine", response_model=list[dict])
async def get_quarantine():
    """Lists currently blocked IPs, reasons, and remaining TTL leases."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, app.state.mitigation_engine.get_active_quarantine)

@app.post("/api/quarantine/release/{ip}", response_model=ReleaseResponse)
async def release_ip(ip: str):
    """Manual SOC analyst override to instantly unblock an IP."""
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(None, app.state.mitigation_engine.release_ip, ip)
    return {"success": success, "ip": ip}

@app.post("/api/pipeline/toggle", response_model=ToggleResponse)
async def toggle_pipeline():
    """Pause or resume background network capture."""
    app.state.is_paused = not app.state.is_paused
    return {"paused": app.state.is_paused}


# --------------------------------------------------------------------------- #
# Server Lifecycle & CLI Entry Point                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser(description="NIDS Backend Server")
    parser.add_argument("--interface", type=str, default="lo", help="Network interface to capture")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--mock", action="store_true", help="Use mock flow generator for safe dashboard testing")
    ARGS = parser.parse_args()
    
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)