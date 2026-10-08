# Autonomous IoT-23 Network Intrusion Detection & Response System (NIDS)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)](#)

An enterprise-grade, end-to-end Autonomous Network Intrusion Detection and Incident Response System (NIDS/NIPS) aka Net Attack Deduction (NAD). Built on the foundational architecture of **Dutta et al. (2020)** (*"A Deep Learning Ensemble for Network Anomaly and Cyber-Attack Detection"*), this system couples a **Deep Sparse Autoencoder (DSAE)** with a **heterogeneous Level-0 ensemble (`SpatialDNN` + `TemporalLSTM`)**, a multinomial stacking meta-learner, real-time bidirectional packet streaming via **`NFStream`**, explainable AI (**SHAP**), and automated, reversible Linux **`nftables`** kernel-level firewall containment.

---

## Table of Contents

- [Key Architectural Highlights](#key-architectural-highlights)
- [System Architecture](#system-architecture)
- [Directory Layout](#directory-layout)
- [Installation & Environment Setup](#installation--environment-setup)
- [Quickstart: Zero-Training Mock Mode](#quickstart-zero-training-mock-mode)
- [Full Training Pipeline (from Raw IoT-23 Logs)](#full-training-pipeline-from-raw-iot-23-logs)
  - [Stage 1: Dataset Parsing, Sanitization & Resampling](#stage-1-dataset-parsing-sanitization--resampling)
  - [Stage 2: Deep Sparse Autoencoder (DSAE) Training](#stage-2-deep-sparse-autoencoder-dsae-training)
  - [Stage 3: Out-of-Fold Stacking Ensemble Training](#stage-3-out-of-fold-stacking-ensemble-training)
- [Live Traffic Capture & Autonomous Containment](#live-traffic-capture--autonomous-containment)
  - [Running the Live Engine](#running-the-live-engine)
  - [Simulating an Attack & Live Verification](#simulating-an-attack--live-verification)
- [Mathematical & Algorithmic Formulation](#mathematical--algorithmic-formulation)
  - [1. Deep Sparse Autoencoder (KL Penalty)](#1-deep-sparse-autoencoder-kl-penalty)
  - [2. Multi-Factor Composite Risk Scoring ($R$)](#2-multi-factor-composite-risk-scoring-r)
  - [3. Native Kernel-Level Firewall Quarantine (nftables)](#3-native-kernel-level-firewall-quarantine-nftables)
- [API & WebSocket Specification](#api--websocket-specification)
- [License](#license)

---

## Key Architectural Highlights

1. **Strict Leakage-Prevention Contract**: Train/test splitting occurs *before* statistical median estimation, categorical one-hot encoding dictionaries, or `RobustScaler` quantiles are calculated. Hybrid **SMOTE-ENN** resampling touches only the training fold; the test split retains its natural class imbalance.
2. **Heterogeneous Level-0 + Level-1 Stacking**:
   - **`SpatialDNN`**: Fully-connected hourglass evaluating per-flow spatial patterns independently from compressed 32-dimensional latent vectors.
   - **`TemporalLSTM`**: 2-layer stacked LSTM scoring sliding windows of consecutive flows ($L=5$) to capture temporal multi-step attack signatures.
   - **Level-1 Stacking Meta-Learner**: Fit using **5-fold Out-of-Fold (OOF)** cross-validation to ensure the meta-learner never evaluates predictions from models trained on the same data.
3. **Sub-Millisecond Inference & Explainability**: Integrated with Tree/Kernel SHAP (`FlowExplainer`) yielding per-flow directional feature attributions (`increases_risk` vs. `decreases_risk`) streamed in real time.
4. **Autonomous, Reversible Mitigation**: Translates multi-factor risk scores into native Linux `nftables` dynamic sets equipped with **kernel-level TTL leases**. Threat actors are automatically isolated at the packet filter without risk of permanent network lockout.
5. **Dark-Themed SOC Dashboard**: Single-page application powered by TailwindCSS, Chart.js, and Lucide icons communicating over persistent WebSockets.

---

## System Architecture

```text
                                   LIVE NETWORK TRAFFIC (NIC / PCAP)
                                                  │
                                                  ▼
                                      NFStream Flow Aggregator
                                  (Bidirectional Feature Extraction)
                                                  │
                                                  ▼
                                     RobustScaler & One-Hot Align
                                                  │
                                                  ▼
                                    Deep Sparse Autoencoder (DSAE)
                                    (Compressed to 32-dim Latent Space)
                                                  │
                                 ┌────────────────┴────────────────┐
                                 ▼                                 ▼
                         SpatialDNN (1x32)               TemporalLSTM (5x32)
                       [Spatial Flow Geometry]        [Sliding-Window Temporal]
                                 │                                 │
                                 └────────────────┬────────────────┘
                                                  ▼
                                      Softmax Posteriors (10-dim)
                                                  │
                                                  ▼
                                     Level-1 Stacking Meta-Learner
                                      (Multinomial Logistic Reg.)
                                                  │
                                                  ▼
                                      Threat Class & Confidence
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
               SHAP Flow Explainer                                Multi-Factor Risk Engine
           (Per-Feature Attributions)                      (Severity, Volume, Conf, Asset)
                         │                                                 │
                         └────────────────────────┬────────────────────────┘
                                                  ▼
                                      FastAPI Orchestration Core
                                                  │
                       ┌──────────────────────────┴──────────────────────────┐
                       ▼                                                     ▼
           WebSocket Telemetry Hub                                 Linux nftables Engine
          (Streamed to SOC Dashboard)                           (Dynamic Timed Quarantine Set)
```

---

## Directory Layout

```text
├── data/
│   ├── processed/               # Preprocessed NumPy matrices, scalers, schema (generated)
│   └── raw/                     # Raw Zeek conn.log.labeled files (user-provided)
├── models/
│   └── saved/                   # PyTorch checkpoints and joblib meta-learners (generated)
├── src/
│   ├── autoencoder.py           # DSAE architecture and SparseLoss (MSE + KL divergence)
│   ├── classifiers.py           # Level-0 base learners: SpatialDNN & TemporalLSTM
│   ├── dataset.py               # Zeek ingestion, sanitization, split, SMOTE-ENN
│   ├── explainer.py             # SHAP explainability engine for latent flow features
│   ├── flow_engine.py           # Live NFStream aggregation and inference engine
│   ├── mitigation.py            # Automated, reversible nftables firewall engine
│   ├── risk_engine.py           # Multi-factor composite risk scoring engine
│   └── server.py                # Asynchronous FastAPI WebSocket and REST server
├── static/
│   └── index.html               # Reactive SOC analyst dashboard
├── train_dsae.py                # Training and latent extraction script for DSAE
├── train_ensemble.py            # 5-fold OOF training script for stacking ensemble
├── requirements.txt             # Pinned project dependencies
└── README.md
```

---

## Installation & Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/benu-zukuto/Net-Attack-Detection.git
cd Net-Attack-Detection
```

### 2. Create and Activate a Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Quickstart: Zero-Training Mock Mode

If you have just cloned the repository and want to run the SOC dashboard immediately without training the neural networks or capturing live packets, start the server in **mock mode**.

This mode runs the entire FastAPI server, simulated telemetry generator, multi-factor risk engine, and dry-run firewall containment without requiring root privileges or pre-trained models.

```bash
python3 -m src.server --mock --port 8000
```

1. Open your web browser and navigate to: **`http://localhost:8000`**
2. You will observe:
   - Live traffic anomaly waveforms rendering over WebSockets.
   - Attack detection cards updating in real time.
   - Explainable AI (XAI) feature attribution bars populating on alert click.
   - Dynamic quarantine additions with real-time TTL countdown timers.

---

## Full Training Pipeline (from Raw IoT-23 Logs)

Follow these steps to train the complete model stack from scratch using labelled connection logs from the **Stratosphere IoT-23 dataset**.

### Stage 1: Dataset Parsing, Sanitization & Resampling

Place one or more `conn.log.labeled` files (or `.gz` archives) under `data/raw/iot_23/`, then execute the leakage-safe preprocessor:

```bash
python -m src.dataset \
  --conn-log data/raw/iot_23 \
  --output-dir data/processed \
  --sample-size 100000 \
  --test-size 0.2 \
  --random-state 42
```

**Artifacts Generated in `data/processed/`:**
- `X_train.npy` & `y_train.npy` (SMOTE-ENN resampled training set)
- `X_test.npy` & `y_test.npy` (Held-out, untouched evaluation set)
- `scaler.joblib` (Fitted `RobustScaler`)
- `feature_names.joblib` (Feature schema, reference encodings, and column medians)

---

### Stage 2: Deep Sparse Autoencoder (DSAE) Training

Train the symmetric hourglass autoencoder using the sparse KL divergence objective, checkpoint the best model on held-out validation loss, and export 32-dimensional sparse latent representations:

```bash
python train_dsae.py \
  --epochs 25 \
  --batch-size 256 \
  --lr 0.001 \
  --latent-dim 32 \
  --device auto
```

**Artifacts Generated:**
- `models/saved/dsae_full.pt` (Complete autoencoder checkpoint)
- `models/saved/dsae_encoder.pt` (Encoder weights for inference)
- `data/processed/z_train.npy` (32-dim latent training vectors)
- `data/processed/z_test.npy` (32-dim latent testing vectors)

---

### Stage 3: Out-of-Fold Stacking Ensemble Training

Train the Level-0 classifiers (`SpatialDNN` and `TemporalLSTM`) across 5 stratified folds to generate honest Out-of-Fold (OOF) meta-features. Fit the Level-1 multinomial `LogisticRegression` meta-learner on the OOF posteriors, persist the production models, and generate the final classification report:

```bash
python train_ensemble.py \
  --epochs 15 \
  --batch-size 128 \
  --k-folds 5 \
  --lr 0.001 \
  --device auto
```

**Artifacts Generated:**
- `models/saved/dnn_level0.pt` (Full-set SpatialDNN weights)
- `models/saved/lstm_level0.pt` (Full-set TemporalLSTM weights)
- `models/saved/meta_learner.joblib` (Fitted Level-1 meta-learner)

---

## Live Traffic Capture & Autonomous Containment

### Running the Live Engine

Once the models are generated under `models/saved/`, execute the server with root privileges to attach `NFStream` to your active network interface and enable live kernel-level `nftables` isolation.

1. **Identify your target network interface**:
   ```bash
   ip -br link
   ```
   *(e.g., `eth0`, `wlan0`, or `wlx688fc9200f08`)*

2. **Launch the server with live packet sniffing**:
   ```bash
   sudo $(which python3) -m src.server --interface <YOUR_INTERFACE> --port 8000
   ```

*Note: If `nftables` (`nft`) is not available or if run without root privileges, `MitigationEngine` automatically falls back to `dry_run = True`, logging the exact containment commands without modifying system firewall rules.*

---

### Simulating an Attack & Live Verification

With the server running on your network interface (e.g., target host IP `victim's IP address`), launch a port scan or reconnaissance probe from an external client on the same subnet:

```bash
sudo nmap -O <victim's IP address>
```

#### What Happens in Real Time:
1. **Detection**: `NFStream` aggregates the TCP SYN probe flows and passes them to `FlowInferenceEngine`.
2. **Classification**: The ensemble identifies the signature as **Recon / PortScan** (Class 1) with $>99\%$ confidence.
3. **Attribution**: `FlowExplainer` isolates the top contributing latent dimensions responsible for the classification.
4. **Risk Calculation**: `RiskEngine` calculates a composite risk score (e.g., `40.6 MEDIUM`).
5. **Mitigation**: `MitigationEngine` issues an atomic rule into the `quarantine_v4` set:
   ```bash
   nft add element inet idps_filter quarantine_v4 { <ATTACKER_IP> timeout 60s }
   ```
6. **Telemetry**: The incident is broadcast over WebSockets to the SOC dashboard. The attacker's packets are dropped at the kernel boundary until the lease expires or is manually revoked.

---

## Mathematical & Algorithmic Formulation

### 1. Deep Sparse Autoencoder (KL Penalty)

The DSAE compresses $D$-dimensional preprocessed flow vectors into a 32-dimensional Sigmoid bottleneck $z \in (0, 1)^d$. The optimization objective balances mean-squared reconstruction error with Kullback-Leibler (KL) divergence sparsity:

$$L(x, \hat{x}, z) = \text{MSE}(x, \hat{x}) + \beta \sum_{j=1}^{d} \text{KL}(\rho \parallel \hat{\rho}_j)$$

Where:
- $\rho = 0.05$ (target average neuron activation)
- $\beta = 2.0$ (sparsity penalty weight)
- $\hat{\rho}_j = \frac{1}{B} \sum_{i=1}^{B} z_{i, j}$ (average batch activation of latent unit $j$)
- $\text{KL}(\rho \parallel \hat{\rho}_j) = \rho \log \frac{\rho}{\hat{\rho}_j} + (1 - \rho) \log \frac{1 - \rho}{1 - \hat{\rho}_j}$

---

### 2. Multi-Factor Composite Risk Scoring ($R$)

To eliminate false-positive-driven automated lockouts, mitigations are gated by a deterministic composite risk score $R \in [0, 100]$:

$$R = \min\left(100.0,\; w_1 (S_{\text{attack}} \cdot 20) + w_2 (C_{\text{model}} \cdot 100) + w_3 A_{\text{vol}} + w_4 I_{\text{asset}}\right)$$

Where:
- **$S_{\text{attack}} \in [0, 5]$**: MITRE ATT&CK base severity ($0.0$ for Benign, $1.5$ for Recon, $4.0$ for DoS, $4.5$ for Malware, $5.0$ for Botnet C&C).
- **$C_{\text{model}} \in [0, 1]$**: Calibrated Level-1 meta-learner probability.
- **$A_{\text{vol}} \in [0, 100)$**: Saturated volume anomaly:
  $$A_{\text{vol}} = 100 \cdot \left(1.0 - \exp\left(-\frac{\text{packet\_rate}}{\text{baseline\_rate} + \varepsilon}\right)\right)$$
- **$I_{\text{asset}} \in [0, 100]$**: Target destination asset priority from CIDR lookup tables.
- **Weights ($\sum w_i = 1.0$)**: $w_1 = 0.35$, $w_2 = 0.25$, $w_3 = 0.20$, $w_4 = 0.20$.

#### Action Tiers

| Score Range | Risk Tier | Recommended Mitigation | Action Executed |
| :--- | :--- | :--- | :--- |
| **$[0.0, 30.0)$** | `LOW` | `ALLOW` | Standard telemetry logging |
| **$[30.0, 60.0)$** | `MEDIUM` | `LOG_RATE_LIMIT` | Dashboard notification, rate limiting |
| **$[60.0, 85.0)$** | `HIGH` | `QUARANTINE_TEMP` | Audible alert, 60-second timed quarantine |
| **$[85.0, 100.0]$** | `CRITICAL` | `QUARANTINE_ISOLATE` | 300-second strict isolation, session drop |

---

### 3. Native Kernel-Level Firewall Quarantine (nftables)

Mitigations configure a dedicated table hierarchy without affecting host firewall definitions:

```text
table inet idps_filter {
    set quarantine_v4 {
        type ipv4_addr
        flags timeout
    }
    chain input {
        type filter hook input priority 0; policy accept;
        ip saddr @quarantine_v4 drop
    }
    chain forward {
        type filter hook forward priority 0; policy accept;
        ip saddr @quarantine_v4 drop
    }
}
```

Because elements in `quarantine_v4` carry kernel-level timeouts (`timeout 60s`), hosts are unblocked automatically upon lease expiration even if the python process crashes.

---

## API & WebSocket Specification

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the single-page SOC dashboard (`static/index.html`). |
| `GET` | `/api/status` | Returns system health, uptime, flows analyzed, and active mitigations. |
| `GET` | `/api/alerts?limit=50` | Returns the most recent $N$ threat incidents. |
| `GET` | `/api/quarantine` | Lists currently blocked IP addresses, reasons, and remaining TTLs. |
| `POST`| `/api/quarantine/release/{ip}` | Manual SOC analyst override to instantly unblock an IP. |
| `POST`| `/api/pipeline/toggle` | Toggles background packet capture (Pause/Resume). |
| `WS`  | `/ws/telemetry` | Persistent WebSocket streaming live flow detection events to the dashboard. |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
