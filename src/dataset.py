"""IoT-23 dataset pipeline: Zeek ``conn.log.labeled`` → ML-ready NumPy arrays.

This module implements the data-ingestion stage of the deep-learning network
intrusion detection system (NIDS) of *Dutta et al. (2020)*, targeting the
**IoT-23** dataset (Garcia, Parmisano & Erquiaga, 2020 — Stratosphere Labs):
labelled ``conn.log.labeled`` connection logs produced by the Zeek (formerly
Bro) network security monitor.

Pipeline
--------
1. :func:`parse_zeek_conn_log`
   Stream one or more Zeek logs (plain or gzip), gracefully bypass every
   ``#``-prefixed header/metadata line, reconcile the classic IoT-23 header
   quirks, and draw a uniform, reproducible random sample (reservoir
   sampling — memory bounded by ``O(sample_size)``).
2. :func:`map_labels`
   Normalise the raw ``label`` / ``detailed-label`` fields into a 5-class
   integer target column.
3. :func:`preprocess_and_split`
   Stratified 80/20 split, numeric sanitisation (Zeek ``'-'`` nulls and
   infinities → column medians), one-hot encoding, ``RobustScaler`` scaling,
   SMOTE-ENN resampling of the training fold, and persistence of every
   artifact under ``data/processed/``.

Leakage-prevention contract (critical for honest evaluation)
------------------------------------------------------------
* The train/test split happens **before** any statistic is estimated.
* Median imputation values, the one-hot vocabulary and the ``RobustScaler``
  quantiles are estimated from the **training fold only**; the test fold is
  only ever *transformed*.
* SMOTE-ENN touches **only** the scaled ``(X_train, y_train)`` pair. The
  test fold deliberately keeps its natural class imbalance so that offline
  metrics reflect production network conditions.

Generated artifacts (``data/processed/``)
-----------------------------------------
========================  ===================================================
``X_train.npy``           SMOTE-ENN-resampled, RobustScaled float32 matrix.
``y_train.npy``           Resampled int64 target vector.
``X_test.npy``            Untouched (never resampled), scaled float32 matrix.
``y_test.npy``            Ground-truth int64 target vector.
``scaler.joblib``         Fitted ``sklearn.preprocessing.RobustScaler``.
``feature_names.joblib``  Feature schema, reference levels, imputation
                          medians and class-name map (for inference time).
========================  ===================================================

Usage
-----
.. code-block:: console

    python src/dataset.py --conn-log data/raw/iot_23 --sample-size 100000
    python -m src.dataset -i data/raw/iot_23/CTU-IoT-Mal-Capture-34-1/bro/conn.log.labeled -v

Dependencies: ``numpy``, ``pandas``, ``scikit-learn``, ``imbalanced-learn``,
``joblib`` (Python >= 3.10).
"""

from __future__ import annotations

import argparse
import gzip
import logging
import random
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, TextIO

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# --------------------------------------------------------------------------- #
# Public configuration                                                         #
# --------------------------------------------------------------------------- #

LOGGER: Final[logging.Logger] = logging.getLogger("iot23.dataset")

#: Canonical 23-field schema of IoT-23 ``conn.log.labeled`` files.
ZEEK_CONN_COLUMNS: Final[list[str]] = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents", "label", "detailed-label",
]

#: Older 22-field variant (pre-``tunnel_parents`` Zeek, e.g. CTU-13 logs).
ZEEK_CONN_COLUMNS_LEGACY: Final[list[str]] = [
    column for column in ZEEK_CONN_COLUMNS if column != "tunnel_parents"
]

#: Model inputs — numeric block (Dutta et al., 2020 feature set).
NUMERICAL_FEATURES: Final[list[str]] = [
    "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
    "orig_ip_bytes", "resp_ip_bytes",
]

#: Model inputs — categorical block.
CATEGORICAL_FEATURES: Final[list[str]] = ["proto", "conn_state"]

LABEL_COLUMN: Final[str] = "label"
DETAILED_LABEL_COLUMN: Final[str] = "detailed-label"
TARGET_COLUMN: Final[str] = "target"

TARGET_BENIGN: Final[int] = 0
TARGET_RECON: Final[int] = 1
TARGET_DOS: Final[int] = 2
TARGET_MALWARE: Final[int] = 3
TARGET_C2: Final[int] = 4

CLASS_NAMES: Final[dict[int, str]] = {
    TARGET_BENIGN: "Benign",
    TARGET_RECON: "Recon / Horizontal PortScan",
    TARGET_DOS: "DoS / DDoS",
    TARGET_MALWARE: "Malware / Okiru",
    TARGET_C2: "Botnet C&C",
}

#: Class assigned to malicious flows whose detailed label is not recognised
#: (spec §3: default ambiguous attack labels to class 2).
AMBIGUOUS_ATTACK_CLASS: Final[int] = TARGET_DOS

#: Artifact filenames (spec §4 / §6).
SCALER_FILENAME: Final[str] = "scaler.joblib"
FEATURE_NAMES_FILENAME: Final[str] = "feature_names.joblib"
X_TRAIN_FILENAME: Final[str] = "X_train.npy"
Y_TRAIN_FILENAME: Final[str] = "y_train.npy"
X_TEST_FILENAME: Final[str] = "X_test.npy"
Y_TEST_FILENAME: Final[str] = "y_test.npy"

# Internal configuration ---------------------------------------------------- #

_NULL_TOKENS: Final[frozenset[str]] = frozenset({"-", "(empty)", "", "nan", "none"})
_BENIGN_LABEL_TOKENS: Final[frozenset[str]] = frozenset({"benign", "normal", "legitimate"})

#: Substring rules (case-insensitive) applied to "<label> <detailed-label>".
#: Order == priority, e.g. "C&C-PartOfAHorizontalPortScan" resolves to C&C.
_ATTACK_TAG_RULES: Final[list[tuple[str, int]]] = [
    ("c&c", TARGET_C2),
    ("cnc", TARGET_C2),
    ("okiru", TARGET_MALWARE),
    ("partofahorizontalportscan", TARGET_RECON),
    ("horizontalportscan", TARGET_RECON),
    ("portscan", TARGET_RECON),
    ("ddos", TARGET_DOS),
    ("attack", TARGET_DOS),
    ("dos", TARGET_DOS),
]

_LOG_FILE_PATTERNS: Final[tuple[str, ...]] = ("*.log.labeled", "*.log.labeled.gz")
_PROGRESS_LOG_INTERVAL: Final[int] = 1_000_000

__all__: Final[list[str]] = [
    "parse_zeek_conn_log",
    "map_labels",
    "preprocess_and_split",
    "main",
    "CLASS_NAMES",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "ZEEK_CONN_COLUMNS",
]


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


def _open_text(path: Path) -> TextIO:
    """Open a plain-text or gzip-compressed Zeek log for reading.

    ``utf-8-sig`` transparently strips a leading BOM, and undecodable bytes
    are replaced instead of aborting a multi-gigabyte ingestion run.
    """
    if path.name.endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8-sig", errors="replace")
    return open(path, mode="r", encoding="utf-8-sig", errors="replace")


def _resolve_log_paths(path: str | Path) -> list[Path]:
    """Resolve a user-supplied path into an ordered, de-duplicated file list.

    Args:
        path: A single log file, or a directory searched recursively for
            ``*.log.labeled`` / ``*.log.labeled.gz`` files.

    Returns:
        Ordered list of log file paths.

    Raises:
        FileNotFoundError: If the path does not exist or no labelled logs are
            discovered beneath a directory.
        NotADirectoryError: If the path is neither a file nor a directory.
    """
    root = Path(path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root!s}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a file or directory, got: {root!s}")

    discovered: list[Path] = []
    for pattern in _LOG_FILE_PATTERNS:
        discovered.extend(sorted(root.rglob(pattern)))
    unique = list(dict.fromkeys(discovered))  # order-preserving de-duplication
    if not unique:
        raise FileNotFoundError(
            f"No files matching {_LOG_FILE_PATTERNS} were found below '{root}'. "
            "Point --conn-log at a single file or at a scenario directory."
        )
    LOGGER.info("Discovered %d labelled Zeek log file(s) below '%s'.", len(unique), root)
    return unique


def _resolve_row_columns(declared: list[str] | None, width: int) -> list[str]:
    """Return the column names governing a data row of ``width`` fields.

    Handles three real-world situations:
      * rows whose width matches the ``#fields`` declaration;
      * the original IoT-23 release, where ``label`` / ``detailed-label``
        were appended to every data row but *omitted* from ``#fields``;
      * header-less files, where the schema is guessed from the row width
        (23-field modern vs. 22-field legacy).
    """
    if declared is not None:
        if len(declared) == width or LABEL_COLUMN in declared:
            return declared
        if width == len(declared) + 2:
            return [*declared, LABEL_COLUMN, DETAILED_LABEL_COLUMN]
        if width == len(declared) + 1:
            return [*declared, LABEL_COLUMN]
        return declared
    if width == len(ZEEK_CONN_COLUMNS_LEGACY):
        return ZEEK_CONN_COLUMNS_LEGACY
    return ZEEK_CONN_COLUMNS


def _iter_zeek_records(path: Path) -> Iterator[dict[str, str]]:
    """Stream parsed connection records from a single Zeek log file.

    All ``#``-prefixed header/metadata lines are bypassed; a ``#fields`` line
    updates the active column declaration. Rows are split on tabs first and
    on generic whitespace as a fallback (some distributions are
    space-separated). Malformed rows are padded/truncated and counted rather
    than aborting the run.
    """
    declared_fields: list[str] | None = None
    comment_lines = 0
    data_lines = 0
    records = 0
    malformed = 0
    try:
        with _open_text(path) as handle:
            for line in handle:
                if line.startswith("#"):  # header / metadata — bypass gracefully
                    comment_lines += 1
                    if line.startswith("#fields"):
                        declared_fields = [token for token in line.split()[1:] if token]
                    continue
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                data_lines += 1
                if data_lines % _PROGRESS_LOG_INTERVAL == 0:
                    LOGGER.info("  … '%s': %d lines streamed so far.", path.name, data_lines)

                values = line.split("\t")
                while values and values[-1].strip() == "":  # trailing tabs
                    values.pop()
                if declared_fields is not None and len(values) != len(declared_fields):
                    whitespace_split = line.split()  # space-separated fallback
                    if len(whitespace_split) == len(declared_fields):
                        values = whitespace_split

                columns = _resolve_row_columns(declared_fields, len(values))
                if len(values) != len(columns):
                    malformed += 1
                    if len(values) < len(columns):
                        values = values + ["-"] * (len(columns) - len(values))
                    else:
                        values = values[: len(columns)]

                records += 1
                yield dict(zip(columns, values))
    finally:
        LOGGER.info(
            "Parsed '%s' — header/comment lines skipped: %d, data lines: %d, "
            "records emitted: %d, malformed rows repaired: %d.",
            path, comment_lines, data_lines, records, malformed,
        )


# --------------------------------------------------------------------------- #
# 1. File parsing (spec §1)                                                    #
# --------------------------------------------------------------------------- #


def parse_zeek_conn_log(
    path: str | Path,
    sample_size: int | None = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Parse Zeek ``conn.log.labeled`` file(s) into a raw pandas DataFrame.

    The parser streams lines, gracefully bypassing every header/metadata line
    prefixed with ``#`` (``#separator``, ``#path``, ``#fields``, …). Column
    names come from the file's own ``#fields`` declaration when present, with
    the canonical 23-field IoT-23 schema as fallback. The original IoT-23
    release declared 21 columns but appended two label columns to every data
    row — this quirk is reconciled transparently. Gzip-compressed logs
    (``*.gz``) are decompressed on the fly.

    Sampling uses classic reservoir sampling (Algorithm R): memory is bounded
    by ``O(sample_size)`` regardless of input size, every connection record
    has an equal retention probability (preserving the global class
    distribution in expectation), and the sample is reproducible for a given
    ``random_state``.

    Args:
        path: A single log file (plain or ``.gz``), or a directory searched
            recursively for ``*.log.labeled`` / ``*.log.labeled.gz``.
        sample_size: Number of records to keep. ``None`` or a value ``<= 0``
            retains every record (memory permitting). Defaults to ``100000``.
        random_state: Seed making the uniform sample reproducible.

    Returns:
        A DataFrame with one row per connection record (values left as raw
        strings — sanitisation happens later, leakage-safely) and guaranteed
        ``label`` / ``detailed-label`` columns.

    Raises:
        FileNotFoundError: If ``path`` does not exist or no labelled logs are
            discovered beneath a directory.
    """
    log_paths = _resolve_log_paths(path)
    rng = random.Random(random_state)
    reservoir: list[dict[str, str]] = []
    total_seen = 0

    with _stage_timer(f"Zeek ingestion of {len(log_paths)} log file(s)"):
        for log_path in log_paths:
            for record in _iter_zeek_records(log_path):
                total_seen += 1
                if sample_size is None or sample_size <= 0 or total_seen <= sample_size:
                    reservoir.append(record)
                else:
                    # Algorithm R: uniform over the full stream, bounded memory.
                    replace_at = rng.randrange(total_seen)
                    if replace_at < sample_size:
                        reservoir[replace_at] = record

    if not reservoir:
        LOGGER.warning("No connection records could be parsed from '%s'.", path)
        return pd.DataFrame(columns=ZEEK_CONN_COLUMNS)

    frame = pd.DataFrame.from_records(reservoir)
    for column in (LABEL_COLUMN, DETAILED_LABEL_COLUMN):
        if column not in frame.columns:
            frame[column] = "-"
        else:
            frame[column] = frame[column].fillna("-")

    LOGGER.info(
        "Sampled %d record(s) out of %d parsed connection(s) (sample_size=%s).",
        len(frame), total_seen, sample_size,
    )
    LOGGER.info("Raw DataFrame: shape=%s.", frame.shape)
    LOGGER.info("Raw label distribution: %s", frame[LABEL_COLUMN].value_counts().to_dict())
    LOGGER.info(
        "Top detailed labels: %s", frame[DETAILED_LABEL_COLUMN].value_counts().head(10).to_dict()
    )
    return frame


# --------------------------------------------------------------------------- #
# 2/3. Multi-class label mapping (spec §3)                                     #
# --------------------------------------------------------------------------- #


def map_labels(df: pd.DataFrame, *, drop_ambiguous: bool = False) -> pd.DataFrame:
    """Map Zeek ``label`` / ``detailed-label`` onto the 5-class integer target.

    Both fields are lower-cased and matched as *substrings* of the combined
    string ``"<label> <detailed-label>"``, which makes the matcher robust to
    IoT-23 tag suffixes such as ``C&C-HeartBeat-FileDownload`` or
    ``PartOfAHorizontalPortScan-ProofPoint``.

    Label taxonomy (Dutta et al., 2020):

    ======  ============================  =======================================
    Class  Name                           Matching tokens (priority order)
    ======  ============================  =======================================
    0       Benign / Normal               benign, normal, legitimate, null pairs
    1       Recon / Horizontal PortScan   partofahorizontalportscan, portscan
    2       DoS / DDoS / generic attack   ddos, attack, dos, **default fallback**
    3       Malware / Okiru               okiru
    4       Botnet C&C                    c&c, cnc
    ======  ============================  =======================================

    Policy notes:
      * A benign ``label`` always wins over any detailed tag.
      * Unlabelled pairs (``-`` / ``-``) are conservatively treated as benign
        (class 0); in the IoT-23 releases that emit Zeek-labeler style logs,
        benign background traffic carries exactly those null tokens.
      * Malicious rows whose tags are not recognised default to class 2
        (spec §3) unless ``drop_ambiguous=True`` drops them cleanly.
      * Combined tags resolve by rule priority: C&C > Okiru > PortScan >
        DDoS (e.g. ``C&C-PartOfAHorizontalPortScan`` → class 4).

    Args:
        df: Raw frame from :func:`parse_zeek_conn_log` (or any frame with the
            ``label`` and ``detailed-label`` columns).
        drop_ambiguous: Drop unrecognised attack rows instead of defaulting
            them to class 2.

    Returns:
        A copy of ``df`` with an additional integer ``target`` column.

    Raises:
        ValueError: If the label columns are missing.
    """
    _require_columns(df, [LABEL_COLUMN, DETAILED_LABEL_COLUMN])
    work = df.copy()

    labels = work[LABEL_COLUMN].astype(str).str.strip().str.lower()
    details = work[DETAILED_LABEL_COLUMN].astype(str).str.strip().str.lower()
    combined = labels.str.cat(details, sep=" ")

    benign_mask = (
        labels.isin(_BENIGN_LABEL_TOKENS)
        | (labels.isin(_NULL_TOKENS) & details.isin(_NULL_TOKENS))
        | combined.str.contains("benign", regex=False)
    ).to_numpy()

    tag_masks = [combined.str.contains(tag, regex=False).to_numpy() for tag, _ in _ATTACK_TAG_RULES]
    tag_classes = [klass for _, klass in _ATTACK_TAG_RULES]

    targets = np.select(condlist=tag_masks, choicelist=tag_classes, default=AMBIGUOUS_ATTACK_CLASS)
    targets = np.where(benign_mask, TARGET_BENIGN, targets).astype(np.int64)

    any_tag = np.zeros(len(work), dtype=bool)
    for mask in tag_masks:
        any_tag |= mask
    recognized = benign_mask | any_tag

    n_ambiguous = int((~recognized).sum())
    if n_ambiguous:
        top_offenders = (
            pd.DataFrame({LABEL_COLUMN: labels, DETAILED_LABEL_COLUMN: details})
            .loc[~recognized]
            .value_counts()
            .head(10)
            .to_dict()
        )
        LOGGER.warning(
            "%d row(s) carry unrecognised attack labels — defaulting to class %d (%s). "
            "Most frequent raw values: %s",
            n_ambiguous, AMBIGUOUS_ATTACK_CLASS, CLASS_NAMES[AMBIGUOUS_ATTACK_CLASS], top_offenders,
        )
        if drop_ambiguous:
            work = work[recognized].reset_index(drop=True)
            targets = targets[recognized]
            LOGGER.info("Dropped %d ambiguous row(s) (drop_ambiguous=True).", n_ambiguous)

    work[TARGET_COLUMN] = targets
    _log_class_distribution(work[TARGET_COLUMN].to_numpy(), "Label mapping result")
    return work


# --------------------------------------------------------------------------- #
# Internal preprocessing helpers (leakage-safe)                                #
# --------------------------------------------------------------------------- #


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise ``ValueError`` listing any required columns missing from ``frame``."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required column(s) {missing}; "
            f"available columns: {list(frame.columns)}"
        )


def _log_class_distribution(y: np.ndarray | pd.Series, title: str) -> None:
    """Log per-class sample counts together with human-readable class names."""
    counts = pd.Series(y).value_counts().sort_index()
    rendered = ", ".join(
        f"{int(klass)}:{CLASS_NAMES.get(int(klass), '?')}={int(count)}"
        for klass, count in counts.items()
    )
    LOGGER.info("%s — total=%d | %s", title, int(counts.sum()), rendered)


def _sanitize_numeric_block(
    frame: pd.DataFrame,
    columns: Sequence[str],
    medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sanitise a numeric feature block and impute missing values (spec §2).

    Zeek's null placeholders (``'-'``, ``'(empty)'``) and any other
    non-numeric junk are coerced to ``NaN`` by ``pd.to_numeric``; infinite
    values are neutralised; ``NaN`` entries are imputed with the column
    median (falling back to ``0.0`` for entirely-empty columns) and the
    block is coerced to ``float32``.

    Args:
        frame: Frame holding the raw (string-valued) numeric columns.
        columns: Ordered numeric feature names.
        medians: When supplied (test fold), these training-derived medians
            are applied verbatim. When ``None`` (training fold), medians are
            estimated from ``frame`` itself and returned.

    Returns:
        ``(sanitised_block, medians)`` — the medians are fit on the training
        fold and reused unchanged for the test fold (leakage prevention).
    """
    block = frame.loc[:, list(columns)].copy()
    for column in columns:
        # pd.to_numeric(errors="coerce") turns '-', '(empty)' and any other
        # Zeek null indicator into NaN in one deterministic step.
        block[column] = pd.to_numeric(block[column], errors="coerce")
    block = block.replace(to_replace=[np.inf, -np.inf], value=np.nan)
    if medians is None:
        medians = block.median(axis=0).fillna(0.0)
        LOGGER.debug("Numeric imputation medians fit on training fold: %s", medians.to_dict())
    block = block.fillna(value=medians)
    return block.astype(np.float32), medians


def _normalize_categories(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Normalise categorical fields: trimmed, lower-cased, nulls → 'unknown'."""
    normalized = frame.loc[:, list(columns)].copy()
    for column in columns:
        values = normalized[column].astype(str).str.strip().str.lower()
        normalized[column] = values.where(~values.isin(_NULL_TOKENS), other="unknown")
    return normalized


def _encode_categoricals(
    train_categories: pd.DataFrame,
    test_categories: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """One-hot encode categorical features with a training-fold vocabulary.

    ``pd.get_dummies(..., drop_first=True)`` defines the reference levels and
    the dummy column set on the *training* fold (spec §2). The test fold is
    encoded without dropping a level and then re-indexed onto the training
    columns so that:

    * the dropped reference level is identical across both folds, and
    * categories unseen during training encode to an all-zero vector instead
      of shifting the feature space (logged as a warning).
    """
    columns = list(train_categories.columns)
    train_dummies = pd.get_dummies(train_categories, columns=columns, drop_first=True, dtype=np.float32)

    unseen = {
        column: sorted(set(test_categories[column].unique()) - set(train_categories[column].unique()))
        for column in columns
        if not set(test_categories[column].unique()).issubset(set(train_categories[column].unique()))
    }
    if unseen:
        LOGGER.warning("Test-fold categories unseen in training (encoded as all-zero): %s", unseen)

    test_dummies = pd.get_dummies(test_categories, columns=columns, drop_first=False, dtype=np.float32)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0.0).astype(np.float32)
    return train_dummies, test_dummies, list(train_dummies.columns)


def _apply_smote_enn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE-ENN hybrid resampling to a training fold (spec §5).

    ``SMOTE.k_neighbors`` is clamped to the smallest class support so the
    oversampler cannot crash on ultra-rare classes; when the smallest class
    has fewer than two samples, resampling is skipped with a warning instead
    of failing the build. Note that synthetic interpolation also affects the
    one-hot block, which is accepted by the reference methodology.

    Args:
        x_train: Scaled training feature matrix.
        y_train: Training target vector.
        random_state: Seed for reproducible resampling.

    Returns:
        ``(X_resampled, y_resampled)`` as ``(float32, int64)``.
    """
    _log_class_distribution(y_train, "y_train — before SMOTE-ENN")
    classes, counts = np.unique(y_train, return_counts=True)
    minority_support = int(counts.min())

    if minority_support < 2:
        LOGGER.warning(
            "Smallest training class (%s) has %d sample(s); SMOTE needs >= 2 — "
            "skipping resampling, the training fold stays imbalanced.",
            CLASS_NAMES.get(int(classes[int(np.argmin(counts))]), "?"), minority_support,
        )
        return x_train, y_train

    k_neighbors = int(max(1, min(5, minority_support - 1)))
    sampler = SMOTEENN(
        random_state=random_state,
        smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state),
        enn=EditedNearestNeighbours(n_neighbors=3),
    )
    x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
    _log_class_distribution(y_resampled, "y_train — after SMOTE-ENN")
    LOGGER.info(
        "SMOTE-ENN (k_neighbors=%d): training samples %d → %d.",
        k_neighbors, len(y_train), len(y_resampled),
    )
    return np.asarray(x_resampled, dtype=np.float32), np.asarray(y_resampled, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 4/5/6. Split, scale, resample, persist (spec §4–§6)                          #
# --------------------------------------------------------------------------- #


def preprocess_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str | Path = "data/processed",
    resample: bool = True,
    min_class_support: int = 2,
) -> dict[str, np.ndarray]:
    """Split, sanitise, encode, scale and resample the parsed IoT-23 frame.

    Leakage-prevention contract enforced here (spec §4–§5):

    1. A stratified 80/20 train/test split is drawn **before** any statistic
       (medians, one-hot vocabulary, scaler quantiles) is estimated.
    2. ``RobustScaler`` is fitted on the training fold only
       (``fit_transform``) and merely *transformed* onto the test fold.
    3. ``SMOTEENN(random_state=42)`` is applied to the scaled training fold
       only — the test fold is never resampled, so its natural class
       imbalance mirrors production network conditions.

    Artifacts written to ``output_dir`` (spec §6): ``X_train.npy``,
    ``y_train.npy``, ``X_test.npy``, ``y_test.npy``, ``scaler.joblib`` and
    ``feature_names.joblib``.

    Args:
        df: Labelled frame from :func:`map_labels` (a ``target`` column is
            mapped on the fly if absent).
        test_size: Held-out fraction for the test split.
        random_state: Seed shared by the split and the resampler.
        output_dir: Destination directory for every artifact.
        resample: Apply SMOTE-ENN to the training fold (default ``True``).
        min_class_support: Classes with fewer samples than this are dropped
            (a stratified split requires at least two per class).

    Returns:
        ``{"X_train": …, "y_train": …, "X_test": …, "y_test": …}`` — the
        exact arrays persisted to disk, for in-memory consumption.

    Raises:
        ValueError: If arguments are invalid, required columns are missing,
            or fewer than two classes survive rare-class filtering.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must lie in (0, 1), got {test_size}.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    if TARGET_COLUMN not in work.columns:
        LOGGER.info("No '%s' column present — running label mapping first.", TARGET_COLUMN)
        work = map_labels(work)
    _require_columns(work, [*NUMERICAL_FEATURES, *CATEGORICAL_FEATURES, TARGET_COLUMN])

    unexpected = set(np.unique(work[TARGET_COLUMN].to_numpy())) - set(CLASS_NAMES)
    if unexpected:
        raise ValueError(f"Target contains classes outside {sorted(CLASS_NAMES)}: {sorted(unexpected)}.")

    # -- Stage A: drop classes too rare for a stratified split -------------- #
    counts = work[TARGET_COLUMN].value_counts()
    rare = counts[counts < min_class_support]
    if not rare.empty:
        LOGGER.warning(
            "Dropping %d row(s) from classes with fewer than %d sample(s): %s",
            int(rare.sum()), min_class_support, {int(k): int(v) for k, v in rare.items()},
        )
        work = work[work[TARGET_COLUMN].isin(counts[counts >= min_class_support].index)]
    if work[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Fewer than two classes remain — cannot split or train.")
    _log_class_distribution(work[TARGET_COLUMN].to_numpy(), "Dataset (post mapping, post rare-class filter)")

    # -- Stage B: stratified 80/20 split — BEFORE any statistic is fit ------ #
    y = work[TARGET_COLUMN].to_numpy(dtype=np.int64)
    feature_frame = work.loc[:, [*NUMERICAL_FEATURES, *CATEGORICAL_FEATURES]]
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        feature_frame, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=True,
    )
    x_train_raw = x_train_raw.reset_index(drop=True)
    x_test_raw = x_test_raw.reset_index(drop=True)
    LOGGER.info(
        "Stratified split (test_size=%.2f): X_train%s, X_test%s, y_train=%d, y_test=%d.",
        test_size, x_train_raw.shape, x_test_raw.shape, len(y_train), len(y_test),
    )
    _log_class_distribution(y_train, "y_train — natural imbalance (pre-resampling)")
    _log_class_distribution(y_test, "y_test — intentionally left imbalanced")

    # -- Stage C: numeric sanitisation (medians fit on TRAIN only) ---------- #
    with _stage_timer("Numeric sanitisation & median imputation"):
        train_numeric, train_medians = _sanitize_numeric_block(x_train_raw, NUMERICAL_FEATURES)
        test_numeric, _ = _sanitize_numeric_block(x_test_raw, NUMERICAL_FEATURES, medians=train_medians)

    # -- Stage D: one-hot encoding (vocabulary from TRAIN only) ------------- #
    with _stage_timer("Categorical one-hot encoding"):
        train_categories = _normalize_categories(x_train_raw, CATEGORICAL_FEATURES)
        test_categories = _normalize_categories(x_test_raw, CATEGORICAL_FEATURES)
        train_dummies, test_dummies, dummy_columns = _encode_categoricals(train_categories, test_categories)

    feature_names: list[str] = [*NUMERICAL_FEATURES, *dummy_columns]
    x_train_frame = pd.concat([train_numeric, train_dummies], axis=1).loc[:, feature_names].astype(np.float32)
    x_test_frame = pd.concat([test_numeric, test_dummies], axis=1).loc[:, feature_names].astype(np.float32)
    LOGGER.info(
        "Feature space assembled: %d features (%d numeric + %d one-hot) — %s",
        len(feature_names), len(NUMERICAL_FEATURES), len(dummy_columns), feature_names,
    )

    # -- Stage E: RobustScaler — fit on TRAIN, transform TEST --------------- #
    # STRICT RULE (spec §4): the scaler is fitted on the training fold ONLY
    # via fit_transform; the test fold is transformed with the frozen
    # training statistics. Nothing below ever re-fits on test data.
    with _stage_timer("RobustScaler fit (train) / transform (test)"):
        scaler = RobustScaler()  # median / IQR — robust to heavy-tailed byte counts
        numeric_span = len(NUMERICAL_FEATURES)
        x_train = x_train_frame.to_numpy(dtype=np.float32)
        x_test = x_test_frame.to_numpy(dtype=np.float32)
        scaled_train_numeric = scaler.fit_transform(x_train[:, :numeric_span])
        scaled_test_numeric = scaler.transform(x_test[:, :numeric_span])  # leakage-safe
        x_train = np.concatenate([scaled_train_numeric, x_train[:, numeric_span:]], axis=1).astype(np.float32, copy=False)
        x_test = np.concatenate([scaled_test_numeric, x_test[:, numeric_span:]], axis=1).astype(np.float32, copy=False)
        LOGGER.info(
            "Scaled feature matrices: X_train%s, X_test%s (dtype=%s).",
            x_train.shape, x_test.shape, x_train.dtype,
        )

    # Persist the fitted transformer and feature schema for inference time.
    scaler_path = output_path / SCALER_FILENAME
    joblib.dump(scaler, scaler_path)
    feature_metadata: dict[str, Any] = {
        "feature_names": feature_names,
        "numeric_features": list(NUMERICAL_FEATURES),  # column order expected by the scaler
        "categorical_features": list(CATEGORICAL_FEATURES),
        "dummy_columns": dummy_columns,
        "categorical_reference_levels": {
            column: sorted(train_categories[column].unique())[0]
            for column in CATEGORICAL_FEATURES
        },
        "train_medians": {column: float(value) for column, value in train_medians.items()},
        "class_names": dict(CLASS_NAMES),
        "split": {"test_size": test_size, "random_state": random_state},
        "smote_enn_applied": bool(resample),
    }
    feature_names_path = output_path / FEATURE_NAMES_FILENAME
    joblib.dump(feature_metadata, feature_names_path)
    LOGGER.info("Saved fitted RobustScaler → %s", scaler_path)
    LOGGER.info("Saved feature schema (%d features) → %s", len(feature_names), feature_names_path)

    # -- Stage F: SMOTE-ENN — training fold ONLY (spec §5) ------------------ #
    # STRICT RULE: the test fold (x_test, y_test) is never resampled.
    if resample:
        with _stage_timer("SMOTE-ENN resampling (training fold only)"):
            x_train, y_train = _apply_smote_enn(x_train, y_train, random_state=random_state)
    else:
        LOGGER.warning("Resampling disabled — the training fold keeps its natural imbalance.")

    # -- Stage G: persist final artifacts (spec §6) -------------------------- #
    np.save(output_path / X_TRAIN_FILENAME, x_train)
    np.save(output_path / Y_TRAIN_FILENAME, y_train)
    np.save(output_path / X_TEST_FILENAME, x_test)
    np.save(output_path / Y_TEST_FILENAME, y_test)
    for filename, array in (
        (X_TRAIN_FILENAME, x_train),
        (Y_TRAIN_FILENAME, y_train),
        (X_TEST_FILENAME, x_test),
        (Y_TEST_FILENAME, y_test),
    ):
        LOGGER.info(
            "Saved %s → %s (shape=%s, dtype=%s)",
            filename, output_path / filename, array.shape, array.dtype,
        )

    return {"X_train": x_train, "y_train": y_train, "X_test": x_test, "y_test": y_test}


# --------------------------------------------------------------------------- #
# Console entry point                                                          #
# --------------------------------------------------------------------------- #


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser used when the module is executed directly."""
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset",
        description=(
            "Ingest IoT-23 Zeek conn.log.labeled file(s) and emit leakage-safe, "
            "SMOTE-ENN-resampled train/test NumPy artifacts for the deep-learning "
            "NIDS of Dutta et al. (2020)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--conn-log", type=Path, required=True,
        help="Zeek conn.log.labeled file (plain or .gz), or a directory searched "
             "recursively for '*.log.labeled[.gz]'.",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("data/processed"),
                        help="Directory for the generated artifacts.")
    parser.add_argument("--sample-size", type=int, default=100_000,
                        help="Uniform random sample of connection records (<= 0 keeps all).")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Held-out fraction for the stratified test split.")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Global seed for sampling, splitting and resampling.")
    parser.add_argument("--no-resample", action="store_true",
                        help="Disable SMOTE-ENN resampling of the training fold.")
    parser.add_argument("--drop-ambiguous", action="store_true",
                        help="Drop rows with unrecognised attack labels instead of "
                             "defaulting them to class 2.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full pipeline from the command line.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code — ``0`` on success, ``1`` on failure.
    """
    args = _build_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER.info("=" * 78)
    LOGGER.info("IoT-23 dataset builder — Dutta et al. (2020) preprocessing pipeline")
    LOGGER.info(
        "input=%s | output_dir=%s | sample_size=%s | test_size=%.2f | "
        "random_state=%d | resample=%s | drop_ambiguous=%s",
        args.conn_log, args.output_dir, args.sample_size, args.test_size, args.random_state,
        not args.no_resample, args.drop_ambiguous,
    )
    LOGGER.info("=" * 78)

    try:
        raw_frame = parse_zeek_conn_log(
            args.conn_log, sample_size=args.sample_size, random_state=args.random_state
        )
        if raw_frame.empty:
            LOGGER.error("No connection records were parsed from '%s' — aborting.", args.conn_log)
            return 1
        labelled_frame = map_labels(raw_frame, drop_ambiguous=args.drop_ambiguous)
        arrays = preprocess_and_split(
            labelled_frame,
            test_size=args.test_size,
            random_state=args.random_state,
            output_dir=args.output_dir,
            resample=not args.no_resample,
        )
    except Exception:  # Deliberate top-level CLI guard — log and exit cleanly.
        LOGGER.exception("Dataset build failed.")
        return 1

    LOGGER.info("=" * 78)
    LOGGER.info("Pipeline finished — artifacts written to '%s':", args.output_dir.resolve())
    for name in ("X_train", "y_train", "X_test", "y_test"):
        array = arrays[name]
        LOGGER.info("  %-7s shape=%-16s dtype=%s", name, array.shape, array.dtype)
    LOGGER.info(
        "Train the NIDS on (X_train, y_train); evaluate ONLY against the "
        "untouched, naturally-imbalanced (X_test, y_test)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())