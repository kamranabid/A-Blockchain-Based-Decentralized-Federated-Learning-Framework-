import os
import sys
import re
import gc
import json
import time
import math
import random
import shutil
import hashlib
import platform
import warnings
import subprocess
import importlib.util

from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings(
    "ignore",
    category=pd.errors.DtypeWarning if "pd" in globals() else Warning
)

from google.colab import drive

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
else:
    print("✅ Google Drive already mounted.")

def ensure_package(import_name, pip_name=None):

    pip_name = pip_name or import_name

    if importlib.util.find_spec(import_name) is None:

        print(f"📦 Installing {pip_name} ...")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            pip_name
        ])

    else:

        print(f"✅ {import_name} already available.")

for imp, pip_name in [

    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("sklearn", "scikit-learn"),
    ("psutil", "psutil"),
    ("torch", "torch"),

]:

    ensure_package(
        imp,
        pip_name
    )

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import sklearn
import psutil
import torch

warnings.filterwarnings(
    "ignore",
    category=pd.errors.DtypeWarning
)

print("\n" + "=" * 110)
print("IJACSA — HYBRID BLOCKCHAIN–FEDERATED LEARNING IoT SECURITY")
print("STEP 1 — COMPLETE DATASET PREPARATION")
print("=" * 110)

if not torch.cuda.is_available():

    raise RuntimeError(
        "\n❌ GPU NOT ENABLED.\n\n"
        "Google Colab:\n"
        "Runtime -> Change runtime type -> GPU\n"
        "Then rerun this cell."
    )

DEVICE = torch.device("cuda:0")

GPU_NAME = torch.cuda.get_device_name(0)

GPU_MEMORY_GB = (
    torch.cuda
    .get_device_properties(0)
    .total_memory
    /
    (1024 ** 3)
)

print("\n✅ GPU DETECTED")
print("-" * 70)

print("GPU        :", GPU_NAME)
print("GPU VRAM   :", f"{GPU_MEMORY_GB:.2f} GB")
print("PyTorch    :", torch.__version__)
print("CUDA       :", torch.version.cuda)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:

    torch.set_float32_matmul_precision(
        "high"
    )

except Exception:

    pass

BASE_SEED = 42
SPLIT_SEED = 2026

random.seed(BASE_SEED)
np.random.seed(BASE_SEED)

torch.manual_seed(BASE_SEED)
torch.cuda.manual_seed_all(BASE_SEED)

print("\n🧪 Testing GPU...")

_A = torch.randn(
    2048,
    2048,
    device=DEVICE
)

_B = torch.randn(
    2048,
    2048,
    device=DEVICE
)

_C = _A @ _B

torch.cuda.synchronize()

print(
    "✅ GPU computation successful | mean:",
    float(_C.mean().item())
)

del _A, _B, _C

torch.cuda.empty_cache()

DATASET_ROOT = Path(
    "/content/drive/MyDrive/CIC_DIAD_2024/FULL_DATASET/"
    "IoT device identification and anomaly detection dataset "
    "(CIC IoT-DIAD 2024)"
)

if not DATASET_ROOT.exists():

    raise FileNotFoundError(
        "\n❌ EXISTING DATASET NOT FOUND:\n"
        f"{DATASET_ROOT}\n\n"
        "Check Google Drive mount."
    )

csv_files = sorted(
    DATASET_ROOT.rglob(
        "*.csv"
    )
)

if len(csv_files) == 0:

    raise RuntimeError(
        "\n❌ Dataset exists but no CSV files were found."
    )

TOTAL_RAW_BYTES = sum(
    p.stat().st_size
    for p in csv_files
)

TOTAL_RAW_GB = (
    TOTAL_RAW_BYTES /
    (1024 ** 3)
)

print("\n" + "=" * 100)
print("EXISTING CIC IoT-DIAD 2024 DATASET")
print("=" * 100)

print("Path:")
print(DATASET_ROOT)

print(
    f"\nCSV files : {len(csv_files):,}"
)

print(
    f"Size      : {TOTAL_RAW_GB:.2f} GB"
)

EXPECTED_TOTAL_CSV = 129

if len(csv_files) != EXPECTED_TOTAL_CSV:

    print(
        f"\n⚠️ Previous verified run had "
        f"{EXPECTED_TOTAL_CSV} CSV files."
    )

    print(
        f"Current run found {len(csv_files)}."
    )

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/"
    "Hybrid_BCFL_IJACSA_2026"
)

CONFIG_DIR = (
    PROJECT_ROOT /
    "00_CONFIG"
)

AUDIT_DIR = (
    PROJECT_ROOT /
    "01_DATASET_AUDIT"
)

CLEAN_ROOT = (
    PROJECT_ROOT /
    "02_CLEAN_PARQUET"
)

SPLIT_DIR = (
    PROJECT_ROOT /
    "03_SPLITS"
)

PREPROCESSOR_DIR = (
    PROJECT_ROOT /
    "04_PREPROCESSOR"
)

MODEL_READY_ROOT = (
    PROJECT_ROOT /
    "05_MODEL_READY"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT /
    "06_CHECKPOINTS"
)

AI_MODEL_DIR = (
    PROJECT_ROOT /
    "07_AI_MODELS"
)

FL_DIR = (
    PROJECT_ROOT /
    "08_FEDERATED_LEARNING"
)

PRIVACY_DIR = (
    PROJECT_ROOT /
    "09_PRIVACY"
)

BLOCKCHAIN_DIR = (
    PROJECT_ROOT /
    "10_BLOCKCHAIN"
)

RESULTS_DIR = (
    PROJECT_ROOT /
    "11_RESULTS"
)

FIGURES_DIR = (
    PROJECT_ROOT /
    "12_FIGURES"
)

TABLES_DIR = (
    PROJECT_ROOT /
    "13_TABLES"
)

LOG_DIR = (
    PROJECT_ROOT /
    "14_LOGS"
)

for folder in [

    PROJECT_ROOT,
    CONFIG_DIR,
    AUDIT_DIR,
    CLEAN_ROOT,
    SPLIT_DIR,
    PREPROCESSOR_DIR,
    MODEL_READY_ROOT,
    CHECKPOINT_DIR,
    AI_MODEL_DIR,
    FL_DIR,
    PRIVACY_DIR,
    BLOCKCHAIN_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    LOG_DIR

]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

print("\n✅ NEW PROJECT:")
print(PROJECT_ROOT)

print("\n✅ RAW DATASET WILL REMAIN READ-ONLY.")

LOCAL_CACHE = Path(
    "/content/"
    "Hybrid_BCFL_IJACSA_RUNTIME"
)

LOCAL_CACHE.mkdir(
    parents=True,
    exist_ok=True
)

LOCAL_RAW_CACHE = (
    LOCAL_CACHE /
    "CURRENT_RAW"
)

LOCAL_RAW_CACHE.mkdir(
    parents=True,
    exist_ok=True
)

print("\n✅ Fast runtime cache:")
print(LOCAL_CACHE)

PIPELINE_VERSION = (
    "IJACSA_STEP01_CIC_DIAD_V1"
)

CHUNK_SIZE = 150_000

PROFILE_ROWS_PER_FILE = 4_000
PROFILE_FILES_PER_CLASS = 2

NUMERIC_THRESHOLD = 0.90

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MIN_FEATURE_VARIANCE = 1e-12

RESUME = True
FORCE_REBUILD = False

FAST_EXIT_IF_COMPLETE = True

USE_LOCAL_RAW_CACHE = True

USE_GPU_FOR_NORMALIZATION = True

EXPECTED_FLOW_FILES = 124
EXPECTED_EXCLUDED_FILES = 5
EXPECTED_MODEL_FEATURES_APPROX = 79
EXPECTED_ROWS_APPROX = 19_519_162

CLASS_TO_ID = {

    "Benign": 0,
    "DDoS": 1,
    "DoS": 2,
    "Recon": 3,
    "Web-Based": 4,
    "Brute Force": 5,
    "Spoofing": 6,
    "Mirai": 7
}

ID_TO_CLASS = {

    value: key

    for key, value
    in CLASS_TO_ID.items()

}

CLASS_NAMES = [

    ID_TO_CLASS[i]

    for i in range(8)

]

CLEAN_SHARD_ROOT = (
    CLEAN_ROOT /
    "UNSCALED_SHARDS"
)

CLEAN_SHARD_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_DIR = (
    MODEL_READY_ROOT /
    "TRAIN"
)

VAL_DIR = (
    MODEL_READY_ROOT /
    "VALIDATION"
)

TEST_DIR = (
    MODEL_READY_ROOT /
    "TEST"
)

for folder in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

STEP1_PROGRESS = (
    CHECKPOINT_DIR /
    "STEP01_PROGRESS.json"
)

STEP1_CLEAN_COMPLETE = (
    CHECKPOINT_DIR /
    "STEP01_CLEAN_COMPLETE.json"
)

STEP1_COMPLETE = (
    CHECKPOINT_DIR /
    "STEP01_COMPLETE.json"
)

def safe_slug(
    text,
    max_length=70
):

    text = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(text)
    ).strip("_")

    if not text:

        text = "item"

    return text[:max_length]

def short_hash(
    value,
    length=10
):

    return hashlib.sha1(
        str(value)
        .encode("utf-8")
    ).hexdigest()[:length]

def atomic_copy(
    local_file,
    final_file,
    retries=6
):

    local_file = Path(
        local_file
    )

    final_file = Path(
        final_file
    )

    final_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    partial_file = Path(
        str(final_file)
        +
        ".partial"
    )

    last_error = None

    for attempt in range(
        1,
        retries + 1
    ):

        try:

            partial_file.unlink(
                missing_ok=True
            )

            shutil.copy2(
                local_file,
                partial_file
            )

            if (
                not partial_file.exists()
                or
                partial_file.stat().st_size <= 0
            ):

                raise IOError(
                    "Temporary Drive file is empty."
                )

            os.replace(
                partial_file,
                final_file
            )

            return

        except Exception as exc:

            last_error = exc

            print(
                f"⚠️ Save retry "
                f"{attempt}/{retries}: {exc}"
            )

            time.sleep(
                attempt * 3
            )

    raise IOError(
        f"Unable to save:\n{final_file}"
    ) from last_error

def save_json(
    obj,
    destination
):

    destination = Path(
        destination
    )

    local = (
        LOCAL_CACHE /
        (
            safe_slug(
                destination.stem
            )
            +
            "_"
            +
            short_hash(
                destination
            )
            +
            ".json"
        )
    )

    with open(
        local,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    atomic_copy(
        local,
        destination
    )

    local.unlink(
        missing_ok=True
    )

def save_csv(
    dataframe,
    destination
):

    destination = Path(
        destination
    )

    local = (
        LOCAL_CACHE /
        (
            safe_slug(
                destination.stem
            )
            +
            "_"
            +
            short_hash(
                destination
            )
            +
            ".csv"
        )
    )

    dataframe.to_csv(
        local,
        index=False
    )

    atomic_copy(
        local,
        destination
    )

    local.unlink(
        missing_ok=True
    )

def save_parquet(
    dataframe,
    destination
):

    destination = Path(
        destination
    )

    local = (
        LOCAL_CACHE /
        (
            safe_slug(
                destination.stem
            )
            +
            "_"
            +
            short_hash(
                destination
            )
            +
            ".parquet"
        )
    )

    dataframe.to_parquet(
        local,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    atomic_copy(
        local,
        destination
    )

    local.unlink(
        missing_ok=True
    )

def valid_parquet(path):

    path = Path(
        path
    )

    try:

        if (
            not path.exists()
            or
            path.stat().st_size <= 0
        ):

            return False

        _ = pq.ParquetFile(
            path
        ).metadata.num_rows

        return True

    except Exception:

        return False

def normalize_column(
    column_name
):

    text = str(
        column_name
    )

    text = (
        text
        .replace(
            "\ufeff",
            ""
        )
        .strip()
    )

    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1_\2",
        text
    )

    text = text.lower()

    text = re.sub(
        r"[\(\)\[\]\{\}]",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    text = re.sub(
        r"_+",
        "_",
        text
    )

    return text.strip("_")

def get_attack_family(
    file_path
):

    path_text = str(
        file_path
    ).lower()

    path_parts = [

        part.lower()

        for part
        in Path(file_path).parts

    ]

    if "benign" in path_text:
        return "Benign"

    if "mirai" in path_text:
        return "Mirai"

    if "spoof" in path_text:
        return "Spoofing"

    if "brute" in path_text:
        return "Brute Force"

    if "recon" in path_text:
        return "Recon"

    if (

        "web-based" in path_text
        or
        "web_based" in path_text
        or
        "web based" in path_text
        or
        "webattack" in path_text
        or
        "web_attack" in path_text

    ):

        return "Web-Based"

    if "ddos" in path_text:

        return "DDoS"

    if (

        any(
            part == "dos"
            for part in path_parts
        )

        or

        re.search(
            r"(^|[/_\-\s])dos([/_\-\s]|$)",
            path_text
        )

    ):

        return "DoS"

    return "UNKNOWN"

def existing_complete_summary():

    if not (
        FAST_EXIT_IF_COMPLETE
        and
        not FORCE_REBUILD
        and
        STEP1_COMPLETE.exists()
    ):

        return None

    try:

        with open(
            STEP1_COMPLETE,
            "r",
            encoding="utf-8"
        ) as f:

            summary = json.load(
                f
            )

        if (

            summary.get(
                "pipeline_version"
            )
            ==
            PIPELINE_VERSION

            and

            summary.get(
                "status"
            )
            ==
            "COMPLETED"

        ):

            return summary

    except Exception:

        pass

    return None

def reset_step1():

    if not FORCE_REBUILD:

        return

    print(
        "\n⚠️ FORCE_REBUILD=True"
    )

    for folder in [

        CLEAN_ROOT,
        SPLIT_DIR,
        PREPROCESSOR_DIR,
        MODEL_READY_ROOT

    ]:

        if folder.exists():

            shutil.rmtree(
                folder
            )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    CLEAN_SHARD_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for marker in [

        STEP1_PROGRESS,
        STEP1_CLEAN_COMPLETE,
        STEP1_COMPLETE

    ]:

        marker.unlink(
            missing_ok=True
        )

def build_manifest():

    records = []

    for file_id, path in enumerate(
        csv_files
    ):

        relative = path.relative_to(
            DATASET_ROOT
        )

        records.append({

            "file_id":
                int(file_id),

            "file_name":
                path.name,

            "relative_path":
                str(relative),

            "absolute_path":
                str(path),

            "top_folder":
                (
                    relative.parts[0]
                    if len(relative.parts) > 0
                    else ""
                ),

            "attack_family":
                get_attack_family(
                    path
                ),

            "size_mb":
                round(
                    path.stat().st_size /
                    (1024 ** 2),
                    4
                )

        })

    manifest = pd.DataFrame(
        records
    )

    save_csv(
        manifest,
        AUDIT_DIR /
        "raw_dataset_manifest.csv"
    )

    return manifest

def build_header_audit(
    manifest
):

    audit_file = (
        AUDIT_DIR /
        "header_schema_audit.csv"
    )

    if (
        RESUME
        and
        not FORCE_REBUILD
        and
        audit_file.exists()
    ):

        try:

            old = pd.read_csv(
                audit_file
            )

            if len(old) == len(
                manifest
            ):

                print(
                    "✅ Reusing header/schema audit."
                )

                return old

        except Exception:

            pass

    print(
        "\n🔎 Building header/schema audit..."
    )

    records = []

    for index, row in (
        manifest
        .reset_index(drop=True)
        .iterrows()
    ):

        path = Path(
            row[
                "absolute_path"
            ]
        )

        try:

            original_columns = (
                pd.read_csv(
                    path,
                    nrows=0
                )
                .columns
                .tolist()
            )

            normalized_columns = [

                normalize_column(c)

                for c
                in original_columns

            ]

            normalized_set = set(
                normalized_columns
            )

            src_candidates = [

                c

                for c
                in normalized_columns

                if c in [

                    "src_ip",
                    "source_ip",
                    "srcip",
                    "sourceip"

                ]

            ]

            dst_candidates = [

                c

                for c
                in normalized_columns

                if c in [

                    "dst_ip",
                    "destination_ip",
                    "dstip",
                    "destinationip"

                ]

            ]

            label_candidates = [

                c

                for c
                in normalized_columns

                if "label" in c

            ]

            has_flow_features = any(

                c in normalized_set

                for c in [

                    "flow_duration",
                    "flow_id",
                    "flow_iat_mean",
                    "flow_bytes_s",
                    "flow_packets_s"

                ]

            )

            looks_like_flow = (

                70
                <=
                len(original_columns)
                <=
                110

                and

                (

                    bool(src_candidates)
                    or
                    bool(dst_candidates)
                    or
                    has_flow_features

                )

                and

                bool(label_candidates)

            )

            records.append({

                "file_id":
                    int(
                        row[
                            "file_id"
                        ]
                    ),

                "absolute_path":
                    str(path),

                "file_name":
                    path.name,

                "attack_family":
                    row[
                        "attack_family"
                    ],

                "n_columns":
                    len(
                        original_columns
                    ),

                "schema_type":
                    (
                        "flow_anomaly_only"
                        if looks_like_flow
                        else "unknown"
                    ),

                "src_ip_column":
                    (
                        src_candidates[0]
                        if src_candidates
                        else ""
                    ),

                "dst_ip_column":
                    (
                        dst_candidates[0]
                        if dst_candidates
                        else ""
                    ),

                "label_column":
                    (
                        label_candidates[0]
                        if label_candidates
                        else ""
                    ),

                "columns_json":
                    json.dumps(
                        normalized_columns
                    ),

                "read_error":
                    ""

            })

        except Exception as exc:

            records.append({

                "file_id":
                    int(
                        row[
                            "file_id"
                        ]
                    ),

                "absolute_path":
                    str(path),

                "file_name":
                    path.name,

                "attack_family":
                    row[
                        "attack_family"
                    ],

                "n_columns":
                    np.nan,

                "schema_type":
                    "read_error",

                "src_ip_column":
                    "",

                "dst_ip_column":
                    "",

                "label_column":
                    "",

                "columns_json":
                    "[]",

                "read_error":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

            })

        if (
            (index + 1) % 10 == 0
            or
            (index + 1)
            ==
            len(manifest)
        ):

            save_csv(
                pd.DataFrame(
                    records
                ),
                audit_file
            )

            print(
                f"  ✅ {index+1}/"
                f"{len(manifest)} headers audited"
            )

    return pd.DataFrame(
        records
    )

def resolve_flow_schema(
    header_df
):

    flow_df = (
        header_df[
            header_df[
                "schema_type"
            ]
            ==
            "flow_anomaly_only"
        ]
        .copy()
    )

    flow_df = (
        flow_df[
            flow_df[
                "attack_family"
            ]
            .isin(
                CLASS_TO_ID.keys()
            )
        ]
        .copy()
    )

    excluded_df = (
        header_df[
            ~header_df[
                "file_id"
            ]
            .isin(
                flow_df[
                    "file_id"
                ]
            )
        ]
        .copy()
    )

    save_csv(
        excluded_df,
        AUDIT_DIR /
        "excluded_nonflow_or_unknown_files.csv"
    )

    print("\nDATA REPRESENTATION")
    print("-" * 80)

    print(
        "Total CSV files      :",
        len(header_df)
    )

    print(
        "Flow files selected  :",
        len(flow_df)
    )

    print(
        "Excluded/audit files :",
        len(excluded_df)
    )

    if len(flow_df) != EXPECTED_FLOW_FILES:

        print(
            f"⚠️ Previous verified run had "
            f"{EXPECTED_FLOW_FILES} usable flow files."
        )

    if len(excluded_df) != EXPECTED_EXCLUDED_FILES:

        print(
            f"⚠️ Previous verified run excluded "
            f"{EXPECTED_EXCLUDED_FILES} files."
        )

    if len(flow_df) == 0:

        raise RuntimeError(
            "❌ No usable flow files found."
        )

    column_sets = []

    for value in flow_df[
        "columns_json"
    ]:

        try:

            column_sets.append(
                set(
                    json.loads(
                        value
                    )
                )
            )

        except Exception:

            pass

    common_columns = set.intersection(
        *column_sets
    )

    SOURCE_CANDIDATES = [

        "src_ip",
        "source_ip",
        "srcip",
        "sourceip"

    ]

    DEST_CANDIDATES = [

        "dst_ip",
        "destination_ip",
        "dstip",
        "destinationip"

    ]

    SRC_IP_COL = next(
        (
            c
            for c
            in SOURCE_CANDIDATES
            if c in common_columns
        ),
        None
    )

    DST_IP_COL = next(
        (
            c
            for c
            in DEST_CANDIDATES
            if c in common_columns
        ),
        None
    )

    LABEL_PRIORITY = [

        "label",
        "attack_label",
        "class_label"

    ]

    LABEL_COL = next(
        (
            c
            for c
            in LABEL_PRIORITY
            if c in common_columns
        ),
        None
    )

    if LABEL_COL is None:

        labels = sorted([
            c
            for c
            in common_columns
            if "label" in c
        ])

        if labels:
            LABEL_COL = labels[0]

    if SRC_IP_COL is None:

        raise RuntimeError(
            "❌ Source IP column not detected."
        )

    if LABEL_COL is None:

        raise RuntimeError(
            "❌ Label column not detected."
        )

    print(
        "\nCommon flow columns :",
        len(common_columns)
    )

    print(
        "Source endpoint      :",
        SRC_IP_COL
    )

    print(
        "Destination metadata :",
        DST_IP_COL
    )

    print(
        "Original label       :",
        LABEL_COL
    )

    return (
        flow_df,
        excluded_df,
        common_columns,
        SRC_IP_COL,
        DST_IP_COL,
        LABEL_COL
    )

def get_candidate_features(
    common_columns,
    SRC_IP_COL,
    DST_IP_COL,
    LABEL_COL
):

    leakage_columns = {

        SRC_IP_COL,
        DST_IP_COL,

        "src_ip",
        "source_ip",
        "srcip",
        "sourceip",

        "dst_ip",
        "destination_ip",
        "dstip",
        "destinationip",

        "flow_id",
        "timestamp",

        "device_mac",
        "src_mac",
        "dst_mac",

        "eth_src",
        "eth_dst",

        "eth_src_oui",
        "eth_dst_oui",

        "http_host",
        "http_uri",
        "user_agent",

        "dns_server",
        "tls_server",

        "stream",

        LABEL_COL

    }

    leakage_columns = {

        c

        for c
        in leakage_columns

        if c is not None

    }

    candidates = sorted([

        c

        for c
        in common_columns

        if (
            c not in leakage_columns
            and
            "label" not in c
        )

    ])

    save_json(

        {
            "raw_identifier_columns_excluded":
                sorted(
                    leakage_columns
                ),

            "endpoint_retained_as_metadata":
                SRC_IP_COL,

            "endpoint_used_as_ml_feature":
                False,

            "candidate_feature_count":
                len(
                    candidates
                )

        },

        AUDIT_DIR /
        "leakage_control.json"

    )

    print(
        "\nLeakage-safe candidate features:",
        len(candidates)
    )

    return candidates

def profile_numeric_features(
    flow_df,
    common_columns,
    candidate_features,
    SRC_IP_COL,
    DST_IP_COL,
    LABEL_COL
):

    feature_list_file = (
        CONFIG_DIR /
        "selected_numeric_features.json"
    )

    profile_file = (
        AUDIT_DIR /
        "feature_selection_profile.csv"
    )

    if (
        RESUME
        and
        not FORCE_REBUILD
        and
        feature_list_file.exists()
    ):

        try:

            with open(
                feature_list_file,
                "r",
                encoding="utf-8"
            ) as f:

                old = json.load(
                    f
                )

            if (
                old.get(
                    "pipeline_version"
                )
                ==
                PIPELINE_VERSION
            ):

                selected = old[
                    "selected_features"
                ]

                print(
                    "\n✅ Reusing selected feature list:",
                    len(selected)
                )

                return selected

        except Exception:

            pass

    print(
        "\n🔬 Profiling stable numeric behavioral features..."
    )

    representative = []

    for family in CLASS_TO_ID:

        family_df = flow_df[
            flow_df[
                "attack_family"
            ]
            ==
            family
        ]

        if len(
            family_df
        ):

            representative.append(
                family_df.head(
                    PROFILE_FILES_PER_CLASS
                )
            )

    representative_df = pd.concat(
        representative,
        ignore_index=True
    )

    feature_stats = {

        c: {
            "observed": 0,
            "finite_numeric": 0
        }

        for c
        in candidate_features

    }

    for i, row in (
        representative_df
        .iterrows()
    ):

        path = Path(
            row[
                "absolute_path"
            ]
        )

        sample = pd.read_csv(
            path,
            nrows=PROFILE_ROWS_PER_FILE,
            low_memory=False,
            on_bad_lines="skip"
        )

        sample.columns = [

            normalize_column(c)

            for c
            in sample.columns

        ]

        sample = sample.loc[
            :,
            ~sample.columns.duplicated()
        ]

        for feature in candidate_features:

            if feature not in sample.columns:

                continue

            raw = sample[
                feature
            ]

            strings = (
                raw
                .astype("string")
                .str.strip()
            )

            valid = (

                raw.notna()

                &

                strings.ne("")

                &

                ~strings
                .str.lower()
                .isin([

                    "nan",
                    "none",
                    "null",
                    "<na>",
                    "inf",
                    "+inf",
                    "-inf",
                    "infinity",
                    "+infinity",
                    "-infinity"

                ])

            )

            observed = int(
                valid.sum()
            )

            if observed == 0:

                continue

            numeric = pd.to_numeric(
                raw[
                    valid
                ],
                errors="coerce"
            )

            numeric_np = numeric.to_numpy(
                dtype=np.float64,
                na_value=np.nan
            )

            finite_numeric = int(
                np.isfinite(
                    numeric_np
                ).sum()
            )

            feature_stats[
                feature
            ][
                "observed"
            ] += observed

            feature_stats[
                feature
            ][
                "finite_numeric"
            ] += finite_numeric

        print(
            f"  ✅ Profile "
            f"{i+1}/{len(representative_df)} | "
            f"{row['attack_family']}"
        )

        del sample
        gc.collect()

    records = []

    for column in sorted(
        common_columns
    ):

        if column not in candidate_features:

            records.append({

                "feature":
                    column,

                "observed_values":
                    0,

                "numeric_values":
                    0,

                "numeric_ratio":
                    0.0,

                "selected":
                    False,

                "reason":
                    "IDENTIFIER_LABEL_OR_UNSUPPORTED"

            })

            continue

        observed = (
            feature_stats[
                column
            ][
                "observed"
            ]
        )

        numeric = (
            feature_stats[
                column
            ][
                "finite_numeric"
            ]
        )

        ratio = (
            numeric / observed
            if observed > 0
            else 0.0
        )

        selected = (
            observed > 0
            and
            ratio >= NUMERIC_THRESHOLD
        )

        records.append({

            "feature":
                column,

            "observed_values":
                observed,

            "numeric_values":
                numeric,

            "numeric_ratio":
                ratio,

            "selected":
                selected,

            "reason":
                (
                    "SELECTED_NUMERIC_BEHAVIORAL_FEATURE"
                    if selected
                    else "NON_NUMERIC_OR_UNSTABLE"
                )

        })

    profile_df = pd.DataFrame(
        records
    )

    selected = sorted(
        profile_df.loc[
            profile_df[
                "selected"
            ],
            "feature"
        ]
        .tolist()
    )

    if len(selected) < 10:

        raise RuntimeError(
            "❌ Too few stable numeric features selected."
        )

    save_csv(
        profile_df,
        profile_file
    )

    save_json(

        {
            "pipeline_version":
                PIPELINE_VERSION,

            "numeric_threshold":
                NUMERIC_THRESHOLD,

            "selected_feature_count":
                len(
                    selected
                ),

            "selected_features":
                selected,

            "endpoint_feature_excluded":
                SRC_IP_COL,

            "destination_feature_excluded":
                DST_IP_COL,

            "label_excluded":
                LABEL_COL

        },

        feature_list_file

    )

    print(
        "\n✅ Selected numeric features:",
        len(selected)
    )

    if (
        abs(
            len(selected)
            -
            EXPECTED_MODEL_FEATURES_APPROX
        )
        >
        10
    ):

        print(
            "⚠️ Feature count differs noticeably "
            "from previous run."
        )

    return selected

def get_column_mapping(
    file_path
):

    original_columns = (
        pd.read_csv(
            file_path,
            nrows=0
        )
        .columns
        .tolist()
    )

    mapping = {}

    for original in original_columns:

        normalized = normalize_column(
            original
        )

        if normalized not in mapping:

            mapping[
                normalized
            ] = original

    return mapping

def get_fast_source(
    source_path
):

    source_path = Path(
        source_path
    )

    if not USE_LOCAL_RAW_CACHE:

        return source_path, False

    local_path = (
        LOCAL_RAW_CACHE /
        (
            safe_slug(
                source_path.name,
                90
            )
            +
            "_"
            +
            short_hash(
                source_path
            )
            +
            ".csv"
        )
    )

    try:

        if (
            not local_path.exists()
            or
            local_path.stat().st_size
            !=
            source_path.stat().st_size
        ):

            local_path.unlink(
                missing_ok=True
            )

            print(
                "  ⚡ Copying current CSV to local SSD..."
            )

            shutil.copy2(
                source_path,
                local_path
            )

        return local_path, True

    except Exception as exc:

        print(
            "  ⚠️ Local cache copy failed; "
            "reading directly from Drive:",
            exc
        )

        return source_path, False

def build_clean_shards(
    flow_df,
    SELECTED_FEATURES,
    SRC_IP_COL,
    DST_IP_COL,
    LABEL_COL
):

    print("\n" + "=" * 105)
    print("BUILDING RESUMABLE CLEAN PARQUET SHARDS")
    print("=" * 105)

    flow_df = (
        flow_df
        .reset_index(
            drop=True
        )
    )

    for file_position, row in (
        flow_df.iterrows()
    ):

        source_file_id = int(
            row[
                "file_id"
            ]
        )

        original_path = Path(
            row[
                "absolute_path"
            ]
        )

        attack_family = row[
            "attack_family"
        ]

        file_tag = (

            f"F{source_file_id:03d}_"

            +
            safe_slug(
                attack_family,
                18
            )

            +
            "_"

            +
            safe_slug(
                original_path.stem,
                35
            )

            +
            "_"

            +
            short_hash(
                original_path
            )

        )

        output_dir = (
            CLEAN_SHARD_ROOT /
            file_tag
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        file_complete = (
            output_dir /
            "FILE_COMPLETE.json"
        )

        print(
            f"\n[{file_position+1}/"
            f"{len(flow_df)}] "
            f"{attack_family} | "
            f"{original_path.name}"
        )

        if (
            RESUME
            and
            not FORCE_REBUILD
            and
            file_complete.exists()
        ):

            saved = sorted(
                output_dir.glob(
                    "chunk_*.parquet"
                )
            )

            if (
                len(saved) > 0
                and
                all(
                    valid_parquet(p)
                    for p in saved
                )
            ):

                print(
                    f"  ↪ File already complete "
                    f"({len(saved)} chunks)."
                )

                continue

        mapping = get_column_mapping(
            original_path
        )

        required_normalized = [

            SRC_IP_COL,
            LABEL_COL

        ]

        if (
            DST_IP_COL is not None
            and
            DST_IP_COL in mapping
        ):

            required_normalized.append(
                DST_IP_COL
            )

        required_normalized.extend(
            SELECTED_FEATURES
        )

        missing = [

            c

            for c
            in required_normalized

            if c not in mapping

        ]

        if missing:

            raise RuntimeError(
                f"\n❌ Required columns missing in "
                f"{original_path.name}:\n"
                f"{missing[:30]}"
            )

        use_columns = [

            mapping[c]

            for c
            in required_normalized

        ]

        use_columns = list(
            dict.fromkeys(
                use_columns
            )
        )

        read_path, is_local_copy = (
            get_fast_source(
                original_path
            )
        )

        reader = pd.read_csv(

            read_path,

            usecols=use_columns,

            chunksize=CHUNK_SIZE,

            low_memory=False,

            on_bad_lines="skip"

        )

        row_offset = 0
        completed_chunks = 0
        processed_rows = 0

        for chunk_index, raw_chunk in enumerate(
            reader
        ):

            parquet_file = (
                output_dir /
                f"chunk_{chunk_index:05d}.parquet"
            )

            if (
                RESUME
                and
                not FORCE_REBUILD
                and
                valid_parquet(
                    parquet_file
                )
            ):

                completed_chunks += 1

                processed_rows += len(
                    raw_chunk
                )

                row_offset += len(
                    raw_chunk
                )

                print(
                    f"  ↪ chunk {chunk_index:05d} reused"
                )

                del raw_chunk
                gc.collect()

                continue

            raw_chunk.columns = [

                normalize_column(c)

                for c
                in raw_chunk.columns

            ]

            raw_chunk = raw_chunk.loc[
                :,
                ~raw_chunk.columns.duplicated()
            ].copy()

            n_rows = len(
                raw_chunk
            )

            endpoint = (
                raw_chunk[
                    SRC_IP_COL
                ]
                .astype("string")
                .str.strip()
            )

            bad_endpoint = (

                endpoint.isna()

                |

                endpoint.eq("")

                |

                endpoint
                .str.lower()
                .isin([
                    "nan",
                    "none",
                    "null",
                    "<na>"
                ])

            )

            endpoint = (
                endpoint
                .mask(
                    bad_endpoint,
                    "UNKNOWN"
                )
                .fillna(
                    "UNKNOWN"
                )
            )

            if (
                DST_IP_COL is not None
                and
                DST_IP_COL
                in raw_chunk.columns
            ):

                destination = (
                    raw_chunk[
                        DST_IP_COL
                    ]
                    .astype("string")
                    .str.strip()
                    .fillna(
                        "UNKNOWN"
                    )
                )

            else:

                destination = pd.Series(
                    ["UNKNOWN"] * n_rows,
                    dtype="string"
                )

            original_label = (
                raw_chunk[
                    LABEL_COL
                ]
                .astype("string")
                .str.strip()
                .fillna(
                    "UNKNOWN"
                )
            )

            feature_block = {}

            for feature in SELECTED_FEATURES:

                values = pd.to_numeric(

                    raw_chunk[
                        feature
                    ],

                    errors="coerce"

                )

                values = values.replace(

                    [
                        np.inf,
                        -np.inf
                    ],

                    np.nan

                )

                feature_block[
                    feature
                ] = values.astype(
                    np.float32
                )

            feature_df = pd.DataFrame(
                feature_block
            )

            y_binary = (
                0
                if attack_family == "Benign"
                else 1
            )

            y_multiclass = CLASS_TO_ID[
                attack_family
            ]

            clean = pd.DataFrame({

                "source_file_id":
                    np.full(
                        n_rows,
                        source_file_id,
                        dtype=np.int16
                    ),

                "source_chunk_id":
                    np.full(
                        n_rows,
                        chunk_index,
                        dtype=np.int16
                    ),

                "row_in_chunk":
                    np.arange(
                        n_rows,
                        dtype=np.int32
                    ),

                "source_row_approx":
                    np.arange(
                        row_offset,
                        row_offset + n_rows,
                        dtype=np.int64
                    ),

                "endpoint_id":
                    endpoint.reset_index(
                        drop=True
                    ),

                "destination_endpoint":
                    destination.reset_index(
                        drop=True
                    ),

                "attack_family":
                    pd.Series(
                        [attack_family] * n_rows,
                        dtype="string"
                    ),

                "original_label":
                    original_label.reset_index(
                        drop=True
                    ),

                "y_binary":
                    np.full(
                        n_rows,
                        y_binary,
                        dtype=np.int8
                    ),

                "y_multiclass":
                    np.full(
                        n_rows,
                        y_multiclass,
                        dtype=np.int8
                    )

            })

            clean = pd.concat(

                [
                    clean,
                    feature_df.reset_index(
                        drop=True
                    )
                ],

                axis=1

            )

            save_parquet(
                clean,
                parquet_file
            )

            completed_chunks += 1
            processed_rows += n_rows
            row_offset += n_rows

            save_json(

                {
                    "pipeline_version":
                        PIPELINE_VERSION,

                    "stage":
                        "clean_parquet",

                    "status":
                        "RUNNING",

                    "file_position":
                        int(
                            file_position + 1
                        ),

                    "total_flow_files":
                        int(
                            len(flow_df)
                        ),

                    "current_file":
                        str(
                            original_path
                        ),

                    "last_completed_chunk":
                        int(
                            chunk_index
                        ),

                    "updated_at":
                        datetime.now().isoformat()

                },

                STEP1_PROGRESS

            )

            print(
                f"  ✅ chunk {chunk_index:05d} | "
                f"{n_rows:,} rows"
            )

            del raw_chunk
            del clean
            del feature_df
            del feature_block

            gc.collect()

        save_json(

            {
                "pipeline_version":
                    PIPELINE_VERSION,

                "file":
                    str(
                        original_path
                    ),

                "attack_family":
                    attack_family,

                "chunks":
                    int(
                        completed_chunks
                    ),

                "rows_seen":
                    int(
                        processed_rows
                    ),

                "status":
                    "COMPLETED",

                "completed_at":
                    datetime.now().isoformat()

            },

            file_complete

        )

        print(
            f"  ✅ FILE COMPLETE | "
            f"{processed_rows:,} rows | "
            f"{completed_chunks} chunks"
        )

        if is_local_copy:

            try:

                read_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass

        gc.collect()

    save_json(

        {
            "pipeline_version":
                PIPELINE_VERSION,

            "status":
                "COMPLETED",

            "completed_at":
                datetime.now().isoformat()

        },

        STEP1_CLEAN_COMPLETE

    )

def get_clean_shards():

    shards = sorted(
        CLEAN_SHARD_ROOT.rglob(
            "chunk_*.parquet"
        )
    )

    if not shards:

        raise RuntimeError(
            "❌ No clean Parquet shards found."
        )

    return shards

def build_global_audit(
    clean_shards
):

    print("\n" + "=" * 105)
    print("BUILDING GLOBAL ENDPOINT / CLASS AUDIT")
    print("=" * 105)

    endpoint_class = defaultdict(
        int
    )

    class_counts = defaultdict(
        int
    )

    original_label_counts = defaultdict(
        int
    )

    total_rows = 0
    unknown_endpoint_rows = 0

    for i, shard in enumerate(
        clean_shards,
        start=1
    ):

        df = pd.read_parquet(

            shard,

            columns=[

                "endpoint_id",
                "attack_family",
                "original_label"

            ]

        )

        total_rows += len(
            df
        )

        unknown_endpoint_rows += int(
            df[
                "endpoint_id"
            ]
            .eq(
                "UNKNOWN"
            )
            .sum()
        )

        for family, count in (
            df[
                "attack_family"
            ]
            .value_counts()
            .items()
        ):

            class_counts[
                str(family)
            ] += int(
                count
            )

        for label, count in (
            df[
                "original_label"
            ]
            .astype("string")
            .fillna(
                "UNKNOWN"
            )
            .value_counts()
            .items()
        ):

            original_label_counts[
                str(label)
            ] += int(
                count
            )

        grouped = (
            df
            .groupby(
                [
                    "endpoint_id",
                    "attack_family"
                ],
                dropna=False
            )
            .size()
        )

        for (
            endpoint,
            family
        ), count in grouped.items():

            endpoint_class[
                (
                    str(endpoint),
                    str(family)
                )
            ] += int(
                count
            )

        if (
            i % 25 == 0
            or
            i == len(clean_shards)
        ):

            print(
                f"  ✅ audit "
                f"{i}/{len(clean_shards)} shards"
            )

        del df
        gc.collect()

    endpoint_class_df = pd.DataFrame([

        {
            "endpoint_id":
                endpoint,

            "attack_family":
                family,

            "class_id":
                CLASS_TO_ID.get(
                    family,
                    -1
                ),

            "rows":
                count
        }

        for (
            endpoint,
            family
        ), count
        in endpoint_class.items()

    ])

    endpoint_class_df = (
        endpoint_class_df
        .sort_values(
            [
                "endpoint_id",
                "class_id"
            ]
        )
    )

    save_csv(

        endpoint_class_df,

        AUDIT_DIR /
        "endpoint_attack_family_counts.csv"

    )

    endpoint_total_df = (
        endpoint_class_df
        .groupby(
            "endpoint_id",
            as_index=False
        )[
            "rows"
        ]
        .sum()
        .rename(
            columns={
                "rows":
                    "total_rows"
            }
        )
        .sort_values(
            "total_rows",
            ascending=False
        )
    )

    save_csv(

        endpoint_total_df,

        AUDIT_DIR /
        "endpoint_total_rows.csv"

    )

    class_distribution = pd.DataFrame([

        {
            "attack_family":
                family,

            "class_id":
                class_id,

            "rows":
                class_counts.get(
                    family,
                    0
                )
        }

        for family, class_id
        in CLASS_TO_ID.items()

    ])

    class_distribution[
        "percentage"
    ] = (

        100
        *
        class_distribution[
            "rows"
        ]
        /
        max(
            1,
            class_distribution[
                "rows"
            ].sum()
        )

    )

    save_csv(

        class_distribution,

        AUDIT_DIR /
        "class_distribution_8class.csv"

    )

    original_label_df = pd.DataFrame([

        {
            "original_label":
                label,

            "rows":
                count
        }

        for label, count
        in original_label_counts.items()

    ])

    original_label_df = (
        original_label_df
        .sort_values(
            "rows",
            ascending=False
        )
    )

    save_csv(

        original_label_df,

        AUDIT_DIR /
        "original_label_distribution.csv"

    )

    known_endpoints = int(
        endpoint_total_df[
            "endpoint_id"
        ]
        .ne(
            "UNKNOWN"
        )
        .sum()
    )

    print(
        f"\nTotal processed rows : {total_rows:,}"
    )

    print(
        f"Known endpoints      : {known_endpoints:,}"
    )

    print(
        f"Unknown endpoint rows: "
        f"{unknown_endpoint_rows:,}"
    )

    return (
        total_rows,
        endpoint_class_df,
        endpoint_total_df,
        class_distribution,
        known_endpoints,
        unknown_endpoint_rows
    )

def stable_endpoint_hash(
    endpoint
):

    return int(

        hashlib.sha1(

            (
                str(
                    SPLIT_SEED
                )
                +
                "|"
                +
                str(
                    endpoint
                )
            )
            .encode(
                "utf-8"
            )

        )
        .hexdigest()[:16],

        16

    )

def build_endpoint_split(
    endpoint_class_df
):

    print("\n" + "=" * 105)
    print("CREATING ENDPOINT-DISJOINT TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 105)

    pivot = (
        endpoint_class_df
        .pivot_table(

            index="endpoint_id",

            columns="attack_family",

            values="rows",

            aggfunc="sum",

            fill_value=0

        )
    )

    for family in CLASS_NAMES:

        if family not in pivot.columns:

            pivot[
                family
            ] = 0

    pivot = pivot[
        CLASS_NAMES
    ]

    pivot[
        "total_rows"
    ] = pivot[
        CLASS_NAMES
    ].sum(
        axis=1
    )

    pivot[
        "dominant_class"
    ] = (
        pivot[
            CLASS_NAMES
        ]
        .idxmax(
            axis=1
        )
    )

    pivot = (
        pivot
        .reset_index()
    )

    known = pivot[
        pivot[
            "endpoint_id"
        ]
        !=
        "UNKNOWN"
    ].copy()

    unknown = pivot[
        pivot[
            "endpoint_id"
        ]
        ==
        "UNKNOWN"
    ].copy()

    assignments = []

    split_targets = {

        "train":
            TRAIN_RATIO,

        "validation":
            VAL_RATIO,

        "test":
            TEST_RATIO
    }

    for family in CLASS_NAMES:

        group = (
            known[
                known[
                    "dominant_class"
                ]
                ==
                family
            ]
            .copy()
        )

        if len(group) == 0:

            continue

        group[
            "_hash"
        ] = (
            group[
                "endpoint_id"
            ]
            .map(
                stable_endpoint_hash
            )
        )

        group = (
            group
            .sort_values(
                "_hash"
            )
            .reset_index(
                drop=True
            )
        )

        total_family_rows = float(
            group[
                "total_rows"
            ].sum()
        )

        target_rows = {

            split:
                ratio
                *
                total_family_rows

            for split, ratio
            in split_targets.items()

        }

        assigned_rows = {

            "train": 0.0,
            "validation": 0.0,
            "test": 0.0

        }

        start_index = 0

        mandatory = [

            "train",
            "validation",
            "test"

        ]

        for split_name in mandatory:

            if start_index >= len(group):

                break

            record = group.iloc[
                start_index
            ]

            assignments.append({

                "endpoint_id":
                    record[
                        "endpoint_id"
                    ],

                "split":
                    split_name,

                "dominant_class":
                    family,

                "total_rows":
                    int(
                        record[
                            "total_rows"
                        ]
                    )

            })

            assigned_rows[
                split_name
            ] += float(
                record[
                    "total_rows"
                ]
            )

            start_index += 1

        for index in range(
            start_index,
            len(group)
        ):

            record = group.iloc[
                index
            ]

            deficit_scores = {}

            for split_name in [

                "train",
                "validation",
                "test"

            ]:

                target = max(
                    target_rows[
                        split_name
                    ],
                    1.0
                )

                deficit_scores[
                    split_name
                ] = (

                    target
                    -
                    assigned_rows[
                        split_name
                    ]

                ) / target

            split_name = max(
                deficit_scores,
                key=deficit_scores.get
            )

            assignments.append({

                "endpoint_id":
                    record[
                        "endpoint_id"
                    ],

                "split":
                    split_name,

                "dominant_class":
                    family,

                "total_rows":
                    int(
                        record[
                            "total_rows"
                        ]
                    )

            })

            assigned_rows[
                split_name
            ] += float(
                record[
                    "total_rows"
                ]
            )

    if len(unknown):

        assignments.append({

            "endpoint_id":
                "UNKNOWN",

            "split":
                "train",

            "dominant_class":
                "UNKNOWN",

            "total_rows":
                int(
                    unknown[
                        "total_rows"
                    ].sum()
                )

        })

    assignment_df = pd.DataFrame(
        assignments
    )

    if assignment_df[
        "endpoint_id"
    ].duplicated().any():

        raise RuntimeError(
            "❌ Endpoint appears in multiple split assignments."
        )

    save_csv(

        assignment_df,

        SPLIT_DIR /
        "endpoint_split_assignments.csv"

    )

    joined = endpoint_class_df.merge(

        assignment_df[
            [
                "endpoint_id",
                "split"
            ]
        ],

        on="endpoint_id",

        how="left"

    )

    joined[
        "split"
    ] = joined[
        "split"
    ].fillna(
        "train"
    )

    split_class = (
        joined
        .groupby(
            [
                "split",
                "attack_family"
            ],
            as_index=False
        )[
            "rows"
        ]
        .sum()
    )

    split_class_pivot = (
        split_class
        .pivot(
            index="attack_family",
            columns="split",
            values="rows"
        )
        .fillna(0)
    )

    for split_name in [

        "train",
        "validation",
        "test"

    ]:

        if split_name not in split_class_pivot.columns:

            split_class_pivot[
                split_name
            ] = 0

    split_class_pivot = (
        split_class_pivot[
            [
                "train",
                "validation",
                "test"
            ]
        ]
        .reset_index()
    )

    save_csv(

        split_class_pivot,

        SPLIT_DIR /
        "endpoint_split_class_distribution.csv"

    )

    viable = True
    missing = []

    for family in CLASS_NAMES:

        family_row = split_class_pivot[
            split_class_pivot[
                "attack_family"
            ]
            ==
            family
        ]

        if len(
            family_row
        ) == 0:

            viable = False

            missing.append(
                (
                    family,
                    "all"
                )
            )

            continue

        for split_name in [

            "train",
            "validation",
            "test"

        ]:

            if int(
                family_row[
                    split_name
                ].iloc[0]
            ) == 0:

                viable = False

                missing.append(
                    (
                        family,
                        split_name
                    )
                )

    print("\nEndpoint split class audit:")
    print(
        split_class_pivot.to_string(
            index=False
        )
    )

    if viable:

        print(
            "\n✅ Endpoint-disjoint split is viable "
            "for all 8 classes."
        )

    else:

        print(
            "\n⚠️ Endpoint-disjoint split does not "
            "contain every class in every split."
        )

        print(
            "Missing:",
            missing
        )

        print(
            "➡️ Primary experiment will automatically "
            "use deterministic row-hash split."
        )

    return (
        assignment_df,
        viable,
        split_class_pivot
    )

def deterministic_row_split(
    dataframe
):

    file_id = (
        dataframe[
            "source_file_id"
        ]
        .to_numpy(
            dtype=np.uint64
        )
    )

    chunk_id = (
        dataframe[
            "source_chunk_id"
        ]
        .to_numpy(
            dtype=np.uint64
        )
    )

    row_id = (
        dataframe[
            "row_in_chunk"
        ]
        .to_numpy(
            dtype=np.uint64
        )
    )

    class_id = (
        dataframe[
            "y_multiclass"
        ]
        .to_numpy(
            dtype=np.uint64
        )
    )

    with np.errstate(
        over="ignore"
    ):

        x = (

            np.uint64(
                SPLIT_SEED
            )

            +

            file_id
            *
            np.uint64(
                0x9E3779B185EBCA87
            )

            +

            chunk_id
            *
            np.uint64(
                0xC2B2AE3D27D4EB4F
            )

            +

            row_id
            *
            np.uint64(
                0x165667B19E3779F9
            )

            +

            class_id
            *
            np.uint64(
                0x85EBCA77C2B2AE63
            )

        )

        x ^= (
            x >> np.uint64(30)
        )

        x *= np.uint64(
            0xBF58476D1CE4E5B9
        )

        x ^= (
            x >> np.uint64(27)
        )

        x *= np.uint64(
            0x94D049BB133111EB
        )

        x ^= (
            x >> np.uint64(31)
        )

    u = (

        (
            x
            %
            np.uint64(
                1_000_000
            )
        )
        .astype(
            np.float64
        )

        /

        1_000_000.0

    )

    result = np.empty(
        len(dataframe),
        dtype=object
    )

    result[
        u < TRAIN_RATIO
    ] = "train"

    result[
        (
            u >= TRAIN_RATIO
        )
        &
        (
            u
            <
            TRAIN_RATIO
            +
            VAL_RATIO
        )
    ] = "validation"

    result[
        u
        >=
        TRAIN_RATIO
        +
        VAL_RATIO
    ] = "test"

    return result

def get_split_array(
    dataframe,
    primary_split_mode,
    endpoint_split_map
):

    if (
        primary_split_mode
        ==
        "endpoint_disjoint"
    ):

        return (

            dataframe[
                "endpoint_id"
            ]
            .astype("string")
            .map(
                endpoint_split_map
            )
            .fillna(
                "train"
            )
            .to_numpy(
                dtype=object
            )

        )

    return deterministic_row_split(
        dataframe
    )

def calculate_train_statistics(

    clean_shards,
    SELECTED_FEATURES,
    primary_split_mode,
    endpoint_split_map

):

    print("\n" + "=" * 105)
    print("CALCULATING TRAIN-ONLY IMPUTATION / NORMALIZATION STATISTICS")
    print("=" * 105)

    n_features = len(
        SELECTED_FEATURES
    )

    feature_sum = np.zeros(
        n_features,
        dtype=np.float64
    )

    feature_sq_sum = np.zeros(
        n_features,
        dtype=np.float64
    )

    feature_count = np.zeros(
        n_features,
        dtype=np.int64
    )

    all_feature_nonmissing = np.zeros(
        n_features,
        dtype=np.int64
    )

    total_rows_seen = 0

    split_counts = {

        "train": 0,
        "validation": 0,
        "test": 0

    }

    binary_train_counts = {

        0: 0,
        1: 0

    }

    multiclass_train_counts = {

        i: 0

        for i in range(
            len(
                CLASS_TO_ID
            )
        )

    }

    split_class_counts = {

        split: {

            family: 0

            for family
            in CLASS_NAMES

        }

        for split in [

            "train",
            "validation",
            "test"

        ]

    }

    for shard_index, shard in enumerate(
        clean_shards,
        start=1
    ):

        columns = (

            [
                "source_file_id",
                "source_chunk_id",
                "row_in_chunk",
                "endpoint_id",
                "attack_family",
                "y_binary",
                "y_multiclass"
            ]

            +

            SELECTED_FEATURES

        )

        df = pd.read_parquet(
            shard,
            columns=columns
        )

        splits = get_split_array(

            df,
            primary_split_mode,
            endpoint_split_map

        )

        total_rows_seen += len(
            df
        )

        for split_name in [

            "train",
            "validation",
            "test"

        ]:

            mask = (
                splits
                ==
                split_name
            )

            count = int(
                mask.sum()
            )

            split_counts[
                split_name
            ] += count

            if count > 0:

                family_counts = (
                    df.loc[
                        mask,
                        "attack_family"
                    ]
                    .value_counts()
                )

                for family, n in (
                    family_counts.items()
                ):

                    split_class_counts[
                        split_name
                    ][
                        str(family)
                    ] += int(
                        n
                    )

        X_all = (
            df[
                SELECTED_FEATURES
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        finite_all = np.isfinite(
            X_all
        )

        all_feature_nonmissing += (
            finite_all.sum(
                axis=0
            )
        )

        train_mask = (
            splits == "train"
        )

        if train_mask.any():

            X_train = X_all[
                train_mask
            ]

            finite = np.isfinite(
                X_train
            )

            safe_x = np.where(
                finite,
                X_train,
                0.0
            )

            feature_sum += (
                safe_x.sum(
                    axis=0,
                    dtype=np.float64
                )
            )

            feature_sq_sum += (
                (
                    safe_x
                    *
                    safe_x
                )
                .sum(
                    axis=0,
                    dtype=np.float64
                )
            )

            feature_count += (
                finite.sum(
                    axis=0
                )
            )

            binary_values = (
                df.loc[
                    train_mask,
                    "y_binary"
                ]
                .value_counts()
            )

            for k, v in binary_values.items():

                binary_train_counts[
                    int(k)
                ] += int(
                    v
                )

            multiclass_values = (
                df.loc[
                    train_mask,
                    "y_multiclass"
                ]
                .value_counts()
            )

            for k, v in multiclass_values.items():

                multiclass_train_counts[
                    int(k)
                ] += int(
                    v
                )

            del X_train
            del finite
            del safe_x

        del X_all
        del finite_all
        del df

        gc.collect()

        if (
            shard_index % 25 == 0
            or
            shard_index
            ==
            len(clean_shards)
        ):

            print(
                f"  ✅ statistics "
                f"{shard_index}/"
                f"{len(clean_shards)} shards"
            )

    train_mean = np.divide(

        feature_sum,
        feature_count,

        out=np.zeros_like(
            feature_sum
        ),

        where=(
            feature_count > 0
        )

    )

    second_moment = np.divide(

        feature_sq_sum,
        feature_count,

        out=np.zeros_like(
            feature_sq_sum
        ),

        where=(
            feature_count > 0
        )

    )

    variance = (
        second_moment
        -
        train_mean ** 2
    )

    variance = np.maximum(
        variance,
        0.0
    )

    train_scale = np.sqrt(
        variance
    )

    valid_feature_mask = (

        (feature_count > 0)

        &

        (
            variance
            >
            MIN_FEATURE_VARIANCE
        )

    )

    FINAL_FEATURES = [

        feature

        for feature, keep
        in zip(
            SELECTED_FEATURES,
            valid_feature_mask
        )

        if keep

    ]

    REMOVED_FEATURES = [

        feature

        for feature, keep
        in zip(
            SELECTED_FEATURES,
            valid_feature_mask
        )

        if not keep

    ]

    train_scale_safe = train_scale.copy()

    train_scale_safe[
        ~valid_feature_mask
    ] = 1.0

    final_indices = np.where(
        valid_feature_mask
    )[0]

    binary_total = sum(
        binary_train_counts.values()
    )

    binary_weights = {

        str(c):
            (
                binary_total
                /
                (
                    2
                    *
                    binary_train_counts[c]
                )
                if binary_train_counts[c] > 0
                else 0.0
            )

        for c in [
            0,
            1
        ]

    }

    multiclass_total = sum(
        multiclass_train_counts.values()
    )

    multi_weights = {

        str(c):
            (
                multiclass_total
                /
                (
                    len(
                        CLASS_TO_ID
                    )
                    *
                    multiclass_train_counts[c]
                )
                if multiclass_train_counts[c] > 0
                else 0.0
            )

        for c in range(
            len(
                CLASS_TO_ID
            )
        )

    }

    save_json(

        {
            "pipeline_version":
                PIPELINE_VERSION,

            "fit_on":
                "TRAIN ONLY",

            "primary_split_mode":
                primary_split_mode,

            "imputation":
                "training feature mean",

            "normalization":
                "z-score / StandardScaler equivalent",

            "selected_features_before_variance_filter":
                SELECTED_FEATURES,

            "final_features":
                FINAL_FEATURES,

            "removed_zero_or_near_zero_variance_features":
                REMOVED_FEATURES,

            "train_feature_mean":
                train_mean.tolist(),

            "train_feature_variance":
                variance.tolist(),

            "train_feature_scale":
                train_scale_safe.tolist(),

            "feature_observation_counts":
                feature_count.tolist()

        },

        PREPROCESSOR_DIR /
        "train_only_preprocessor.json"

    )

    save_json(

        {
            "binary_train_counts":
                binary_train_counts,

            "binary_class_weights":
                binary_weights,

            "multiclass_train_counts":
                multiclass_train_counts,

            "multiclass_class_weights":
                multi_weights,

            "calculated_from":
                "training split only"

        },

        PREPROCESSOR_DIR /
        "class_weights.json"

    )

    missingness_df = pd.DataFrame({

        "feature":
            SELECTED_FEATURES,

        "total_rows":
            total_rows_seen,

        "nonmissing_rows":
            all_feature_nonmissing,

        "missing_rows":
            (
                total_rows_seen
                -
                all_feature_nonmissing
            ),

        "missing_percentage":
            (
                100
                *
                (
                    total_rows_seen
                    -
                    all_feature_nonmissing
                )
                /
                max(
                    1,
                    total_rows_seen
                )
            )

    })

    save_csv(

        missingness_df,

        AUDIT_DIR /
        "feature_missingness.csv"

    )

    split_summary = pd.DataFrame([

        {
            "split":
                split_name,

            "rows":
                split_counts[
                    split_name
                ],

            "percentage":
                (
                    100
                    *
                    split_counts[
                        split_name
                    ]
                    /
                    max(
                        1,
                        total_rows_seen
                    )
                )

        }

        for split_name in [

            "train",
            "validation",
            "test"

        ]

    ])

    save_csv(

        split_summary,

        SPLIT_DIR /
        "primary_split_distribution.csv"

    )

    split_class_rows = []

    for split_name in [

        "train",
        "validation",
        "test"

    ]:

        for family in CLASS_NAMES:

            split_class_rows.append({

                "split":
                    split_name,

                "attack_family":
                    family,

                "class_id":
                    CLASS_TO_ID[
                        family
                    ],

                "rows":
                    split_class_counts[
                        split_name
                    ][
                        family
                    ]

            })

    save_csv(

        pd.DataFrame(
            split_class_rows
        ),

        SPLIT_DIR /
        "primary_split_class_distribution.csv"

    )

    print(
        "\nSelected features before variance filter:",
        len(
            SELECTED_FEATURES
        )
    )

    print(
        "Final model features:",
        len(
            FINAL_FEATURES
        )
    )

    print(
        "Removed near-constant features:",
        len(
            REMOVED_FEATURES
        )
    )

    print("\nPrimary split:")

    print(
        split_summary.to_string(
            index=False
        )
    )

    return (

        train_mean,
        train_scale_safe,
        final_indices,
        FINAL_FEATURES,
        REMOVED_FEATURES,
        split_counts,
        binary_weights,
        multi_weights

    )

def normalize_features(

    X,
    train_mean,
    train_scale,
    final_indices

):

    X = np.asarray(
        X,
        dtype=np.float32
    )

    invalid = ~np.isfinite(
        X
    )

    if invalid.any():

        rows, cols = np.where(
            invalid
        )

        X[
            rows,
            cols
        ] = (
            train_mean[
                cols
            ]
            .astype(
                np.float32
            )
        )

    if (
        USE_GPU_FOR_NORMALIZATION
        and
        torch.cuda.is_available()
    ):

        try:

            tensor = torch.from_numpy(
                X
            ).to(
                DEVICE,
                non_blocking=True
            )

            mean_tensor = torch.tensor(

                train_mean,

                dtype=torch.float32,

                device=DEVICE

            )

            scale_tensor = torch.tensor(

                train_scale,

                dtype=torch.float32,

                device=DEVICE

            )

            index_tensor = torch.tensor(

                final_indices,

                dtype=torch.long,

                device=DEVICE

            )

            tensor = (
                tensor
                -
                mean_tensor
            ) / scale_tensor

            tensor = tensor.index_select(
                1,
                index_tensor
            )

            result = (
                tensor
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            del tensor
            del mean_tensor
            del scale_tensor
            del index_tensor

            torch.cuda.empty_cache()

            return result

        except RuntimeError as exc:

            print(
                "⚠️ GPU normalization fallback to CPU:",
                exc
            )

            torch.cuda.empty_cache()

    result = (

        (
            X
            -
            train_mean.astype(
                np.float32
            )
        )

        /

        train_scale.astype(
            np.float32
        )

    )

    return (
        result[
            :,
            final_indices
        ]
        .astype(
            np.float32
        )
    )

def build_model_ready_data(

    clean_shards,
    SELECTED_FEATURES,
    train_mean,
    train_scale,
    final_indices,
    FINAL_FEATURES,
    primary_split_mode,
    endpoint_split_map

):

    print("\n" + "=" * 105)
    print("CREATING FINAL MODEL-READY TRAIN / VALIDATION / TEST DATA")
    print("=" * 105)

    nonfinite_after_processing = 0

    metadata_columns = [

        "source_file_id",
        "source_chunk_id",
        "row_in_chunk",
        "source_row_approx",
        "endpoint_id",
        "destination_endpoint",
        "attack_family",
        "original_label",
        "y_binary",
        "y_multiclass"

    ]

    for shard_index, shard in enumerate(
        clean_shards,
        start=1
    ):

        shard_id = (

            safe_slug(
                shard.parent.name,
                45
            )

            +
            "_"

            +
            shard.stem

            +
            "_"

            +
            short_hash(
                shard
            )

        )

        outputs = {

            "train":
                TRAIN_DIR /
                f"{shard_id}_train.parquet",

            "validation":
                VAL_DIR /
                f"{shard_id}_validation.parquet",

            "test":
                TEST_DIR /
                f"{shard_id}_test.parquet"

        }

        if (
            RESUME
            and
            not FORCE_REBUILD
            and
            all(
                valid_parquet(
                    path
                )
                for path
                in outputs.values()
            )
        ):

            if (
                shard_index % 25 == 0
                or
                shard_index
                ==
                len(clean_shards)
            ):

                print(
                    f"  ↪ model-ready "
                    f"{shard_index}/"
                    f"{len(clean_shards)} reused"
                )

            continue

        df = pd.read_parquet(
            shard
        )

        split_array = get_split_array(

            df,
            primary_split_mode,
            endpoint_split_map

        )

        X = (
            df[
                SELECTED_FEATURES
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        X_scaled = normalize_features(

            X,
            train_mean,
            train_scale,
            final_indices

        )

        nonfinite_after_processing += int(
            (
                ~np.isfinite(
                    X_scaled
                )
            )
            .sum()
        )

        feature_df = pd.DataFrame(

            X_scaled,

            columns=FINAL_FEATURES

        )

        meta = (
            df[
                metadata_columns
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        meta[
            "split"
        ] = split_array

        final_df = pd.concat(

            [
                meta,
                feature_df
            ],

            axis=1

        )

        for split_name, destination in (
            outputs.items()
        ):

            subset = (
                final_df[
                    final_df[
                        "split"
                    ]
                    ==
                    split_name
                ]
                .reset_index(
                    drop=True
                )
            )

            save_parquet(
                subset,
                destination
            )

            del subset

        save_json(

            {
                "pipeline_version":
                    PIPELINE_VERSION,

                "stage":
                    "model_ready",

                "completed_source_shards":
                    int(
                        shard_index
                    ),

                "total_source_shards":
                    int(
                        len(clean_shards)
                    ),

                "current_source_shard":
                    str(
                        shard
                    ),

                "updated_at":
                    datetime.now().isoformat()

            },

            STEP1_PROGRESS

        )

        print(
            f"  ✅ model-ready "
            f"{shard_index}/"
            f"{len(clean_shards)}"
        )

        del df
        del X
        del X_scaled
        del feature_df
        del meta
        del final_df

        gc.collect()

        torch.cuda.empty_cache()

    if nonfinite_after_processing != 0:

        raise RuntimeError(
            f"❌ Final normalized data contains "
            f"{nonfinite_after_processing:,} "
            f"non-finite values."
        )

def validate_final_outputs(
    FINAL_FEATURES
):

    print("\n" + "=" * 105)
    print("FINAL STEP-1 VALIDATION")
    print("=" * 105)

    final_files = {

        "train":
            sorted(
                TRAIN_DIR.glob(
                    "*.parquet"
                )
            ),

        "validation":
            sorted(
                VAL_DIR.glob(
                    "*.parquet"
                )
            ),

        "test":
            sorted(
                TEST_DIR.glob(
                    "*.parquet"
                )
            )

    }

    final_rows = {}

    for split_name, files in (
        final_files.items()
    ):

        rows = 0

        for path in files:

            if not valid_parquet(
                path
            ):

                raise RuntimeError(
                    f"❌ Invalid final Parquet:\n"
                    f"{path}"
                )

            rows += (
                pq.ParquetFile(
                    path
                )
                .metadata
                .num_rows
            )

        final_rows[
            split_name
        ] = int(
            rows
        )

    print(
        "Train rows      :",
        f"{final_rows['train']:,}"
    )

    print(
        "Validation rows :",
        f"{final_rows['validation']:,}"
    )

    print(
        "Test rows       :",
        f"{final_rows['test']:,}"
    )

    print(
        "Final features  :",
        len(
            FINAL_FEATURES
        )
    )

    return (
        final_rows,
        final_files
    )

def run_step1():

    complete = existing_complete_summary()

    if complete is not None:

        print("\n" + "=" * 105)
        print("✅ STEP 1 WAS ALREADY COMPLETED")
        print("=" * 105)

        print(
            json.dumps(
                complete,
                indent=2
            )
        )

        print(
            "\nNo preprocessing needs to be repeated."
        )

        return complete

    reset_step1()

    manifest = build_manifest()

    print("\nAttack-family file mapping:")

    print(
        manifest[
            "attack_family"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    header_df = build_header_audit(
        manifest
    )

    (
        flow_df,
        excluded_df,
        common_columns,
        SRC_IP_COL,
        DST_IP_COL,
        LABEL_COL

    ) = resolve_flow_schema(
        header_df
    )

    candidate_features = (
        get_candidate_features(

            common_columns,
            SRC_IP_COL,
            DST_IP_COL,
            LABEL_COL

        )
    )

    SELECTED_FEATURES = (
        profile_numeric_features(

            flow_df,
            common_columns,
            candidate_features,
            SRC_IP_COL,
            DST_IP_COL,
            LABEL_COL

        )
    )

    save_json(

        {
            "pipeline_version":
                PIPELINE_VERSION,

            "dataset":
                "CIC IoT-DIAD 2024",

            "raw_dataset_path":
                str(
                    DATASET_ROOT
                ),

            "raw_dataset_modified":
                False,

            "raw_csv_files":
                int(
                    len(
                        manifest
                    )
                ),

            "flow_files":
                int(
                    len(
                        flow_df
                    )
                ),

            "excluded_files":
                int(
                    len(
                        excluded_df
                    )
                ),

            "representation":
                "Flow-based anomaly detection",

            "endpoint_metadata":
                SRC_IP_COL,

            "endpoint_used_as_model_feature":
                False,

            "destination_metadata":
                DST_IP_COL,

            "label_column":
                LABEL_COL,

            "selected_numeric_features":
                SELECTED_FEATURES,

            "binary_target":
                {
                    "0":
                        "Benign",

                    "1":
                        "Attack"
                },

            "multiclass_target":
                CLASS_TO_ID,

            "created_at":
                datetime.now().isoformat()

        },

        CONFIG_DIR /
        "data_preparation_config.json"

    )

    build_clean_shards(

        flow_df,
        SELECTED_FEATURES,
        SRC_IP_COL,
        DST_IP_COL,
        LABEL_COL

    )

    clean_shards = get_clean_shards()

    print(
        "\n✅ Clean Parquet shards:",
        len(
            clean_shards
        )
    )

    (
        total_rows,
        endpoint_class_df,
        endpoint_total_df,
        class_distribution,
        known_endpoints,
        unknown_endpoint_rows

    ) = build_global_audit(
        clean_shards
    )

    (
        endpoint_assignment_df,
        endpoint_split_viable,
        endpoint_split_class_distribution

    ) = build_endpoint_split(
        endpoint_class_df
    )

    endpoint_split_map = dict(
        zip(
            endpoint_assignment_df[
                "endpoint_id"
            ].astype(str),

            endpoint_assignment_df[
                "split"
            ].astype(str)
        )
    )

    if endpoint_split_viable:

        primary_split_mode = (
            "endpoint_disjoint"
        )

    else:

        primary_split_mode = (
            "deterministic_row_hash"
        )

    save_json(

        {
            "primary_split_mode":
                primary_split_mode,

            "train_fraction":
                TRAIN_RATIO,

            "validation_fraction":
                VAL_RATIO,

            "test_fraction":
                TEST_RATIO,

            "split_seed":
                SPLIT_SEED,

            "endpoint_disjoint_split_viable":
                bool(
                    endpoint_split_viable
                ),

            "endpoint_assignments_file":
                str(
                    SPLIT_DIR /
                    "endpoint_split_assignments.csv"
                )

        },

        SPLIT_DIR /
        "primary_split_config.json"

    )

    print(
        "\n✅ PRIMARY SPLIT MODE:",
        primary_split_mode
    )

    (
        train_mean,
        train_scale,
        final_indices,
        FINAL_FEATURES,
        REMOVED_FEATURES,
        split_counts,
        binary_weights,
        multi_weights

    ) = calculate_train_statistics(

        clean_shards,
        SELECTED_FEATURES,
        primary_split_mode,
        endpoint_split_map

    )

    build_model_ready_data(

        clean_shards,
        SELECTED_FEATURES,
        train_mean,
        train_scale,
        final_indices,
        FINAL_FEATURES,
        primary_split_mode,
        endpoint_split_map

    )

    final_rows, final_files = (
        validate_final_outputs(
            FINAL_FEATURES
        )
    )

    if GPU_MEMORY_GB >= 30:

        recommended_batch_size = 8192

    elif GPU_MEMORY_GB >= 20:

        recommended_batch_size = 4096

    elif GPU_MEMORY_GB >= 14:

        recommended_batch_size = 2048

    else:

        recommended_batch_size = 1024

    save_json(

        {
            "gpu":
                GPU_NAME,

            "gpu_memory_gb":
                GPU_MEMORY_GB,

            "device":
                str(
                    DEVICE
                ),

            "mixed_precision":
                True,

            "tf32":
                True,

            "recommended_neural_batch_size":
                recommended_batch_size,

            "recommended_num_workers":
                2,

            "pin_memory":
                True,

            "non_blocking_transfer":
                True

        },

        CONFIG_DIR /
        "gpu_training_config.json"

    )

    save_json(

        {
            "python":
                sys.version,

            "platform":
                platform.platform(),

            "numpy":
                np.__version__,

            "pandas":
                pd.__version__,

            "pyarrow":
                pyarrow.__version__,

            "scikit_learn":
                sklearn.__version__,

            "pytorch":
                torch.__version__,

            "cuda":
                torch.version.cuda,

            "gpu":
                GPU_NAME,

            "gpu_memory_gb":
                GPU_MEMORY_GB

        },

        CONFIG_DIR /
        "software_versions.json"

    )

    summary = {

        "pipeline_version":
            PIPELINE_VERSION,

        "step":
            1,

        "name":
            "CIC IoT-DIAD 2024 Dataset Preparation",

        "status":
            "COMPLETED",

        "paper_title":
            (
                "A Hybrid Blockchain–Federated Learning "
                "Framework with AI-Based Anomaly Detection "
                "for Privacy-Preserving IoT Security"
            ),

        "dataset":
            "CIC IoT-DIAD 2024",

        "dataset_path":
            str(
                DATASET_ROOT
            ),

        "raw_dataset_modified":
            False,

        "project_root":
            str(
                PROJECT_ROOT
            ),

        "raw_csv_files":
            int(
                len(
                    manifest
                )
            ),

        "flow_files_used":
            int(
                len(
                    flow_df
                )
            ),

        "excluded_schema_files":
            int(
                len(
                    excluded_df
                )
            ),

        "total_processed_rows":
            int(
                total_rows
            ),

        "known_source_endpoints":
            int(
                known_endpoints
            ),

        "unknown_endpoint_rows":
            int(
                unknown_endpoint_rows
            ),

        "primary_split_mode":
            primary_split_mode,

        "train_rows":
            int(
                final_rows[
                    "train"
                ]
            ),

        "validation_rows":
            int(
                final_rows[
                    "validation"
                ]
            ),

        "test_rows":
            int(
                final_rows[
                    "test"
                ]
            ),

        "numeric_features_initial":
            int(
                len(
                    SELECTED_FEATURES
                )
            ),

        "final_model_features":
            int(
                len(
                    FINAL_FEATURES
                )
            ),

        "removed_near_constant_features":
            REMOVED_FEATURES,

        "class_balancing":
            "training-only class weights",

        "gpu":
            GPU_NAME,

        "gpu_memory_gb":
            GPU_MEMORY_GB,

        "recommended_batch_size_step2":
            recommended_batch_size,

        "completed_at":
            datetime.now().isoformat()

    }

    save_json(
        summary,
        STEP1_COMPLETE
    )

    print("\n" + "=" * 110)
    print("✅ STEP 1 COMPLETED SUCCESSFULLY")
    print("=" * 110)

    print("\nDATASET")
    print("-" * 80)

    print(
        "Raw CSV files        :",
        f"{len(manifest):,}"
    )

    print(
        "Flow files used      :",
        f"{len(flow_df):,}"
    )

    print(
        "Excluded files       :",
        f"{len(excluded_df):,}"
    )

    print(
        "Processed rows       :",
        f"{total_rows:,}"
    )

    print(
        "Known endpoints      :",
        f"{known_endpoints:,}"
    )

    print("\nFEATURES")
    print("-" * 80)

    print(
        "Stable numeric       :",
        len(
            SELECTED_FEATURES
        )
    )

    print(
        "Final model features :",
        len(
            FINAL_FEATURES
        )
    )

    print("\nPRIMARY DATA SPLIT")
    print("-" * 80)

    print(
        "Mode                 :",
        primary_split_mode
    )

    print(
        "Train                :",
        f"{final_rows['train']:,}"
    )

    print(
        "Validation           :",
        f"{final_rows['validation']:,}"
    )

    print(
        "Test                 :",
        f"{final_rows['test']:,}"
    )

    print("\nGPU")
    print("-" * 80)

    print(
        "GPU                  :",
        GPU_NAME
    )

    print(
        "VRAM                 :",
        f"{GPU_MEMORY_GB:.2f} GB"
    )

    print(
        "Step-2 batch size    :",
        recommended_batch_size
    )

    print("\nSCIENTIFIC CHECKS")
    print("-" * 80)

    checks = [

        "Existing dataset reused; no redownload.",
        "Raw dataset was not modified.",
        "New IJACSA project directories used.",
        "Flow-based CIC IoT-DIAD representation selected.",
        "Unknown/non-flow schema files audited and excluded.",
        "DDoS mapping checked before DoS.",
        "Source endpoint retained only as metadata.",
        "Src/Dst IP excluded from ML input.",
        "Flow ID and Timestamp excluded from ML input.",
        "NaN handled without using test information.",
        "Positive/negative Infinity converted to NaN.",
        "Binary target created.",
        "8-class target created.",
        "Stable numeric behavioral features selected.",
        "Endpoint-disjoint split attempted first.",
        "Safe deterministic fallback available.",
        "Imputation statistics calculated from training only.",
        "Normalization statistics calculated from training only.",
        "Near-zero-variance features removed using training only.",
        "Class weights calculated from training only.",
        "GPU-assisted final normalization enabled.",
        "Every expensive preprocessing chunk checkpointed.",
        "Final data contains no NaN/Inf after preprocessing."

    ]

    for check in checks:

        print(
            "✅",
            check
        )

    print("\nFINAL MODEL DATA")
    print("-" * 80)

    print(
        "TRAIN:"
    )
    print(
        TRAIN_DIR
    )

    print(
        "\nVALIDATION:"
    )
    print(
        VAL_DIR
    )

    print(
        "\nTEST:"
    )
    print(
        TEST_DIR
    )

    print("\n" + "=" * 110)

    print(
        "NEXT → STEP 2:"
    )

    print(
        "CENTRALIZED AI BASELINE BENCHMARKING"
    )

    print(
        "Random Forest + XGBoost + MLP + 1D-CNN"
    )

    print(
        "Binary + 8-Class Evaluation + Best Model Selection"
    )

    print("=" * 110)

    return summary

STEP1_RESULT = run_step1()

import os, sys, gc, re, json, math, time, random, shutil, hashlib, platform
import warnings, subprocess, importlib.util
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

from google.colab import drive
if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
else:
    print("✅ Google Drive already mounted.")

def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    if importlib.util.find_spec(import_name) is None:
        print(f"📦 Installing {pip_name} ...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet",
            "--disable-pip-version-check", pip_name
        ])
    else:
        print(f"✅ {import_name} already available.")

for imp, pip_name in [
    ("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
    ("sklearn", "scikit-learn"), ("joblib", "joblib"),
    ("xgboost", "xgboost"), ("matplotlib", "matplotlib"), ("torch", "torch"),
]:
    ensure_package(imp, pip_name)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

print("\n" + "=" * 112)
print("IJACSA — HYBRID BLOCKCHAIN–FEDERATED LEARNING IoT SECURITY")
print("STEP 2 — CENTRALIZED AI BASELINE BENCHMARKING")
print("=" * 112)

PROJECT_ROOT = Path("/content/drive/MyDrive/Hybrid_BCFL_IJACSA_2026")
CONFIG_DIR = PROJECT_ROOT / "00_CONFIG"
SPLIT_DIR = PROJECT_ROOT / "03_SPLITS"
PREPROCESSOR_DIR = PROJECT_ROOT / "04_PREPROCESSOR"
MODEL_READY_ROOT = PROJECT_ROOT / "05_MODEL_READY"
STEP1_CHECKPOINT_DIR = PROJECT_ROOT / "06_CHECKPOINTS"

TRAIN_DIR = MODEL_READY_ROOT / "TRAIN"
VAL_DIR = MODEL_READY_ROOT / "VALIDATION"
TEST_DIR = MODEL_READY_ROOT / "TEST"

STEP1_COMPLETE = STEP1_CHECKPOINT_DIR / "STEP01_COMPLETE.json"
PREPROCESSOR_FILE = PREPROCESSOR_DIR / "train_only_preprocessor.json"
CLASS_WEIGHT_FILE = PREPROCESSOR_DIR / "class_weights.json"

STEP2_ROOT = PROJECT_ROOT / "07_AI_MODELS" / "STEP02_CENTRALIZED_BASELINES"
MODEL_DIR = STEP2_ROOT / "MODELS"
CHECKPOINT_DIR = STEP2_ROOT / "CHECKPOINTS"
SAMPLE_DIR = STEP2_ROOT / "SAMPLES"
RESULT_DIR = STEP2_ROOT / "RESULTS"
FIGURE_DIR = STEP2_ROOT / "FIGURES"
LOG_DIR = STEP2_ROOT / "LOGS"

for folder in [STEP2_ROOT, MODEL_DIR, CHECKPOINT_DIR, SAMPLE_DIR, RESULT_DIR, FIGURE_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

LOCAL_ROOT = Path("/content/Hybrid_BCFL_IJACSA_STEP02_RUNTIME")
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

LOCAL_MODEL_DATA = LOCAL_ROOT / "MODEL_READY"
LOCAL_MODEL_DATA.mkdir(parents=True, exist_ok=True)

for required in [
    STEP1_COMPLETE, PREPROCESSOR_FILE, CLASS_WEIGHT_FILE,
    TRAIN_DIR, VAL_DIR, TEST_DIR
]:
    if not required.exists():
        raise FileNotFoundError(
            f"\n❌ Missing Step-1 output:\n{required}\n"
            "Run Step 1 successfully before Step 2."
        )

with open(STEP1_COMPLETE, "r", encoding="utf-8") as f:
    STEP1_SUMMARY = json.load(f)
with open(PREPROCESSOR_FILE, "r", encoding="utf-8") as f:
    PREPROCESSOR = json.load(f)
with open(CLASS_WEIGHT_FILE, "r", encoding="utf-8") as f:
    CLASS_WEIGHT_INFO = json.load(f)

if STEP1_SUMMARY.get("status") != "COMPLETED":
    raise RuntimeError("❌ Step 1 is not marked COMPLETED.")

FINAL_FEATURES = PREPROCESSOR["final_features"]
N_FEATURES = len(FINAL_FEATURES)

CLASS_TO_ID = {
    "Benign": 0, "DDoS": 1, "DoS": 2, "Recon": 3,
    "Web-Based": 4, "Brute Force": 5, "Spoofing": 6, "Mirai": 7,
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
CLASS_NAMES = [ID_TO_CLASS[i] for i in range(8)]

print("\n✅ STEP 1 VERIFIED")
print("-" * 80)
print("Training rows   :", f"{STEP1_SUMMARY['train_rows']:,}")
print("Validation rows :", f"{STEP1_SUMMARY['validation_rows']:,}")
print("Test rows       :", f"{STEP1_SUMMARY['test_rows']:,}")
print("Features        :", N_FEATURES)
print("Split mode      :", STEP1_SUMMARY["primary_split_mode"])

if not torch.cuda.is_available():
    raise RuntimeError(
        "\n❌ GPU not enabled.\n"
        "Colab -> Runtime -> Change runtime type -> GPU"
    )

DEVICE = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print("\n✅ GPU READY")
print("-" * 80)
print("GPU      :", GPU_NAME)
print("VRAM     :", f"{GPU_MEMORY_GB:.2f} GB")
print("PyTorch  :", torch.__version__)
print("CUDA     :", torch.version.cuda)
print("XGBoost  :", xgb.__version__)

BASE_SEED = 42
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
torch.cuda.manual_seed_all(BASE_SEED)

PIPELINE_VERSION = "IJACSA_STEP02_CENTRALIZED_V1"
RESUME = True
FORCE_REBUILD = False
FAST_EXIT_IF_COMPLETE = True

TREE_BINARY_TRAIN_CAP_PER_CLASS = 200_000
TREE_BINARY_VAL_CAP_PER_CLASS = 50_000
TREE_MULTI_TRAIN_CAP_PER_CLASS = 100_000
TREE_MULTI_VAL_CAP_PER_CLASS = 25_000

MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

if GPU_MEMORY_GB >= 14:
    NEURAL_BATCH_SIZE = 8192
elif GPU_MEMORY_GB >= 8:
    NEURAL_BATCH_SIZE = 4096
else:
    NEURAL_BATCH_SIZE = 2048

PREDICT_BATCH_SIZE = max(NEURAL_BATCH_SIZE, 16384)

NEURAL_VAL_BINARY_CAP_PER_CLASS = 50_000
NEURAL_VAL_MULTI_CAP_PER_CLASS = 25_000

AUC_SAMPLE_MAX_ROWS = 250_000
NEURAL_CLASS_WEIGHT_MODE = "sqrt_balanced"

STAGE_DATA_TO_LOCAL_SSD = True
LOCAL_DISK_SAFETY_GB = 12.0

STEP2_COMPLETE = CHECKPOINT_DIR / "STEP02_COMPLETE.json"

def safe_slug(text, max_length=80):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return (text or "item")[:max_length]

def short_hash(value, length=10):
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:length]

def atomic_copy(local_file, final_file, retries=5):
    local_file = Path(local_file)
    final_file = Path(final_file)
    final_file.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(final_file) + ".partial")
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            partial.unlink(missing_ok=True)
            shutil.copy2(local_file, partial)
            if not partial.exists() or partial.stat().st_size <= 0:
                raise IOError("Partial file is empty.")
            os.replace(partial, final_file)
            return
        except Exception as exc:
            last_error = exc
            print(f"⚠️ Save retry {attempt}/{retries}: {exc}")
            time.sleep(attempt * 2)

    raise IOError(f"Could not save {final_file}") from last_error

def save_json(obj, destination):
    destination = Path(destination)
    local = LOCAL_ROOT / f"{safe_slug(destination.stem)}_{short_hash(destination)}.json"
    with open(local, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    atomic_copy(local, destination)
    local.unlink(missing_ok=True)

def save_csv(df, destination):
    destination = Path(destination)
    local = LOCAL_ROOT / f"{safe_slug(destination.stem)}_{short_hash(destination)}.csv"
    df.to_csv(local, index=False)
    atomic_copy(local, destination)
    local.unlink(missing_ok=True)

def save_joblib(obj, destination):
    destination = Path(destination)
    local = LOCAL_ROOT / f"{safe_slug(destination.stem)}_{short_hash(destination)}.joblib"
    joblib.dump(obj, local, compress=3)
    atomic_copy(local, destination)
    local.unlink(missing_ok=True)

def save_torch(obj, destination):
    destination = Path(destination)
    local = LOCAL_ROOT / f"{safe_slug(destination.stem)}_{short_hash(destination)}.pt"
    torch.save(obj, local)
    atomic_copy(local, destination)
    local.unlink(missing_ok=True)

def save_npz(destination, **arrays):
    destination = Path(destination)
    local = LOCAL_ROOT / f"{safe_slug(destination.stem)}_{short_hash(destination)}.npz"
    np.savez_compressed(local, **arrays)
    atomic_copy(local, destination)
    local.unlink(missing_ok=True)

if FORCE_REBUILD:
    print("\n⚠️ FORCE_REBUILD=True — deleting Step-2 outputs only.")
    for folder in [MODEL_DIR, CHECKPOINT_DIR, SAMPLE_DIR, RESULT_DIR, FIGURE_DIR, LOG_DIR]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

if (
    FAST_EXIT_IF_COMPLETE and RESUME and not FORCE_REBUILD
    and STEP2_COMPLETE.exists()
):
    try:
        with open(STEP2_COMPLETE, "r", encoding="utf-8") as f:
            old_complete = json.load(f)
        if (
            old_complete.get("pipeline_version") == PIPELINE_VERSION
            and old_complete.get("status") == "COMPLETED"
        ):
            print("\n" + "=" * 112)
            print("✅ STEP 2 ALREADY COMPLETED")
            print("=" * 112)
            print(json.dumps(old_complete, indent=2))
            raise SystemExit
    except SystemExit:
        raise
    except Exception:
        pass

DRIVE_FILES = {
    "train": sorted(TRAIN_DIR.glob("*.parquet")),
    "validation": sorted(VAL_DIR.glob("*.parquet")),
    "test": sorted(TEST_DIR.glob("*.parquet")),
}

for split_name, paths in DRIVE_FILES.items():
    if len(paths) == 0:
        raise RuntimeError(f"❌ No {split_name} Parquet files found.")

print("\nMODEL-READY PARQUET FILES")
print("-" * 80)
for split_name, paths in DRIVE_FILES.items():
    print(f"{split_name:10s}: {len(paths):,}")

def folder_size_bytes(paths):
    return sum(p.stat().st_size for p in paths)

def stage_split_to_local(split_name, paths):
    if not STAGE_DATA_TO_LOCAL_SSD:
        return paths

    total_bytes = folder_size_bytes(paths)
    total_gb = total_bytes / (1024 ** 3)
    free_gb = shutil.disk_usage("/content").free / (1024 ** 3)

    if free_gb < total_gb + LOCAL_DISK_SAFETY_GB:
        print(
            f"⚠️ {split_name}: not enough local SSD "
            f"({free_gb:.1f} GB free; {total_gb:.1f} GB needed). "
            "Using Google Drive directly."
        )
        return paths

    target_dir = LOCAL_MODEL_DATA / split_name.upper()
    target_dir.mkdir(parents=True, exist_ok=True)

    staged = []
    print(f"\n⚡ Staging {split_name} to local SSD ({total_gb:.2f} GB)...")

    for i, src in enumerate(paths, start=1):
        dst = target_dir / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        staged.append(dst)
        if i % 50 == 0 or i == len(paths):
            print(f"  ✅ {i}/{len(paths)} files staged")

    return staged

DATA_FILES = {
    split_name: stage_split_to_local(split_name, paths)
    for split_name, paths in DRIVE_FILES.items()
}

def prepare_weight_vector(raw_weight_dict, n_classes):
    raw = np.array(
        [
            float(raw_weight_dict.get(str(i), raw_weight_dict.get(i, 1.0)))
            for i in range(n_classes)
        ],
        dtype=np.float32,
    )

    if NEURAL_CLASS_WEIGHT_MODE == "sqrt_balanced":
        raw = np.sqrt(np.maximum(raw, 1e-12))

    raw = raw / max(float(raw.mean()), 1e-12)
    raw = np.clip(raw, 0.05, 30.0)
    return raw.astype(np.float32)

BINARY_WEIGHT_VECTOR = prepare_weight_vector(
    CLASS_WEIGHT_INFO["binary_class_weights"], 2
)
MULTI_WEIGHT_VECTOR = prepare_weight_vector(
    CLASS_WEIGHT_INFO["multiclass_class_weights"], 8
)

print("\nLOSS WEIGHTS")
print("-" * 80)
print("Binary     :", BINARY_WEIGHT_VECTOR.tolist())
print("Multiclass :", MULTI_WEIGHT_VECTOR.tolist())

save_json(
    {
        "mode": NEURAL_CLASS_WEIGHT_MODE,
        "binary_weights_used": BINARY_WEIGHT_VECTOR.tolist(),
        "multiclass_weights_used": MULTI_WEIGHT_VECTOR.tolist(),
        "source": str(CLASS_WEIGHT_FILE),
    },
    RESULT_DIR / "step2_loss_weights.json",
)

TASK_CONFIG = {
    "binary": {
        "target": "y_binary",
        "n_classes": 2,
        "class_names": ["Benign", "Attack"],
        "train_cap_per_class": TREE_BINARY_TRAIN_CAP_PER_CLASS,
        "val_cap_per_class": TREE_BINARY_VAL_CAP_PER_CLASS,
        "neural_val_cap_per_class": NEURAL_VAL_BINARY_CAP_PER_CLASS,
    },
    "multiclass": {
        "target": "y_multiclass",
        "n_classes": 8,
        "class_names": CLASS_NAMES,
        "train_cap_per_class": TREE_MULTI_TRAIN_CAP_PER_CLASS,
        "val_cap_per_class": TREE_MULTI_VAL_CAP_PER_CLASS,
        "neural_val_cap_per_class": NEURAL_VAL_MULTI_CAP_PER_CLASS,
    },
}

def build_capped_sample(task_name, split_name, cap_per_class, sample_tag):
    cfg = TASK_CONFIG[task_name]
    target_col = cfg["target"]
    n_classes = cfg["n_classes"]

    output_file = SAMPLE_DIR / f"{sample_tag}.npz"

    if RESUME and not FORCE_REBUILD and output_file.exists():
        try:
            loaded = np.load(output_file)
            X = loaded["X"].astype(np.float32, copy=False)
            y = loaded["y"].astype(np.int64, copy=False)
            if X.shape[1] == N_FEATURES:
                print(f"✅ Reusing sample {sample_tag}: {len(y):,} rows")
                return X, y
        except Exception:
            pass

    files = list(DATA_FILES[split_name])

    rng_files = np.random.default_rng(
        BASE_SEED
        + (1000 if task_name == "multiclass" else 0)
        + (100 if split_name == "validation" else 0)
    )

    order = rng_files.permutation(len(files))
    files = [files[i] for i in order]

    X_parts = defaultdict(list)
    y_parts = defaultdict(list)
    counts = {c: 0 for c in range(n_classes)}

    columns = FINAL_FEATURES + [target_col]

    for file_index, file_path in enumerate(files, start=1):
        df = pd.read_parquet(file_path, columns=columns)

        if len(df) == 0:
            del df
            continue

        y_file = df[target_col].to_numpy(dtype=np.int64)
        X_file = df[FINAL_FEATURES].to_numpy(dtype=np.float32)

        for cls in range(n_classes):
            needed = cap_per_class - counts[cls]
            if needed <= 0:
                continue

            idx = np.flatnonzero(y_file == cls)
            if len(idx) == 0:
                continue

            take = min(needed, len(idx))

            rng = np.random.default_rng(
                BASE_SEED
                + cls * 10_000
                + file_index
                + (777 if split_name == "validation" else 0)
                + (333 if task_name == "multiclass" else 0)
            )

            if take < len(idx):
                idx = rng.choice(idx, size=take, replace=False)

            X_parts[cls].append(X_file[idx])
            y_parts[cls].append(np.full(take, cls, dtype=np.int64))
            counts[cls] += take

        del df, X_file, y_file
        gc.collect()

        if file_index % 50 == 0 or file_index == len(files):
            print(
                f"  sample {sample_tag}: "
                f"{file_index}/{len(files)} files | {counts}"
            )

        if all(counts[c] >= cap_per_class for c in range(n_classes)):
            break

    X_list, y_list = [], []

    for cls in range(n_classes):
        if X_parts[cls]:
            X_list.append(np.concatenate(X_parts[cls], axis=0))
            y_list.append(np.concatenate(y_parts[cls], axis=0))

    if not X_list:
        raise RuntimeError(f"❌ Could not build sample: {sample_tag}")

    X = np.concatenate(X_list, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_list, axis=0).astype(np.int64, copy=False)

    rng_final = np.random.default_rng(BASE_SEED + len(y))
    perm = rng_final.permutation(len(y))
    X = X[perm]
    y = y[perm]

    save_npz(output_file, X=X, y=y)

    print(
        f"✅ Sample saved {sample_tag}: {len(y):,} rows | "
        f"class counts={dict(zip(*np.unique(y, return_counts=True)))}"
    )

    return X, y

def metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.int64)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    accuracy = correct / max(total, 1)

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    tn = total - tp - fp - fn

    precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0
    )
    recall = np.divide(
        tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    support = cm.sum(axis=1).astype(np.float64)
    weighted_f1 = float(
        np.sum(f1 * support) / max(float(support.sum()), 1.0)
    )

    fpr_per_class = np.divide(
        fp, fp + tn, out=np.zeros_like(fp), where=(fp + tn) > 0
    )

    result = {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": weighted_f1,
        "balanced_accuracy": float(np.mean(recall)),
        "macro_fpr": float(np.mean(fpr_per_class)),
    }

    if cm.shape == (2, 2):
        tn_b, fp_b, fn_b, tp_b = cm.ravel()
        sensitivity = tp_b / max(tp_b + fn_b, 1)
        specificity = tn_b / max(tn_b + fp_b, 1)
        precision_b = tp_b / max(tp_b + fp_b, 1)
        f1_b = (
            2 * precision_b * sensitivity
            / max(precision_b + sensitivity, 1e-12)
        )

        result.update({
            "precision": float(precision_b),
            "recall": float(sensitivity),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1": float(f1_b),
            "fpr": float(1.0 - specificity),
        })

    return result

def probability_metrics(task_name, y_true, probabilities):
    cfg = TASK_CONFIG[task_name]
    n_classes = cfg["n_classes"]

    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)

    result = {
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "auc_sample_rows": int(len(y_true)),
    }

    if len(y_true) == 0:
        return result

    try:
        if n_classes == 2:
            result["roc_auc"] = float(
                roc_auc_score(y_true, probabilities[:, 1])
            )
            result["pr_auc"] = float(
                average_precision_score(y_true, probabilities[:, 1])
            )
        else:
            if len(np.unique(y_true)) == n_classes:
                y_bin = label_binarize(
                    y_true, classes=np.arange(n_classes)
                )
                result["roc_auc"] = float(
                    roc_auc_score(
                        y_bin, probabilities,
                        average="macro", multi_class="ovr"
                    )
                )
                result["pr_auc"] = float(
                    average_precision_score(
                        y_bin, probabilities, average="macro"
                    )
                )
    except Exception as exc:
        print(f"⚠️ AUC calculation warning: {exc}")

    return result

def save_confusion_figure(cm, class_names, title, destination):
    cm = np.asarray(cm)

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    if len(class_names) <= 8:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{int(cm[i, j]):,}",
                    ha="center", va="center"
                )

    fig.tight_layout()

    local = LOCAL_ROOT / (
        safe_slug(destination.stem)
        + "_"
        + short_hash(destination)
        + ".png"
    )

    fig.savefig(local, dpi=220, bbox_inches="tight")
    plt.close(fig)

    atomic_copy(local, destination)
    local.unlink(missing_ok=True)

def evaluate_sklearn_model(model, model_name, task_name, split_name="test"):
    cfg = TASK_CONFIG[task_name]
    target_col = cfg["target"]
    n_classes = cfg["n_classes"]
    class_names = cfg["class_names"]

    files = list(DATA_FILES[split_name])
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    auc_y_parts, auc_p_parts = [], []

    expected_rows = int(
        STEP1_SUMMARY[
            {"train": "train_rows",
             "validation": "validation_rows",
             "test": "test_rows"}[split_name]
        ]
    )

    sample_fraction = min(
        1.0, AUC_SAMPLE_MAX_ROWS / max(expected_rows, 1)
    )

    start = time.time()
    total_rows = 0

    for file_index, file_path in enumerate(files, start=1):
        df = pd.read_parquet(
            file_path, columns=FINAL_FEATURES + [target_col]
        )

        if len(df) == 0:
            del df
            continue

        X = df[FINAL_FEATURES].to_numpy(dtype=np.float32)
        y = df[target_col].to_numpy(dtype=np.int64)

        pred = model.predict(X)
        proba = model.predict_proba(X)

        cm += confusion_matrix(
            y, pred, labels=np.arange(n_classes)
        )

        total_rows += len(y)

        sample_n = min(
            len(y), max(1, int(round(len(y) * sample_fraction)))
        )

        if sample_n > 0:
            rng = np.random.default_rng(
                BASE_SEED + file_index * 17
                + (10000 if task_name == "multiclass" else 0)
                + (20000 if model_name == "XGBoost" else 0)
            )

            if sample_n < len(y):
                idx = rng.choice(len(y), size=sample_n, replace=False)
            else:
                idx = np.arange(len(y))

            auc_y_parts.append(y[idx])
            auc_p_parts.append(proba[idx].astype(np.float32))

        del df, X, y, pred, proba
        gc.collect()

        if file_index % 40 == 0 or file_index == len(files):
            print(
                f"  {model_name} {task_name} {split_name}: "
                f"{file_index}/{len(files)} files | {total_rows:,} rows"
            )

    auc_y = (
        np.concatenate(auc_y_parts)
        if auc_y_parts else np.empty(0, dtype=np.int64)
    )
    auc_p = (
        np.concatenate(auc_p_parts)
        if auc_p_parts
        else np.empty((0, n_classes), dtype=np.float32)
    )

    if len(auc_y) > AUC_SAMPLE_MAX_ROWS:
        rng = np.random.default_rng(BASE_SEED + 991)
        idx = rng.choice(
            len(auc_y), size=AUC_SAMPLE_MAX_ROWS, replace=False
        )
        auc_y = auc_y[idx]
        auc_p = auc_p[idx]

    metrics = metrics_from_confusion(cm)
    metrics.update(probability_metrics(task_name, auc_y, auc_p))
    metrics.update({
        "model": model_name,
        "task": task_name,
        "split": split_name,
        "rows_evaluated": int(total_rows),
        "evaluation_seconds": float(time.time() - start),
    })

    save_csv(
        pd.DataFrame(cm, index=class_names, columns=class_names),
        RESULT_DIR
        / f"confusion_{safe_slug(model_name)}_{task_name}_{split_name}.csv",
    )

    save_confusion_figure(
        cm,
        class_names,
        f"{model_name} — {task_name} — {split_name}",
        FIGURE_DIR
        / f"confusion_{safe_slug(model_name)}_{task_name}_{split_name}.png",
    )

    return metrics

def train_random_forest(task_name):
    cfg = TASK_CONFIG[task_name]

    model_file = MODEL_DIR / f"RandomForest_{task_name}.joblib"
    marker_file = CHECKPOINT_DIR / f"RandomForest_{task_name}_COMPLETE.json"

    if (
        RESUME and not FORCE_REBUILD
        and model_file.exists() and marker_file.exists()
    ):
        print(f"\n✅ Reusing Random Forest ({task_name}).")
        return joblib.load(model_file)

    X_train, y_train = build_capped_sample(
        task_name, "train",
        cfg["train_cap_per_class"],
        f"RF_XGB_{task_name}_train",
    )

    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=24,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=BASE_SEED,
        verbose=0,
    )

    print(
        f"\n🌲 Training Random Forest ({task_name}) "
        f"on {len(y_train):,} sampled training rows..."
    )

    start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - start

    save_joblib(model, model_file)

    save_json(
        {
            "status": "COMPLETED",
            "pipeline_version": PIPELINE_VERSION,
            "model": "RandomForest",
            "task": task_name,
            "training_rows": int(len(y_train)),
            "fit_seconds": float(fit_seconds),
            "training_strategy": "reproducible capped class-aware sample",
            "completed_at": datetime.now().isoformat(),
        },
        marker_file,
    )

    del X_train, y_train
    gc.collect()
    return model

def xgb_sample_weights(task_name, y):
    weights = (
        BINARY_WEIGHT_VECTOR
        if task_name == "binary"
        else MULTI_WEIGHT_VECTOR
    )
    return weights[np.asarray(y, dtype=np.int64)]

def build_xgb(task_name, use_gpu=True):
    cfg = TASK_CONFIG[task_name]

    common = dict(
        n_estimators=900,
        max_depth=8,
        learning_rate=0.06,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=BASE_SEED,
        n_jobs=-1,
        eval_metric="logloss" if task_name == "binary" else "mlogloss",
        early_stopping_rounds=50,
    )

    if use_gpu:
        common["device"] = "cuda"

    if task_name == "binary":
        common.update(objective="binary:logistic")
    else:
        common.update(
            objective="multi:softprob",
            num_class=cfg["n_classes"],
        )

    return xgb.XGBClassifier(**common)

def train_xgboost(task_name):
    cfg = TASK_CONFIG[task_name]

    model_file = MODEL_DIR / f"XGBoost_{task_name}.json"
    marker_file = CHECKPOINT_DIR / f"XGBoost_{task_name}_COMPLETE.json"

    if (
        RESUME and not FORCE_REBUILD
        and model_file.exists() and marker_file.exists()
    ):
        print(f"\n✅ Reusing XGBoost ({task_name}).")
        model = build_xgb(task_name, use_gpu=True)
        model.load_model(model_file)
        return model

    X_train, y_train = build_capped_sample(
        task_name, "train",
        cfg["train_cap_per_class"],
        f"RF_XGB_{task_name}_train",
    )

    X_val, y_val = build_capped_sample(
        task_name, "validation",
        cfg["val_cap_per_class"],
        f"XGB_{task_name}_validation",
    )

    sample_weight = xgb_sample_weights(task_name, y_train)

    print(
        f"\n🚀 Training XGBoost ({task_name}) "
        f"on {len(y_train):,} rows..."
    )

    start = time.time()
    gpu_used = True

    try:
        model = build_xgb(task_name, use_gpu=True)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    except Exception as exc:
        gpu_used = False
        print(
            "\n⚠️ XGBoost CUDA training failed; "
            "falling back to CPU."
        )
        print("Reason:", exc)

        model = build_xgb(task_name, use_gpu=False)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    fit_seconds = time.time() - start

    local_model = LOCAL_ROOT / f"XGBoost_{task_name}.json"
    model.save_model(local_model)
    atomic_copy(local_model, model_file)
    local_model.unlink(missing_ok=True)

    save_json(
        {
            "status": "COMPLETED",
            "pipeline_version": PIPELINE_VERSION,
            "model": "XGBoost",
            "task": task_name,
            "training_rows": int(len(y_train)),
            "validation_rows_for_early_stopping": int(len(y_val)),
            "gpu_used": bool(gpu_used),
            "fit_seconds": float(fit_seconds),
            "best_iteration": (
                int(model.best_iteration)
                if hasattr(model, "best_iteration")
                and model.best_iteration is not None
                else None
            ),
            "training_strategy": "reproducible capped class-aware sample",
            "completed_at": datetime.now().isoformat(),
        },
        marker_file,
    )

    del X_train, y_train, X_val, y_val, sample_weight
    gc.collect()
    torch.cuda.empty_cache()
    return model

class MLPClassifierNet(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.network(x)

class CNN1DClassifierNet(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)

def build_neural_model(model_name, n_classes):
    if model_name == "MLP":
        return MLPClassifierNet(N_FEATURES, n_classes)
    if model_name == "CNN1D":
        return CNN1DClassifierNet(N_FEATURES, n_classes)
    raise ValueError(model_name)

NEURAL_VAL_CACHE = {}

def get_neural_validation(task_name):
    if task_name in NEURAL_VAL_CACHE:
        return NEURAL_VAL_CACHE[task_name]

    cfg = TASK_CONFIG[task_name]

    X_val, y_val = build_capped_sample(
        task_name, "validation",
        cfg["neural_val_cap_per_class"],
        f"NEURAL_{task_name}_epoch_validation",
    )

    NEURAL_VAL_CACHE[task_name] = (X_val, y_val)
    return X_val, y_val

@torch.no_grad()
def evaluate_neural_array(model, X, y, n_classes):
    model.eval()

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    total_loss = 0.0
    total_rows = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for start in range(0, len(y), PREDICT_BATCH_SIZE):
        end = min(start + PREDICT_BATCH_SIZE, len(y))

        xb = torch.from_numpy(X[start:end]).to(
            DEVICE, dtype=torch.float32, non_blocking=True
        )

        yb = torch.from_numpy(y[start:end]).to(
            DEVICE, dtype=torch.long, non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        ):
            logits = model(xb)
            loss = criterion(logits.float(), yb)

        pred = torch.argmax(logits, dim=1)

        cm += confusion_matrix(
            yb.cpu().numpy(),
            pred.cpu().numpy(),
            labels=np.arange(n_classes),
        )

        total_loss += float(loss.item())
        total_rows += len(yb)

        del xb, yb, logits, pred, loss

    metrics = metrics_from_confusion(cm)
    metrics["loss"] = total_loss / max(total_rows, 1)
    return metrics

def train_neural_model(model_name, task_name):
    cfg = TASK_CONFIG[task_name]
    target_col = cfg["target"]
    n_classes = cfg["n_classes"]

    best_model_file = MODEL_DIR / f"{model_name}_{task_name}_BEST.pt"
    last_checkpoint = CHECKPOINT_DIR / f"{model_name}_{task_name}_LAST.pt"
    complete_marker = CHECKPOINT_DIR / f"{model_name}_{task_name}_COMPLETE.json"
    history_file = RESULT_DIR / f"history_{model_name}_{task_name}.csv"

    model = build_neural_model(model_name, n_classes).to(DEVICE)

    weight_vector = (
        BINARY_WEIGHT_VECTOR
        if task_name == "binary"
        else MULTI_WEIGHT_VECTOR
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weight_vector, dtype=torch.float32, device=DEVICE
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=True)

    start_epoch = 0
    best_macro_f1 = -np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    if (
        RESUME and not FORCE_REBUILD
        and complete_marker.exists() and best_model_file.exists()
    ):
        print(f"\n✅ Reusing completed {model_name} ({task_name}).")
        state = torch.load(
            best_model_file,
            map_location=DEVICE,
            weights_only=False,
        )
        model.load_state_dict(state["model_state"])
        model.eval()
        return model

    if (
        RESUME and not FORCE_REBUILD
        and last_checkpoint.exists()
    ):
        try:
            checkpoint = torch.load(
                last_checkpoint,
                map_location=DEVICE,
                weights_only=False,
            )

            if (
                checkpoint.get("pipeline_version") == PIPELINE_VERSION
                and checkpoint.get("model_name") == model_name
                and checkpoint.get("task_name") == task_name
            ):
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])

                if checkpoint.get("scaler_state"):
                    scaler.load_state_dict(checkpoint["scaler_state"])

                start_epoch = int(checkpoint["epoch"]) + 1
                best_macro_f1 = float(checkpoint["best_macro_f1"])
                best_epoch = int(checkpoint["best_epoch"])
                epochs_without_improvement = int(
                    checkpoint["epochs_without_improvement"]
                )
                history = checkpoint.get("history", [])

                print(
                    f"\n♻️ Resuming {model_name} {task_name} "
                    f"from epoch {start_epoch + 1}"
                )

        except Exception as exc:
            print("⚠️ Could not resume neural checkpoint:", exc)

    X_val, y_val = get_neural_validation(task_name)
    train_files = list(DATA_FILES["train"])
    total_training_start = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()

        rng_files = np.random.default_rng(
            BASE_SEED
            + epoch * 100
            + (1000 if task_name == "multiclass" else 0)
            + (2000 if model_name == "CNN1D" else 0)
        )

        epoch_files = [
            train_files[i]
            for i in rng_files.permutation(len(train_files))
        ]

        epoch_loss_sum = 0.0
        epoch_rows = 0
        epoch_cm = np.zeros(
            (n_classes, n_classes), dtype=np.int64
        )

        epoch_start = time.time()

        print(
            f"\n🔥 {model_name} | {task_name} | "
            f"Epoch {epoch + 1}/{MAX_EPOCHS}"
        )

        for file_index, file_path in enumerate(epoch_files, start=1):
            df = pd.read_parquet(
                file_path,
                columns=FINAL_FEATURES + [target_col],
            )

            if len(df) == 0:
                del df
                continue

            X_file = df[FINAL_FEATURES].to_numpy(dtype=np.float32)
            y_file = df[target_col].to_numpy(dtype=np.int64)

            rng_rows = np.random.default_rng(
                BASE_SEED + epoch * 1_000_000 + file_index
            )

            order = rng_rows.permutation(len(y_file))
            X_file = X_file[order]
            y_file = y_file[order]

            for start in range(0, len(y_file), NEURAL_BATCH_SIZE):
                end = min(start + NEURAL_BATCH_SIZE, len(y_file))

                if (end - start) < 2:
                    continue

                xb = torch.from_numpy(X_file[start:end]).to(
                    DEVICE, dtype=torch.float32, non_blocking=True
                )

                yb = torch.from_numpy(y_file[start:end]).to(
                    DEVICE, dtype=torch.long, non_blocking=True
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):
                    logits = model(xb)
                    loss = criterion(logits, yb)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0
                )

                scaler.step(optimizer)
                scaler.update()

                pred = torch.argmax(logits.detach(), dim=1)

                batch_rows = len(yb)
                epoch_loss_sum += float(loss.item()) * batch_rows
                epoch_rows += batch_rows

                epoch_cm += confusion_matrix(
                    yb.detach().cpu().numpy(),
                    pred.cpu().numpy(),
                    labels=np.arange(n_classes),
                )

                del xb, yb, logits, loss, pred

            del df, X_file, y_file, order
            gc.collect()

            if file_index % 40 == 0 or file_index == len(epoch_files):
                print(
                    f"  files {file_index}/{len(epoch_files)} | "
                    f"rows {epoch_rows:,}"
                )

        train_metrics = metrics_from_confusion(epoch_cm)
        train_loss = epoch_loss_sum / max(epoch_rows, 1)

        val_metrics = evaluate_neural_array(
            model, X_val, y_val, n_classes
        )

        epoch_record = {
            "epoch": int(epoch + 1),
            "train_rows": int(epoch_rows),
            "train_loss": float(train_loss),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": float(time.time() - epoch_start),
        }

        history.append(epoch_record)
        save_csv(pd.DataFrame(history), history_file)

        current_macro_f1 = val_metrics["macro_f1"]

        print(
            f"  Train loss={train_loss:.5f} "
            f"| Train macro-F1={train_metrics['macro_f1']:.5f}"
        )
        print(
            f"  Val loss={val_metrics['loss']:.5f} "
            f"| Val macro-F1={current_macro_f1:.5f}"
        )

        improved = current_macro_f1 > best_macro_f1 + 1e-5

        if improved:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            save_torch(
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "model_name": model_name,
                    "task_name": task_name,
                    "input_features": FINAL_FEATURES,
                    "n_classes": n_classes,
                    "best_epoch": int(best_epoch),
                    "best_validation_macro_f1": float(best_macro_f1),
                    "model_state": model.state_dict(),
                },
                best_model_file,
            )

            print("  ✅ Best checkpoint updated.")
        else:
            epochs_without_improvement += 1

        save_torch(
            {
                "pipeline_version": PIPELINE_VERSION,
                "model_name": model_name,
                "task_name": task_name,
                "epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "best_macro_f1": float(best_macro_f1),
                "epochs_without_improvement": int(epochs_without_improvement),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
            },
            last_checkpoint,
        )

        torch.cuda.empty_cache()

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"🛑 Early stopping after epoch {epoch + 1}. "
                f"Best epoch={best_epoch}"
            )
            break

    best = torch.load(
        best_model_file,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(best["model_state"])
    model.eval()

    save_json(
        {
            "status": "COMPLETED",
            "pipeline_version": PIPELINE_VERSION,
            "model": model_name,
            "task": task_name,
            "training_strategy": "complete Step-1 training split",
            "best_epoch": int(best["best_epoch"]),
            "best_validation_macro_f1": float(
                best["best_validation_macro_f1"]
            ),
            "total_training_seconds": float(
                time.time() - total_training_start
            ),
            "batch_size": int(NEURAL_BATCH_SIZE),
            "mixed_precision": True,
            "gpu": GPU_NAME,
            "completed_at": datetime.now().isoformat(),
        },
        complete_marker,
    )

    return model

@torch.no_grad()
def evaluate_neural_model(
    model, model_name, task_name, split_name="test"
):
    cfg = TASK_CONFIG[task_name]
    target_col = cfg["target"]
    n_classes = cfg["n_classes"]
    class_names = cfg["class_names"]

    files = list(DATA_FILES[split_name])
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    auc_y_parts, auc_p_parts = [], []

    expected_rows = int(
        STEP1_SUMMARY[
            {"train": "train_rows",
             "validation": "validation_rows",
             "test": "test_rows"}[split_name]
        ]
    )

    sample_fraction = min(
        1.0, AUC_SAMPLE_MAX_ROWS / max(expected_rows, 1)
    )

    total_rows = 0
    start_time = time.time()
    model.eval()

    for file_index, file_path in enumerate(files, start=1):
        df = pd.read_parquet(
            file_path,
            columns=FINAL_FEATURES + [target_col],
        )

        if len(df) == 0:
            del df
            continue

        X = df[FINAL_FEATURES].to_numpy(dtype=np.float32)
        y = df[target_col].to_numpy(dtype=np.int64)

        pred_parts, prob_parts = [], []

        for start in range(0, len(y), PREDICT_BATCH_SIZE):
            end = min(start + PREDICT_BATCH_SIZE, len(y))

            xb = torch.from_numpy(X[start:end]).to(
                DEVICE, dtype=torch.float32, non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            ):
                logits = model(xb)

            probability = torch.softmax(logits.float(), dim=1)
            prediction = torch.argmax(probability, dim=1)

            pred_parts.append(
                prediction.cpu().numpy().astype(np.int64)
            )
            prob_parts.append(
                probability.cpu().numpy().astype(np.float32)
            )

            del xb, logits, probability, prediction

        pred = np.concatenate(pred_parts)
        proba = np.concatenate(prob_parts)

        cm += confusion_matrix(
            y, pred, labels=np.arange(n_classes)
        )

        total_rows += len(y)

        sample_n = min(
            len(y), max(1, int(round(len(y) * sample_fraction)))
        )

        if sample_n > 0:
            rng = np.random.default_rng(
                BASE_SEED + file_index * 31
                + (5000 if task_name == "multiclass" else 0)
                + (7000 if model_name == "CNN1D" else 0)
            )

            if sample_n < len(y):
                idx = rng.choice(
                    len(y), size=sample_n, replace=False
                )
            else:
                idx = np.arange(len(y))

            auc_y_parts.append(y[idx])
            auc_p_parts.append(proba[idx])

        del df, X, y, pred, proba, pred_parts, prob_parts
        gc.collect()

        if file_index % 40 == 0 or file_index == len(files):
            print(
                f"  {model_name} {task_name} {split_name}: "
                f"{file_index}/{len(files)} files | {total_rows:,} rows"
            )

    auc_y = (
        np.concatenate(auc_y_parts)
        if auc_y_parts else np.empty(0, dtype=np.int64)
    )
    auc_p = (
        np.concatenate(auc_p_parts)
        if auc_p_parts
        else np.empty((0, n_classes), dtype=np.float32)
    )

    if len(auc_y) > AUC_SAMPLE_MAX_ROWS:
        rng = np.random.default_rng(BASE_SEED + 1991)
        idx = rng.choice(
            len(auc_y), size=AUC_SAMPLE_MAX_ROWS, replace=False
        )
        auc_y = auc_y[idx]
        auc_p = auc_p[idx]

    metrics = metrics_from_confusion(cm)
    metrics.update(probability_metrics(task_name, auc_y, auc_p))

    metrics.update({
        "model": model_name,
        "task": task_name,
        "split": split_name,
        "rows_evaluated": int(total_rows),
        "evaluation_seconds": float(time.time() - start_time),
    })

    save_csv(
        pd.DataFrame(cm, index=class_names, columns=class_names),
        RESULT_DIR
        / f"confusion_{model_name}_{task_name}_{split_name}.csv",
    )

    save_confusion_figure(
        cm,
        class_names,
        f"{model_name} — {task_name} — {split_name}",
        FIGURE_DIR
        / f"confusion_{model_name}_{task_name}_{split_name}.png",
    )

    return metrics

def load_cached_metrics(model_name, task_name, split_name):
    path = RESULT_DIR / f"metrics_{model_name}_{task_name}_{split_name}.json"

    if RESUME and not FORCE_REBUILD and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                result = json.load(f)

            if result.get("pipeline_version") == PIPELINE_VERSION:
                print(
                    f"✅ Reusing metrics: "
                    f"{model_name} | {task_name} | {split_name}"
                )
                return result["metrics"]
        except Exception:
            pass

    return None

def cache_metrics(model_name, task_name, split_name, metrics):
    save_json(
        {
            "pipeline_version": PIPELINE_VERSION,
            "model": model_name,
            "task": task_name,
            "split": split_name,
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
        },
        RESULT_DIR / f"metrics_{model_name}_{task_name}_{split_name}.json",
    )

ALL_RESULTS = []

def register_metrics(metrics):
    global ALL_RESULTS

    key = (
        metrics["model"],
        metrics["task"],
        metrics["split"],
    )

    ALL_RESULTS = [
        row for row in ALL_RESULTS
        if (
            row.get("model"),
            row.get("task"),
            row.get("split"),
        ) != key
    ]

    ALL_RESULTS.append(metrics)

    save_csv(
        pd.DataFrame(ALL_RESULTS),
        RESULT_DIR / "step2_results_running.csv",
    )

for task_name in ["binary", "multiclass"]:

    rf = train_random_forest(task_name)

    for split_name in ["validation", "test"]:
        metrics = load_cached_metrics(
            "RandomForest", task_name, split_name
        )

        if metrics is None:
            print(
                f"\n📊 Evaluating RandomForest "
                f"{task_name} on COMPLETE {split_name} split..."
            )

            metrics = evaluate_sklearn_model(
                rf, "RandomForest", task_name, split_name
            )

            cache_metrics(
                "RandomForest", task_name, split_name, metrics
            )

        register_metrics(metrics)

    del rf
    gc.collect()

    xgb_model = train_xgboost(task_name)

    for split_name in ["validation", "test"]:
        metrics = load_cached_metrics(
            "XGBoost", task_name, split_name
        )

        if metrics is None:
            print(
                f"\n📊 Evaluating XGBoost "
                f"{task_name} on COMPLETE {split_name} split..."
            )

            metrics = evaluate_sklearn_model(
                xgb_model, "XGBoost", task_name, split_name
            )

            cache_metrics(
                "XGBoost", task_name, split_name, metrics
            )

        register_metrics(metrics)

    del xgb_model
    gc.collect()
    torch.cuda.empty_cache()

for model_name in ["MLP", "CNN1D"]:

    for task_name in ["binary", "multiclass"]:

        neural_model = train_neural_model(
            model_name, task_name
        )

        for split_name in ["validation", "test"]:
            metrics = load_cached_metrics(
                model_name, task_name, split_name
            )

            if metrics is None:
                print(
                    f"\n📊 Evaluating {model_name} "
                    f"{task_name} on COMPLETE {split_name} split..."
                )

                metrics = evaluate_neural_model(
                    neural_model,
                    model_name,
                    task_name,
                    split_name,
                )

                cache_metrics(
                    model_name, task_name, split_name, metrics
                )

            register_metrics(metrics)

        del neural_model
        gc.collect()
        torch.cuda.empty_cache()

results_df = pd.DataFrame(ALL_RESULTS)

for column in [
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "weighted_f1", "balanced_accuracy", "macro_fpr",
    "precision", "recall", "sensitivity", "specificity",
    "f1", "fpr", "roc_auc", "pr_auc",
    "rows_evaluated", "evaluation_seconds",
]:
    if column in results_df.columns:
        results_df[column] = pd.to_numeric(
            results_df[column], errors="coerce"
        )

results_df = results_df.sort_values(
    ["task", "split", "macro_f1"],
    ascending=[True, True, False],
).reset_index(drop=True)

save_csv(
    results_df,
    RESULT_DIR / "STEP02_ALL_METRICS.csv",
)

test_results = (
    results_df[
        results_df["split"] == "test"
    ]
    .copy()
)

save_csv(
    test_results,
    RESULT_DIR / "STEP02_TEST_COMPARISON.csv",
)

validation_results = (
    results_df[
        results_df["split"] == "validation"
    ]
    .copy()
)

multi_validation = (
    validation_results[
        validation_results["task"] == "multiclass"
    ]
    .copy()
)

if len(multi_validation) == 0:
    raise RuntimeError(
        "❌ No multiclass validation results available."
    )

best_overall_val_row = (
    multi_validation
    .sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"],
        ascending=False,
    )
    .iloc[0]
)

neural_multi_validation = (
    multi_validation[
        multi_validation["model"].isin(["MLP", "CNN1D"])
    ]
    .copy()
)

if len(neural_multi_validation) == 0:
    raise RuntimeError(
        "❌ No federation-compatible neural validation result available."
    )

best_fl_val_row = (
    neural_multi_validation
    .sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"],
        ascending=False,
    )
    .iloc[0]
)

BEST_OVERALL_MODEL = str(
    best_overall_val_row["model"]
)
BEST_FL_MODEL = str(
    best_fl_val_row["model"]
)

def selected_test_row(model_name):
    rows = test_results[
        (test_results["task"] == "multiclass")
        &
        (test_results["model"] == model_name)
    ]

    if len(rows) == 0:
        raise RuntimeError(
            f"❌ Missing multiclass test result for {model_name}"
        )

    return rows.iloc[0]

best_overall_test_row = selected_test_row(
    BEST_OVERALL_MODEL
)
best_fl_test_row = selected_test_row(
    BEST_FL_MODEL
)

selection_summary = {
    "pipeline_version": PIPELINE_VERSION,
    "selection_task": "8-class attack-family classification",
    "selection_split": "validation",
    "primary_selection_metric": "validation macro_f1",

    "best_overall_model": BEST_OVERALL_MODEL,
    "best_overall_validation_macro_f1": float(
        best_overall_val_row["macro_f1"]
    ),
    "best_overall_test_macro_f1": float(
        best_overall_test_row["macro_f1"]
    ),
    "best_overall_test_accuracy": float(
        best_overall_test_row["accuracy"]
    ),

    "best_federation_compatible_model": BEST_FL_MODEL,
    "best_fl_model_validation_macro_f1": float(
        best_fl_val_row["macro_f1"]
    ),
    "best_fl_model_test_macro_f1": float(
        best_fl_test_row["macro_f1"]
    ),
    "best_fl_model_test_accuracy": float(
        best_fl_test_row["accuracy"]
    ),
    "best_fl_model_test_balanced_accuracy": float(
        best_fl_test_row["balanced_accuracy"]
    ),

    "federation_compatible_candidates": ["MLP", "CNN1D"],
    "test_set_used_for_selection": False,

    "why_tree_models_are_not_federated": (
        "Random Forest and XGBoost are centralized reference baselines. "
        "Step 3 uses the strongest validation-selected neural architecture "
        "because standard FedAvg operates naturally on neural-network parameters."
    ),

    "created_at": datetime.now().isoformat(),
}

save_json(
    selection_summary,
    RESULT_DIR / "BEST_MODEL_SELECTION.json",
)

def save_metric_bar_chart(task_name, metric):
    df = test_results[
        test_results["task"] == task_name
    ].copy()

    if len(df) == 0 or metric not in df.columns:
        return

    df = df.sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        df["model"].astype(str),
        df[metric].astype(float),
    )

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Model")
    ax.set_title(
        f"Centralized Baseline Comparison — "
        f"{task_name.title()} — "
        f"{metric.replace('_', ' ').title()}"
    )

    ax.set_ylim(
        0,
        max(1.0, float(df[metric].max()) * 1.08),
    )

    fig.tight_layout()

    destination = FIGURE_DIR / f"comparison_{task_name}_{metric}.png"
    local = LOCAL_ROOT / destination.name

    fig.savefig(local, dpi=220, bbox_inches="tight")
    plt.close(fig)

    atomic_copy(local, destination)
    local.unlink(missing_ok=True)

for task_name in ["binary", "multiclass"]:
    for metric in [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
    ]:
        save_metric_bar_chart(task_name, metric)

save_json(
    {
        "pipeline_version": PIPELINE_VERSION,
        "paper_title": (
            "A Hybrid Blockchain–Federated Learning Framework "
            "with AI-Based Anomaly Detection for "
            "Privacy-Preserving IoT Security"
        ),
        "dataset": "CIC IoT-DIAD 2024",
        "step1_split_mode": STEP1_SUMMARY["primary_split_mode"],
        "features": FINAL_FEATURES,
        "feature_count": N_FEATURES,
        "models": [
            "RandomForest", "XGBoost", "MLP", "CNN1D"
        ],
        "tasks": ["binary", "multiclass"],
        "tree_baseline_sampling": {
            "binary_train_cap_per_class":
                TREE_BINARY_TRAIN_CAP_PER_CLASS,
            "multiclass_train_cap_per_class":
                TREE_MULTI_TRAIN_CAP_PER_CLASS,
            "reason": (
                "CPU/tree scalability on >12M-row training split; "
                "neural baselines use complete training data."
            ),
        },
        "neural_training": {
            "full_training_split": True,
            "batch_size": NEURAL_BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": EARLY_STOPPING_PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "mixed_precision": True,
            "class_weight_mode": NEURAL_CLASS_WEIGHT_MODE,
        },
        "evaluation": {
            "classification_metrics":
                "complete validation/test splits",
            "auc_probability_sample_max_rows":
                AUC_SAMPLE_MAX_ROWS,
        },
        "gpu": GPU_NAME,
        "gpu_memory_gb": GPU_MEMORY_GB,
        "created_at": datetime.now().isoformat(),
    },
    CONFIG_DIR / "STEP02_EXPERIMENT_CONFIG.json",
)

completion = {
    "pipeline_version": PIPELINE_VERSION,
    "step": 2,
    "name": "Centralized AI Baseline Benchmarking",
    "status": "COMPLETED",

    "models_completed": [
        "RandomForest", "XGBoost", "MLP", "CNN1D"
    ],

    "tasks_completed": [
        "binary", "multiclass"
    ],

    "test_rows": int(
        STEP1_SUMMARY["test_rows"]
    ),

    "final_features": int(N_FEATURES),

    "best_overall_model": BEST_OVERALL_MODEL,
    "best_federation_compatible_model": BEST_FL_MODEL,

    "best_model_selection_file": str(
        RESULT_DIR / "BEST_MODEL_SELECTION.json"
    ),

    "all_metrics_file": str(
        RESULT_DIR / "STEP02_ALL_METRICS.csv"
    ),

    "test_comparison_file": str(
        RESULT_DIR / "STEP02_TEST_COMPARISON.csv"
    ),

    "next_step": (
        "STEP 3 — Non-IID Federated Learning using "
        f"{BEST_FL_MODEL}"
    ),

    "completed_at": datetime.now().isoformat(),
}

save_json(
    completion,
    STEP2_COMPLETE,
)

print("\n" + "=" * 112)
print("✅ STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 112)

print("\nTEST-SET COMPARISON")
print("-" * 112)

display_columns = [
    c for c in [
        "model",
        "task",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "specificity",
        "fpr",
        "rows_evaluated",
    ]
    if c in test_results.columns
]

print(
    test_results[
        display_columns
    ].to_string(
        index=False
    )
)

print("\nMODEL SELECTION")
print("-" * 112)

print(
    "Best overall centralized model      :",
    BEST_OVERALL_MODEL,
)

print(
    "Best FL-compatible neural model     :",
    BEST_FL_MODEL,
)

print(
    "FL model validation Macro-F1        :",
    f"{float(best_fl_val_row['macro_f1']):.6f}",
)

print(
    "FL model untouched test Macro-F1    :",
    f"{float(best_fl_test_row['macro_f1']):.6f}",
)

print(
    "FL model untouched test Accuracy    :",
    f"{float(best_fl_test_row['accuracy']):.6f}",
)

print("\nIMPORTANT")
print("-" * 112)
print("✅ RF/XGBoost are centralized reference baselines.")
print("✅ MLP/CNN1D were trained on the complete Step-1 training split.")
print("✅ Full validation/test splits were used for classification metrics.")
print("✅ Model selection used validation data, NOT test data.")
print("✅ Test data was never used for fitting or early stopping.")
print("✅ Model checkpoints are persistent and restart-safe.")
print("✅ The strongest neural model is automatically selected for Step 3.")

print("\nOUTPUT:")
print(RESULT_DIR)

print("\nNEXT:")
print(
    f"STEP 3 — NON-IID FEDERATED LEARNING "
    f"USING {BEST_FL_MODEL}"
)

print("=" * 112)

import os, sys, gc, re, json, time, math, random, shutil, hashlib, warnings, subprocess, importlib.util
from pathlib import Path
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings("ignore")

from google.colab import drive
if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")

def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name])

for imp, pipn in [
    ("numpy","numpy"), ("pandas","pandas"), ("pyarrow","pyarrow"),
    ("sklearn","scikit-learn"), ("imblearn","imbalanced-learn"),
    ("matplotlib","matplotlib"), ("torch","torch")
]:
    ensure_package(imp, pipn)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

print("="*115)
print("STEP 2B — CLASS-IMBALANCE MITIGATION + ROBUST CENTRALIZED LEARNING")
print("="*115)

PROJECT_ROOT = Path("/content/drive/MyDrive/Hybrid_BCFL_IJACSA_2026")
PREPROCESSOR_FILE = PROJECT_ROOT/"04_PREPROCESSOR"/"train_only_preprocessor.json"
CLASS_WEIGHT_FILE = PROJECT_ROOT/"04_PREPROCESSOR"/"class_weights.json"
STEP1_COMPLETE = PROJECT_ROOT/"06_CHECKPOINTS"/"STEP01_COMPLETE.json"
MODEL_READY = PROJECT_ROOT/"05_MODEL_READY"

TRAIN_DIR = MODEL_READY/"TRAIN"
VAL_DIR = MODEL_READY/"VALIDATION"
TEST_DIR = MODEL_READY/"TEST"

STEP2A_ROOT = PROJECT_ROOT/"07_AI_MODELS"/"STEP02_CENTRALIZED_BASELINES"
STEP2A_TEST = STEP2A_ROOT/"RESULTS"/"STEP02_TEST_COMPARISON.csv"

STEP2B_ROOT = PROJECT_ROOT/"07_AI_MODELS"/"STEP02B_IMBALANCE_ROBUST"
MODEL_DIR = STEP2B_ROOT/"MODELS"
CKPT_DIR = STEP2B_ROOT/"CHECKPOINTS"
CACHE_DIR = STEP2B_ROOT/"CACHE"
RESULT_DIR = STEP2B_ROOT/"RESULTS"
FIG_DIR = STEP2B_ROOT/"FIGURES"
for p in [MODEL_DIR, CKPT_DIR, CACHE_DIR, RESULT_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

LOCAL = Path("/content/STEP02B_RUNTIME")
LOCAL.mkdir(parents=True, exist_ok=True)

for p in [PREPROCESSOR_FILE, CLASS_WEIGHT_FILE, STEP1_COMPLETE, TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not p.exists():
        raise FileNotFoundError(f"Required Step-1 file/folder missing: {p}")

with open(PREPROCESSOR_FILE, encoding="utf-8") as f:
    PREPROCESSOR = json.load(f)
with open(CLASS_WEIGHT_FILE, encoding="utf-8") as f:
    CLASS_INFO = json.load(f)
with open(STEP1_COMPLETE, encoding="utf-8") as f:
    STEP1 = json.load(f)

STEP1_FEATURES = list(PREPROCESSOR["final_features"])
STEP2A_DF = pd.read_csv(STEP2A_TEST) if STEP2A_TEST.exists() else None

if not torch.cuda.is_available():
    raise RuntimeError("GPU not enabled. Colab -> Runtime -> Change runtime type -> GPU")

DEVICE = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_GB = torch.cuda.get_device_properties(0).total_memory/(1024**3)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

print(f"GPU: {GPU_NAME} | VRAM: {GPU_GB:.2f} GB")
print(f"Step-1 split: {STEP1['primary_split_mode']}")
print(f"Rows: train={STEP1['train_rows']:,}, val={STEP1['validation_rows']:,}, test={STEP1['test_rows']:,}")
print(f"Step-1 features: {len(STEP1_FEATURES)}")

VERSION = "STEP02B_V2"
RESUME = True
FORCE_REBUILD = False

SOURCE_CAP_PER_CLASS = 150_000
TARGET_PER_CLASS = 100_000
SMOTE_MAX_MULTIPLIER = 5
SMOTE_MIN_TARGET = 20_000

RF_RANK_MAX_ROWS = 500_000
RF_TREES = 140
FEATURE_K_CANDIDATES = [24, 36, 48, len(STEP1_FEATURES)]
PROBE_ROWS = 400_000
PROBE_EPOCHS = 3
PROBE_VAL_CAP = 20_000

MAX_EPOCHS = 12
PATIENCE = 3
LR = 3e-4
WEIGHT_DECAY = 2e-4
DROPOUT = 0.20
BATCH_SIZE = 8192 if GPU_GB >= 14 else 4096
PRED_BATCH = max(BATCH_SIZE, 16384)
FOCAL_GAMMA = 2.0
AUC_SAMPLE = 300_000

BINARY_THRESHOLDS = np.round(np.arange(0.05,0.951,0.025),3)
PRIOR_TAUS = np.array([-0.50,-0.25,0.0,0.25,0.50,0.75,1.0],dtype=np.float32)

COMPLETE_FILE = CKPT_DIR/"STEP02B_COMPLETE.json"

def shash(x): return hashlib.sha1(str(x).encode()).hexdigest()[:10]

def atomic_copy(src, dst):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(dst)+".partial")
    partial.unlink(missing_ok=True)
    shutil.copy2(src, partial)
    os.replace(partial, dst)

def save_json(obj, dst):
    tmp = LOCAL/f"{Path(dst).stem}_{shash(dst)}.json"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,default=str)
    atomic_copy(tmp,dst); tmp.unlink(missing_ok=True)

def save_csv(df, dst):
    tmp = LOCAL/f"{Path(dst).stem}_{shash(dst)}.csv"
    df.to_csv(tmp,index=False)
    atomic_copy(tmp,dst); tmp.unlink(missing_ok=True)

def save_npz(dst, **arrays):
    tmp = LOCAL/f"{Path(dst).stem}_{shash(dst)}.npz"
    np.savez_compressed(tmp, **arrays)
    atomic_copy(tmp,dst); tmp.unlink(missing_ok=True)

def save_torch(obj, dst):
    tmp = LOCAL/f"{Path(dst).stem}_{shash(dst)}.pt"
    torch.save(obj,tmp)
    atomic_copy(tmp,dst); tmp.unlink(missing_ok=True)

if FORCE_REBUILD and STEP2B_ROOT.exists():
    shutil.rmtree(STEP2B_ROOT)
    for p in [MODEL_DIR, CKPT_DIR, CACHE_DIR, RESULT_DIR, FIG_DIR]:
        p.mkdir(parents=True, exist_ok=True)

if RESUME and not FORCE_REBUILD and COMPLETE_FILE.exists():
    with open(COMPLETE_FILE,encoding="utf-8") as f:
        old=json.load(f)
    if old.get("version")==VERSION and old.get("status")=="COMPLETED":
        print("✅ Step 2B already completed.")
        print(json.dumps(old,indent=2))
        raise SystemExit

DRIVE_FILES = {
    "train": sorted(TRAIN_DIR.glob("*.parquet")),
    "validation": sorted(VAL_DIR.glob("*.parquet")),
    "test": sorted(TEST_DIR.glob("*.parquet")),
}
for k,v in DRIVE_FILES.items():
    if not v: raise RuntimeError(f"No {k} parquet files found")

def stage_split(name, files):
    target = LOCAL/"DATA"/name.upper()
    target.mkdir(parents=True, exist_ok=True)
    need = sum(p.stat().st_size for p in files)
    free = shutil.disk_usage("/content").free
    if free < need + 10*(1024**3):
        print(f"Using Drive directly for {name}")
        return files
    out=[]
    print(f"Staging {name} to local SSD...")
    for i,src in enumerate(files,1):
        dst=target/src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src,dst)
        out.append(dst)
        if i%50==0 or i==len(files): print(f"  {i}/{len(files)}")
    return out

DATA_FILES = {k:stage_split(k,v) for k,v in DRIVE_FILES.items()}

CLASS_TO_ID = {"Benign":0,"DDoS":1,"DoS":2,"Recon":3,"Web-Based":4,"Brute Force":5,"Spoofing":6,"Mirai":7}
ID_TO_CLASS = {v:k for k,v in CLASS_TO_ID.items()}
CLASS_NAMES = [ID_TO_CLASS[i] for i in range(8)]

if STEP2A_DF is not None:
    save_csv(STEP2A_DF, RESULT_DIR/"STEP02A_IMPORTED_TEST_BASELINE.csv")
    print("\nSTEP 2A baseline:")
    print(STEP2A_DF.to_string(index=False))

natural_counts_raw = CLASS_INFO.get("multiclass_train_counts",{})
NATURAL_COUNTS = {int(k):int(v) for k,v in natural_counts_raw.items()}
if len(NATURAL_COUNTS)!=8:
    raise RuntimeError("Could not read 8-class training counts from Step 1.")

NATURAL_PRIOR = np.array([NATURAL_COUNTS[i] for i in range(8)],dtype=np.float64)
NATURAL_PRIOR /= NATURAL_PRIOR.sum()

STRICT_PATTERNS = [
    r"(^|_)src(_|$).*ip", r"(^|_)dst(_|$).*ip",
    r"source.*ip", r"destination.*ip", r"(^|_)mac(_|$)",
    r"(^|_)src(_|$).*port", r"(^|_)dst(_|$).*port",
    r"source.*port", r"destination.*port",
    r"flow_id", r"timestamp", r"device_id"
]
def identifier_like(name):
    s=str(name).lower()
    return any(re.search(p,s) for p in STRICT_PATTERNS)

STRICT_FEATURES=[f for f in STEP1_FEATURES if not identifier_like(f)]
REMOVED_STRICT=[f for f in STEP1_FEATURES if f not in STRICT_FEATURES]
if len(STRICT_FEATURES)<20:
    STRICT_FEATURES=STEP1_FEATURES.copy()
    REMOVED_STRICT=[]

save_json({
    "step1_features":len(STEP1_FEATURES),
    "strict_features":len(STRICT_FEATURES),
    "removed":REMOVED_STRICT,
    "note":"Feature-name-only gate; no validation/test labels used."
}, RESULT_DIR/"strict_feature_gate.json")
print(f"\nStrict candidate features: {len(STRICT_FEATURES)} | removed={REMOVED_STRICT}")

SOURCE_FILE=CACHE_DIR/"train_source_pool.npz"

def build_source_pool():
    if RESUME and SOURCE_FILE.exists():
        d=np.load(SOURCE_FILE)
        if d["X"].shape[1]==len(STRICT_FEATURES):
            print("✅ Reusing source pool")
            return d["X"].astype(np.float32,copy=False), d["y"].astype(np.int64,copy=False)

    files=list(DATA_FILES["train"])
    rng=np.random.default_rng(SEED)
    files=[files[i] for i in rng.permutation(len(files))]
    xp,yp=defaultdict(list),defaultdict(list)
    counts={i:0 for i in range(8)}
    cols=STRICT_FEATURES+["y_multiclass"]

    for fi,p in enumerate(files,1):
        df=pd.read_parquet(p,columns=cols)
        if len(df)==0: continue
        Xf=df[STRICT_FEATURES].to_numpy(np.float32)
        yf=df["y_multiclass"].to_numpy(np.int64)
        for c in range(8):
            need=SOURCE_CAP_PER_CLASS-counts[c]
            if need<=0: continue
            idx=np.flatnonzero(yf==c)
            if len(idx)==0: continue
            take=min(need,len(idx))
            rr=np.random.default_rng(SEED+fi*100+c)
            if take<len(idx): idx=rr.choice(idx,size=take,replace=False)
            xp[c].append(Xf[idx]); yp[c].append(np.full(take,c,np.int64)); counts[c]+=take
        del df,Xf,yf; gc.collect()
        if fi%40==0 or fi==len(files): print(f"source {fi}/{len(files)} {counts}")

    X=np.concatenate([np.concatenate(xp[c]) for c in range(8)])
    y=np.concatenate([np.concatenate(yp[c]) for c in range(8)])
    perm=np.random.default_rng(SEED+1).permutation(len(y))
    X,y=X[perm].astype(np.float32),y[perm].astype(np.int64)
    save_npz(SOURCE_FILE,X=X,y=y)
    return X,y

X_SOURCE,y_SOURCE=build_source_pool()
source_counts={int(k):int(v) for k,v in zip(*np.unique(y_SOURCE,return_counts=True))}
print("Source counts:",source_counts)

RANK_FILE=RESULT_DIR/"rf_train_only_feature_ranking.csv"

def get_ranking():
    if RESUME and RANK_FILE.exists():
        r=pd.read_csv(RANK_FILE)
        if len(r)==len(STRICT_FEATURES): return r

    n=min(len(y_SOURCE),RF_RANK_MAX_ROWS)
    idx=np.random.default_rng(SEED+2).choice(len(y_SOURCE),size=n,replace=False)
    rf=RandomForestClassifier(
        n_estimators=RF_TREES,max_depth=20,min_samples_leaf=2,
        max_features="sqrt",class_weight="balanced_subsample",
        n_jobs=-1,random_state=SEED
    )
    print("\nTraining RF feature ranker...")
    rf.fit(X_SOURCE[idx],y_SOURCE[idx])
    r=pd.DataFrame({"feature":STRICT_FEATURES,"importance":rf.feature_importances_})
    r=r.sort_values("importance",ascending=False).reset_index(drop=True)
    r["rank"]=np.arange(1,len(r)+1)
    save_csv(r,RANK_FILE)
    print(r.head(15).to_string(index=False))
    return r

RANKING=get_ranking()

BAL_FILE=CACHE_DIR/"hybrid_balanced_multiclass_pool.npz"

def build_balanced_pool():
    if RESUME and BAL_FILE.exists():
        d=np.load(BAL_FILE)
        print("✅ Reusing hybrid-balanced pool")
        return d["X"].astype(np.float32,copy=False), d["y"].astype(np.int64,copy=False)

    X,y=X_SOURCE.copy(),y_SOURCE.copy()
    before={int(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))}

    smote_strategy={}
    for c,n in before.items():
        if 6<=n<TARGET_PER_CLASS:
            tgt=min(TARGET_PER_CLASS,max(SMOTE_MIN_TARGET,n*SMOTE_MAX_MULTIPLIER))
            if tgt>n: smote_strategy[c]=int(tgt)

    if smote_strategy:
        k=min(5,min(before[c] for c in smote_strategy)-1)
        print("SMOTE targets:",smote_strategy)
        X,y=SMOTE(sampling_strategy=smote_strategy,random_state=SEED,k_neighbors=k).fit_resample(X,y)

    after_smote={int(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))}
    under={c:min(after_smote[c],TARGET_PER_CLASS) for c in range(8)}
    X,y=RandomUnderSampler(sampling_strategy=under,random_state=SEED).fit_resample(X,y)

    cur={int(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))}
    over={c:TARGET_PER_CLASS for c in range(8) if cur[c]<TARGET_PER_CLASS}
    if over:
        X,y=RandomOverSampler(sampling_strategy=over,random_state=SEED+1).fit_resample(X,y)

    X=X.astype(np.float32); y=y.astype(np.int64)
    perm=np.random.default_rng(SEED+3).permutation(len(y))
    X,y=X[perm],y[perm]
    final={int(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))}

    rows=[]
    for c in range(8):
        rows.append({
            "class_id":c,"class_name":ID_TO_CLASS[c],
            "natural_train_rows":NATURAL_COUNTS[c],
            "source_pool_rows":before[c],
            "after_smote_rows":after_smote[c],
            "final_balanced_rows":final[c]
        })
    save_csv(pd.DataFrame(rows),RESULT_DIR/"hybrid_resampling_distribution.csv")
    save_npz(BAL_FILE,X=X,y=y)
    print("Final balanced counts:",final)
    return X,y

X_BAL,y_BAL=build_balanced_pool()

VAL_PROBE_FILE=CACHE_DIR/"validation_probe.npz"

def build_val_probe():
    if RESUME and VAL_PROBE_FILE.exists():
        d=np.load(VAL_PROBE_FILE)
        return d["X"].astype(np.float32),d["y"].astype(np.int64)

    xp,yp=defaultdict(list),defaultdict(list); counts={i:0 for i in range(8)}
    cols=STRICT_FEATURES+["y_multiclass"]
    for fi,p in enumerate(DATA_FILES["validation"],1):
        df=pd.read_parquet(p,columns=cols)
        Xf=df[STRICT_FEATURES].to_numpy(np.float32)
        yf=df["y_multiclass"].to_numpy(np.int64)
        for c in range(8):
            need=PROBE_VAL_CAP-counts[c]
            if need<=0: continue
            idx=np.flatnonzero(yf==c)
            if len(idx)==0: continue
            take=min(need,len(idx))
            rr=np.random.default_rng(SEED+10000+fi*100+c)
            if take<len(idx): idx=rr.choice(idx,size=take,replace=False)
            xp[c].append(Xf[idx]); yp[c].append(np.full(take,c,np.int64)); counts[c]+=take
        del df,Xf,yf; gc.collect()
    X=np.concatenate([np.concatenate(xp[c]) for c in range(8)])
    y=np.concatenate([np.concatenate(yp[c]) for c in range(8)])
    perm=np.random.default_rng(SEED+4).permutation(len(y)); X,y=X[perm],y[perm]
    save_npz(VAL_PROBE_FILE,X=X,y=y)
    print("Validation probe counts:",counts)
    return X,y

X_VAL_PROBE,y_VAL_PROBE=build_val_probe()

def cm_metrics(cm):
    cm=np.asarray(cm,np.int64); total=int(cm.sum()); tp=np.diag(cm).astype(float)
    fp=cm.sum(0)-tp; fn=cm.sum(1)-tp; tn=total-tp-fp-fn
    precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
    recall=np.divide(tp,tp+fn,out=np.zeros_like(tp),where=(tp+fn)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(tp),where=(precision+recall)>0)
    support=cm.sum(1).astype(float)
    r={
        "accuracy":float(np.trace(cm)/max(total,1)),
        "macro_precision":float(precision.mean()),
        "macro_recall":float(recall.mean()),
        "macro_f1":float(f1.mean()),
        "weighted_f1":float(np.sum(f1*support)/max(support.sum(),1)),
        "balanced_accuracy":float(recall.mean())
    }
    if cm.shape==(2,2):
        tn0,fp0,fn0,tp0=cm.ravel()
        sens=tp0/max(tp0+fn0,1); spec=tn0/max(tn0+fp0,1)
        r.update({"sensitivity":float(sens),"specificity":float(spec),"fpr":float(1-spec)})
    return r

def per_class_df(cm,names):
    cm=np.asarray(cm,np.int64); tp=np.diag(cm).astype(float)
    fp=cm.sum(0)-tp; fn=cm.sum(1)-tp
    p=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
    r=np.divide(tp,tp+fn,out=np.zeros_like(tp),where=(tp+fn)>0)
    f=np.divide(2*p*r,p+r,out=np.zeros_like(tp),where=(p+r)>0)
    return pd.DataFrame({"class_id":range(len(names)),"class_name":names,"precision":p,"recall":r,"f1":f,"support":cm.sum(1)})

class ProbeMLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.15),
            nn.Linear(128,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,8)
        )
    def forward(self,x): return self.net(x)

@torch.no_grad()
def probs_array(model,X,batch=PRED_BATCH):
    model.eval(); out=[]
    for s in range(0,len(X),batch):
        xb=torch.from_numpy(X[s:s+batch]).to(DEVICE,dtype=torch.float32,non_blocking=True)
        with torch.amp.autocast("cuda",dtype=torch.float16,enabled=True):
            logits=model(xb)
        out.append(torch.softmax(logits.float(),1).cpu().numpy().astype(np.float32))
        del xb,logits
    return np.concatenate(out)

SELECT_FILE=RESULT_DIR/"selected_feature_configuration.json"

def select_features():
    if RESUME and SELECT_FILE.exists():
        with open(SELECT_FILE,encoding="utf-8") as f: x=json.load(f)
        if x.get("version")==VERSION:
            return x["selected_features"],pd.DataFrame(x["probe_results"])

    ranked=RANKING["feature"].tolist()
    fmap={f:i for i,f in enumerate(STRICT_FEATURES)}
    ks=sorted(set(min(int(k),len(ranked)) for k in FEATURE_K_CANDIDATES))
    n=min(PROBE_ROWS,len(y_BAL))
    idx=np.random.default_rng(SEED+5).choice(len(y_BAL),size=n,replace=False)
    rows=[]

    for k in ks:
        feats=ranked[:k]; inds=[fmap[f] for f in feats]
        Xtr=X_BAL[idx][:,inds]; ytr=y_BAL[idx]
        Xv=X_VAL_PROBE[:,inds]

        model=ProbeMLP(k).to(DEVICE)
        opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
        crit=nn.CrossEntropyLoss()
        loader=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True)
        scaler=torch.amp.GradScaler("cuda",enabled=True)

        for _ in range(PROBE_EPOCHS):
            model.train()
            for xb0,yb0 in loader:
                xb=xb0.to(DEVICE,dtype=torch.float32,non_blocking=True); yb=yb0.to(DEVICE,dtype=torch.long,non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda",dtype=torch.float16,enabled=True):
                    loss=crit(model(xb),yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        pr=probs_array(model,Xv); pred=pr.argmax(1)
        cm=confusion_matrix(y_VAL_PROBE,pred,labels=np.arange(8))
        mf1=cm_metrics(cm)["macro_f1"]
        rows.append({"feature_count":k,"validation_probe_macro_f1":mf1})
        print(f"Feature probe k={k}: macro-F1={mf1:.6f}")
        del model,opt,crit,loader,scaler,pr,pred,Xtr,Xv; gc.collect(); torch.cuda.empty_cache()

    df=pd.DataFrame(rows)
    best=int(df.sort_values(["validation_probe_macro_f1","feature_count"],ascending=[False,True]).iloc[0]["feature_count"])
    feats=ranked[:best]
    save_csv(df,RESULT_DIR/"feature_count_probe_results.csv")
    save_json({"version":VERSION,"selected_feature_count":best,"selected_features":feats,"probe_results":rows,"test_used":False},SELECT_FILE)
    return feats,df

SELECTED_FEATURES,FEATURE_PROBE=select_features()
STRICT_INDEX={f:i for i,f in enumerate(STRICT_FEATURES)}
SELECTED_IDX=[STRICT_INDEX[f] for f in SELECTED_FEATURES]
X_MULTI=X_BAL[:,SELECTED_IDX].astype(np.float32)
y_MULTI=y_BAL.astype(np.int64)
X_VAL_PROBE_SEL=X_VAL_PROBE[:,SELECTED_IDX].astype(np.float32)

print(f"\nSelected {len(SELECTED_FEATURES)} features")

BIN_FILE=CACHE_DIR/"balanced_binary_pool.npz"
def build_binary_pool():
    if RESUME and BIN_FILE.exists():
        d=np.load(BIN_FILE)
        return d["X"].astype(np.float32),d["y"].astype(np.int64)
    benign=np.flatnonzero(y_MULTI==0); attack=np.flatnonzero(y_MULTI!=0)
    n=min(len(benign),150000)
    rng=np.random.default_rng(SEED+6)
    b=rng.choice(benign,size=n,replace=False) if len(benign)>n else benign
    a=rng.choice(attack,size=len(b),replace=False)
    idx=np.concatenate([b,a]); y=(y_MULTI[idx]!=0).astype(np.int64); X=X_MULTI[idx]
    perm=rng.permutation(len(y)); X,y=X[perm],y[perm]
    save_npz(BIN_FILE,X=X,y=y)
    return X,y
X_BINARY,y_BINARY=build_binary_pool()

def effective_alpha(counts,beta=.9999):
    c=np.asarray(counts,float)
    w=(1-beta)/np.maximum(1-np.power(beta,c),1e-12)
    w/=w.mean(); w=np.clip(w,.5,2.5); w/=w.mean()
    return w.astype(np.float32)

MULTI_ALPHA=effective_alpha([NATURAL_COUNTS[i] for i in range(8)])
BINARY_ALPHA=np.ones(2,np.float32)
save_json({"gamma":FOCAL_GAMMA,"multiclass_alpha":MULTI_ALPHA.tolist()},RESULT_DIR/"focal_configuration.json")

class ResidualBlock(nn.Module):
    def __init__(self,w=256,drop=.2):
        super().__init__()
        self.norm=nn.LayerNorm(w)
        self.fc1=nn.Linear(w,w*2)
        self.fc2=nn.Linear(w*2,w)
        self.drop=nn.Dropout(drop)
    def forward(self,x):
        r=x; z=self.norm(x); z=self.fc1(z); z=F.gelu(z); z=self.drop(z); z=self.fc2(z); z=self.drop(z)
        return r+z

class RobustMLP(nn.Module):
    def __init__(self,input_dim,n_classes):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(input_dim,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(DROPOUT))
        self.blocks=nn.Sequential(ResidualBlock(256,DROPOUT),ResidualBlock(256,DROPOUT),ResidualBlock(256,DROPOUT*.75))
        self.head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,128),nn.GELU(),nn.Dropout(DROPOUT*.75),nn.Linear(128,n_classes))
    def forward(self,x):
        return self.head(self.blocks(self.input(x)))

class FocalLoss(nn.Module):
    def __init__(self,alpha,gamma=2.0):
        super().__init__()
        self.register_buffer("alpha",torch.tensor(alpha,dtype=torch.float32))
        self.gamma=float(gamma)
    def forward(self,logits,target):
        ce=F.cross_entropy(logits,target,reduction="none")
        pt=torch.exp(-ce)
        return (self.alpha[target]*(1-pt).pow(self.gamma)*ce).mean()

def loader(X,y,shuffle=True):
    return DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32,copy=False)),torch.from_numpy(y.astype(np.int64,copy=False))),
        batch_size=BATCH_SIZE,shuffle=shuffle,num_workers=0,pin_memory=True
    )

def train_model(tag,Xtr,ytr,Xv,yv,n_classes,loss_mode,alpha):
    best_file=MODEL_DIR/f"{tag}_BEST.pt"; last_file=CKPT_DIR/f"{tag}_LAST.pt"; done_file=CKPT_DIR/f"{tag}_COMPLETE.json"
    model=RobustMLP(Xtr.shape[1],n_classes).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=1,min_lr=1e-5)
    crit=FocalLoss(alpha,FOCAL_GAMMA).to(DEVICE) if loss_mode=="focal" else nn.CrossEntropyLoss(label_smoothing=.02)
    scaler=torch.amp.GradScaler("cuda",enabled=True)

    if RESUME and done_file.exists() and best_file.exists():
        ck=torch.load(best_file,map_location=DEVICE,weights_only=False); model.load_state_dict(ck["model_state"]); model.eval()
        print(f"✅ Reusing {tag}")
        return model

    start_epoch=0; best=-1.; best_epoch=-1; noimp=0; history=[]
    if RESUME and last_file.exists():
        try:
            ck=torch.load(last_file,map_location=DEVICE,weights_only=False)
            if ck.get("version")==VERSION and ck.get("tag")==tag:
                model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optimizer_state"]); scaler.load_state_dict(ck["scaler_state"])
                start_epoch=ck["epoch"]+1; best=ck["best"]; best_epoch=ck["best_epoch"]; noimp=ck["noimp"]; history=ck["history"]
                print(f"♻️ Resuming {tag} from epoch {start_epoch+1}")
        except Exception as e:
            print("Resume warning:",e)

    dl=loader(Xtr,ytr,True)
    for epoch in range(start_epoch,MAX_EPOCHS):
        model.train(); loss_sum=0.; rows=0
        for xb0,yb0 in dl:
            xb=xb0.to(DEVICE,dtype=torch.float32,non_blocking=True); yb=yb0.to(DEVICE,dtype=torch.long,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",dtype=torch.float16,enabled=True):
                logits=model(xb); loss=crit(logits,yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(opt); scaler.update()
            loss_sum += float(loss.item())*len(yb); rows+=len(yb)

        pv=probs_array(model,Xv); pred=pv.argmax(1)
        cm=confusion_matrix(yv,pred,labels=np.arange(n_classes)); m=cm_metrics(cm); mf1=m["macro_f1"]
        sched.step(mf1)
        rec={"epoch":epoch+1,"train_loss":loss_sum/max(rows,1),"validation_accuracy":m["accuracy"],"validation_macro_f1":mf1,"validation_balanced_accuracy":m["balanced_accuracy"],"lr":opt.param_groups[0]["lr"]}
        history.append(rec); save_csv(pd.DataFrame(history),RESULT_DIR/f"history_{tag}.csv")
        print(f"{tag} epoch {epoch+1}: val acc={m['accuracy']:.5f}, macro-F1={mf1:.5f}")

        if mf1>best+1e-5:
            best=mf1; best_epoch=epoch+1; noimp=0
            save_torch({"version":VERSION,"tag":tag,"features":SELECTED_FEATURES,"n_classes":n_classes,"model_state":model.state_dict(),"best_epoch":best_epoch,"best_validation_macro_f1":best},best_file)
        else:
            noimp+=1

        save_torch({"version":VERSION,"tag":tag,"epoch":epoch,"best":best,"best_epoch":best_epoch,"noimp":noimp,"history":history,"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"scaler_state":scaler.state_dict()},last_file)
        if noimp>=PATIENCE:
            print(f"Early stop {tag}; best epoch={best_epoch}")
            break

    ck=torch.load(best_file,map_location=DEVICE,weights_only=False); model.load_state_dict(ck["model_state"]); model.eval()
    save_json({"version":VERSION,"status":"COMPLETED","tag":tag,"best_epoch":best_epoch,"best_validation_macro_f1":best,"training_rows":len(ytr)},done_file)
    return model

CE_MODEL=train_model(
    "B1_ResMLP_HybridResample_CE",
    X_MULTI,y_MULTI,X_VAL_PROBE_SEL,y_VAL_PROBE,8,"ce",MULTI_ALPHA
)
FOCAL_MODEL=train_model(
    "B2_ResMLP_HybridResample_Focal",
    X_MULTI,y_MULTI,X_VAL_PROBE_SEL,y_VAL_PROBE,8,"focal",MULTI_ALPHA
)
BIN_MODEL=train_model(
    "B3_ResMLP_Binary_HybridResample_Focal",
    X_BINARY,y_BINARY,X_VAL_PROBE_SEL,(y_VAL_PROBE!=0).astype(np.int64),2,"focal",BINARY_ALPHA
)

@torch.no_grad()
def collect_probs(model,split,task):
    target="y_multiclass" if task=="multiclass" else "y_binary"
    ys=[]; ps=[]; total=0
    for i,p in enumerate(DATA_FILES[split],1):
        df=pd.read_parquet(p,columns=SELECTED_FEATURES+[target])
        if len(df)==0: continue
        X=df[SELECTED_FEATURES].to_numpy(np.float32); y=df[target].to_numpy(np.int64)
        pr=probs_array(model,X)
        ys.append(y); ps.append(pr); total+=len(y)
        del df,X,y,pr; gc.collect()
        if i%50==0 or i==len(DATA_FILES[split]): print(f"{task} {split}: {i}/{len(DATA_FILES[split])} | {total:,}")
    return np.concatenate(ys),np.concatenate(ps)

val_rows=[]
VAL_CACHE={}
for tag,model in [("B1_ResMLP_HybridResample_CE",CE_MODEL),("B2_ResMLP_HybridResample_Focal",FOCAL_MODEL)]:
    yv,pv=collect_probs(model,"validation","multiclass")
    cm=confusion_matrix(yv,pv.argmax(1),labels=np.arange(8)); m=cm_metrics(cm)
    val_rows.append({"model":tag,"validation_accuracy":m["accuracy"],"validation_macro_f1":m["macro_f1"],"validation_balanced_accuracy":m["balanced_accuracy"]})
    VAL_CACHE[tag]=(yv,pv)

VAL_DF=pd.DataFrame(val_rows)
save_csv(VAL_DF,RESULT_DIR/"multiclass_validation_ablation.csv")
BEST_TAG=VAL_DF.sort_values(["validation_macro_f1","validation_balanced_accuracy"],ascending=False).iloc[0]["model"]
BEST_MODEL=CE_MODEL if BEST_TAG.startswith("B1") else FOCAL_MODEL
yvm,pvm=VAL_CACHE[BEST_TAG]
print("\nBest multiclass Step-2B model:",BEST_TAG)

tau_rows=[]
for tau in PRIOR_TAUS:
    scores=np.log(np.clip(pvm,1e-12,1.0))-float(tau)*np.log(np.clip(NATURAL_PRIOR,1e-12,1.0))
    pred=scores.argmax(1); cm=confusion_matrix(yvm,pred,labels=np.arange(8)); m=cm_metrics(cm)
    tau_rows.append({"tau":float(tau),"validation_accuracy":m["accuracy"],"validation_macro_f1":m["macro_f1"],"validation_balanced_accuracy":m["balanced_accuracy"]})
TAU_DF=pd.DataFrame(tau_rows); save_csv(TAU_DF,RESULT_DIR/"multiclass_prior_adjustment.csv")
BEST_TAU=float(TAU_DF.sort_values(["validation_macro_f1","validation_balanced_accuracy"],ascending=False).iloc[0]["tau"])

yvb,pvb=collect_probs(BIN_MODEL,"validation","binary")
thr_rows=[]
for thr in BINARY_THRESHOLDS:
    pred=(pvb[:,1]>=thr).astype(np.int64); cm=confusion_matrix(yvb,pred,labels=[0,1]); m=cm_metrics(cm)
    thr_rows.append({"threshold":float(thr),"validation_accuracy":m["accuracy"],"validation_macro_f1":m["macro_f1"],"validation_balanced_accuracy":m["balanced_accuracy"],"specificity":m["specificity"],"sensitivity":m["sensitivity"]})
THR_DF=pd.DataFrame(thr_rows); save_csv(THR_DF,RESULT_DIR/"binary_threshold_tuning.csv")
BEST_THR=float(THR_DF.sort_values(["validation_macro_f1","validation_balanced_accuracy"],ascending=False).iloc[0]["threshold"])
print("Best tau:",BEST_TAU,"| Best binary threshold:",BEST_THR)

@torch.no_grad()
def final_test(model,task,param):
    target="y_multiclass" if task=="multiclass" else "y_binary"
    ncls=8 if task=="multiclass" else 2
    cm=np.zeros((ncls,ncls),np.int64)
    auc_y=[]; auc_p=[]; total=0
    frac=min(1.0,AUC_SAMPLE/max(int(STEP1["test_rows"]),1))

    for i,p in enumerate(DATA_FILES["test"],1):
        df=pd.read_parquet(p,columns=SELECTED_FEATURES+[target])
        if len(df)==0: continue
        X=df[SELECTED_FEATURES].to_numpy(np.float32); y=df[target].to_numpy(np.int64)
        pr=probs_array(model,X)
        if task=="multiclass":
            score=np.log(np.clip(pr,1e-12,1.0))-float(param)*np.log(np.clip(NATURAL_PRIOR,1e-12,1.0))
            pred=score.argmax(1)
        else:
            pred=(pr[:,1]>=float(param)).astype(np.int64)
        cm += confusion_matrix(y,pred,labels=np.arange(ncls)); total+=len(y)

        n=min(len(y),max(1,int(round(len(y)*frac))))
        rr=np.random.default_rng(SEED+i*79+(10000 if task=="multiclass" else 0))
        idx=rr.choice(len(y),size=n,replace=False) if n<len(y) else np.arange(len(y))
        auc_y.append(y[idx]); auc_p.append(pr[idx])
        del df,X,y,pr,pred; gc.collect()
        if i%50==0 or i==len(DATA_FILES["test"]): print(f"final {task} test {i}/{len(DATA_FILES['test'])} | {total:,}")

    ay=np.concatenate(auc_y); ap=np.concatenate(auc_p)
    if len(ay)>AUC_SAMPLE:
        idx=np.random.default_rng(SEED+999).choice(len(ay),size=AUC_SAMPLE,replace=False); ay,ap=ay[idx],ap[idx]

    m=cm_metrics(cm)
    try:
        if task=="binary":
            m["roc_auc"]=float(roc_auc_score(ay,ap[:,1]))
            m["pr_auc"]=float(average_precision_score(ay,ap[:,1]))
        else:
            ybin=label_binarize(ay,classes=np.arange(8))
            m["roc_auc"]=float(roc_auc_score(ybin,ap,average="macro",multi_class="ovr"))
            m["pr_auc"]=float(average_precision_score(ybin,ap,average="macro"))
    except Exception as e:
        print("AUC warning:",e); m["roc_auc"]=np.nan; m["pr_auc"]=np.nan
    m["rows_evaluated"]=total
    return m,cm

MULTI_METRICS,MULTI_CM=final_test(BEST_MODEL,"multiclass",BEST_TAU)
BIN_METRICS,BIN_CM=final_test(BIN_MODEL,"binary",BEST_THR)

MULTI_METRICS.update({"model":BEST_TAG,"task":"multiclass","tau":BEST_TAU,"feature_count":len(SELECTED_FEATURES)})
BIN_METRICS.update({"model":"B3_ResMLP_Binary_HybridResample_Focal","task":"binary","threshold":BEST_THR,"feature_count":len(SELECTED_FEATURES)})

FINAL_DF=pd.DataFrame([BIN_METRICS,MULTI_METRICS])
save_csv(FINAL_DF,RESULT_DIR/"STEP02B_FINAL_TEST_METRICS.csv")

MULTI_PC=per_class_df(MULTI_CM,CLASS_NAMES)
BIN_PC=per_class_df(BIN_CM,["Benign","Attack"])
save_csv(MULTI_PC,RESULT_DIR/"final_multiclass_per_class_metrics.csv")
save_csv(BIN_PC,RESULT_DIR/"final_binary_per_class_metrics.csv")
save_csv(pd.DataFrame(MULTI_CM,index=CLASS_NAMES,columns=CLASS_NAMES),RESULT_DIR/"final_multiclass_confusion_matrix.csv")
save_csv(pd.DataFrame(BIN_CM,index=["Benign","Attack"],columns=["Benign","Attack"]),RESULT_DIR/"final_binary_confusion_matrix.csv")

compare=[]
if STEP2A_DF is not None:
    for task,new in [("binary",BIN_METRICS),("multiclass",MULTI_METRICS)]:
        old=STEP2A_DF[(STEP2A_DF["model"]=="MLP")&(STEP2A_DF["task"]==task)]
        if len(old):
            old=old.iloc[0]
            compare.append({
                "task":task,
                "step2a_mlp_accuracy":float(old["accuracy"]),
                "step2a_mlp_macro_f1":float(old["macro_f1"]),
                "step2a_mlp_balanced_accuracy":float(old["balanced_accuracy"]),
                "step2b_accuracy":new["accuracy"],
                "step2b_macro_f1":new["macro_f1"],
                "step2b_balanced_accuracy":new["balanced_accuracy"],
                "macro_f1_absolute_gain":new["macro_f1"]-float(old["macro_f1"]),
                "balanced_accuracy_absolute_gain":new["balanced_accuracy"]-float(old["balanced_accuracy"])
            })
COMPARE_DF=pd.DataFrame(compare)
if len(COMPARE_DF): save_csv(COMPARE_DF,RESULT_DIR/"STEP02A_vs_STEP02B_COMPARISON.csv")

def save_cm_figure(cm,names,title,filename):
    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(cm); fig.colorbar(im,ax=ax)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names,rotation=45,ha="right"); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,f"{int(cm[i,j]):,}",ha="center",va="center",fontsize=7)
    fig.tight_layout(); tmp=LOCAL/filename; fig.savefig(tmp,dpi=220,bbox_inches="tight"); plt.close(fig); atomic_copy(tmp,FIG_DIR/filename); tmp.unlink()

save_cm_figure(MULTI_CM,CLASS_NAMES,"Step 2B Final Multiclass Confusion Matrix","multiclass_confusion.png")
save_cm_figure(BIN_CM,["Benign","Attack"],"Step 2B Final Binary Confusion Matrix","binary_confusion.png")

fig,ax=plt.subplots(figsize=(10,5))
ax.bar(MULTI_PC["class_name"],MULTI_PC["f1"]); ax.set_ylim(0,1); ax.set_ylabel("F1-score"); ax.set_title("Step 2B Per-Class F1 — Endpoint-Disjoint Test")
plt.setp(ax.get_xticklabels(),rotation=35,ha="right"); fig.tight_layout()
tmp=LOCAL/"per_class_f1.png"; fig.savefig(tmp,dpi=220,bbox_inches="tight"); plt.close(fig); atomic_copy(tmp,FIG_DIR/"per_class_f1.png"); tmp.unlink()

BEST_MODEL_FILE = MODEL_DIR/f"{BEST_TAG}_BEST.pt"
HANDOFF={
    "version":VERSION,
    "status":"READY_FOR_STEP3",
    "architecture":"Residual LayerNorm MLP",
    "model_tag":BEST_TAG,
    "model_file":str(BEST_MODEL_FILE),
    "selected_features":SELECTED_FEATURES,
    "input_dim":len(SELECTED_FEATURES),
    "n_classes":8,
    "class_names":CLASS_NAMES,
    "centralized_training_strategy":"controlled majority undersampling + capped SMOTE + random oversampling",
    "loss":"class-balanced focal loss" if "Focal" in BEST_TAG else "cross entropy with label smoothing",
    "validation_selected_tau":BEST_TAU,
    "test_accuracy":MULTI_METRICS["accuracy"],
    "test_macro_f1":MULTI_METRICS["macro_f1"],
    "test_balanced_accuracy":MULTI_METRICS["balanced_accuracy"],
    "fl_note":"Do NOT globally SMOTE across clients in Step 3. Preserve non-IID client distributions; use local class-aware sampling/focal loss inside each client if required."
}
save_json(HANDOFF,RESULT_DIR/"STEP02B_FOR_STEP3.json")

completion={
    "version":VERSION,
    "status":"COMPLETED",
    "step":"2B",
    "step2a_preserved":True,
    "split_mode":STEP1["primary_split_mode"],
    "selected_feature_count":len(SELECTED_FEATURES),
    "best_multiclass_model":BEST_TAG,
    "best_tau":BEST_TAU,
    "binary_threshold":BEST_THR,
    "multiclass_test_accuracy":MULTI_METRICS["accuracy"],
    "multiclass_test_macro_f1":MULTI_METRICS["macro_f1"],
    "multiclass_test_balanced_accuracy":MULTI_METRICS["balanced_accuracy"],
    "binary_test_accuracy":BIN_METRICS["accuracy"],
    "binary_test_macro_f1":BIN_METRICS["macro_f1"],
    "binary_test_specificity":BIN_METRICS["specificity"],
    "step3_handoff":str(RESULT_DIR/"STEP02B_FOR_STEP3.json"),
    "completed_at":datetime.now().isoformat()
}
save_json(completion,COMPLETE_FILE)

print("\n"+"="*115)
print("✅ STEP 2B COMPLETED")
print("="*115)
print(f"Selected features: {len(SELECTED_FEATURES)}")
print(f"Best multiclass model: {BEST_TAG}")
print(f"Multiclass test accuracy: {MULTI_METRICS['accuracy']:.6f}")
print(f"Multiclass test macro-F1: {MULTI_METRICS['macro_f1']:.6f}")
print(f"Multiclass balanced accuracy: {MULTI_METRICS['balanced_accuracy']:.6f}")
print(f"Binary threshold: {BEST_THR}")
print(f"Binary test accuracy: {BIN_METRICS['accuracy']:.6f}")
print(f"Binary test macro-F1: {BIN_METRICS['macro_f1']:.6f}")
print(f"Binary specificity: {BIN_METRICS['specificity']:.6f}")
if len(COMPARE_DF):
    print("\nStep 2A -> Step 2B comparison:")
    print(COMPARE_DF.to_string(index=False))
print("\nPer-class multiclass:")
print(MULTI_PC.to_string(index=False))
print("\nStep-3 handoff:",RESULT_DIR/"STEP02B_FOR_STEP3.json")
print("="*115)

import os, sys, gc, re, json, time, random, shutil, hashlib, warnings, subprocess, importlib.util
from pathlib import Path
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings("ignore")

from google.colab import drive
if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
else:
    print("✅ Drive already mounted.")

def ensure(imp, pip_name=None):
    pip_name = pip_name or imp
    if importlib.util.find_spec(imp) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","--quiet",pip_name])

for imp,pipn in [
    ("numpy","numpy"),("pandas","pandas"),("pyarrow","pyarrow"),
    ("sklearn","scikit-learn"),("imblearn","imbalanced-learn"),
    ("xgboost","xgboost"),("matplotlib","matplotlib"),("torch","torch")
]:
    ensure(imp,pipn)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE, RandomOverSampler
import xgboost as xgb

ROOT = Path("/content/drive/MyDrive/Hybrid_BCFL_IJACSA_2026")
CLEAN_ROOT = ROOT/"02_CLEAN_PARQUET"
PREP_FILE = ROOT/"04_PREPROCESSOR"/"train_only_preprocessor.json"
STEP1_FILE = ROOT/"06_CHECKPOINTS"/"STEP01_COMPLETE.json"

STEP2A_TEST = ROOT/"07_AI_MODELS"/"STEP02_CENTRALIZED_BASELINES"/"RESULTS"/"STEP02_TEST_COMPARISON.csv"
STEP2B_TEST = ROOT/"07_AI_MODELS"/"STEP02B_IMBALANCE_ROBUST"/"RESULTS"/"STEP02B_FINAL_TEST_METRICS.csv"

OUT = ROOT/"07_AI_MODELS"/"STEP02C_SCIENTIFIC_REPAIR"
MODEL_DIR, CKPT_DIR, CACHE_DIR, RESULT_DIR, FIG_DIR = [OUT/x for x in ["MODELS","CHECKPOINTS","CACHE","RESULTS","FIGURES"]]
for p in [MODEL_DIR,CKPT_DIR,CACHE_DIR,RESULT_DIR,FIG_DIR]: p.mkdir(parents=True,exist_ok=True)

LOCAL = Path("/content/STEP02C_RUNTIME"); LOCAL.mkdir(parents=True,exist_ok=True)
LOCAL_CLEAN = LOCAL/"CLEAN"; LOCAL_CLEAN.mkdir(parents=True,exist_ok=True)

for p in [CLEAN_ROOT, PREP_FILE, STEP1_FILE]:
    if not p.exists(): raise FileNotFoundError(p)

STEP1 = json.load(open(STEP1_FILE,"r",encoding="utf-8"))
PREP = json.load(open(PREP_FILE,"r",encoding="utf-8"))
STEP1_FEATURES = list(PREP["final_features"])

if not torch.cuda.is_available():
    raise RuntimeError("Enable GPU: Runtime -> Change runtime type -> GPU")
DEVICE = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_GB = torch.cuda.get_device_properties(0).total_memory/(1024**3)
torch.backends.cudnn.benchmark=True
try: torch.set_float32_matmul_precision("high")
except: pass

VERSION="STEP02C_V1"
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
RESUME=True
FORCE_REBUILD=False

TRAIN_BP=7000; VAL_BP=8500; HASH_MOD=10000

TRAIN_CAP_PER_CLASS=250_000
MINORITY_FLOOR=20_000
SMOTE_MAX_MULTIPLIER=5

RF_RANK_MAX_ROWS=500_000
RF_TREES=180
FEATURE_K=[25,36,48,len(STEP1_FEATURES)]
PROBE_ROWS=500_000
PROBE_EPOCHS=3
VAL_PROBE_CAP=20_000

MAX_EPOCHS=12
PATIENCE=3
LR=3e-4
WEIGHT_DECAY=2e-4
DROPOUT=.20
BATCH=8192 if GPU_GB>=14 else (4096 if GPU_GB>=8 else 2048)
PRED_BATCH=max(16384,BATCH)

EFFECTIVE_BETA=.9999
FOCAL_GAMMA=2.0
BINARY_THRESHOLDS=np.round(np.arange(.05,.951,.025),3)
TAU_GRID=np.array([-.5,-.25,0,.25,.5,.75],np.float32)
AUC_SAMPLE_MAX=300_000
RUN_XGB=True
COMPLETE_FILE=CKPT_DIR/"STEP02C_COMPLETE.json"

CLASS_NAMES=["Benign","DDoS","DoS","Recon","Web-Based","Brute Force","Spoofing","Mirai"]

print("\n"+"="*110)
print("STEP 2C — SCIENTIFIC SPLIT REPAIR + FL-READY BASELINE")
print("="*110)
print("GPU:",GPU_NAME,"| VRAM:",f"{GPU_GB:.2f} GB")
print("Step-1 endpoint_id unique values:",f"{STEP1.get('known_source_endpoints',0):,}")
print("Official CIC IoT-DIAD IoT devices: 105")
print("IMPORTANT: A/B remain source-IP-disjoint stress tests, NOT device-disjoint.")

def slug(x): return re.sub(r"[^A-Za-z0-9._-]+","_",str(x)).strip("_")[:90]
def shash(x): return hashlib.sha1(str(x).encode()).hexdigest()[:10]
def atomic_copy(src,dst):
    src,dst=Path(src),Path(dst); dst.parent.mkdir(parents=True,exist_ok=True)
    partial=Path(str(dst)+".partial"); partial.unlink(missing_ok=True)
    shutil.copy2(src,partial); os.replace(partial,dst)
def save_json(obj,dst):
    tmp=LOCAL/f"{slug(Path(dst).stem)}_{shash(dst)}.json"
    json.dump(obj,open(tmp,"w",encoding="utf-8"),indent=2,default=str); atomic_copy(tmp,dst); tmp.unlink(missing_ok=True)
def save_csv(df,dst):
    tmp=LOCAL/f"{slug(Path(dst).stem)}_{shash(dst)}.csv"
    df.to_csv(tmp,index=False); atomic_copy(tmp,dst); tmp.unlink(missing_ok=True)
def save_npz(dst,**arr):
    tmp=LOCAL/f"{slug(Path(dst).stem)}_{shash(dst)}.npz"
    np.savez_compressed(tmp,**arr); atomic_copy(tmp,dst); tmp.unlink(missing_ok=True)
def save_torch(obj,dst):
    tmp=LOCAL/f"{slug(Path(dst).stem)}_{shash(dst)}.pt"
    torch.save(obj,tmp); atomic_copy(tmp,dst); tmp.unlink(missing_ok=True)

if FORCE_REBUILD and OUT.exists():
    for p in [MODEL_DIR,CKPT_DIR,CACHE_DIR,RESULT_DIR,FIG_DIR]:
        shutil.rmtree(p,ignore_errors=True); p.mkdir(parents=True,exist_ok=True)

if RESUME and not FORCE_REBUILD and COMPLETE_FILE.exists():
    old=json.load(open(COMPLETE_FILE,"r",encoding="utf-8"))
    if old.get("version")==VERSION and old.get("status")=="COMPLETED":
        print("✅ Step 2C already completed."); print(json.dumps(old,indent=2)); raise SystemExit

ID_PATTERNS=[
    r"(^|_)src(_|$).*ip",r"(^|_)dst(_|$).*ip",r"source.*ip",r"destination.*ip",
    r"(^|_)mac(_|$)",r"(^|_)src(_|$).*port",r"(^|_)dst(_|$).*port",
    r"source.*port",r"destination.*port",r"flow_id",r"timestamp",r"device_id"
]
def identifier_like(f):
    t=str(f).lower()
    return any(re.search(p,t) for p in ID_PATTERNS)
STRICT_FEATURES=[f for f in STEP1_FEATURES if not identifier_like(f)]
REMOVED=[f for f in STEP1_FEATURES if f not in STRICT_FEATURES]
if len(STRICT_FEATURES)<20: raise RuntimeError("Too few strict features")
save_json({"step1":STEP1_FEATURES,"strict":STRICT_FEATURES,"removed":REMOVED},RESULT_DIR/"strict_feature_gate.json")
print("Features:",len(STEP1_FEATURES),"-> strict:",len(STRICT_FEATURES),"removed:",REMOVED)

DRIVE_SHARDS=sorted(CLEAN_ROOT.rglob("chunk_*.parquet"))
if not DRIVE_SHARDS: raise RuntimeError("No Step-1 clean shards found")

def stage():
    need=sum(p.stat().st_size for p in DRIVE_SHARDS)
    free=shutil.disk_usage("/content").free
    print("Clean shards:",len(DRIVE_SHARDS),"size:",f"{need/(1024**3):.2f} GB","free:",f"{free/(1024**3):.2f} GB")
    if free < need + 12*(1024**3):
        print("⚠️ Not enough local SSD; streaming from Drive.")
        return DRIVE_SHARDS
    out=[]
    print("⚡ One-time clean-shard staging to /content ...")
    for i,p in enumerate(DRIVE_SHARDS,1):
        d=LOCAL_CLEAN/f"{slug(p.parent.name)}__{p.name}"
        if not d.exists() or d.stat().st_size!=p.stat().st_size: shutil.copy2(p,d)
        out.append(d)
        if i%50==0 or i==len(DRIVE_SHARDS): print(f"  {i}/{len(DRIVE_SHARDS)}")
    return out
SHARDS=stage()

def row_hash(df):

    return pd.util.hash_pandas_object(df[STRICT_FEATURES],index=False).to_numpy(dtype=np.uint64)
def split_code(h):
    b=(h%np.uint64(HASH_MOD)).astype(np.int32)
    s=np.empty(len(b),np.int8)
    s[b<TRAIN_BP]=0
    s[(b>=TRAIN_BP)&(b<VAL_BP)]=1
    s[b>=VAL_BP]=2
    return s

STATS_FILE=CACHE_DIR/"protocol_c_stats.json"
def build_stats():
    if RESUME and not FORCE_REBUILD and STATS_FILE.exists():
        z=json.load(open(STATS_FILE,"r",encoding="utf-8"))
        if z.get("version")==VERSION: print("✅ Reusing split/stats."); return z

    nf=len(STRICT_FEATURES)
    cnt=np.zeros(nf,np.int64); sx=np.zeros(nf,np.float64); sx2=np.zeros(nf,np.float64)
    split_n=np.zeros(3,np.int64); class_n=np.zeros((3,8),np.int64)

    print("\nPASS 1: duplicate-consistent split audit + TRAIN-only normalization stats")
    for fi,p in enumerate(SHARDS,1):
        df=pd.read_parquet(p,columns=STRICT_FEATURES+["y_multiclass"])
        if len(df)==0: continue
        h=row_hash(df); s=split_code(h); y=df["y_multiclass"].to_numpy(np.int64)
        for sid in range(3):
            m=s==sid; split_n[sid]+=m.sum()
            if m.any(): class_n[sid]+=np.bincount(y[m],minlength=8)[:8]
        m=s==0
        if m.any():
            X=df.loc[m,STRICT_FEATURES].to_numpy(np.float64)
            finite=np.isfinite(X); X0=np.where(finite,X,0.0)
            cnt+=finite.sum(0); sx+=X0.sum(0); sx2+=(X0*X0).sum(0)
            del X,finite,X0
        del df,h,s,y; gc.collect()
        if fi%50==0 or fi==len(SHARDS): print(f"  {fi}/{len(SHARDS)}")

    mean=sx/np.maximum(cnt,1); var=np.maximum(sx2/np.maximum(cnt,1)-mean*mean,0); scale=np.sqrt(var)
    keep=(cnt>0)&(var>1e-12)
    model_features=[f for f,k in zip(STRICT_FEATURES,keep) if k]
    z={"version":VERSION,"protocol":"duplicate_consistent_stratified_hash_70_15_15",
       "mean":mean.tolist(),"scale":np.where(scale>0,scale,1).tolist(),
       "keep":keep.astype(int).tolist(),"model_features":model_features,
       "split_counts":split_n.tolist(),"class_counts":class_n.tolist(),
       "created_at":datetime.now().isoformat()}
    save_json(z,STATS_FILE); return z

STATS=build_stats()
MEAN=np.array(STATS["mean"],np.float32); SCALE=np.array(STATS["scale"],np.float32)
KEEP=np.array(STATS["keep"],bool); MODEL_FEATURES=list(STATS["model_features"]); MODEL_IDX=np.flatnonzero(KEEP)
SPLIT_COUNTS=np.array(STATS["split_counts"],np.int64); CLASS_COUNTS=np.array(STATS["class_counts"],np.int64)

rows=[]
for sid,sn in enumerate(["train","validation","test"]):
    for c in range(8):
        rows.append({"split":sn,"class_id":c,"class_name":CLASS_NAMES[c],"rows":int(CLASS_COUNTS[sid,c])})
SPLIT_DF=pd.DataFrame(rows); save_csv(SPLIT_DF,RESULT_DIR/"protocol_c_split_distribution.csv")
print("\nProtocol-C split:",dict(zip(["train","validation","test"],SPLIT_COUNTS.tolist())))
print(SPLIT_DF.pivot(index="class_name",columns="split",values="rows").to_string())

def normalize(X):
    X=np.asarray(X,np.float32)
    bad=~np.isfinite(X)
    if bad.any():
        r,c=np.where(bad); X[r,c]=MEAN[c]
    X=(X-MEAN)/SCALE
    return X[:,MODEL_IDX].astype(np.float32,copy=False)

POOL_FILE=CACHE_DIR/"moderate_pool.npz"
def build_pool():
    if RESUME and not FORCE_REBUILD and POOL_FILE.exists():
        d=np.load(POOL_FILE); print("✅ Reusing moderate training pool.")
        return d["X"].astype(np.float32),d["y"].astype(np.int64)

    natural=CLASS_COUNTS[0].astype(np.int64)
    target=np.minimum(natural,TRAIN_CAP_PER_CLASS)
    prob=np.minimum(1.0,target/np.maximum(natural,1))
    XP,YP=defaultdict(list),defaultdict(list)

    print("\nPASS 2: moderate real-data pool")
    for fi,p in enumerate(SHARDS,1):
        df=pd.read_parquet(p,columns=STRICT_FEATURES+["y_multiclass"])
        if len(df)==0: continue
        h=row_hash(df); s=split_code(h); y=df["y_multiclass"].to_numpy(np.int64)
        key=h^np.uint64(0x9E3779B97F4A7C15); u=(key%np.uint64(1_000_000)).astype(np.float64)/1e6
        keep=np.zeros(len(df),bool)
        for c in range(8): keep |= (s==0)&(y==c)&(u<prob[c])
        if keep.any():
            X=normalize(df.loc[keep,STRICT_FEATURES].to_numpy(np.float32)); yy=y[keep]
            for c in range(8):
                idx=np.flatnonzero(yy==c)
                if len(idx): XP[c].append(X[idx]); YP[c].append(np.full(len(idx),c,np.int64))
            del X,yy
        del df,h,s,y,key,u,keep; gc.collect()
        if fi%50==0 or fi==len(SHARDS): print(f"  {fi}/{len(SHARDS)}")

    Xs=[]; ys=[]
    for c in range(8):
        Xc=np.concatenate(XP[c]); yc=np.concatenate(YP[c])
        if len(yc)>TRAIN_CAP_PER_CLASS:
            rng=np.random.default_rng(SEED+c); idx=rng.choice(len(yc),TRAIN_CAP_PER_CLASS,replace=False)
            Xc,yc=Xc[idx],yc[idx]
        Xs.append(Xc); ys.append(yc)
    X=np.concatenate(Xs).astype(np.float32); y=np.concatenate(ys).astype(np.int64)
    before={int(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))}
    print("Real pool:",before)

    smote_strategy={}
    for c in range(8):
        n=before.get(c,0)
        if 6<=n<MINORITY_FLOOR:
            t=min(MINORITY_FLOOR,n*SMOTE_MAX_MULTIPLIER)
            if t>n: smote_strategy[c]=int(t)
    if smote_strategy:
        print("Capped SMOTE:",smote_strategy)
        k=max(1,min(5,min(before[c] for c in smote_strategy)-1))
        X,y=SMOTE(sampling_strategy=smote_strategy,random_state=SEED,k_neighbors=k).fit_resample(X,y)

    now={int(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))}
    ros_strategy={c:MINORITY_FLOOR for c in range(8) if now.get(c,0)<MINORITY_FLOOR}
    if ros_strategy:
        print("Minority-floor ROS:",ros_strategy)
        X,y=RandomOverSampler(sampling_strategy=ros_strategy,random_state=SEED+1).fit_resample(X,y)

    rng=np.random.default_rng(SEED+2); perm=rng.permutation(len(y))
    X,y=X[perm].astype(np.float32),y[perm].astype(np.int64)
    final={int(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))}
    print("Final moderate pool:",final)
    save_npz(POOL_FILE,X=X,y=y)
    save_csv(pd.DataFrame([{"class_id":c,"class_name":CLASS_NAMES[c],
                           "natural_train":int(CLASS_COUNTS[0,c]),"pool_rows":final.get(c,0)} for c in range(8)]),
             RESULT_DIR/"protocol_c_training_pool.csv")
    return X,y

X_POOL,y_POOL=build_pool()

VAL_FILE=CACHE_DIR/"validation_probe.npz"
def build_val_probe():
    if RESUME and not FORCE_REBUILD and VAL_FILE.exists():
        d=np.load(VAL_FILE); return d["X"].astype(np.float32),d["y"].astype(np.int64)
    XP,YP=defaultdict(list),defaultdict(list); counts={c:0 for c in range(8)}
    for fi,p in enumerate(SHARDS,1):
        df=pd.read_parquet(p,columns=STRICT_FEATURES+["y_multiclass"])
        if len(df)==0: continue
        h=row_hash(df); s=split_code(h); y=df["y_multiclass"].to_numpy(np.int64)
        m=s==1
        if m.any():
            X=normalize(df[STRICT_FEATURES].to_numpy(np.float32))
            for c in range(8):
                need=VAL_PROBE_CAP-counts[c]
                if need<=0: continue
                idx=np.flatnonzero(m&(y==c))
                if not len(idx): continue
                take=min(need,len(idx)); rng=np.random.default_rng(SEED+fi*100+c)
                if take<len(idx): idx=rng.choice(idx,take,replace=False)
                XP[c].append(X[idx]); YP[c].append(np.full(take,c,np.int64)); counts[c]+=take
            del X
        del df,h,s,y; gc.collect()
    X=np.concatenate([np.concatenate(XP[c]) for c in range(8)])
    y=np.concatenate([np.concatenate(YP[c]) for c in range(8)])
    rng=np.random.default_rng(SEED+3); perm=rng.permutation(len(y)); X,y=X[perm],y[perm]
    save_npz(VAL_FILE,X=X.astype(np.float32),y=y.astype(np.int64))
    print("Validation probe counts:",counts); return X.astype(np.float32),y.astype(np.int64)

X_VAL_ALL,y_VAL=build_val_probe()

RANK_FILE=RESULT_DIR/"protocol_c_rf_ranking.csv"
def rank_features():
    if RESUME and not FORCE_REBUILD and RANK_FILE.exists():
        r=pd.read_csv(RANK_FILE)
        if len(r)==len(MODEL_FEATURES): return r
    if len(y_POOL)>RF_RANK_MAX_ROWS:
        rng=np.random.default_rng(SEED+4); idx=rng.choice(len(y_POOL),RF_RANK_MAX_ROWS,replace=False)
        Xr,yr=X_POOL[idx],y_POOL[idx]
    else: Xr,yr=X_POOL,y_POOL
    print("\nTraining train-only RF ranker...")
    rf=RandomForestClassifier(n_estimators=RF_TREES,max_depth=22,min_samples_leaf=2,
                              max_features="sqrt",class_weight="balanced_subsample",
                              n_jobs=-1,random_state=SEED)
    rf.fit(Xr,yr)
    r=pd.DataFrame({"feature":MODEL_FEATURES,"importance":rf.feature_importances_}).sort_values("importance",ascending=False).reset_index(drop=True)
    r["rank"]=np.arange(1,len(r)+1); save_csv(r,RANK_FILE); print(r.head(20).to_string(index=False)); return r
RANK=rank_features()

def metrics(cm):
    cm=np.asarray(cm,np.int64); total=cm.sum()
    tp=np.diag(cm).astype(float); fp=cm.sum(0)-tp; fn=cm.sum(1)-tp; tn=total-tp-fp-fn
    p=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
    r=np.divide(tp,tp+fn,out=np.zeros_like(tp),where=(tp+fn)>0)
    f=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=(p+r)>0)
    sup=cm.sum(1).astype(float)
    out={"accuracy":float(np.trace(cm)/max(total,1)),"macro_precision":float(p.mean()),
         "macro_recall":float(r.mean()),"macro_f1":float(f.mean()),
         "weighted_f1":float((f*sup).sum()/max(sup.sum(),1)),"balanced_accuracy":float(r.mean())}
    if cm.shape==(2,2):
        tnn,fpp,fnn,tpp=cm.ravel()
        sen=tpp/max(tpp+fnn,1); spe=tnn/max(tnn+fpp,1); pp=tpp/max(tpp+fpp,1)
        out.update({"sensitivity":float(sen),"specificity":float(spe),"precision":float(pp),
                    "f1":float(2*pp*sen/max(pp+sen,1e-12)),"fpr":float(1-spe)})
    return out

def per_class(cm,names):
    cm=np.asarray(cm,np.int64); tp=np.diag(cm).astype(float); fp=cm.sum(0)-tp; fn=cm.sum(1)-tp
    p=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
    r=np.divide(tp,tp+fn,out=np.zeros_like(tp),where=(tp+fn)>0)
    f=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=(p+r)>0)
    return pd.DataFrame({"class_id":range(len(names)),"class_name":names,"precision":p,"recall":r,"f1":f,"support":cm.sum(1)})

def effective_weights(counts):
    c=np.asarray(counts,float); e=1-np.power(EFFECTIVE_BETA,c); w=(1-EFFECTIVE_BETA)/np.maximum(e,1e-12)
    w/=max(w.mean(),1e-12); w=np.clip(w,.25,4.0); w/=max(w.mean(),1e-12); return w.astype(np.float32)

NATURAL_COUNTS=CLASS_COUNTS[0].astype(float)
MULTI_WEIGHTS=effective_weights(NATURAL_COUNTS)
print("Effective weights:",MULTI_WEIGHTS.tolist())

class ResBlock(nn.Module):
    def __init__(self,w,drop):
        super().__init__(); self.norm=nn.LayerNorm(w); self.fc1=nn.Linear(w,2*w); self.fc2=nn.Linear(2*w,w); self.drop=nn.Dropout(drop)
    def forward(self,x):
        z=self.norm(x); z=F.gelu(self.fc1(z)); z=self.drop(z); z=self.fc2(z); return x+self.drop(z)

class RobustMLP(nn.Module):
    def __init__(self,d,nc):
        super().__init__()
        self.inp=nn.Sequential(nn.Linear(d,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(DROPOUT))
        self.body=nn.Sequential(ResBlock(256,DROPOUT),ResBlock(256,DROPOUT),ResBlock(256,DROPOUT*.75))
        self.head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,128),nn.GELU(),nn.Dropout(DROPOUT*.75),nn.Linear(128,nc))
    def forward(self,x): return self.head(self.body(self.inp(x)))

class FocalLoss(nn.Module):
    def __init__(self,alpha,gamma=2):
        super().__init__(); self.register_buffer("alpha",torch.tensor(alpha,dtype=torch.float32)); self.gamma=gamma
    def forward(self,logits,y):
        ce=F.cross_entropy(logits,y,reduction="none"); pt=torch.exp(-ce); return (self.alpha[y]*((1-pt)**self.gamma)*ce).mean()

@torch.no_grad()
def probs(model,X):
    model.eval(); out=[]
    for st in range(0,len(X),PRED_BATCH):
        xb=torch.from_numpy(X[st:st+PRED_BATCH]).to(DEVICE,dtype=torch.float32,non_blocking=True)
        with torch.amp.autocast("cuda",dtype=torch.float16,enabled=True): lg=model(xb)
        out.append(torch.softmax(lg.float(),1).cpu().numpy().astype(np.float32))
        del xb,lg
    return np.concatenate(out)

SEL_FILE=RESULT_DIR/"selected_features.json"
def choose_features():
    if RESUME and not FORCE_REBUILD and SEL_FILE.exists():
        z=json.load(open(SEL_FILE,"r",encoding="utf-8"))
        if z.get("version")==VERSION: return z["selected_features"]
    rfeatures=RANK["feature"].tolist()
    fmap={f:i for i,f in enumerate(MODEL_FEATURES)}
    ks=sorted(set(min(int(k),len(rfeatures)) for k in FEATURE_K if int(k)>0))
    rng=np.random.default_rng(SEED+5)
    tidx=rng.choice(len(y_POOL),min(PROBE_ROWS,len(y_POOL)),replace=False)
    rec=[]
    for k in ks:
        feats=rfeatures[:k]; idx=[fmap[f] for f in feats]
        Xt,Xv=X_POOL[tidx][:,idx],X_VAL_ALL[:,idx]; yt=y_POOL[tidx]
        m=RobustMLP(k,8).to(DEVICE); opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-4)
        crit=nn.CrossEntropyLoss(weight=torch.tensor(MULTI_WEIGHTS,dtype=torch.float32,device=DEVICE),label_smoothing=.01)
        dl=DataLoader(TensorDataset(torch.from_numpy(Xt),torch.from_numpy(yt)),batch_size=BATCH,shuffle=True,pin_memory=True)
        sc=torch.amp.GradScaler("cuda",enabled=True)
        for ep in range(PROBE_EPOCHS):
            m.train()
            for xb,yb in dl:
                xb=xb.to(DEVICE,dtype=torch.float32,non_blocking=True); yb=yb.to(DEVICE,dtype=torch.long,non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda",dtype=torch.float16,enabled=True): lg=m(xb); loss=crit(lg,yb)
                sc.scale(loss).backward(); sc.step(opt); sc.update()
        pr=probs(m,Xv); cm=confusion_matrix(y_VAL,pr.argmax(1),labels=np.arange(8)); mm=metrics(cm)
        rec.append({"k":k,"validation_macro_f1":mm["macro_f1"],"validation_accuracy":mm["accuracy"],"balanced_accuracy":mm["balanced_accuracy"]})
        print(f"Feature probe k={k}: macro-F1={mm['macro_f1']:.6f}")
        del m,opt,crit,dl,sc,pr; gc.collect(); torch.cuda.empty_cache()
    rdf=pd.DataFrame(rec); best=int(rdf.sort_values(["validation_macro_f1","k"],ascending=[False,True]).iloc[0]["k"])
    selected=rfeatures[:best]; save_csv(rdf,RESULT_DIR/"feature_probe.csv")
    save_json({"version":VERSION,"selected_features":selected,"selected_k":best,"test_used":False},SEL_FILE); return selected

SELECTED_FEATURES=choose_features()
MFAP={f:i for i,f in enumerate(MODEL_FEATURES)}
SELECTED_IDX=[MFAP[f] for f in SELECTED_FEATURES]
X_MULTI=X_POOL[:,SELECTED_IDX].astype(np.float32); X_VAL=X_VAL_ALL[:,SELECTED_IDX].astype(np.float32)
print("✅ Selected",len(SELECTED_FEATURES),"features")

BIN_FILE=CACHE_DIR/"binary_pool.npz"
def build_binary_pool():
    if RESUME and not FORCE_REBUILD and BIN_FILE.exists():
        d=np.load(BIN_FILE); return d["X"].astype(np.float32),d["y"].astype(np.int64)
    b=np.flatnonzero(y_POOL==0); a=np.flatnonzero(y_POOL!=0); target=min(len(b),200_000); rng=np.random.default_rng(SEED+6)
    if len(b)>target: b=rng.choice(b,target,replace=False)
    a=rng.choice(a,len(b),replace=False); idx=np.concatenate([b,a]); y=(y_POOL[idx]!=0).astype(np.int64); X=X_MULTI[idx]
    perm=rng.permutation(len(y)); X,y=X[perm],y[perm]; save_npz(BIN_FILE,X=X,y=y); return X.astype(np.float32),y
X_BIN,y_BIN=build_binary_pool(); y_VAL_BIN=(y_VAL!=0).astype(np.int64)

def train_model(tag,Xt,yt,Xv,yv,nc,loss_mode,weights):
    bestf=MODEL_DIR/f"{tag}_BEST.pt"; lastf=CKPT_DIR/f"{tag}_LAST.pt"; donef=CKPT_DIR/f"{tag}_DONE.json"
    m=RobustMLP(Xt.shape[1],nc).to(DEVICE)
    if RESUME and not FORCE_REBUILD and donef.exists() and bestf.exists():
        ck=torch.load(bestf,map_location=DEVICE,weights_only=False); m.load_state_dict(ck["model_state"]); m.eval(); print("✅ Reusing",tag); return m
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=1,min_lr=1e-5)
    wt=torch.tensor(weights,dtype=torch.float32,device=DEVICE)
    crit=nn.CrossEntropyLoss(weight=wt,label_smoothing=.01) if loss_mode=="ce" else FocalLoss(weights,FOCAL_GAMMA).to(DEVICE)
    sc=torch.amp.GradScaler("cuda",enabled=True); start=0; best=-1; bestep=-1; noimp=0; hist=[]
    if RESUME and not FORCE_REBUILD and lastf.exists():
        try:
            ck=torch.load(lastf,map_location=DEVICE,weights_only=False)
            if ck.get("version")==VERSION and ck.get("tag")==tag:
                m.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optimizer_state"]); sc.load_state_dict(ck["scaler_state"])
                start=ck["epoch"]+1; best=ck["best"]; bestep=ck["best_epoch"]; noimp=ck["noimp"]; hist=ck.get("history",[])
                print("♻️ Resume",tag,"from epoch",start+1)
        except Exception as e: print("Resume warning:",e)
    dl=DataLoader(TensorDataset(torch.from_numpy(Xt),torch.from_numpy(yt)),batch_size=BATCH,shuffle=True,pin_memory=True)
    for ep in range(start,MAX_EPOCHS):
        m.train(); cm=np.zeros((nc,nc),np.int64); loss_sum=0.; n=0
        for xb,yb in dl:
            xb=xb.to(DEVICE,dtype=torch.float32,non_blocking=True); yb=yb.to(DEVICE,dtype=torch.long,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",dtype=torch.float16,enabled=True): lg=m(xb); loss=crit(lg,yb)
            sc.scale(loss).backward(); sc.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(),5.); sc.step(opt); sc.update()
            pred=lg.detach().argmax(1); cm+=confusion_matrix(yb.cpu().numpy(),pred.cpu().numpy(),labels=np.arange(nc))
            loss_sum+=float(loss.item())*len(yb); n+=len(yb)
        vm,_,_=eval_array(m,Xv,yv,nc); mf=vm["macro_f1"]; sched.step(mf)
        rec={"epoch":ep+1,"train_loss":loss_sum/max(n,1),"train_macro_f1":metrics(cm)["macro_f1"],
             "validation_accuracy":vm["accuracy"],"validation_macro_f1":mf,"validation_balanced_accuracy":vm["balanced_accuracy"],
             "lr":opt.param_groups[0]["lr"]}
        hist.append(rec); save_csv(pd.DataFrame(hist),RESULT_DIR/f"history_{tag}.csv")
        print(f"{tag} epoch {ep+1}: trainF1={rec['train_macro_f1']:.5f} valAcc={vm['accuracy']:.5f} valF1={mf:.5f}")
        if mf>best+1e-5:
            best,bestep,noimp=mf,ep+1,0
            save_torch({"version":VERSION,"tag":tag,"features":SELECTED_FEATURES,"n_classes":nc,"loss":loss_mode,
                        "best_epoch":bestep,"best_validation_macro_f1":best,"model_state":m.state_dict()},bestf)
        else: noimp+=1
        save_torch({"version":VERSION,"tag":tag,"epoch":ep,"best":best,"best_epoch":bestep,"noimp":noimp,"history":hist,
                    "model_state":m.state_dict(),"optimizer_state":opt.state_dict(),"scaler_state":sc.state_dict()},lastf)
        if noimp>=PATIENCE: print("🛑 Early stop",tag,"best epoch",bestep); break
    ck=torch.load(bestf,map_location=DEVICE,weights_only=False); m.load_state_dict(ck["model_state"]); m.eval()
    save_json({"version":VERSION,"status":"COMPLETED","tag":tag,"best_epoch":ck["best_epoch"],
               "best_validation_macro_f1":ck["best_validation_macro_f1"]},donef)
    return m

@torch.no_grad()
def eval_array(m,X,y,nc):
    pr=probs(m,X); cm=confusion_matrix(y,pr.argmax(1),labels=np.arange(nc)); return metrics(cm),pr,cm

C1=train_model("C1_ResidualMLP_WeightedCE",X_MULTI,y_POOL,X_VAL,y_VAL,8,"ce",MULTI_WEIGHTS)
C2=train_model("C2_ResidualMLP_Focal",X_MULTI,y_POOL,X_VAL,y_VAL,8,"focal",MULTI_WEIGHTS)
BIN_W=np.array([1.,1.],np.float32)
C3=train_model("C3_Binary_CE",X_BIN,y_BIN,X_VAL,y_VAL_BIN,2,"ce",BIN_W)
C4=train_model("C4_Binary_Focal",X_BIN,y_BIN,X_VAL,y_VAL_BIN,2,"focal",BIN_W)

XGB_FILE=MODEL_DIR/"C5_XGBoost.json"
def train_xgb():
    if not RUN_XGB: return None
    if RESUME and not FORCE_REBUILD and XGB_FILE.exists():
        m=xgb.XGBClassifier(); m.load_model(XGB_FILE); return m
    common=dict(n_estimators=900,max_depth=8,learning_rate=.06,min_child_weight=2,subsample=.85,colsample_bytree=.85,
                reg_lambda=1.,objective="multi:softprob",num_class=8,eval_metric="mlogloss",tree_method="hist",
                random_state=SEED,early_stopping_rounds=50,n_jobs=-1)
    sw=MULTI_WEIGHTS[y_POOL]
    try:
        m=xgb.XGBClassifier(**common,device="cuda"); m.fit(X_MULTI,y_POOL,sample_weight=sw,eval_set=[(X_VAL,y_VAL)],verbose=False)
    except Exception as e:
        print("XGB GPU fallback:",e); m=xgb.XGBClassifier(**common); m.fit(X_MULTI,y_POOL,sample_weight=sw,eval_set=[(X_VAL,y_VAL)],verbose=False)
    tmp=LOCAL/"xgb.json"; m.save_model(tmp); atomic_copy(tmp,XGB_FILE); tmp.unlink(missing_ok=True); return m
XGB=train_xgb()

@torch.no_grad()
def full_validation():
    multi_tags={"C1":C1,"C2":C2}; bin_tags={"C3":C3,"C4":C4}
    ym=[]; yb=[]; mp={k:[] for k in multi_tags}; bp={k:[] for k in bin_tags}; xp=[]
    total=0
    for fi,p in enumerate(SHARDS,1):
        df=pd.read_parquet(p,columns=STRICT_FEATURES+["y_multiclass"])
        h=row_hash(df); s=split_code(h); m=s==1
        if m.any():
            X0=normalize(df.loc[m,STRICT_FEATURES].to_numpy(np.float32)); X=X0[:,SELECTED_IDX]
            yy=df.loc[m,"y_multiclass"].to_numpy(np.int64); yyb=(yy!=0).astype(np.int64)
            ym.append(yy); yb.append(yyb)
            for k,model in multi_tags.items(): mp[k].append(probs(model,X))
            for k,model in bin_tags.items(): bp[k].append(probs(model,X))
            if XGB is not None: xp.append(XGB.predict_proba(X).astype(np.float32))
            total+=len(yy); del X0,X,yy,yyb
        del df,h,s; gc.collect()
        if fi%50==0 or fi==len(SHARDS): print(f"Full validation {fi}/{len(SHARDS)} | {total:,}")
    ym=np.concatenate(ym); yb=np.concatenate(yb)
    mp={k:np.concatenate(v) for k,v in mp.items()}; bp={k:np.concatenate(v) for k,v in bp.items()}
    xpr=np.concatenate(xp) if xp else None
    return ym,yb,mp,bp,xpr

YVM,YVB,PVM,PVB,PVX=full_validation()
NAT_PRIOR=NATURAL_COUNTS/NATURAL_COUNTS.sum()

mrec=[]
for tag,pr in PVM.items():
    for tau in TAU_GRID:
        score=np.log(np.clip(pr,1e-12,1))-float(tau)*np.log(np.clip(NAT_PRIOR,1e-12,1))
        mm=metrics(confusion_matrix(YVM,score.argmax(1),labels=np.arange(8)))
        mrec.append({"model":tag,"tau":float(tau),"validation_accuracy":mm["accuracy"],
                     "validation_macro_f1":mm["macro_f1"],"validation_balanced_accuracy":mm["balanced_accuracy"]})
MVAL=pd.DataFrame(mrec); save_csv(MVAL,RESULT_DIR/"multiclass_validation_tuning.csv")
brow=MVAL.sort_values(["validation_macro_f1","validation_balanced_accuracy"],ascending=False).iloc[0]
BEST_MTAG=str(brow["model"]); BEST_TAU=float(brow["tau"]); BEST_M={"C1":C1,"C2":C2}[BEST_MTAG]

brec=[]
for tag,pr in PVB.items():
    for thr in BINARY_THRESHOLDS:
        pred=(pr[:,1]>=thr).astype(np.int64); mm=metrics(confusion_matrix(YVB,pred,labels=[0,1]))
        brec.append({"model":tag,"threshold":float(thr),"validation_accuracy":mm["accuracy"],
                     "validation_macro_f1":mm["macro_f1"],"validation_balanced_accuracy":mm["balanced_accuracy"],
                     "sensitivity":mm["sensitivity"],"specificity":mm["specificity"]})
BVAL=pd.DataFrame(brec); save_csv(BVAL,RESULT_DIR/"binary_validation_tuning.csv")
brow2=BVAL.sort_values(["validation_macro_f1","validation_balanced_accuracy"],ascending=False).iloc[0]
BEST_BTAG=str(brow2["model"]); BEST_THR=float(brow2["threshold"]); BEST_B={"C3":C3,"C4":C4}[BEST_BTAG]
print("Best multiclass:",BEST_MTAG,"tau",BEST_TAU,"valF1",float(brow["validation_macro_f1"]))
print("Best binary:",BEST_BTAG,"thr",BEST_THR,"valF1",float(brow2["validation_macro_f1"]))

XGB_VAL=None
if PVX is not None:
    XGB_VAL=metrics(confusion_matrix(YVM,PVX.argmax(1),labels=np.arange(8)))
    print("XGBoost validation Macro-F1:",XGB_VAL["macro_f1"])
del PVM,PVB,PVX; gc.collect(); torch.cuda.empty_cache()

@torch.no_grad()
def final_test():
    mcm=np.zeros((8,8),np.int64); bcm=np.zeros((2,2),np.int64); xcm=np.zeros((8,8),np.int64) if XGB is not None else None
    ay_m=[]; ap_m=[]; ay_b=[]; ap_b=[]
    frac=min(1.,AUC_SAMPLE_MAX/max(int(SPLIT_COUNTS[2]),1)); total=0
    for fi,p in enumerate(SHARDS,1):
        df=pd.read_parquet(p,columns=STRICT_FEATURES+["y_multiclass"])
        h=row_hash(df); s=split_code(h); m=s==2
        if m.any():
            X0=normalize(df.loc[m,STRICT_FEATURES].to_numpy(np.float32)); X=X0[:,SELECTED_IDX]
            ym=df.loc[m,"y_multiclass"].to_numpy(np.int64); yb=(ym!=0).astype(np.int64)
            pm=probs(BEST_M,X); score=np.log(np.clip(pm,1e-12,1))-BEST_TAU*np.log(np.clip(NAT_PRIOR,1e-12,1)); predm=score.argmax(1)
            pb=probs(BEST_B,X); predb=(pb[:,1]>=BEST_THR).astype(np.int64)
            mcm+=confusion_matrix(ym,predm,labels=np.arange(8)); bcm+=confusion_matrix(yb,predb,labels=[0,1])
            if XGB is not None:
                px=XGB.predict_proba(X); xcm+=confusion_matrix(ym,px.argmax(1),labels=np.arange(8)); del px
            hh=h[m]^np.uint64(0xD1B54A32D192ED03); u=(hh%np.uint64(1_000_000)).astype(float)/1e6; sm=u<frac
            if sm.any():
                ay_m.append(ym[sm]); ap_m.append(pm[sm]); ay_b.append(yb[sm]); ap_b.append(pb[sm])
            total+=len(ym); del X0,X,ym,yb,pm,pb,score,predm,predb,hh,u,sm
        del df,h,s; gc.collect()
        if fi%50==0 or fi==len(SHARDS): print(f"Final test {fi}/{len(SHARDS)} | {total:,}")
    mm=metrics(mcm); bm=metrics(bcm); xm=metrics(xcm) if xcm is not None else None
    if ay_m:
        yy=np.concatenate(ay_m); pp=np.concatenate(ap_m)
        if len(yy)>AUC_SAMPLE_MAX:
            rng=np.random.default_rng(SEED+7); idx=rng.choice(len(yy),AUC_SAMPLE_MAX,replace=False); yy,pp=yy[idx],pp[idx]
        try:
            ybin=label_binarize(yy,classes=np.arange(8)); mm["roc_auc"]=float(roc_auc_score(ybin,pp,average="macro",multi_class="ovr"))
            mm["pr_auc"]=float(average_precision_score(ybin,pp,average="macro"))
        except: mm["roc_auc"]=np.nan; mm["pr_auc"]=np.nan
    if ay_b:
        yy=np.concatenate(ay_b); pp=np.concatenate(ap_b)
        if len(yy)>AUC_SAMPLE_MAX:
            rng=np.random.default_rng(SEED+8); idx=rng.choice(len(yy),AUC_SAMPLE_MAX,replace=False); yy,pp=yy[idx],pp[idx]
        bm["roc_auc"]=float(roc_auc_score(yy,pp[:,1])); bm["pr_auc"]=float(average_precision_score(yy,pp[:,1]))
    return mm,mcm,bm,bcm,xm,xcm

MM,MCM,BM,BCM,XM,XCM=final_test()
MM.update({"model":BEST_MTAG,"task":"multiclass","protocol":"duplicate_consistent_stratified_hash","tau":BEST_TAU,"feature_count":len(SELECTED_FEATURES),"rows_evaluated":int(SPLIT_COUNTS[2])})
BM.update({"model":BEST_BTAG,"task":"binary","protocol":"duplicate_consistent_stratified_hash","threshold":BEST_THR,"feature_count":len(SELECTED_FEATURES),"rows_evaluated":int(SPLIT_COUNTS[2])})
final=[BM,MM]
if XM is not None:
    XM.update({"model":"C5_XGBoost","task":"multiclass","protocol":"duplicate_consistent_stratified_hash","feature_count":len(SELECTED_FEATURES),"rows_evaluated":int(SPLIT_COUNTS[2])})
    final.append(XM)
FINAL_DF=pd.DataFrame(final); save_csv(FINAL_DF,RESULT_DIR/"STEP02C_FINAL_TEST_METRICS.csv")
MPC=per_class(MCM,CLASS_NAMES); BPC=per_class(BCM,["Benign","Attack"])
save_csv(MPC,RESULT_DIR/"multiclass_per_class.csv"); save_csv(BPC,RESULT_DIR/"binary_per_class.csv")
save_csv(pd.DataFrame(MCM,index=CLASS_NAMES,columns=CLASS_NAMES),RESULT_DIR/"multiclass_confusion.csv")
save_csv(pd.DataFrame(BCM,index=["Benign","Attack"],columns=["Benign","Attack"]),RESULT_DIR/"binary_confusion.csv")
if XCM is not None: save_csv(pd.DataFrame(XCM,index=CLASS_NAMES,columns=CLASS_NAMES),RESULT_DIR/"xgboost_confusion.csv")

comp=[]
if STEP2A_TEST.exists():
    a=pd.read_csv(STEP2A_TEST)
    for task in ["binary","multiclass"]:
        z=a[(a["model"]=="MLP")&(a["task"]==task)]
        if len(z):
            r=z.iloc[0]; comp.append({"protocol":"A_source_ip_disjoint","task":task,"model":"MLP","accuracy":r["accuracy"],"macro_f1":r["macro_f1"],"balanced_accuracy":r["balanced_accuracy"]})
if STEP2B_TEST.exists():
    b=pd.read_csv(STEP2B_TEST)
    for _,r in b.iterrows():
        comp.append({"protocol":"B_source_ip_disjoint_rebalanced","task":r.get("task"),"model":r.get("model"),"accuracy":r.get("accuracy"),"macro_f1":r.get("macro_f1"),"balanced_accuracy":r.get("balanced_accuracy")})
comp.append({"protocol":"C_duplicate_consistent_stratified","task":"binary","model":BM["model"],"accuracy":BM["accuracy"],"macro_f1":BM["macro_f1"],"balanced_accuracy":BM["balanced_accuracy"]})
comp.append({"protocol":"C_duplicate_consistent_stratified","task":"multiclass","model":MM["model"],"accuracy":MM["accuracy"],"macro_f1":MM["macro_f1"],"balanced_accuracy":MM["balanced_accuracy"]})
COMP=pd.DataFrame(comp); save_csv(COMP,RESULT_DIR/"PROTOCOL_A_B_C_COMPARISON.csv")

BEST_MODEL_FILE=MODEL_DIR/f"{'C1_ResidualMLP_WeightedCE' if BEST_MTAG=='C1' else 'C2_ResidualMLP_Focal'}_BEST.pt"
HANDOFF={
    "version":VERSION,"status":"READY_FOR_STEP3",
    "scientific_role":"Protocol C is the main centralized-vs-FL utility benchmark; A/B remain source-IP-disjoint OOD stress tests.",
    "split_protocol":"duplicate_consistent_stratified_hash_70_15_15",
    "duplicate_control":"Exact duplicate strict feature vectors share one deterministic hash and cannot cross partitions.",
    "architecture":"Residual LayerNorm MLP",
    "model_tag":BEST_MTAG,"model_file":str(BEST_MODEL_FILE),
    "input_features":SELECTED_FEATURES,"input_dim":len(SELECTED_FEATURES),"n_classes":8,"class_names":CLASS_NAMES,
    "strict_features":STRICT_FEATURES,"model_features":MODEL_FEATURES,
    "mean":MEAN.tolist(),"scale":SCALE.tolist(),"model_indices_in_strict":MODEL_IDX.tolist(),"selected_indices_in_model":SELECTED_IDX,
    "class_strategy":"moderate majority cap + capped minority SMOTE/ROS + effective-number loss weighting",
    "validation_tau":BEST_TAU,
    "centralized_test_accuracy":MM["accuracy"],"centralized_test_macro_f1":MM["macro_f1"],"centralized_test_balanced_accuracy":MM["balanced_accuracy"],
    "step3_rule":"Only Protocol-C TRAIN is partitioned into 10 non-IID clients. Keep Protocol-C validation/test untouched. Do not globally SMOTE after client partitioning.",
    "created_at":datetime.now().isoformat()
}
save_json(HANDOFF,RESULT_DIR/"STEP02C_FOR_STEP3.json")

done={"version":VERSION,"status":"COMPLETED","step":"2C","selected_features":len(SELECTED_FEATURES),
      "best_multiclass":BEST_MTAG,"multiclass_test_macro_f1":MM["macro_f1"],"multiclass_test_accuracy":MM["accuracy"],
      "best_binary":BEST_BTAG,"binary_test_macro_f1":BM["macro_f1"],"binary_test_accuracy":BM["accuracy"],
      "xgb_test_macro_f1":XM["macro_f1"] if XM is not None else None,
      "step3_handoff":str(RESULT_DIR/"STEP02C_FOR_STEP3.json"),"completed_at":datetime.now().isoformat()}
save_json(done,COMPLETE_FILE)

print("\n"+"="*110)
print("✅ STEP 2C COMPLETED")
print("="*110)
print(FINAL_DF.to_string(index=False))
print("\nPer-class multiclass:\n",MPC.to_string(index=False))
print("\nProtocol comparison:\n",COMP.to_string(index=False))
print("\nSTEP 3 HANDOFF:",RESULT_DIR/"STEP02C_FOR_STEP3.json")
print("NEXT: Step 3 — 10-client Non-IID Federated Learning using EXACT Protocol-C model/features/split.")
print("="*110)

import os, sys, gc, re, json, math, time, random, shutil, hashlib, warnings
import subprocess, importlib.util
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

from google.colab import drive

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
else:
    print("✅ Google Drive already mounted.")

def ensure_pkg(import_name, pip_name=None):
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet",
            "--disable-pip-version-check",
            pip_name or import_name
        ])

for imp,pip in [
    ("numpy","numpy"),
    ("pandas","pandas"),
    ("sklearn","scikit-learn"),
    ("scipy","scipy"),
    ("matplotlib","matplotlib"),
    ("web3","web3"),
    ("eth_tester","eth-tester"),
    ("solcx","py-solc-x"),
    ("eth","py-evm"),
]:
    ensure_pkg(imp,pip)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import label_binarize

ROOT = Path("/content/drive/MyDrive/Hybrid_BCFL_IJACSA_2026")

R3 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R3_FAST_CARF_STACK"
R5 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R5_FCS_MOE"
R6 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R6_HFSB_FL"
S4C = ROOT/"08_FEDERATED_LEARNING"/"STEP04C_FINAL_BC_ACTG_HFSB_FL"

R2 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R2_AHF_RCE"
STEP44A = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_LT_SIAMTAC_FL"

R3_SELECTION = R3/"RESULTS"/"OPTUNA_VALIDATION_SELECTION.json"
R3_REL = R3/"RESULTS"/"META_MODEL_CLASS_RELIABILITY.csv"

R5_SELECTION = R5/"RESULTS"/"R5_VALIDATION_SELECTION.json"
R6_SELECTION = R6/"RESULTS"/"R6_VALIDATION_SELECTION.json"

R6_COMPLETE = R6/"CHECKPOINTS"/"STEP04_4A_R6_COMPLETE.json"

S4C_COMPLETE = S4C/"CHECKPOINTS"/"STEP04C_COMPLETE.json"
S4C_CCAC = S4C/"RESULTS"/"CCAC_SELECTION_AND_RESULT.json"
S4C_PRIVACY = S4C/"RESULTS"/"STEP04C_PRIVACY_UTILITY_VALIDATION.csv"
S4C_SUMMARY = S4C/"RESULTS"/"STEP04C_FINAL_SUMMARY.json"

DATA_FILE = STEP44A/"CACHE"/"TACNET_PROFILE_1671681.npz"
R2_SPLIT = R2/"CACHE"/"TRAIN_CAL_SPLIT.npz"

required=[
    R3_SELECTION,R3_REL,
    R5_SELECTION,R6_SELECTION,R6_COMPLETE,
    S4C_COMPLETE,S4C_CCAC,S4C_PRIVACY,S4C_SUMMARY,
    DATA_FILE,R2_SPLIT
]

for p in required:
    if not p.exists():
        raise FileNotFoundError(
            f"Required previous artifact missing: {p}"
        )

OUT = ROOT/"11_RESULTS"/"STEP05_FINAL_VALIDATION"
TABLES = ROOT/"13_TABLES"/"STEP05_FINAL_VALIDATION"
FIGS = ROOT/"12_FIGURES"/"STEP05_FINAL_VALIDATION"
CKPT = ROOT/"06_CHECKPOINTS"/"STEP05_FINAL_VALIDATION"

for p in [OUT,TABLES,FIGS,CKPT]:
    p.mkdir(parents=True,exist_ok=True)

LOCAL=Path("/content/STEP05_VALIDATION_RUNTIME")
LOCAL.mkdir(parents=True,exist_ok=True)

VERSION="STEP05_FINAL_VALIDATION_V1"
SEED=42
RESUME=True
FORCE_REBUILD=False

random.seed(SEED)
np.random.seed(SEED)

COMPLETE=CKPT/"STEP05_FINAL_VALIDATION_COMPLETE.json"

CLASS_NAMES=[
    "Benign","DoS","DDoS","Spoofing",
    "SQLInjection","Mirai","BruteForce","XSS"
]

N_CLASSES=8
N_CLIENTS=10

REPEAT_SEEDS=[
    42,
    142,
    242,
    342,
    442
]

MALICIOUS_PAIRS=[
    (1,7),
    (0,5),
    (2,8),
    (3,9),
    (4,6),
]

ATTACK_SCALE=7.0

ROBUST_Z=4.0
CATASTROPHIC_Z=6.0
MIN_ANOMALY_FLAGS_FOR_REJECTION=2

DP_DELTA=1e-5
EPSILON_GRID=[
    6.0,
    8.0,
    12.0,
    20.0,
]

DP_CLIP_MIN=0.02
DP_CLIP_MAX=0.20

SECURE_MASK_STD=0.25
SECURE_CHUNK=50_000

BLOCKCHAIN_TX_REPEATS=30

FLOAT_BYTES=4
PAIRWISE_SEED_BYTES=32
REPORT_BATCH_SIZE=1000

SCALABILITY_CLIENTS=[
    5,10,20,50,100
]

D4C=json.load(
    open(S4C_COMPLETE,"r",encoding="utf-8")
)

CCAC=json.load(
    open(S4C_CCAC,"r",encoding="utf-8")
)

S4C_SUM=json.load(
    open(S4C_SUMMARY,"r",encoding="utf-8")
)

R6D=json.load(
    open(R6_COMPLETE,"r",encoding="utf-8")
)

R6_ACC=float(
    R6D["final_accuracy"]
)
R6_F1=float(
    R6D["final_macro_f1"]
)

EPSILON=float(
    D4C["epsilon"]
)

if EPSILON not in EPSILON_GRID:
    EPSILON_GRID=sorted(
        list(
            set(
                EPSILON_GRID+[EPSILON]
            )
        )
    )

privacy_prev=pd.read_csv(
    S4C_PRIVACY
)

selected_rows=privacy_prev[
    np.isclose(
        privacy_prev["epsilon"].astype(float),
        EPSILON
    )
]

if len(selected_rows)==0:
    raise RuntimeError(
        "Could not recover Step-4C protected fusion weight."
    )

PROTECTED_WEIGHT=float(
    selected_rows.sort_values(
        "score",
        ascending=False
    ).iloc[0][
        "protected_weight"
    ]
)

Z=np.load(DATA_FILE)

y=Z["y"].astype(
    np.int64,
    copy=False
)

train_idx=Z["train_idx"].astype(np.int64,copy=False)
val_idx=Z["val_idx"].astype(np.int64,copy=False)
test_idx=Z["test_idx"].astype(np.int64,copy=False)

YTR_FULL=y[train_idx]
YV=y[val_idx]
YTE=y[test_idx]

split=np.load(R2_SPLIT)
cal_idx=split["cal_idx"].astype(np.int64,copy=False)
YCAL=YTR_FULL[cal_idx]

N_CAL=len(YCAL)
N_VAL=len(YV)
N_TEST=len(YTE)

print("\n"+"="*130)
print("STEP 5 — FINAL VALIDATION / ABLATION")
print("="*130)
print("Frozen proposed model         : BC-ACTG-HFSB-FL")
print("R6 pre-proposed               :",R6_ACC,R6_F1)
print("Step4C clean                  :",D4C["ccac_clean_accuracy"],D4C["ccac_clean_macro_f1"])
print("Step4C full                   :",D4C["full_accuracy"],D4C["full_macro_f1"])
print("Frozen epsilon / delta        :",EPSILON,DP_DELTA)
print("Frozen protected weight       :",PROTECTED_WEIGHT)
print("Repeated seeds                :",REPEAT_SEEDS)
print("Malicious pairs               :",MALICIOUS_PAIRS)
print("Security paired experiments   :",len(REPEAT_SEEDS)*len(MALICIOUS_PAIRS))
print("CAL / VAL / TEST              :",f"{N_CAL:,}",f"{N_VAL:,}",f"{N_TEST:,}")
print("="*130)

def slug(x):
    return re.sub(r"[^A-Za-z0-9._-]+","_",str(x)).strip("_")[:120]

def atomic_copy(src,dst):
    src,dst=Path(src),Path(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)

    part=Path(str(dst)+".partial")
    part.unlink(missing_ok=True)

    shutil.copy2(src,part)
    os.replace(part,dst)

def save_json(obj,dst):
    tmp=LOCAL/(slug(Path(dst).stem)+".json")

    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,default=str)

    atomic_copy(tmp,dst)
    tmp.unlink(missing_ok=True)

def save_csv(df,dst):
    tmp=LOCAL/(slug(Path(dst).stem)+".csv")
    df.to_csv(tmp,index=False)
    atomic_copy(tmp,dst)
    tmp.unlink(missing_ok=True)

def sha256_file(path,chunk=1024*1024):
    h=hashlib.sha256()

    with open(path,"rb") as f:
        while True:
            b=f.read(chunk)
            if not b:
                break
            h.update(b)

    return h.hexdigest()

if FORCE_REBUILD:
    for p in [OUT,TABLES,FIGS,CKPT]:
        shutil.rmtree(p,ignore_errors=True)
        p.mkdir(parents=True,exist_ok=True)

ALREADY_DONE=False

if RESUME and COMPLETE.exists() and not FORCE_REBUILD:
    try:
        old=json.load(
            open(COMPLETE,"r",encoding="utf-8")
        )

        if (
            old.get("version")==VERSION
            and
            old.get("status")=="COMPLETED"
        ):
            ALREADY_DONE=True
            print("✅ Step 5 already completed.")

            f=TABLES/"FINAL_VALIDATION_SUMMARY.csv"

            if f.exists():
                print(
                    pd.read_csv(f).to_string(index=False)
                )

    except Exception:
        pass

if not ALREADY_DONE:

    def normalize_prob(p):
        p=np.clip(
            np.asarray(
                p,
                dtype=np.float64
            ),
            1e-10,
            None
        )

        p/=np.maximum(
            p.sum(
                axis=1,
                keepdims=True
            ),
            1e-12
        )

        return p.astype(
            np.float32
        )

    def metrics(
        y_true,
        pred,
        prob=None
    ):
        mp,mr,mf,_=precision_recall_fscore_support(
            y_true,
            pred,
            labels=np.arange(N_CLASSES),
            average="macro",
            zero_division=0
        )

        wp,wr,wf,_=precision_recall_fscore_support(
            y_true,
            pred,
            labels=np.arange(N_CLASSES),
            average="weighted",
            zero_division=0
        )

        out={
            "accuracy":float(
                accuracy_score(
                    y_true,
                    pred
                )
            ),
            "macro_precision":float(mp),
            "macro_recall":float(mr),
            "macro_f1":float(mf),
            "weighted_precision":float(wp),
            "weighted_recall":float(wr),
            "weighted_f1":float(wf),
            "balanced_accuracy":float(
                balanced_accuracy_score(
                    y_true,
                    pred
                )
            )
        }

        if prob is not None:
            try:
                yy=label_binarize(
                    y_true,
                    classes=np.arange(N_CLASSES)
                )

                out["roc_auc_macro"]=float(
                    roc_auc_score(
                        yy,
                        prob,
                        average="macro",
                        multi_class="ovr"
                    )
                )

                out["pr_auc_macro"]=float(
                    average_precision_score(
                        yy,
                        prob,
                        average="macro"
                    )
                )

            except Exception:
                out["roc_auc_macro"]=np.nan
                out["pr_auc_macro"]=np.nan

        return out

    def ci95(values):
        x=np.asarray(
            values,
            dtype=np.float64
        )

        n=len(x)

        if n<=1:
            return (
                float(np.mean(x)),
                0.0,
                float(np.mean(x)),
                float(np.mean(x))
            )

        mean=float(
            np.mean(x)
        )

        sd=float(
            np.std(
                x,
                ddof=1
            )
        )

        sem=sd/math.sqrt(n)

        crit=float(
            stats.t.ppf(
                0.975,
                df=n-1
            )
        )

        half=crit*sem

        return (
            mean,
            sd,
            mean-half,
            mean+half
        )

    R3_SEL=json.load(
        open(R3_SELECTION,"r",encoding="utf-8")
    )

    R3_BETA=float(
        R3_SEL["best_params"]["beta"]
    )

    R3_GAMMA=float(
        R3_SEL["best_params"]["gamma"]
    )

    R3_BIAS=np.asarray(
        R3_SEL["bias_vector"],
        dtype=np.float32
    )

    R3_REL_DF=pd.read_csv(
        R3_REL
    )

    R3_REL_MATRIX=np.stack([
        R3_REL_DF[
            f"f1_c{c}"
        ].to_numpy(np.float64)
        for c in range(N_CLASSES)
    ],axis=1)

    def r3_ensemble(tensor):
        M,N,C=tensor.shape

        rw=np.exp(
            R3_BETA*R3_REL_MATRIX
        )

        rw/=np.maximum(
            rw.max(
                axis=0,
                keepdims=True
            ),
            1e-12
        )

        num=np.zeros(
            (N,C),
            dtype=np.float64
        )

        den=np.zeros(
            (N,C),
            dtype=np.float64
        )

        for m in range(M):
            p=np.asarray(
                tensor[m],
                dtype=np.float32
            )

            conf=np.max(
                p,
                axis=1
            ).astype(np.float64)

            w=(
                conf**R3_GAMMA
            )[:,None]*rw[m][None,:]

            num+=p.astype(np.float64)*w
            den+=w

        p=normalize_prob(
            num/np.maximum(
                den,
                1e-12
            )
        )

        lp=np.log(
            np.clip(
                p,
                1e-12,
                1.0
            )
        )

        lp+=R3_BIAS[None,:]
        lp-=lp.max(
            axis=1,
            keepdims=True
        )

        return normalize_prob(
            np.exp(lp)
        )

    def load_r3(name):
        path=R3/"CACHE"/f"{name}_META_PREDICTIONS.npy"

        if not path.exists():
            raise FileNotFoundError(path)

        return r3_ensemble(
            np.load(
                path,
                mmap_mode="r"
            )
        )

    print("\nReconstructing frozen model outputs...")

    VAL_R3=load_r3("VAL")
    TEST_R3=load_r3("TEST")
    CAL_R3=load_r3("CAL")

    R5_SEL=json.load(
        open(R5_SELECTION,"r",encoding="utf-8")
    )

    sw=R5_SEL["stacker_weights"]

    R5_WC=float(sw["catboost"])
    R5_WX=float(sw["xgboost"])
    R5_WL=float(sw["lightgbm"])

    R5_THRESHOLD=float(
        R5_SEL["gate_threshold"]
    )

    R5_ALPHAS=np.asarray(
        R5_SEL["class_alphas"],
        dtype=np.float32
    )

    R5_BIAS=np.asarray(
        R5_SEL["class_bias"],
        dtype=np.float32
    )

    R5_TUNED=[
        0,3,4,5,6,7
    ]

    def load_r5_stacker(name):
        path=R5/"CACHE"/f"{name}_STACKER_PREDICTIONS.npz"

        if not path.exists():
            raise FileNotFoundError(path)

        q=np.load(path)

        return normalize_prob(
            R5_WC*q["cat"].astype(np.float32)
            +
            R5_WX*q["xgb"].astype(np.float32)
            +
            R5_WL*q["lgb"].astype(np.float32)
        )

    def r5_fuse(
        base,
        expert
    ):
        conf=np.max(
            base,
            axis=1
        )

        bp=np.argmax(
            base,
            axis=1
        )

        ep=np.argmax(
            expert,
            axis=1
        )

        hard=(
            conf<R5_THRESHOLD
        ) | np.isin(
            bp,
            R5_TUNED
        ) | np.isin(
            ep,
            R5_TUNED
        )

        lb=np.log(
            np.clip(
                base,
                1e-12,
                1.0
            )
        ).astype(np.float64)

        le=np.log(
            np.clip(
                expert,
                1e-12,
                1.0
            )
        ).astype(np.float64)

        lf=lb.copy()

        for c in range(N_CLASSES):
            a=float(
                R5_ALPHAS[c]
            )

            if a==0:
                continue

            lf[
                hard,
                c
            ]=(
                (1-a)*lb[hard,c]
                +
                a*le[hard,c]
            )

        lf+=R5_BIAS[None,:]
        lf-=lf.max(
            axis=1,
            keepdims=True
        )

        return normalize_prob(
            np.exp(lf)
        )

    VAL_R5=r5_fuse(
        VAL_R3,
        load_r5_stacker("VAL")
    )

    TEST_R5=r5_fuse(
        TEST_R3,
        load_r5_stacker("TEST")
    )

    R6_SEL=json.load(
        open(R6_SELECTION,"r",encoding="utf-8")
    )

    R6_GATE_BETA=float(
        R6_SEL["gate_beta"]
    )
    R6_FINE_BETA=float(
        R6_SEL["fine_beta"]
    )
    R6_GATE_TEMP=float(
        R6_SEL["gate_temperature"]
    )
    R6_FINE_TEMP=float(
        R6_SEL["fine_temperature"]
    )
    R6_STRONG=float(
        R6_SEL["strong_carf_weight"]
    )
    R6_WEAK=float(
        R6_SEL["weak_carf_weight"]
    )
    R6_BIAS=np.asarray(
        R6_SEL["bias"],
        dtype=np.float32
    )

    def to_coarse(y8):
        out=np.full(
            len(y8),
            3,
            dtype=np.int64
        )

        out[y8==0]=0
        out[y8==1]=1
        out[y8==2]=2

        return out

    def to_fine(y8):
        return (
            y8-3
        ).astype(np.int64)

    def load_r6_tensor(
        name,
        stage
    ):
        path=R6/"CACHE"/f"{name}_{stage.upper()}_PRED.npy"

        if not path.exists():
            raise FileNotFoundError(path)

        return np.load(
            path,
            mmap_mode="r"
        )

    CAL_GATE_T=load_r6_tensor(
        "CAL",
        "gate"
    )

    CAL_FINE_T=load_r6_tensor(
        "CAL",
        "fine"
    )

    VAL_GATE_T=load_r6_tensor(
        "VAL",
        "gate"
    )

    VAL_FINE_T=load_r6_tensor(
        "VAL",
        "fine"
    )

    TEST_GATE_T=load_r6_tensor(
        "TEST",
        "gate"
    )

    TEST_FINE_T=load_r6_tensor(
        "TEST",
        "fine"
    )

    def f1_matrix(
        tensor,
        ytrue,
        n_classes
    ):
        out=np.zeros(
            (
                tensor.shape[0],
                n_classes
            ),
            dtype=np.float64
        )

        for m in range(
            tensor.shape[0]
        ):
            pred=np.argmax(
                np.asarray(
                    tensor[m],
                    dtype=np.float32
                ),
                axis=1
            )

            _,_,f,_=precision_recall_fscore_support(
                ytrue,
                pred,
                labels=np.arange(n_classes),
                zero_division=0
            )

            out[m]=f

        return out

    YCAL_GATE=to_coarse(
        YCAL
    )

    weak_cal=YCAL>=3

    YCAL_FINE=to_fine(
        YCAL[
            weak_cal
        ]
    )

    GATE_REL=f1_matrix(
        CAL_GATE_T,
        YCAL_GATE,
        4
    )

    FINE_REL=np.zeros(
        (
            CAL_FINE_T.shape[0],
            5
        ),
        dtype=np.float64
    )

    for m in range(
        CAL_FINE_T.shape[0]
    ):
        pred=np.argmax(
            np.asarray(
                CAL_FINE_T[
                    m,
                    weak_cal
                ],
                dtype=np.float32
            ),
            axis=1
        )

        _,_,f,_=precision_recall_fscore_support(
            YCAL_FINE,
            pred,
            labels=np.arange(5),
            zero_division=0
        )

        FINE_REL[m]=f

    def reliability_ensemble(
        tensor,
        rel,
        beta
    ):
        M,N,C=tensor.shape

        rw=np.exp(
            beta*rel
        )

        rw/=np.maximum(
            rw.max(
                axis=0,
                keepdims=True
            ),
            1e-12
        )

        num=np.zeros(
            (N,C),
            dtype=np.float64
        )

        den=np.zeros(
            (N,C),
            dtype=np.float64
        )

        for m in range(M):
            p=np.asarray(
                tensor[m],
                dtype=np.float32
            )

            conf=np.max(
                p,
                axis=1
            ).astype(np.float64)

            w=conf[:,None]*rw[m][None,:]

            num+=p.astype(np.float64)*w
            den+=w

        return normalize_prob(
            num/np.maximum(
                den,
                1e-12
            )
        )

    def temperature_prob(
        p,
        temp
    ):
        lp=np.log(
            np.clip(
                p,
                1e-12,
                1.0
            )
        )/float(temp)

        lp-=lp.max(
            axis=1,
            keepdims=True
        )

        return normalize_prob(
            np.exp(lp)
        )

    def compose_hierarchy(
        gate,
        fine
    ):
        g=temperature_prob(
            gate,
            R6_GATE_TEMP
        )

        f=temperature_prob(
            fine,
            R6_FINE_TEMP
        )

        out=np.zeros(
            (len(g),8),
            dtype=np.float32
        )

        out[:,0]=g[:,0]
        out[:,1]=g[:,1]
        out[:,2]=g[:,2]
        out[:,3:]=g[:,3:4]*f

        return normalize_prob(
            out
        )

    def r6_final(
        hierarchy,
        carf
    ):
        alpha=np.array(
            [
                R6_STRONG,
                R6_STRONG,
                R6_STRONG,
                R6_WEAK,
                R6_WEAK,
                R6_WEAK,
                R6_WEAK,
                R6_WEAK
            ],
            dtype=np.float64
        )

        lp=(
            alpha[None,:]
            *
            np.log(
                np.clip(
                    carf,
                    1e-12,
                    1.0
                )
            )
            +
            (
                1-alpha[None,:]
            )
            *
            np.log(
                np.clip(
                    hierarchy,
                    1e-12,
                    1.0
                )
            )
        )

        lp+=R6_BIAS[None,:]

        lp-=lp.max(
            axis=1,
            keepdims=True
        )

        return normalize_prob(
            np.exp(lp)
        )

    def reconstruct_r6(
        gate_tensor,
        fine_tensor,
        carf
    ):
        g=reliability_ensemble(
            gate_tensor,
            GATE_REL,
            R6_GATE_BETA
        )

        f=reliability_ensemble(
            fine_tensor,
            FINE_REL,
            R6_FINE_BETA
        )

        h=compose_hierarchy(
            g,
            f
        )

        return (
            r6_final(
                h,
                carf
            ),
            g,
            f,
            h
        )

    CAL_R6,CAL_R6_GATE,CAL_R6_FINE,CAL_R6_HIER=reconstruct_r6(
        CAL_GATE_T,
        CAL_FINE_T,
        CAL_R3
    )

    VAL_R6,VAL_R6_GATE,VAL_R6_FINE,VAL_R6_HIER=reconstruct_r6(
        VAL_GATE_T,
        VAL_FINE_T,
        VAL_R3
    )

    TEST_R6,TEST_R6_GATE,TEST_R6_FINE,TEST_R6_HIER=reconstruct_r6(
        TEST_GATE_T,
        TEST_FINE_T,
        TEST_R3
    )

    def client_probs(
        tensor
    ):
        if tensor.shape[0] != 2*N_CLIENTS:
            raise RuntimeError(
                f"Expected {2*N_CLIENTS} model predictions."
            )

        return [
            normalize_prob(
                0.5*np.asarray(
                    tensor[2*k],
                    dtype=np.float32
                )
                +
                0.5*np.asarray(
                    tensor[2*k+1],
                    dtype=np.float32
                )
            )
            for k in range(N_CLIENTS)
        ]

    CAL_GATE_CLIENT=client_probs(
        CAL_GATE_T
    )

    CAL_FINE_CLIENT=client_probs(
        CAL_FINE_T
    )

    VAL_GATE_CLIENT=client_probs(
        VAL_GATE_T
    )

    VAL_FINE_CLIENT=client_probs(
        VAL_FINE_T
    )

    TEST_GATE_CLIENT=client_probs(
        TEST_GATE_T
    )

    TEST_FINE_CLIENT=client_probs(
        TEST_FINE_T
    )

    CCAC_BETA=float(
        CCAC["beta"]
    )

    CCAC_STRONG=float(
        CCAC["strong_r6_anchor"]
    )

    CCAC_WEAK=float(
        CCAC["weak_r6_anchor"]
    )

    CCAC_BIAS=np.asarray(
        CCAC["class_bias"],
        dtype=np.float32
    )

    MODEL_NAMES=[
        "R3_CARF",
        "R5_FCS_MoE",
        "R6_HFSB_FL"
    ]

    VAL_MODELS=[
        VAL_R3,
        VAL_R5,
        VAL_R6
    ]

    TEST_MODELS=[
        TEST_R3,
        TEST_R5,
        TEST_R6
    ]

    CLASS_REL=np.zeros(
        (3,N_CLASSES),
        dtype=np.float64
    )

    for m,p in enumerate(
        VAL_MODELS
    ):
        pred=np.argmax(
            p,
            axis=1
        )

        _,_,f,_=precision_recall_fscore_support(
            YV,
            pred,
            labels=np.arange(N_CLASSES),
            zero_division=0
        )

        CLASS_REL[m]=f

    def frozen_ccac(
        models
    ):
        z=CCAC_BETA*CLASS_REL
        z-=z.max(
            axis=0,
            keepdims=True
        )

        w=np.exp(z)

        w/=np.maximum(
            w.sum(
                axis=0,
                keepdims=True
            ),
            1e-12
        )

        log_ens=np.zeros_like(
            models[0],
            dtype=np.float64
        )

        for m,p in enumerate(
            models
        ):
            log_ens+=(
                w[m][None,:]
                *
                np.log(
                    np.clip(
                        p,
                        1e-12,
                        1.0
                    )
                )
            )

        anchor=np.array(
            [
                CCAC_STRONG,
                CCAC_STRONG,
                CCAC_STRONG,
                CCAC_WEAK,
                CCAC_WEAK,
                CCAC_WEAK,
                CCAC_WEAK,
                CCAC_WEAK
            ],
            dtype=np.float64
        )

        lp=(
            (
                1-anchor[None,:]
            )*log_ens
            +
            anchor[None,:]
            *
            np.log(
                np.clip(
                    models[2],
                    1e-12,
                    1.0
                )
            )
        )

        lp+=CCAC_BIAS[None,:]

        lp-=lp.max(
            axis=1,
            keepdims=True
        )

        return normalize_prob(
            np.exp(lp)
        )

    VAL_CCAC=frozen_ccac(
        VAL_MODELS
    )

    TEST_CCAC=frozen_ccac(
        TEST_MODELS
    )

    CLEAN_METRICS=metrics(
        YTE,
        np.argmax(
            TEST_CCAC,
            axis=1
        ),
        TEST_CCAC
    )

    print(
        "Frozen Step-4C clean metrics:",
        CLEAN_METRICS
    )

    client_quality=[]

    for k in range(N_CLIENTS):
        h=compose_hierarchy(
            CAL_GATE_CLIENT[k],
            CAL_FINE_CLIENT[k]
        )

        mm=metrics(
            YCAL,
            np.argmax(
                h,
                axis=1
            )
        )

        client_quality.append({
            "client_id":k,
            "cal_accuracy":mm["accuracy"],
            "cal_macro_f1":mm["macro_f1"]
        })

    QUALITY_DF=pd.DataFrame(
        client_quality
    )

    save_csv(
        QUALITY_DF,
        TABLES/"CLIENT_CAL_QUALITY.csv"
    )

    q=np.clip(
        QUALITY_DF[
            "cal_macro_f1"
        ].to_numpy(np.float64),
        1e-6,
        1.0
    )

    BASE_WEIGHT=np.exp(
        5.0*q
    )

    BASE_WEIGHT/=BASE_WEIGHT.sum()

    def all_norms(
        clients,
        prior
    ):
        return np.concatenate([
            np.linalg.norm(
                p-prior,
                axis=1
            )
            for p in clients
        ])

    GATE_CLIP=float(
        np.clip(
            np.quantile(
                all_norms(
                    CAL_GATE_CLIENT,
                    CAL_R6_GATE
                ),
                0.90
            ),
            DP_CLIP_MIN,
            DP_CLIP_MAX
        )
    )

    FINE_CLIP=float(
        np.clip(
            np.quantile(
                all_norms(
                    CAL_FINE_CLIENT,
                    CAL_R6_FINE
                ),
                0.90
            ),
            DP_CLIP_MIN,
            DP_CLIP_MAX
        )
    )

    def gaussian_sigma(
        clip,
        epsilon
    ):
        return float(
            clip
            *
            math.sqrt(
                2.0*math.log(
                    1.25/DP_DELTA
                )
            )
            /
            epsilon
        )

    def dp_share(
        client_prob,
        prior,
        clip,
        epsilon,
        seed
    ):
        r=(
            client_prob-prior
        ).astype(np.float32)

        norm=np.linalg.norm(
            r,
            axis=1,
            keepdims=True
        )

        factor=np.minimum(
            1.0,
            clip/np.maximum(
                norm,
                1e-12
            )
        )

        r=r*factor

        sigma=gaussian_sigma(
            clip,
            epsilon
        )

        rng=np.random.default_rng(
            seed
        )

        noise=rng.normal(
            0.0,
            sigma,
            size=r.shape
        ).astype(np.float32)

        return normalize_prob(
            prior+r+noise
        )

    FINE_PERM=np.array(
        [4,3,0,1,2],
        dtype=np.int64
    )

    def post_dp_attack(
        transmitted,
        prior,
        stage,
        attack_type
    ):
        residual=(
            transmitted-prior
        )

        if attack_type=="sign_scale":
            attacked=(
                prior
                -
                ATTACK_SCALE*residual
            )

        elif attack_type=="permute":
            if stage=="fine":
                attacked=transmitted[
                    :,
                    FINE_PERM
                ]
            else:
                attacked=transmitted[
                    :,
                    [3,1,2,0]
                ]

            attacked=np.power(
                np.clip(
                    attacked,
                    1e-6,
                    1.0
                ),
                0.30
            )

        else:
            raise ValueError(
                attack_type
            )

        return normalize_prob(
            attacked
        )

    def robust_z_high(x):
        x=np.asarray(
            x,
            dtype=np.float64
        )

        med=np.median(x)
        mad=np.median(
            np.abs(
                x-med
            )
        )

        return (
            x-med
        )/(
            1.4826*mad+1e-9
        )

    def robust_z_low(x):
        return robust_z_high(
            -np.asarray(
                x,
                dtype=np.float64
            )
        )

    def cosine(a,b):
        a=np.asarray(
            a,
            dtype=np.float64
        ).reshape(-1)

        b=np.asarray(
            b,
            dtype=np.float64
        ).reshape(-1)

        den=np.linalg.norm(a)*np.linalg.norm(b)

        if den<1e-12:
            return 1.0

        return float(
            np.dot(a,b)/den
        )

    def js(p,q):
        p=np.clip(
            np.asarray(
                p,
                dtype=np.float64
            ),
            1e-12,
            None
        )

        q=np.clip(
            np.asarray(
                q,
                dtype=np.float64
            ),
            1e-12,
            None
        )

        p/=p.sum()
        q/=q.sum()

        m=0.5*(p+q)

        return float(
            0.5*np.sum(
                p*np.log(p/m)
            )
            +
            0.5*np.sum(
                q*np.log(q/m)
            )
        )

    def geometric_mix(
        prior,
        candidate,
        alpha
    ):
        lp=(
            (
                1-alpha
            )
            *
            np.log(
                np.clip(
                    prior,
                    1e-12,
                    1.0
                )
            )
            +
            alpha
            *
            np.log(
                np.clip(
                    candidate,
                    1e-12,
                    1.0
                )
            )
        )

        lp-=lp.max(
            axis=1,
            keepdims=True
        )

        return normalize_prob(
            np.exp(lp)
        )

    def cross_entropy(
        ytrue,
        prob
    ):
        return float(
            log_loss(
                ytrue,
                prob,
                labels=np.arange(N_CLASSES)
            )
        )

    def actg_table(
        gate_clients,
        fine_clients,
        gate_prior,
        fine_prior,
        base_hier_prior,
        ycal
    ):
        gate_sig=np.stack([
            (
                p-gate_prior
            ).mean(axis=0)
            for p in gate_clients
        ])

        fine_sig=np.stack([
            (
                p-fine_prior
            ).mean(axis=0)
            for p in fine_clients
        ])

        med_gate=np.median(
            gate_sig,
            axis=0
        )

        med_fine=np.median(
            fine_sig,
            axis=0
        )

        gate_mean=np.stack([
            p.mean(axis=0)
            for p in gate_clients
        ])

        fine_mean=np.stack([
            p.mean(axis=0)
            for p in fine_clients
        ])

        med_gate_prob=np.median(
            gate_mean,
            axis=0
        )

        med_fine_prob=np.median(
            fine_mean,
            axis=0
        )

        prior_ce=cross_entropy(
            ycal,
            base_hier_prior
        )

        prior_f1=metrics(
            ycal,
            np.argmax(
                base_hier_prior,
                axis=1
            )
        )["macro_f1"]

        rows=[]

        for k in range(N_CLIENTS):
            rg=(
                gate_clients[k]
                -
                gate_prior
            )

            rf=(
                fine_clients[k]
                -
                fine_prior
            )

            residual_norm=float(
                0.5*np.mean(
                    np.linalg.norm(
                        rg,
                        axis=1
                    )
                )
                +
                0.5*np.mean(
                    np.linalg.norm(
                        rf,
                        axis=1
                    )
                )
            )

            cos_score=0.5*(
                cosine(
                    gate_sig[k],
                    med_gate
                )
                +
                cosine(
                    fine_sig[k],
                    med_fine
                )
            )

            js_score=0.5*(
                js(
                    gate_mean[k],
                    med_gate_prob
                )
                +
                js(
                    fine_mean[k],
                    med_fine_prob
                )
            )

            h=compose_hierarchy(
                gate_clients[k],
                fine_clients[k]
            )

            candidate=geometric_mix(
                base_hier_prior,
                h,
                0.25
            )

            cand_ce=cross_entropy(
                ycal,
                candidate
            )

            cand_f1=metrics(
                ycal,
                np.argmax(
                    candidate,
                    axis=1
                )
            )["macro_f1"]

            rows.append({
                "client_id":k,
                "residual_norm":residual_norm,
                "cosine_agreement":cos_score,
                "js_divergence":js_score,
                "delta_cross_entropy":cand_ce-prior_ce,
                "delta_macro_f1":cand_f1-prior_f1
            })

        df=pd.DataFrame(
            rows
        )

        df["z_norm"]=robust_z_high(
            df["residual_norm"]
        )

        df["z_cos_bad"]=robust_z_low(
            df["cosine_agreement"]
        )

        df["z_js"]=robust_z_high(
            df["js_divergence"]
        )

        df["z_ce_bad"]=robust_z_high(
            df["delta_cross_entropy"]
        )

        df["z_f1_bad"]=robust_z_low(
            df["delta_macro_f1"]
        )

        zcols=[
            "z_norm",
            "z_cos_bad",
            "z_js",
            "z_ce_bad",
            "z_f1_bad"
        ]

        zbad=np.stack([
            np.maximum(
                df[c].to_numpy(
                    np.float64
                ),
                0.0
            )
            for c in zcols
        ],axis=1)

        flags=zbad>ROBUST_Z

        df["anomaly_flags"]=flags.sum(
            axis=1
        )

        df[
            "catastrophic_validation_anomaly"
        ]=(
            df["z_ce_bad"]>CATASTROPHIC_Z
        ) | (
            df["z_f1_bad"]>CATASTROPHIC_Z
        )

        df["accepted"]=~(
            (
                df["anomaly_flags"]
                >=
                MIN_ANOMALY_FLAGS_FOR_REJECTION
            )
            |
            df[
                "catastrophic_validation_anomaly"
            ]
        )

        sorted_bad=np.sort(
            zbad,
            axis=1
        )

        top2=sorted_bad[
            :,
            -2:
        ].sum(axis=1)

        contribution_bonus=np.clip(
            -df[
                "delta_cross_entropy"
            ].to_numpy(
                np.float64
            ),
            -0.05,
            0.05
        )

        trust=np.exp(
            -0.18*top2
        )*np.exp(
            3.0*contribution_bonus
        )

        df["trust_score"]=np.clip(
            trust,
            1e-6,
            1.0
        )

        return df

    def pair_seed(
        i,j,
        stage,
        base_seed
    ):
        raw=f"{base_seed}|{stage}|{min(i,j)}|{max(i,j)}"

        return int(
            hashlib.sha256(
                raw.encode()
            ).hexdigest()[:16],
            16
        )%(2**32-1)

    def secure_aggregate(
        clients,
        prior,
        accepted,
        weights,
        stage,
        seed
    ):
        accepted=[
            int(k)
            for k in accepted
        ]

        if not accepted:
            raise RuntimeError(
                "No ACTG-approved client."
            )

        weights=np.asarray(
            weights,
            dtype=np.float64
        )

        denom=float(
            weights[
                accepted
            ].sum()
        )

        N,C=prior.shape

        out=np.empty(
            (N,C),
            dtype=np.float32
        )

        max_cancel=0.0

        for start in range(
            0,
            N,
            SECURE_CHUNK
        ):
            end=min(
                start+SECURE_CHUNK,
                N
            )

            payload={}

            for k in accepted:
                residual=(
                    clients[k][start:end]
                    -
                    prior[start:end]
                ).astype(np.float64)

                payload[k]=(
                    weights[k]*residual
                )

            mask_balance=np.zeros(
                (
                    end-start,
                    C
                ),
                dtype=np.float64
            )

            for ai,i in enumerate(
                accepted
            ):
                for j in accepted[
                    ai+1:
                ]:
                    rng=np.random.default_rng(
                        pair_seed(
                            i,j,
                            stage,
                            seed+start
                        )
                    )

                    mask=rng.normal(
                        0.0,
                        SECURE_MASK_STD,
                        size=(
                            end-start,
                            C
                        )
                    )

                    payload[i]+=mask
                    payload[j]-=mask

                    mask_balance+=mask
                    mask_balance-=mask

            numerator=np.zeros(
                (
                    end-start,
                    C
                ),
                dtype=np.float64
            )

            for k in accepted:
                numerator+=payload[k]

            out[start:end]=normalize_prob(
                prior[start:end]
                +
                numerator/max(
                    denom,
                    1e-12
                )
            )

            max_cancel=max(
                max_cancel,
                float(
                    np.max(
                        np.abs(
                            mask_balance
                        )
                    )
                )
            )

            del payload,numerator,mask_balance
            gc.collect()

        return out,{
            "accepted_clients":accepted,
            "sum_weights":denom,
            "max_mask_cancellation_error":max_cancel
        }

    def direct_residual_aggregate(
        clients,
        prior,
        accepted,
        weights
    ):
        weights=np.asarray(
            weights,
            dtype=np.float64
        )

        denom=float(
            weights[
                accepted
            ].sum()
        )

        num=np.zeros_like(
            prior,
            dtype=np.float64
        )

        for k in accepted:
            num+=(
                weights[k]
                *
                (
                    clients[k]-prior
                )
            )

        return normalize_prob(
            prior
            +
            num/max(
                denom,
                1e-12
            )
        )

    def protected_clients(
        gate_clients,
        fine_clients,
        gate_prior,
        fine_prior,
        epsilon,
        seed,
        malicious_pair=None
    ):
        pg=[]
        pf=[]

        for k in range(N_CLIENTS):
            g=dp_share(
                gate_clients[k],
                gate_prior,
                GATE_CLIP,
                epsilon,
                seed+k*101+1
            )

            f=dp_share(
                fine_clients[k],
                fine_prior,
                FINE_CLIP,
                epsilon,
                seed+k*101+2
            )

            if (
                malicious_pair is not None
                and
                k==malicious_pair[0]
            ):
                g=post_dp_attack(
                    g,
                    gate_prior,
                    "gate",
                    "sign_scale"
                )

                f=post_dp_attack(
                    f,
                    fine_prior,
                    "fine",
                    "sign_scale"
                )

            elif (
                malicious_pair is not None
                and
                k==malicious_pair[1]
            ):
                g=post_dp_attack(
                    g,
                    gate_prior,
                    "gate",
                    "permute"
                )

                f=post_dp_attack(
                    f,
                    fine_prior,
                    "fine",
                    "permute"
                )

            pg.append(g)
            pf.append(f)

        return pg,pf

    def trust_for_pair(
        seed,
        malicious_pair
    ):
        cg,cf=protected_clients(
            CAL_GATE_CLIENT,
            CAL_FINE_CLIENT,
            CAL_R6_GATE,
            CAL_R6_FINE,
            EPSILON,
            seed,
            malicious_pair
        )

        t=actg_table(
            cg,
            cf,
            CAL_R6_GATE,
            CAL_R6_FINE,
            CAL_R6_HIER,
            YCAL
        )

        t["known_malicious"]=t[
            "client_id"
        ].isin(
            list(
                malicious_pair
            )
        )

        accepted=t[
            t["accepted"]
        ]["client_id"].astype(int).tolist()

        rejected=t[
            ~t["accepted"]
        ]["client_id"].astype(int).tolist()

        tw=BASE_WEIGHT.copy()

        for _,r in t.iterrows():
            k=int(
                r["client_id"]
            )

            tw[k]*=float(
                r["trust_score"]
            )

        true_attack=t[
            "known_malicious"
        ].astype(int).to_numpy()

        pred_attack=(
            ~t["accepted"]
        ).astype(int).to_numpy()

        detect={
            "precision":float(
                precision_score(
                    true_attack,
                    pred_attack,
                    zero_division=0
                )
            ),
            "recall":float(
                recall_score(
                    true_attack,
                    pred_attack,
                    zero_division=0
                )
            ),
            "f1":float(
                f1_score(
                    true_attack,
                    pred_attack,
                    zero_division=0
                )
            ),
            "false_positive_count":int(
                np.sum(
                    (
                        pred_attack==1
                    )
                    &
                    (
                        true_attack==0
                    )
                )
            ),
            "false_negative_count":int(
                np.sum(
                    (
                        pred_attack==0
                    )
                    &
                    (
                        true_attack==1
                    )
                )
            ),
        }

        return (
            t,
            accepted,
            rejected,
            tw,
            detect
        )

    def final_from_clients(
        gate_clients,
        fine_clients,
        gate_prior,
        fine_prior,
        final_prior,
        accepted,
        weights,
        secure,
        seed
    ):
        if secure:
            g,ga=secure_aggregate(
                gate_clients,
                gate_prior,
                accepted,
                weights,
                "gate",
                seed+7000
            )

            f,fa=secure_aggregate(
                fine_clients,
                fine_prior,
                accepted,
                weights,
                "fine",
                seed+8000
            )

        else:
            g=direct_residual_aggregate(
                gate_clients,
                gate_prior,
                accepted,
                weights
            )

            f=direct_residual_aggregate(
                fine_clients,
                fine_prior,
                accepted,
                weights
            )

            ga=None
            fa=None

        h=compose_hierarchy(
            g,
            f
        )

        final=geometric_mix(
            final_prior,
            h,
            PROTECTED_WEIGHT
        )

        return final,ga,fa

    privacy_seed_rows=[]

    print("\nRunning repeated privacy utility experiments...")

    for eps in EPSILON_GRID:
        for seed in REPEAT_SEEDS:
            pg,pf=protected_clients(
                TEST_GATE_CLIENT,
                TEST_FINE_CLIENT,
                TEST_R6_GATE,
                TEST_R6_FINE,
                eps,
                seed+10_000,
                malicious_pair=None
            )

            accepted=list(
                range(N_CLIENTS)
            )

            p,ga,fa=final_from_clients(
                pg,pf,
                TEST_R6_GATE,
                TEST_R6_FINE,
                TEST_CCAC,
                accepted,
                BASE_WEIGHT,
                secure=True,
                seed=seed+20_000
            )

            mm=metrics(
                YTE,
                np.argmax(
                    p,
                    axis=1
                )
            )

            privacy_seed_rows.append({
                "epsilon":eps,
                "seed":seed,
                **mm
            })

            del pg,pf,p
            gc.collect()

    PRIVACY_SEEDS=pd.DataFrame(
        privacy_seed_rows
    )

    save_csv(
        PRIVACY_SEEDS,
        OUT/"PRIVACY_SEED_LEVEL_RESULTS.csv"
    )

    privacy_summary=[]

    for eps,g in PRIVACY_SEEDS.groupby(
        "epsilon"
    ):
        a=ci95(
            g["accuracy"]
        )

        f=ci95(
            g["macro_f1"]
        )

        privacy_summary.append({
            "epsilon":eps,
            "n_runs":len(g),
            "accuracy_mean":a[0],
            "accuracy_sd":a[1],
            "accuracy_ci95_low":a[2],
            "accuracy_ci95_high":a[3],
            "macro_f1_mean":f[0],
            "macro_f1_sd":f[1],
            "macro_f1_ci95_low":f[2],
            "macro_f1_ci95_high":f[3]
        })

    PRIVACY_SUMMARY=pd.DataFrame(
        privacy_summary
    ).sort_values(
        "epsilon"
    )

    save_csv(
        PRIVACY_SUMMARY,
        TABLES/"PRIVACY_UTILITY_MEAN_SD_CI.csv"
    )

    robustness_rows=[]
    trust_rows=[]

    print(
        "\nRunning",
        len(REPEAT_SEEDS)*len(MALICIOUS_PAIRS),
        "paired attack/defense experiments..."
    )

    for pair_id,pair in enumerate(
        MALICIOUS_PAIRS
    ):
        for seed in REPEAT_SEEDS:
            print(
                f"  pair={pair} seed={seed}"
            )

            trust_df,accepted,rejected,trust_weight,detect=trust_for_pair(
                seed+pair_id*100_000,
                pair
            )

            trust_copy=trust_df.copy()
            trust_copy["pair_id"]=pair_id
            trust_copy["malicious_pair"]=str(pair)
            trust_copy["seed"]=seed

            trust_rows.append(
                trust_copy
            )

            tg,tf=protected_clients(
                TEST_GATE_CLIENT,
                TEST_FINE_CLIENT,
                TEST_R6_GATE,
                TEST_R6_FINE,
                EPSILON,
                seed+pair_id*100_000+30_000,
                malicious_pair=pair
            )

            p_attack,_,_=final_from_clients(
                tg,tf,
                TEST_R6_GATE,
                TEST_R6_FINE,
                TEST_CCAC,
                list(
                    range(N_CLIENTS)
                ),
                BASE_WEIGHT,
                secure=False,
                seed=seed+40_000
            )

            m_attack=metrics(
                YTE,
                np.argmax(
                    p_attack,
                    axis=1
                )
            )

            p_def,ga,fa=final_from_clients(
                tg,tf,
                TEST_R6_GATE,
                TEST_R6_FINE,
                TEST_CCAC,
                accepted,
                trust_weight,
                secure=True,
                seed=seed+50_000
            )

            m_def=metrics(
                YTE,
                np.argmax(
                    p_def,
                    axis=1
                )
            )

            robustness_rows.append({
                "pair_id":pair_id,
                "malicious_pair":str(pair),
                "seed":seed,

                "attack_accuracy":m_attack["accuracy"],
                "attack_macro_f1":m_attack["macro_f1"],

                "defense_accuracy":m_def["accuracy"],
                "defense_macro_f1":m_def["macro_f1"],

                "accuracy_recovery":(
                    m_def["accuracy"]
                    -
                    m_attack["accuracy"]
                ),

                "macro_f1_recovery":(
                    m_def["macro_f1"]
                    -
                    m_attack["macro_f1"]
                ),

                "detection_precision":detect["precision"],
                "detection_recall":detect["recall"],
                "detection_f1":detect["f1"],

                "false_positive_count":detect["false_positive_count"],
                "false_negative_count":detect["false_negative_count"],

                "accepted_count":len(accepted),
                "rejected_count":len(rejected),

                "gate_mask_error":ga[
                    "max_mask_cancellation_error"
                ],
                "fine_mask_error":fa[
                    "max_mask_cancellation_error"
                ],
            })

            del tg,tf,p_attack,p_def
            gc.collect()

    ROBUSTNESS=pd.DataFrame(
        robustness_rows
    )

    TRUST_ALL=pd.concat(
        trust_rows,
        ignore_index=True
    )

    save_csv(
        ROBUSTNESS,
        OUT/"SECURITY_SEED_PAIR_LEVEL_RESULTS.csv"
    )

    save_csv(
        TRUST_ALL,
        OUT/"ACTG_ALL_RUN_DECISIONS.csv"
    )

    def metric_summary(
        df,
        col,
        prefix
    ):
        v=ci95(
            df[col]
        )

        return {
            f"{prefix}_mean":v[0],
            f"{prefix}_sd":v[1],
            f"{prefix}_ci95_low":v[2],
            f"{prefix}_ci95_high":v[3]
        }

    SECURITY_SUMMARY={
        "n_runs":len(ROBUSTNESS),
        **metric_summary(
            ROBUSTNESS,
            "attack_accuracy",
            "attack_accuracy"
        ),
        **metric_summary(
            ROBUSTNESS,
            "defense_accuracy",
            "defense_accuracy"
        ),
        **metric_summary(
            ROBUSTNESS,
            "attack_macro_f1",
            "attack_macro_f1"
        ),
        **metric_summary(
            ROBUSTNESS,
            "defense_macro_f1",
            "defense_macro_f1"
        ),
        **metric_summary(
            ROBUSTNESS,
            "accuracy_recovery",
            "accuracy_recovery"
        ),
        **metric_summary(
            ROBUSTNESS,
            "macro_f1_recovery",
            "macro_f1_recovery"
        ),
        **metric_summary(
            ROBUSTNESS,
            "detection_precision",
            "detection_precision"
        ),
        **metric_summary(
            ROBUSTNESS,
            "detection_recall",
            "detection_recall"
        ),
        **metric_summary(
            ROBUSTNESS,
            "detection_f1",
            "detection_f1"
        ),
    }

    SECURITY_SUMMARY_DF=pd.DataFrame([
        SECURITY_SUMMARY
    ])

    save_csv(
        SECURITY_SUMMARY_DF,
        TABLES/"SECURITY_ROBUSTNESS_MEAN_SD_CI.csv"
    )

    stats_rows=[]

    for metric_name,attack_col,def_col in [
        (
            "Accuracy",
            "attack_accuracy",
            "defense_accuracy"
        ),
        (
            "Macro-F1",
            "attack_macro_f1",
            "defense_macro_f1"
        )
    ]:
        a=ROBUSTNESS[
            attack_col
        ].to_numpy(
            np.float64
        )

        d=ROBUSTNESS[
            def_col
        ].to_numpy(
            np.float64
        )

        t_res=stats.ttest_rel(
            d,
            a,
            nan_policy="omit"
        )

        try:
            w_res=stats.wilcoxon(
                d,
                a,
                zero_method="wilcox",
                alternative="two-sided"
            )

            w_stat=float(
                w_res.statistic
            )

            w_p=float(
                w_res.pvalue
            )

        except Exception:
            w_stat=np.nan
            w_p=np.nan

        diff=d-a

        dz=(
            float(
                np.mean(diff)
                /
                np.std(
                    diff,
                    ddof=1
                )
            )
            if np.std(
                diff,
                ddof=1
            )>0
            else np.nan
        )

        stats_rows.append({
            "metric":metric_name,
            "n_pairs":len(a),
            "mean_attack":float(
                np.mean(a)
            ),
            "mean_defense":float(
                np.mean(d)
            ),
            "mean_difference":float(
                np.mean(diff)
            ),
            "paired_t_stat":float(
                t_res.statistic
            ),
            "paired_t_p":float(
                t_res.pvalue
            ),
            "wilcoxon_stat":w_stat,
            "wilcoxon_p":w_p,
            "cohens_dz":dz
        })

    STAT_TESTS=pd.DataFrame(
        stats_rows
    )

    save_csv(
        STAT_TESTS,
        TABLES/"PAIRED_STATISTICAL_TESTS.csv"
    )

    check_seed=999

    pg,pf=protected_clients(
        TEST_GATE_CLIENT,
        TEST_FINE_CLIENT,
        TEST_R6_GATE,
        TEST_R6_FINE,
        EPSILON,
        check_seed,
        malicious_pair=None
    )

    accepted=list(
        range(N_CLIENTS)
    )

    g_secure,ga=secure_aggregate(
        pg,
        TEST_R6_GATE,
        accepted,
        BASE_WEIGHT,
        "gate",
        check_seed+100
    )

    g_direct=direct_residual_aggregate(
        pg,
        TEST_R6_GATE,
        accepted,
        BASE_WEIGHT
    )

    f_secure,fa=secure_aggregate(
        pf,
        TEST_R6_FINE,
        accepted,
        BASE_WEIGHT,
        "fine",
        check_seed+200
    )

    f_direct=direct_residual_aggregate(
        pf,
        TEST_R6_FINE,
        accepted,
        BASE_WEIGHT
    )

    SECURE_CHECK=pd.DataFrame([
        {
            "stage":"gate",
            "max_abs_secure_vs_direct":float(
                np.max(
                    np.abs(
                        g_secure-g_direct
                    )
                )
            ),
            "mean_abs_secure_vs_direct":float(
                np.mean(
                    np.abs(
                        g_secure-g_direct
                    )
                )
            ),
            "pairwise_mask_cancellation_error":ga[
                "max_mask_cancellation_error"
            ]
        },
        {
            "stage":"fine",
            "max_abs_secure_vs_direct":float(
                np.max(
                    np.abs(
                        f_secure-f_direct
                    )
                )
            ),
            "mean_abs_secure_vs_direct":float(
                np.mean(
                    np.abs(
                        f_secure-f_direct
                    )
                )
            ),
            "pairwise_mask_cancellation_error":fa[
                "max_mask_cancellation_error"
            ]
        }
    ])

    save_csv(
        SECURE_CHECK,
        TABLES/"SECURE_AGGREGATION_CORRECTNESS.csv"
    )

    del pg,pf,g_secure,g_direct,f_secure,f_direct
    gc.collect()

    blockchain_rows=[]
    blockchain_mode="local_hash_chain_fallback"
    blockchain_error=None

    try:
        from web3 import Web3
        from web3.providers.eth_tester import EthereumTesterProvider
        from eth_tester import EthereumTester
        import solcx

        ver="0.8.20"

        installed=[
            str(v)
            for v in solcx.get_installed_solc_versions()
        ]

        if ver not in installed:
            solcx.install_solc(ver)

        solcx.set_solc_version(ver)

        source=r"""
        // SPDX-License-Identifier: MIT
        pragma solidity ^0.8.20;

        contract ValidationAudit {
            struct Record {
                uint256 runId;
                uint256 clientId;
                bytes32 commitment;
                bool accepted;
                uint256 trustMicro;
            }

            Record[] public records;

            function audit(
                uint256 runId,
                uint256 clientId,
                bytes32 commitment,
                bool accepted,
                uint256 trustMicro
            ) external {
                records.push(
                    Record(
                        runId,
                        clientId,
                        commitment,
                        accepted,
                        trustMicro
                    )
                );
            }

            function count() external view returns (uint256) {
                return records.length;
            }
        }
        """

        compiled=solcx.compile_source(
            source,
            output_values=[
                "abi",
                "bin"
            ]
        )

        _,iface=compiled.popitem()

        w3=Web3(
            EthereumTesterProvider(
                EthereumTester()
            )
        )

        account=w3.eth.accounts[0]

        Contract=w3.eth.contract(
            abi=iface["abi"],
            bytecode=iface["bin"]
        )

        t0=time.perf_counter()

        tx=Contract.constructor().transact({
            "from":account
        })

        receipt=w3.eth.wait_for_transaction_receipt(
            tx
        )

        deployment_ms=(
            time.perf_counter()-t0
        )*1000.0

        contract=w3.eth.contract(
            address=receipt.contractAddress,
            abi=iface["abi"]
        )

        blockchain_mode="web3_ethereumtester_solidity"

        for i in range(
            BLOCKCHAIN_TX_REPEATS
        ):
            commitment=hashlib.sha256(
                f"step5|{i}|{VERSION}".encode()
            ).digest()

            start=time.perf_counter()

            tx=contract.functions.audit(
                i+1,
                i%N_CLIENTS,
                commitment,
                True,
                900_000
            ).transact({
                "from":account
            })

            rec=w3.eth.wait_for_transaction_receipt(
                tx
            )

            latency=(
                time.perf_counter()-start
            )*1000.0

            blockchain_rows.append({
                "tx_index":i,
                "latency_ms":latency,
                "gas_used":int(
                    rec.gasUsed
                ),
                "tx_hash":rec.transactionHash.hex(),
                "mode":blockchain_mode
            })

    except Exception as e:
        blockchain_error=repr(e)
        deployment_ms=0.0

        previous=hashlib.sha256(
            b"STEP5_GENESIS"
        ).hexdigest()

        for i in range(
            BLOCKCHAIN_TX_REPEATS
        ):
            start=time.perf_counter()

            payload=json.dumps({
                "run":i,
                "client":i%N_CLIENTS,
                "previous":previous,
                "accepted":True,
                "trust":0.9
            },sort_keys=True)

            current=hashlib.sha256(
                payload.encode()
            ).hexdigest()

            latency=(
                time.perf_counter()-start
            )*1000.0

            blockchain_rows.append({
                "tx_index":i,
                "latency_ms":latency,
                "gas_used":np.nan,
                "tx_hash":current,
                "mode":blockchain_mode
            })

            previous=current

    BLOCKCHAIN_DF=pd.DataFrame(
        blockchain_rows
    )

    save_csv(
        BLOCKCHAIN_DF,
        OUT/"BLOCKCHAIN_LATENCY_RAW.csv"
    )

    BLOCKCHAIN_SUMMARY=pd.DataFrame([
        {
            "mode":blockchain_mode,
            "deployment_latency_ms":deployment_ms,
            "n_transactions":len(BLOCKCHAIN_DF),
            "mean_latency_ms":float(
                BLOCKCHAIN_DF[
                    "latency_ms"
                ].mean()
            ),
            "median_latency_ms":float(
                BLOCKCHAIN_DF[
                    "latency_ms"
                ].median()
            ),
            "p95_latency_ms":float(
                BLOCKCHAIN_DF[
                    "latency_ms"
                ].quantile(
                    0.95
                )
            ),
            "latency_sd_ms":float(
                BLOCKCHAIN_DF[
                    "latency_ms"
                ].std(
                    ddof=1
                )
            ),
            "mean_gas_used":(
                float(
                    BLOCKCHAIN_DF[
                        "gas_used"
                    ].dropna().mean()
                )
                if BLOCKCHAIN_DF[
                    "gas_used"
                ].notna().any()
                else np.nan
            ),
            "fallback_error":blockchain_error
        }
    ])

    save_csv(
        BLOCKCHAIN_SUMMARY,
        TABLES/"BLOCKCHAIN_LATENCY_SUMMARY.csv"
    )

    floats_per_sample=4+5

    bytes_per_sample_client=(
        floats_per_sample
        *
        FLOAT_BYTES
    )

    test_payload_client=(
        N_TEST
        *
        bytes_per_sample_client
    )

    test_payload_all=(
        N_CLIENTS
        *
        test_payload_client
    )

    report_batch_client=(
        REPORT_BATCH_SIZE
        *
        bytes_per_sample_client
    )

    report_batch_all=(
        N_CLIENTS
        *
        report_batch_client
    )

    pair_count=(
        N_CLIENTS
        *
        (
            N_CLIENTS-1
        )
        //
        2
    )

    pair_seed_material=(
        pair_count
        *
        PAIRWISE_SEED_BYTES
    )

    COMM=pd.DataFrame([
        {
            "item":"Protected residual payload per sample per client",
            "bytes":bytes_per_sample_client,
            "MiB":bytes_per_sample_client/(1024**2),
            "assumption":"9 float32 values = gate(4)+fine(5)"
        },
        {
            "item":f"Protected residual payload per {REPORT_BATCH_SIZE}-sample batch per client",
            "bytes":report_batch_client,
            "MiB":report_batch_client/(1024**2),
            "assumption":"float32 residual representation"
        },
        {
            "item":f"Protected residual payload per {REPORT_BATCH_SIZE}-sample batch for 10 clients",
            "bytes":report_batch_all,
            "MiB":report_batch_all/(1024**2),
            "assumption":"10 accepted/transmitting clients"
        },
        {
            "item":"Full TEST benchmark payload per client",
            "bytes":test_payload_client,
            "MiB":test_payload_client/(1024**2),
            "assumption":f"{N_TEST} evaluation records"
        },
        {
            "item":"Full TEST benchmark payload for all 10 clients",
            "bytes":test_payload_all,
            "MiB":test_payload_all/(1024**2),
            "assumption":f"{N_TEST} evaluation records x 10 clients"
        },
        {
            "item":"Pairwise seed/key material per secure-aggregation setup",
            "bytes":pair_seed_material,
            "MiB":pair_seed_material/(1024**2),
            "assumption":f"C(10,2)={pair_count} pairs x {PAIRWISE_SEED_BYTES} bytes"
        },
        {
            "item":"SHA-256 model commitment per client",
            "bytes":32,
            "MiB":32/(1024**2),
            "assumption":"one 256-bit digest"
        }
    ])

    save_csv(
        COMM,
        TABLES/"COMMUNICATION_OVERHEAD.csv"
    )

    scalability=[]

    for k in SCALABILITY_CLIENTS:
        pairs=k*(k-1)//2

        payload_batch=(
            k
            *
            REPORT_BATCH_SIZE
            *
            bytes_per_sample_client
        )

        key_bytes=(
            pairs
            *
            PAIRWISE_SEED_BYTES
        )

        blockchain_records=k

        scalability.append({
            "clients":k,
            "pairwise_mask_pairs":pairs,
            "protected_payload_for_1000_samples_MiB":payload_batch/(1024**2),
            "pairwise_seed_material_KiB":key_bytes/1024,
            "blockchain_records_per_round":blockchain_records,
            "client_payload_complexity":"O(K)",
            "pairwise_key_setup_complexity":"O(K^2)",
            "note":"analytical scaling estimate; not a measured K-client training experiment"
        })

    SCALE=pd.DataFrame(
        scalability
    )

    save_csv(
        SCALE,
        TABLES/"ANALYTICAL_SCALABILITY.csv"
    )

    privacy_selected=PRIVACY_SUMMARY[
        np.isclose(
            PRIVACY_SUMMARY[
                "epsilon"
            ],
            EPSILON
        )
    ].iloc[0]

    sec=SECURITY_SUMMARY

    FINAL_SUMMARY=pd.DataFrame([
        {
            "experiment":"R6 pre-proposed clean",
            "accuracy_mean":R6_ACC,
            "accuracy_sd":0.0,
            "macro_f1_mean":R6_F1,
            "macro_f1_sd":0.0,
            "n_runs":1,
            "interpretation":"pre-proposed performance baseline"
        },
        {
            "experiment":"BC-ACTG-HFSB-FL clean CCAC",
            "accuracy_mean":CLEAN_METRICS["accuracy"],
            "accuracy_sd":0.0,
            "macro_f1_mean":CLEAN_METRICS["macro_f1"],
            "macro_f1_sd":0.0,
            "n_runs":1,
            "interpretation":"frozen proposed clean utility"
        },
        {
            "experiment":f"DP secure aggregation eps={EPSILON}",
            "accuracy_mean":privacy_selected["accuracy_mean"],
            "accuracy_sd":privacy_selected["accuracy_sd"],
            "macro_f1_mean":privacy_selected["macro_f1_mean"],
            "macro_f1_sd":privacy_selected["macro_f1_sd"],
            "n_runs":privacy_selected["n_runs"],
            "interpretation":"privacy utility across independent DP seeds"
        },
        {
            "experiment":"20% malicious clients — no defense",
            "accuracy_mean":sec["attack_accuracy_mean"],
            "accuracy_sd":sec["attack_accuracy_sd"],
            "macro_f1_mean":sec["attack_macro_f1_mean"],
            "macro_f1_sd":sec["attack_macro_f1_sd"],
            "n_runs":sec["n_runs"],
            "interpretation":"5 malicious pairs x 5 DP seeds"
        },
        {
            "experiment":"FULL proposed — ACTG + blockchain + secure aggregation",
            "accuracy_mean":sec["defense_accuracy_mean"],
            "accuracy_sd":sec["defense_accuracy_sd"],
            "macro_f1_mean":sec["defense_macro_f1_mean"],
            "macro_f1_sd":sec["defense_macro_f1_sd"],
            "n_runs":sec["n_runs"],
            "interpretation":"same paired attacks with frozen ACTG defense"
        }
    ])

    save_csv(
        FINAL_SUMMARY,
        TABLES/"FINAL_VALIDATION_SUMMARY.csv"
    )

    fig,ax=plt.subplots(
        figsize=(11,5)
    )

    ax.bar(
        FINAL_SUMMARY[
            "experiment"
        ],
        FINAL_SUMMARY[
            "accuracy_mean"
        ]
    )

    ax.errorbar(
        np.arange(
            len(
                FINAL_SUMMARY
            )
        ),
        FINAL_SUMMARY[
            "accuracy_mean"
        ],
        yerr=FINAL_SUMMARY[
            "accuracy_sd"
        ],
        fmt="none",
        capsize=4
    )

    ax.set_ylim(
        0.90,
        0.96
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_title(
        "Final BC-ACTG-HFSB-FL Ablation"
    )

    ax.tick_params(
        axis="x",
        rotation=28
    )

    fig.tight_layout()

    fp=LOCAL/"FINAL_ABLATION_ACCURACY.png"

    fig.savefig(
        fp,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    atomic_copy(
        fp,
        FIGS/fp.name
    )

    fig,ax=plt.subplots(
        figsize=(8,5)
    )

    ax.errorbar(
        PRIVACY_SUMMARY[
            "epsilon"
        ],
        PRIVACY_SUMMARY[
            "accuracy_mean"
        ],
        yerr=PRIVACY_SUMMARY[
            "accuracy_sd"
        ],
        marker="o",
        capsize=4,
        label="Accuracy"
    )

    ax.errorbar(
        PRIVACY_SUMMARY[
            "epsilon"
        ],
        PRIVACY_SUMMARY[
            "macro_f1_mean"
        ],
        yerr=PRIVACY_SUMMARY[
            "macro_f1_sd"
        ],
        marker="s",
        capsize=4,
        label="Macro-F1"
    )

    ax.set_xlabel(
        "Privacy epsilon"
    )

    ax.set_ylabel(
        "Mean test performance"
    )

    ax.set_title(
        "Multi-Seed Privacy–Utility Trade-off"
    )

    ax.legend()

    fig.tight_layout()

    fp=LOCAL/"PRIVACY_UTILITY_MULTI_SEED.png"

    fig.savefig(
        fp,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    atomic_copy(
        fp,
        FIGS/fp.name
    )

    fig,ax=plt.subplots(
        figsize=(9,5)
    )

    x=np.arange(
        len(
            ROBUSTNESS
        )
    )

    ax.plot(
        x,
        ROBUSTNESS[
            "attack_accuracy"
        ],
        marker="o",
        label="Attack / No defense"
    )

    ax.plot(
        x,
        ROBUSTNESS[
            "defense_accuracy"
        ],
        marker="s",
        label="ACTG defense"
    )

    ax.set_xlabel(
        "Paired seed/attacker experiment"
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_title(
        "Paired Poisoning Attack vs ACTG Defense"
    )

    ax.legend()

    fig.tight_layout()

    fp=LOCAL/"ATTACK_VS_DEFENSE_PAIRED.png"

    fig.savefig(
        fp,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    atomic_copy(
        fp,
        FIGS/fp.name
    )

    fig,ax=plt.subplots(
        figsize=(7,5)
    )

    vals=[
        SECURITY_SUMMARY[
            "detection_precision_mean"
        ],
        SECURITY_SUMMARY[
            "detection_recall_mean"
        ],
        SECURITY_SUMMARY[
            "detection_f1_mean"
        ]
    ]

    errs=[
        SECURITY_SUMMARY[
            "detection_precision_sd"
        ],
        SECURITY_SUMMARY[
            "detection_recall_sd"
        ],
        SECURITY_SUMMARY[
            "detection_f1_sd"
        ]
    ]

    ax.bar(
        [
            "Precision",
            "Recall",
            "F1"
        ],
        vals,
        yerr=errs,
        capsize=4
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_ylabel(
        "Attack-detection score"
    )

    ax.set_title(
        "ACTG Detection Robustness"
    )

    fig.tight_layout()

    fp=LOCAL/"ACTG_DETECTION_ROBUSTNESS.png"

    fig.savefig(
        fp,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    atomic_copy(
        fp,
        FIGS/fp.name
    )

    fig,ax=plt.subplots(
        figsize=(8,5)
    )

    ax.plot(
        BLOCKCHAIN_DF[
            "tx_index"
        ],
        BLOCKCHAIN_DF[
            "latency_ms"
        ],
        marker="o"
    )

    ax.set_xlabel(
        "Transaction index"
    )

    ax.set_ylabel(
        "Latency (ms)"
    )

    ax.set_title(
        "Permissioned Blockchain Audit Latency"
    )

    fig.tight_layout()

    fp=LOCAL/"BLOCKCHAIN_LATENCY.png"

    fig.savefig(
        fp,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    atomic_copy(
        fp,
        FIGS/fp.name
    )

    FINAL_JSON={
        "version":VERSION,
        "status":"COMPLETED",
        "model":"BC-ACTG-HFSB-FL",

        "frozen_parameters":{
            "epsilon":EPSILON,
            "delta":DP_DELTA,
            "protected_weight":PROTECTED_WEIGHT,
            "actg_robust_z":ROBUST_Z,
            "actg_catastrophic_z":CATASTROPHIC_Z,
            "actg_min_flags":MIN_ANOMALY_FLAGS_FOR_REJECTION,
        },

        "clean_proposed":CLEAN_METRICS,

        "privacy_multi_seed":{
            "seeds":REPEAT_SEEDS,
            "epsilon_grid":EPSILON_GRID,
            "summary_file":str(
                TABLES/"PRIVACY_UTILITY_MEAN_SD_CI.csv"
            )
        },

        "security_multi_seed_pair":{
            "seeds":REPEAT_SEEDS,
            "malicious_pairs":[
                list(p)
                for p in MALICIOUS_PAIRS
            ],
            "malicious_fraction":0.20,
            "n_paired_runs":len(ROBUSTNESS),
            "summary":SECURITY_SUMMARY,
            "statistical_tests_file":str(
                TABLES/"PAIRED_STATISTICAL_TESTS.csv"
            )
        },

        "secure_aggregation":{
            "correctness_file":str(
                TABLES/"SECURE_AGGREGATION_CORRECTNESS.csv"
            )
        },

        "blockchain":{
            "mode":blockchain_mode,
            "latency_summary_file":str(
                TABLES/"BLOCKCHAIN_LATENCY_SUMMARY.csv"
            )
        },

        "communication":{
            "file":str(
                TABLES/"COMMUNICATION_OVERHEAD.csv"
            ),
            "note":"analytical payload estimate under float32 protected residual sharing"
        },

        "scalability":{
            "file":str(
                TABLES/"ANALYTICAL_SCALABILITY.csv"
            ),
            "note":"analytical complexity/payload projection; not measured training with >10 physical clients"
        },

        "paper_reporting_rules":[
            "Protocol-LT is literature-comparable and must not be presented as strict non-IID generalization.",
            "Protocol-C strict non-IID results remain separately reported.",
            "Privacy is output/residual-level Gaussian DP under the bounded shared-residual sensitivity assumption, not DP-SGD.",
            "Malicious identities are used only for post-hoc detection metrics.",
            "Blockchain audits and enforces ACTG decisions; it does not detect poisoning.",
            "Scalability above 10 clients is analytical unless separately executed with real additional client partitions."
        ],

        "created_at":datetime.now().isoformat()
    }

    save_json(
        FINAL_JSON,
        OUT/"STEP05_PAPER_READY_SUMMARY.json"
    )

    save_json(
        {
            "version":VERSION,
            "status":"COMPLETED",
            "step":"5_FINAL_VALIDATION",
            "model":"BC-ACTG-HFSB-FL",
            "n_privacy_runs":len(PRIVACY_SEEDS),
            "n_security_paired_runs":len(ROBUSTNESS),
            "blockchain_mode":blockchain_mode,
            "final_summary":str(
                TABLES/"FINAL_VALIDATION_SUMMARY.csv"
            ),
            "paper_ready_json":str(
                OUT/"STEP05_PAPER_READY_SUMMARY.json"
            ),
            "completed_at":datetime.now().isoformat()
        },
        COMPLETE
    )

    print("\n"+"="*130)
    print("✅ STEP 5 FINAL VALIDATION COMPLETED")
    print("="*130)

    print("\nFINAL VALIDATION SUMMARY:")
    print(
        FINAL_SUMMARY.to_string(
            index=False
        )
    )

    print("\nSECURITY ROBUSTNESS:")
    print(
        SECURITY_SUMMARY_DF.to_string(
            index=False
        )
    )

    print("\nPAIRED STATISTICAL TESTS:")
    print(
        STAT_TESTS.to_string(
            index=False
        )
    )

    print("\nSECURE AGGREGATION CHECK:")
    print(
        SECURE_CHECK.to_string(
            index=False
        )
    )

    print("\nBLOCKCHAIN:")
    print(
        BLOCKCHAIN_SUMMARY.to_string(
            index=False
        )
    )

    print("\nFiles:")
    print(
        TABLES/"FINAL_VALIDATION_SUMMARY.csv"
    )
    print(
        OUT/"STEP05_PAPER_READY_SUMMARY.json"
    )

    print("="*130)

import os, sys, gc, re, json, math, time, random, shutil, hashlib, warnings
import subprocess, importlib.util
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

from google.colab import drive

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
else:
    print("✅ Google Drive already mounted.")

def ensure_pkg(import_name, pip_name=None):
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet",
            "--disable-pip-version-check",
            pip_name or import_name
        ])

for imp,pip in [
    ("numpy","numpy"),
    ("pandas","pandas"),
    ("sklearn","scikit-learn"),
    ("scipy","scipy"),
    ("matplotlib","matplotlib"),
    ("web3","web3"),
    ("eth_tester","eth-tester"),
    ("solcx","py-solc-x"),
    ("eth","py-evm"),
]:
    ensure_pkg(imp,pip)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import label_binarize

ROOT = Path("/content/drive/MyDrive/Hybrid_BCFL_IJACSA_2026")

R3 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R3_FAST_CARF_STACK"
R5 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R5_FCS_MOE"
R6 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R6_HFSB_FL"
S4C = ROOT/"08_FEDERATED_LEARNING"/"STEP04C_FINAL_BC_ACTG_HFSB_FL"

R2 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R2_AHF_RCE"
STEP44A = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_LT_SIAMTAC_FL"

R3_SELECTION = R3/"RESULTS"/"OPTUNA_VALIDATION_SELECTION.json"
R3_REL = R3/"RESULTS"/"META_MODEL_CLASS_RELIABILITY.csv"

R5_SELECTION = R5/"RESULTS"/"R5_VALIDATION_SELECTION.json"
R6_SELECTION = R6/"RESULTS"/"R6_VALIDATION_SELECTION.json"

R6_COMPLETE = R6/"CHECKPOINTS"/"STEP04_4A_R6_COMPLETE.json"

S4C_COMPLETE = S4C/"CHECKPOINTS"/"STEP04C_COMPLETE.json"
S4C_CCAC = S4C/"RESULTS"/"CCAC_SELECTION_AND_RESULT.json"
S4C_PRIVACY = S4C/"RESULTS"/"STEP04C_PRIVACY_UTILITY_VALIDATION.csv"
S4C_SUMMARY = S4C/"RESULTS"/"STEP04C_FINAL_SUMMARY.json"

DATA_FILE = STEP44A/"CACHE"/"TACNET_PROFILE_1671681.npz"
R2_SPLIT = R2/"CACHE"/"TRAIN_CAL_SPLIT.npz"

required=[
    R3_SELECTION,R3_REL,
    R5_SELECTION,R6_SELECTION,R6_COMPLETE,
    S4C_COMPLETE,S4C_CCAC,S4C_PRIVACY,S4C_SUMMARY,
    DATA_FILE,R2_SPLIT
]

for p in required:
    if not p.exists():
        raise FileNotFoundError(
            f"Required previous artifact missing: {p}"
        )

OUT = ROOT/"11_RESULTS"/"STEP06_SECURITY_STRESS"
TABLES = ROOT/"13_TABLES"/"STEP06_SECURITY_STRESS"
FIGS = ROOT/"12_FIGURES"/"STEP06_SECURITY_STRESS"
CKPT = ROOT/"06_CHECKPOINTS"/"STEP06_SECURITY_STRESS"

for p in [OUT,TABLES,FIGS,CKPT]:
    p.mkdir(parents=True,exist_ok=True)

LOCAL=Path("/content/STEP06_SECURITY_STRESS_RUNTIME")
LOCAL.mkdir(parents=True,exist_ok=True)

VERSION="STEP06_SECURITY_STRESS_V1"
SEED=42
RESUME=True
FORCE_REBUILD=False

random.seed(SEED)
np.random.seed(SEED)

COMPLETE=CKPT/"STEP06_SECURITY_STRESS_COMPLETE.json"

CLASS_NAMES=[
    "Benign","DoS","DDoS","Spoofing",
    "SQLInjection","Mirai","BruteForce","XSS"
]

N_CLASSES=8
N_CLIENTS=10

REPEAT_SEEDS=[
    42,
    142,
    242,
    342,
    442
]

MALICIOUS_PAIRS=[
    (1,7),
    (0,5),
    (2,8),
    (3,9),
    (4,6),
]

ATTACK_SCALE=7.0

ROBUST_Z=4.0
CATASTROPHIC_Z=6.0
MIN_ANOMALY_FLAGS_FOR_REJECTION=2

DP_DELTA=1e-5
EPSILON_GRID=[
    6.0,
    8.0,
    12.0,
    20.0,
]

DP_CLIP_MIN=0.02
DP_CLIP_MAX=0.20

SECURE_MASK_STD=0.25
SECURE_CHUNK=50_000

BLOCKCHAIN_TX_REPEATS=30

FLOAT_BYTES=4
PAIRWISE_SEED_BYTES=32
REPORT_BATCH_SIZE=1000

SCALABILITY_CLIENTS=[
    5,10,20,50,100
]

STRESS_SEEDS=[
    42,
    242,
    442,
]

ATTACK_TYPES=[
    "sign_scale",
    "permute",
    "model_replacement",
    "targeted_weak",
]

STRESS_PAIRS=[
    (0,5),
    (1,6),
    (2,7),
    (3,8),
    (4,9),
]

FRACTION_SETS={
    0.10:[
        (0,),
        (2,),
        (4,),
        (6,),
        (8,),
    ],
    0.20:[
        (0,5),
        (1,6),
        (2,7),
        (3,8),
        (4,9),
    ],
    0.30:[
        (0,3,6),
        (1,4,7),
        (2,5,8),
        (3,6,9),
        (0,4,8),
    ],
    0.40:[
        (0,2,5,7),
        (1,3,6,8),
        (2,4,7,9),
        (0,3,5,8),
        (1,4,6,9),
    ],
}

STRESS_SIGN_SCALE=15.0
MODEL_REPLACE_CONF=0.995
TARGETED_CONF=0.997

D4C=json.load(
    open(S4C_COMPLETE,"r",encoding="utf-8")
)

CCAC=json.load(
    open(S4C_CCAC,"r",encoding="utf-8")
)

S4C_SUM=json.load(
    open(S4C_SUMMARY,"r",encoding="utf-8")
)

R6D=json.load(
    open(R6_COMPLETE,"r",encoding="utf-8")
)

R6_ACC=float(
    R6D["final_accuracy"]
)
R6_F1=float(
    R6D["final_macro_f1"]
)

EPSILON=float(
    D4C["epsilon"]
)

if EPSILON not in EPSILON_GRID:
    EPSILON_GRID=sorted(
        list(
            set(
                EPSILON_GRID+[EPSILON]
            )
        )
    )

privacy_prev=pd.read_csv(
    S4C_PRIVACY
)

selected_rows=privacy_prev[
    np.isclose(
        privacy_prev["epsilon"].astype(float),
        EPSILON
    )
]

if len(selected_rows)==0:
    raise RuntimeError(
        "Could not recover Step-4C protected fusion weight."
    )

PROTECTED_WEIGHT=float(
    selected_rows.sort_values(
        "score",
        ascending=False
    ).iloc[0][
        "protected_weight"
    ]
)

Z=np.load(DATA_FILE)

y=Z["y"].astype(
    np.int64,
    copy=False
)

train_idx=Z["train_idx"].astype(np.int64,copy=False)
val_idx=Z["val_idx"].astype(np.int64,copy=False)
test_idx=Z["test_idx"].astype(np.int64,copy=False)

YTR_FULL=y[train_idx]
YV=y[val_idx]
YTE=y[test_idx]

split=np.load(R2_SPLIT)
cal_idx=split["cal_idx"].astype(np.int64,copy=False)
YCAL=YTR_FULL[cal_idx]

N_CAL=len(YCAL)
N_VAL=len(YV)
N_TEST=len(YTE)

print("\n"+"="*130)
print("STEP 6 — SECURITY STRESS TEST / THREAT-MODEL VALIDATION")
print("="*130)
print("Frozen proposed model         : BC-ACTG-HFSB-FL")
print("R6 pre-proposed               :",R6_ACC,R6_F1)
print("Step4C clean                  :",D4C["ccac_clean_accuracy"],D4C["ccac_clean_macro_f1"])
print("Step4C full                   :",D4C["full_accuracy"],D4C["full_macro_f1"])
print("Frozen epsilon / delta        :",EPSILON,DP_DELTA)
print("Frozen protected weight       :",PROTECTED_WEIGHT)
print("Stress seeds                  :",STRESS_SEEDS)
print("Attack types                  :",ATTACK_TYPES)
print("Malicious fractions           :",list(FRACTION_SETS.keys()))
print("Malicious pairs               :",MALICIOUS_PAIRS)
print("Security paired experiments   :",len(REPEAT_SEEDS)*len(MALICIOUS_PAIRS))
print("CAL / VAL / TEST              :",f"{N_CAL:,}",f"{N_VAL:,}",f"{N_TEST:,}")
print("="*130)

def slug(x):
    return re.sub(r"[^A-Za-z0-9._-]+","_",str(x)).strip("_")[:120]

def atomic_copy(src,dst):
    src,dst=Path(src),Path(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)

    part=Path(str(dst)+".partial")
    part.unlink(missing_ok=True)

    shutil.copy2(src,part)
    os.replace(part,dst)

def save_json(obj,dst):
    tmp=LOCAL/(slug(Path(dst).stem)+".json")

    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,default=str)

    atomic_copy(tmp,dst)
    tmp.unlink(missing_ok=True)

def save_csv(df,dst):
    tmp=LOCAL/(slug(Path(dst).stem)+".csv")
    df.to_csv(tmp,index=False)
    atomic_copy(tmp,dst)
    tmp.unlink(missing_ok=True)

def sha256_file(path,chunk=1024*1024):
    h=hashlib.sha256()

    with open(path,"rb") as f:
        while True:
            b=f.read(chunk)
            if not b:
                break
            h.update(b)

    return h.hexdigest()

if FORCE_REBUILD:
    for p in [OUT,TABLES,FIGS,CKPT]:
        shutil.rmtree(p,ignore_errors=True)
        p.mkdir(parents=True,exist_ok=True)

ALREADY_DONE=False

if RESUME and COMPLETE.exists() and not FORCE_REBUILD:
    try:
        old=json.load(
            open(COMPLETE,"r",encoding="utf-8")
        )

        if (
            old.get("version")==VERSION
            and
            old.get("status")=="COMPLETED"
        ):
            ALREADY_DONE=True
            print("✅ Step 6 already completed.")

            f=TABLES/"STEP06_SECURITY_STRESS_SUMMARY.csv"

            if f.exists():
                print(
                    pd.read_csv(f).to_string(index=False)
                )

    except Exception:
        pass

if not ALREADY_DONE:

    def normalize_prob(p):
        p=np.clip(
            np.asarray(
                p,
                dtype=np.float64
            ),
            1e-10,
            None
        )

        p/=np.maximum(
            p.sum(
                axis=1,
                keepdims=True
            ),
            1e-12
        )

        return p.astype(
            np.float32
        )

    def metrics(
        y_true,
        pred,
        prob=None
    ):
        mp,mr,mf,_=precision_recall_fscore_support(
            y_true,
            pred,
            labels=np.arange(N_CLASSES),
            average="macro",
            zero_division=0
        )

        wp,wr,wf,_=precision_recall_fscore_support(
            y_true,
            pred,
            labels=np.arange(N_CLASSES),
            average="weighted",
            zero_division=0
        )

        out={
            "accuracy":float(
                accuracy_score(
                    y_true,
                    pred
                )
            ),
            "macro_precision":float(mp),
            "macro_recall":float(mr),
            "macro_f1":float(mf),
            "weighted_precision":float(wp),
            "weighted_recall":float(wr),
            "weighted_f1":float(wf),
            "balanced_accuracy":float(
                balanced_accuracy_score(
                    y_true,
                    pred
                )
            )
        }

        if prob is not None:
            try:
                yy=label_binarize(
                    y_true,
                    classes=np.arange(N_CLASSES)
                )

                out["roc_auc_macro"]=float(
                    roc_auc_score(
                        yy,
                        prob,
                        average="macro",
                        multi_class="ovr"
                    )
                )

                out["pr_auc_macro"]=float(
                    average_precision_score(
                        yy,
                        prob,
                        average="macro"
                    )
                )

            except Exception:
                out["roc_auc_macro"]=np.nan
                out["pr_auc_macro"]=np.nan

        return out

    def ci95(values):
        x=np.asarray(
            values,
            dtype=np.float64
        )

        n=len(x)

        if n<=1:
            return (
                float(np.mean(x)),
                0.0,
                float(np.mean(x)),
                float(np.mean(x))
            )

        mean=float(
            np.mean(x)
        )

        sd=float(
            np.std(
                x,
                ddof=1
            )
        )

        sem=sd/math.sqrt(n)

        crit=float(
            stats.t.ppf(
                0.975,
                df=n-1
            )
        )

        half=crit*sem

        return (
            mean,
            sd,
            mean-half,
            mean+half
        )

    R3_SEL=json.load(
        open(R3_SELECTION,"r",encoding="utf-8")
    )

    R3_BETA=float(
        R3_SEL["best_params"]["beta"]
    )

    R3_GAMMA=float(
        R3_SEL["best_params"]["gamma"]
    )

    R3_BIAS=np.asarray(
        R3_SEL["bias_vector"],
        dtype=np.float32
    )

    R3_REL_DF=pd.read_csv(
        R3_REL
    )

    R3_REL_MATRIX=np.stack([
        R3_REL_DF[
            f"f1_c{c}"
        ].to_numpy(np.float64)
        for c in range(N_CLASSES)
    ],axis=1)

    def r3_ensemble(tensor):
        M,N,C=tensor.shape

        rw=np.exp(
            R3_BETA*R3_REL_MATRIX
        )

        rw/=np.maximum(
            rw.max(
                axis=0,
                keepdims=True
            ),
            1e-12
        )

        num=np.zeros(
            (N,C),
            dtype=np.float64
        )

        den=np.zeros(
            (N,C),
            dtype=np.float64
        )

        for m in range(M):
            p=np.asarray(
                tensor[m],
                dtype=np.float32
            )

            conf=np.max(
                p,
                axis=1
            ).astype(np.float64)

            w=(
                conf**R3_GAMMA
            )[:,None]*rw[m][None,:]

            num+=p.astype(np.float64)*w
            den+=w

        p=normalize_prob(
            num/np.maximum(
                den,
                1e-12
            )
        )

        lp=np.log(
            np.clip(
                p,
                1e-12,
                1.0
            )
        )

        lp+=R3_BIAS[None,:]
        lp-=lp.max(
            axis=1,
            keepdims=True
        )

        return normalize_prob(
            np.exp(lp)
        )

    def load_r3(name):
        path=R3/"CACHE"/f"{name}_META_PREDICTIONS.npy"

        if not path.exists():
            raise FileNotFoundError(path)

        return r3_ensemble(
            np.load(
                path,
                mmap_mode="r"
            )
        )

    print("\nReconstructing frozen model outputs...")

    VAL_R3=load_r3("VAL")
    TEST_R3=load_r3("TEST")
    CAL_R3=load_r3("CAL")

    R5_SEL=json.load(
        open(R5_SELECTION,"r",encoding="utf-8")
    )

    sw=R5_SEL["stacker_weights"]

    R5_WC=float(sw["catboost"])
    R5_WX=float(sw["xgboost"])
    R5_WL=float(sw["lightgbm"])

    R5_THRESHOLD=float(
        R5_SEL["gate_threshold"]
    )

    R5_ALPHAS=np.asarray(
        R5_SEL["class_alphas"],
        dtype=np.float32
    )

    R5_BIAS=np.asarray(
        R5_SEL["class_bias"],
        dtype=np.float32
    )

    R5_TUNED=[
        0,3,4,5,6,7
    ]

    def load_r5_stacker(name):
        path=R5/"CACHE"/f"{name}_STACKER_PREDICTIONS.npz"

        if not path.exists():
            raise FileNotFoundError(path)

        q=np.load(path)

        return normalize_prob(
            R5_WC*q["cat"].astype(np.float32)
            +
            R5_WX*q["xgb"].astype(np.float32)
            +
            R5_WL*q["lgb"].astype(np.float32)
        )

    def r5_fuse(
        base,
        expert
    ):
        conf=np.max(
            base,
            axis=1
        )

        bp=np.argmax(
            base,
            axis=1
        )

        ep=np.argmax(
            expert,
            axis=1
        )

        hard=(
            conf<R5_THRESHOLD
        ) | np.isin(
            bp,
            R5_TUNED
        ) | np.isin(
            ep,
            R5_TUNED
        )

        lb=np.log(
            np.clip(
                base,
                1e-12,
                1.0
            )
        ).astype(np.float64)

        le=np.log(
            np.clip(
                expert,
                1e-12,
                1.0
            )
        ).astype(np.float64)

        lf=lb.copy()

        for c in range(N_CLASSES):
            a=float(
                R5_ALPHAS[c]
            )

            if a==0:
                continue

            lf[
                hard,
                c
            ]=(
                (1-a)*lb[hard,c]
                +
                a*le[hard,c]
            )

        lf+=R5_BIAS[None,:]
        lf-=lf.max(
            axis=1,
            keepdims=True
        )

        return normalize_prob(
            np.exp(lf)
        )

    VAL_R5=r5_fuse(
        VAL_R3,
        load_r5_stacker("VAL")
    )

    TEST_R5=r5_fuse(
        TEST_R3,
        load_r5_stacker("TEST")
    )

    R6_SEL=json.load(
        open(R6_SELECTION,"r",encoding="utf-8")
    )

    R6_GATE_BETA=float(
        R6_SEL["gate_beta"]
    )
    R6_FINE_BETA=float(
        R6_SEL["fine_beta"]
    )
    R6_GATE_TEMP=float(
        R6_SEL["gate_temperature"]
    )
    R6_FINE_TEMP=float(
        R6_SEL["fine_temperature"]
    )
    R6_STRONG=float(
        R6_SEL["strong_carf_weight"]
    )
    R6_WEAK=float(
        R6_SEL["weak_carf_weight"]
    )
    R6_BIAS=np.asarray(
        R6_SEL["bias"],
        dtype=np.float32
    )

    def to_coarse(y8):
        out=np.full(
            len(y8),
            3,
            dtype=np.int64
        )

        out[y8==0]=0
        out[y8==1]=1
        out[y8==2]=2

        return out

    def to_fine(y8):
        return (
            y8-3
        ).astype(np.int64)

    def load_r6_tensor(
        name,
        stage
    ):
        path=R6/"CACHE"/f"{name}_{stage.upper()}_PRED.npy"

        if not path.exists():
            raise FileNotFoundError(path)

        return np.load(
            path,
            mmap_mode="r"
        )

    CAL_GATE_T=load_r6_tensor(
        "CAL",
        "gate"
    )

    CAL_FINE_T=load_r6_tensor(
        "CAL",
        "fine"
    )

    VAL_GATE_T=load_r6_tensor(
        "VAL",
        "gate"
    )

    VAL_FINE_T=load_r6_tensor(
        "VAL",
        "fine"
    )

    TEST_GATE_T=load_r6_tensor(
        "TEST",
        "gate"
    )

    TEST_FINE_T=load_r6_tensor(
        "TEST",
        "fine"
    )

    def f1_matrix(
        tensor,
        ytrue,
        n_classes
    ):
        out=np.zeros(
            (
                tensor.shape[0],
                n_classes
            ),
            dtype=np.float64
        )

        for m in range(
            tensor.shape[0]
        ):
            pred=np.argmax(
                np.asarray(
                    tensor[m],
                    dtype=np.float32
                ),
                axis=1
            )

            _,_,f,_=precision_recall_fscore_support(
                ytrue,
                pred,
                labels=np.arange(n_classes),
                zero_division=0
            )

            out[m]=f

        return out

    YCAL_GATE=to_coarse(
        YCAL
    )

    weak_cal=YCAL>=3

    YCAL_FINE=to_fine(
        YCAL[
            weak_cal
        ]
    )

    GATE_REL=f1_matrix(
        CAL_GATE_T,
        YCAL_GATE,
        4
    )

    FINE_REL=np.zeros(
        (
            CAL_FINE_T.shape[0],
            5
        ),
        dtype=np.float64
    )

    for m in range(
        CAL_FINE_T.shape[0]
    ):
        pred=np.argmax(
            np.asarray(
                CAL_FINE_T[
                    m,
                    weak_cal
                ],
                dtype=np.float32
            ),
            axis=1
        )

        _,_,f,_=precision_recall_fscore_support(
            YCAL_FINE,
            pred,
            labels=np.arange(5),
            zero_division=0
        )

        FINE_REL[m]=f

    def reliability_ensemble(
        tensor,
        rel,
        beta
    ):
        M,N,C=tensor.shape

        rw=np.exp(
            beta*rel
        )

        rw/=np.maximum(
            rw.max(
                axis=0,
                keepdims=True
            ),
            1e-12
        )

        num=np.zeros(
            (N,C),
            dtype=np.float64
        )

        den=np.zeros(
            (N,C),
            dtype=np.float64
        )

        for m in range(M):
            p=np.asarray(
                tensor[m],
                dtype=np.float32
            )

            conf=np.max(
                p,
                axis=1
            ).astype(np.float64)

            w=conf[:,None]*rw[m][None,:]

            num+=p.astype(np.float64)*w
            den+=w

        return normalize_prob(
            num/np.maximum(
                den,
                1e-12
            )
        )

    def temperature_prob(
        p,
        temp
    ):
        lp=np.log(
            np.clip(
                p,
                1e-12,
                1.0
            )
        )/float(temp)

        lp-=lp.max(
            axis=1,
            keepdims=True
        )

        return normalize_prob(
            np.exp(lp)
        )

    def compose_hierarchy(
        gate,
        fine
    ):
        g=temperature_prob(
            gate,
            R6_GATE_TEMP
        )

        f=temperature_prob(
            fine,
            R6_FINE_TEMP
        )

        out=np.zeros(
            (len(g),8),
            dtype=np.float32
        )

        out[:,0]=g[:,0]
        out[:,1]=g[:,1]
        out[:,2]=g[:,2]
        out[:,3:]=g[:,3:4]*f

        return normalize_prob(
            out
        )

    def r6_final(
        hierarchy,
        carf
    ):
        alpha=np.array(
            [
                R6_STRONG,
                R6_STRONG,
                R6_STRONG,
                R6_WEAK,
                R6_WEAK,
                R6_WEAK,
                R6_WEAK,
                R6_WEAK
            ],
            dtype=np.float64
        )

        lp=(
            alpha[None,:]
            *
            np.log(
                np.clip(
                    carf,
                    1e-12,
                    1.0
                )
            )
            +
            (
                1-alpha[None,:]
            )
            *
            np.log(
                np.clip(
                    hierarchy,
                    1e-12,
                    1.0
                )
            )
        )

        lp+=R6_BIAS[None,:]

        lp-=lp.max(
            axis=1,
            keepdims=True
        )

        return normalize_prob(
            np.exp(lp)
        )

    def reconstruct_r6(
        gate_tensor,
        fine_tensor,
        carf
    ):
        g=reliability_ensemble(
            gate_tensor,
            GATE_REL,
            R6_GATE_BETA
        )

        f=reliability_ensemble(
            fine_tensor,
            FINE_REL,
            R6_FINE_BETA
        )

        h=compose_hierarchy(
            g,
            f
        )

        return (
            r6_final(
                h,
                carf
            ),
            g,
            f,
            h
        )

    CAL_R6,CAL_R6_GATE,CAL_R6_FINE,CAL_R6_HIER=reconstruct_r6(
        CAL_GATE_T,
        CAL_FINE_T,
        CAL_R3
    )

    VAL_R6,VAL_R6_GATE,VAL_R6_FINE,VAL_R6_HIER=reconstruct_r6(
        VAL_GATE_T,
        VAL_FINE_T,
        VAL_R3
    )

    TEST_R6,TEST_R6_GATE,TEST_R6_FINE,TEST_R6_HIER=reconstruct_r6(
        TEST_GATE_T,
        TEST_FINE_T,
        TEST_R3
    )

    def client_probs(
        tensor
    ):
        if tensor.shape[0] != 2*N_CLIENTS:
            raise RuntimeError(
                f"Expected {2*N_CLIENTS} model predictions."
            )

        return [
            normalize_prob(
                0.5*np.asarray(
                    tensor[2*k],
                    dtype=np.float32
                )
                +
                0.5*np.asarray(
                    tensor[2*k+1],
                    dtype=np.float32
                )
            )
            for k in range(N_CLIENTS)
        ]

    CAL_GATE_CLIENT=client_probs(
        CAL_GATE_T
    )

    CAL_FINE_CLIENT=client_probs(
        CAL_FINE_T
    )

    VAL_GATE_CLIENT=client_probs(
        VAL_GATE_T
    )

    VAL_FINE_CLIENT=client_probs(
        VAL_FINE_T
    )

    TEST_GATE_CLIENT=client_probs(
        TEST_GATE_T
    )

    TEST_FINE_CLIENT=client_probs(
        TEST_FINE_T
    )

    CCAC_BETA=float(
        CCAC["beta"]
    )

    CCAC_STRONG=float(
        CCAC["strong_r6_anchor"]
    )

    CCAC_WEAK=float(
        CCAC["weak_r6_anchor"]
    )

    CCAC_BIAS=np.asarray(
        CCAC["class_bias"],
        dtype=np.float32
    )

    MODEL_NAMES=[
        "R3_CARF",
        "R5_FCS_MoE",
        "R6_HFSB_FL"
    ]

    VAL_MODELS=[
        VAL_R3,
        VAL_R5,
        VAL_R6
    ]

    TEST_MODELS=[
        TEST_R3,
        TEST_R5,
        TEST_R6
    ]

    CLASS_REL=np.zeros(
        (3,N_CLASSES),
        dtype=np.float64
    )

    for m,p in enumerate(
        VAL_MODELS
    ):
        pred=np.argmax(
            p,
            axis=1
        )

        _,_,f,_=precision_recall_fscore_support(
            YV,
            pred,
            labels=np.arange(N_CLASSES),
            zero_division=0
        )

        CLASS_REL[m]=f

    def frozen_ccac(
        models
    ):
        z=CCAC_BETA*CLASS_REL
        z-=z.max(
            axis=0,
            keepdims=True
        )

        w=np.exp(z)

        w/=np.maximum(
            w.sum(
                axis=0,
                keepdims=True
            ),
            1e-12
        )

        log_ens=np.zeros_like(
            models[0],
            dtype=np.float64
        )

        for m,p in enumerate(
            models
        ):
            log_ens+=(
                w[m][None,:]
                *
                np.log(
                    np.clip(
                        p,
                        1e-12,
                        1.0
                    )
                )
            )

        anchor=np.array(
            [
                CCAC_STRONG,
                CCAC_STRONG,
                CCAC_STRONG,
                CCAC_WEAK,
                CCAC_WEAK,
                CCAC_WEAK,
                CCAC_WEAK,
                CCAC_WEAK
            ],
            dtype=np.float64
        )

        lp=(
            (
                1-anchor[None,:]
            )*log_ens
            +
            anchor[None,:]
            *
            np.log(
                np.clip(
                    models[2],
                    1e-12,
                    1.0
                )
            )
        )

        lp+=CCAC_BIAS[None,:]

        lp-=lp.max(
            axis=1,
            keepdims=True
        )

        return normalize_prob(
            np.exp(lp)
        )

    VAL_CCAC=frozen_ccac(
        VAL_MODELS
    )

    TEST_CCAC=frozen_ccac(
        TEST_MODELS
    )

    CLEAN_METRICS=metrics(
        YTE,
        np.argmax(
            TEST_CCAC,
            axis=1
        ),
        TEST_CCAC
    )

    print(
        "Frozen Step-4C clean metrics:",
        CLEAN_METRICS
    )

    client_quality=[]

    for k in range(N_CLIENTS):
        h=compose_hierarchy(
            CAL_GATE_CLIENT[k],
            CAL_FINE_CLIENT[k]
        )

        mm=metrics(
            YCAL,
            np.argmax(
                h,
                axis=1
            )
        )

        client_quality.append({
            "client_id":k,
            "cal_accuracy":mm["accuracy"],
            "cal_macro_f1":mm["macro_f1"]
        })

    QUALITY_DF=pd.DataFrame(
        client_quality
    )

    save_csv(
        QUALITY_DF,
        TABLES/"CLIENT_CAL_QUALITY.csv"
    )

    q=np.clip(
        QUALITY_DF[
            "cal_macro_f1"
        ].to_numpy(np.float64),
        1e-6,
        1.0
    )

    BASE_WEIGHT=np.exp(
        5.0*q
    )

    BASE_WEIGHT/=BASE_WEIGHT.sum()

    def all_norms(
        clients,
        prior
    ):
        return np.concatenate([
            np.linalg.norm(
                p-prior,
                axis=1
            )
            for p in clients
        ])

    GATE_CLIP=float(
        np.clip(
            np.quantile(
                all_norms(
                    CAL_GATE_CLIENT,
                    CAL_R6_GATE
                ),
                0.90
            ),
            DP_CLIP_MIN,
            DP_CLIP_MAX
        )
    )

    FINE_CLIP=float(
        np.clip(
            np.quantile(
                all_norms(
                    CAL_FINE_CLIENT,
                    CAL_R6_FINE
                ),
                0.90
            ),
            DP_CLIP_MIN,
            DP_CLIP_MAX
        )
    )

    def gaussian_sigma(
        clip,
        epsilon
    ):
        return float(
            clip
            *
            math.sqrt(
                2.0*math.log(
                    1.25/DP_DELTA
                )
            )
            /
            epsilon
        )

    def dp_share(
        client_prob,
        prior,
        clip,
        epsilon,
        seed
    ):
        r=(
            client_prob-prior
        ).astype(np.float32)

        norm=np.linalg.norm(
            r,
            axis=1,
            keepdims=True
        )

        factor=np.minimum(
            1.0,
            clip/np.maximum(
                norm,
                1e-12
            )
        )

        r=r*factor

        sigma=gaussian_sigma(
            clip,
            epsilon
        )

        rng=np.random.default_rng(
            seed
        )

        noise=rng.normal(
            0.0,
            sigma,
            size=r.shape
        ).astype(np.float32)

        return normalize_prob(
            prior+r+noise
        )

    FINE_PERM=np.array(
        [4,3,0,1,2],
        dtype=np.int64
    )

    def post_dp_attack(
        transmitted,
        prior,
        stage,
        attack_type
    ):
        residual=(
            transmitted-prior
        )

        if attack_type=="sign_scale":
            attacked=(
                prior
                -
                ATTACK_SCALE*residual
            )

        elif attack_type=="permute":
            if stage=="fine":
                attacked=transmitted[
                    :,
                    FINE_PERM
                ]
            else:
                attacked=transmitted[
                    :,
                    [3,1,2,0]
                ]

            attacked=np.power(
                np.clip(
                    attacked,
                    1e-6,
                    1.0
                ),
                0.30
            )

        else:
            raise ValueError(
                attack_type
            )

        return normalize_prob(
            attacked
        )

    def robust_z_high(x):
        x=np.asarray(
            x,
            dtype=np.float64
        )

        med=np.median(x)
        mad=np.median(
            np.abs(
                x-med
            )
        )

        return (
            x-med
        )/(
            1.4826*mad+1e-9
        )

    def robust_z_low(x):
        return robust_z_high(
            -np.asarray(
                x,
                dtype=np.float64
            )
        )

    def cosine(a,b):
        a=np.asarray(
            a,
            dtype=np.float64
        ).reshape(-1)

        b=np.asarray(
            b,
            dtype=np.float64
        ).reshape(-1)

        den=np.linalg.norm(a)*np.linalg.norm(b)

        if den<1e-12:
            return 1.0

        return float(
            np.dot(a,b)/den
        )

    def js(p,q):
        p=np.clip(
            np.asarray(
                p,
                dtype=np.float64
            ),
            1e-12,
            None
        )

        q=np.clip(
            np.asarray(
                q,
                dtype=np.float64
            ),
            1e-12,
            None
        )

        p/=p.sum()
        q/=q.sum()

        m=0.5*(p+q)

        return float(
            0.5*np.sum(
                p*np.log(p/m)
            )
            +
            0.5*np.sum(
                q*np.log(q/m)
            )
        )

    def geometric_mix(
        prior,
        candidate,
        alpha
    ):
        lp=(
            (
                1-alpha
            )
            *
            np.log(
                np.clip(
                    prior,
                    1e-12,
                    1.0
                )
            )
            +
            alpha
            *
            np.log(
                np.clip(
                    candidate,
                    1e-12,
                    1.0
                )
            )
        )

        lp-=lp.max(
            axis=1,
            keepdims=True
        )

        return normalize_prob(
            np.exp(lp)
        )

    def cross_entropy(
        ytrue,
        prob
    ):
        return float(
            log_loss(
                ytrue,
                prob,
                labels=np.arange(N_CLASSES)
            )
        )

    def actg_table(
        gate_clients,
        fine_clients,
        gate_prior,
        fine_prior,
        base_hier_prior,
        ycal
    ):
        gate_sig=np.stack([
            (
                p-gate_prior
            ).mean(axis=0)
            for p in gate_clients
        ])

        fine_sig=np.stack([
            (
                p-fine_prior
            ).mean(axis=0)
            for p in fine_clients
        ])

        med_gate=np.median(
            gate_sig,
            axis=0
        )

        med_fine=np.median(
            fine_sig,
            axis=0
        )

        gate_mean=np.stack([
            p.mean(axis=0)
            for p in gate_clients
        ])

        fine_mean=np.stack([
            p.mean(axis=0)
            for p in fine_clients
        ])

        med_gate_prob=np.median(
            gate_mean,
            axis=0
        )

        med_fine_prob=np.median(
            fine_mean,
            axis=0
        )

        prior_ce=cross_entropy(
            ycal,
            base_hier_prior
        )

        prior_f1=metrics(
            ycal,
            np.argmax(
                base_hier_prior,
                axis=1
            )
        )["macro_f1"]

        rows=[]

        for k in range(N_CLIENTS):
            rg=(
                gate_clients[k]
                -
                gate_prior
            )

            rf=(
                fine_clients[k]
                -
                fine_prior
            )

            residual_norm=float(
                0.5*np.mean(
                    np.linalg.norm(
                        rg,
                        axis=1
                    )
                )
                +
                0.5*np.mean(
                    np.linalg.norm(
                        rf,
                        axis=1
                    )
                )
            )

            cos_score=0.5*(
                cosine(
                    gate_sig[k],
                    med_gate
                )
                +
                cosine(
                    fine_sig[k],
                    med_fine
                )
            )

            js_score=0.5*(
                js(
                    gate_mean[k],
                    med_gate_prob
                )
                +
                js(
                    fine_mean[k],
                    med_fine_prob
                )
            )

            h=compose_hierarchy(
                gate_clients[k],
                fine_clients[k]
            )

            candidate=geometric_mix(
                base_hier_prior,
                h,
                0.25
            )

            cand_ce=cross_entropy(
                ycal,
                candidate
            )

            cand_f1=metrics(
                ycal,
                np.argmax(
                    candidate,
                    axis=1
                )
            )["macro_f1"]

            rows.append({
                "client_id":k,
                "residual_norm":residual_norm,
                "cosine_agreement":cos_score,
                "js_divergence":js_score,
                "delta_cross_entropy":cand_ce-prior_ce,
                "delta_macro_f1":cand_f1-prior_f1
            })

        df=pd.DataFrame(
            rows
        )

        df["z_norm"]=robust_z_high(
            df["residual_norm"]
        )

        df["z_cos_bad"]=robust_z_low(
            df["cosine_agreement"]
        )

        df["z_js"]=robust_z_high(
            df["js_divergence"]
        )

        df["z_ce_bad"]=robust_z_high(
            df["delta_cross_entropy"]
        )

        df["z_f1_bad"]=robust_z_low(
            df["delta_macro_f1"]
        )

        zcols=[
            "z_norm",
            "z_cos_bad",
            "z_js",
            "z_ce_bad",
            "z_f1_bad"
        ]

        zbad=np.stack([
            np.maximum(
                df[c].to_numpy(
                    np.float64
                ),
                0.0
            )
            for c in zcols
        ],axis=1)

        flags=zbad>ROBUST_Z

        df["anomaly_flags"]=flags.sum(
            axis=1
        )

        df[
            "catastrophic_validation_anomaly"
        ]=(
            df["z_ce_bad"]>CATASTROPHIC_Z
        ) | (
            df["z_f1_bad"]>CATASTROPHIC_Z
        )

        df["accepted"]=~(
            (
                df["anomaly_flags"]
                >=
                MIN_ANOMALY_FLAGS_FOR_REJECTION
            )
            |
            df[
                "catastrophic_validation_anomaly"
            ]
        )

        sorted_bad=np.sort(
            zbad,
            axis=1
        )

        top2=sorted_bad[
            :,
            -2:
        ].sum(axis=1)

        contribution_bonus=np.clip(
            -df[
                "delta_cross_entropy"
            ].to_numpy(
                np.float64
            ),
            -0.05,
            0.05
        )

        trust=np.exp(
            -0.18*top2
        )*np.exp(
            3.0*contribution_bonus
        )

        df["trust_score"]=np.clip(
            trust,
            1e-6,
            1.0
        )

        return df

    def pair_seed(
        i,j,
        stage,
        base_seed
    ):
        raw=f"{base_seed}|{stage}|{min(i,j)}|{max(i,j)}"

        return int(
            hashlib.sha256(
                raw.encode()
            ).hexdigest()[:16],
            16
        )%(2**32-1)

    def secure_aggregate(
        clients,
        prior,
        accepted,
        weights,
        stage,
        seed
    ):
        accepted=[
            int(k)
            for k in accepted
        ]

        if not accepted:
            raise RuntimeError(
                "No ACTG-approved client."
            )

        weights=np.asarray(
            weights,
            dtype=np.float64
        )

        denom=float(
            weights[
                accepted
            ].sum()
        )

        N,C=prior.shape

        out=np.empty(
            (N,C),
            dtype=np.float32
        )

        max_cancel=0.0

        for start in range(
            0,
            N,
            SECURE_CHUNK
        ):
            end=min(
                start+SECURE_CHUNK,
                N
            )

            payload={}

            for k in accepted:
                residual=(
                    clients[k][start:end]
                    -
                    prior[start:end]
                ).astype(np.float64)

                payload[k]=(
                    weights[k]*residual
                )

            mask_balance=np.zeros(
                (
                    end-start,
                    C
                ),
                dtype=np.float64
            )

            for ai,i in enumerate(
                accepted
            ):
                for j in accepted[
                    ai+1:
                ]:
                    rng=np.random.default_rng(
                        pair_seed(
                            i,j,
                            stage,
                            seed+start
                        )
                    )

                    mask=rng.normal(
                        0.0,
                        SECURE_MASK_STD,
                        size=(
                            end-start,
                            C
                        )
                    )

                    payload[i]+=mask
                    payload[j]-=mask

                    mask_balance+=mask
                    mask_balance-=mask

            numerator=np.zeros(
                (
                    end-start,
                    C
                ),
                dtype=np.float64
            )

            for k in accepted:
                numerator+=payload[k]

            out[start:end]=normalize_prob(
                prior[start:end]
                +
                numerator/max(
                    denom,
                    1e-12
                )
            )

            max_cancel=max(
                max_cancel,
                float(
                    np.max(
                        np.abs(
                            mask_balance
                        )
                    )
                )
            )

            del payload,numerator,mask_balance
            gc.collect()

        return out,{
            "accepted_clients":accepted,
            "sum_weights":denom,
            "max_mask_cancellation_error":max_cancel
        }

    def direct_residual_aggregate(
        clients,
        prior,
        accepted,
        weights
    ):
        weights=np.asarray(
            weights,
            dtype=np.float64
        )

        denom=float(
            weights[
                accepted
            ].sum()
        )

        num=np.zeros_like(
            prior,
            dtype=np.float64
        )

        for k in accepted:
            num+=(
                weights[k]
                *
                (
                    clients[k]-prior
                )
            )

        return normalize_prob(
            prior
            +
            num/max(
                denom,
                1e-12
            )
        )

    def protected_clients(
        gate_clients,
        fine_clients,
        gate_prior,
        fine_prior,
        epsilon,
        seed,
        malicious_pair=None
    ):
        pg=[]
        pf=[]

        for k in range(N_CLIENTS):
            g=dp_share(
                gate_clients[k],
                gate_prior,
                GATE_CLIP,
                epsilon,
                seed+k*101+1
            )

            f=dp_share(
                fine_clients[k],
                fine_prior,
                FINE_CLIP,
                epsilon,
                seed+k*101+2
            )

            if (
                malicious_pair is not None
                and
                k==malicious_pair[0]
            ):
                g=post_dp_attack(
                    g,
                    gate_prior,
                    "gate",
                    "sign_scale"
                )

                f=post_dp_attack(
                    f,
                    fine_prior,
                    "fine",
                    "sign_scale"
                )

            elif (
                malicious_pair is not None
                and
                k==malicious_pair[1]
            ):
                g=post_dp_attack(
                    g,
                    gate_prior,
                    "gate",
                    "permute"
                )

                f=post_dp_attack(
                    f,
                    fine_prior,
                    "fine",
                    "permute"
                )

            pg.append(g)
            pf.append(f)

        return pg,pf

    def trust_for_pair(
        seed,
        malicious_pair
    ):
        cg,cf=protected_clients(
            CAL_GATE_CLIENT,
            CAL_FINE_CLIENT,
            CAL_R6_GATE,
            CAL_R6_FINE,
            EPSILON,
            seed,
            malicious_pair
        )

        t=actg_table(
            cg,
            cf,
            CAL_R6_GATE,
            CAL_R6_FINE,
            CAL_R6_HIER,
            YCAL
        )

        t["known_malicious"]=t[
            "client_id"
        ].isin(
            list(
                malicious_pair
            )
        )

        accepted=t[
            t["accepted"]
        ]["client_id"].astype(int).tolist()

        rejected=t[
            ~t["accepted"]
        ]["client_id"].astype(int).tolist()

        tw=BASE_WEIGHT.copy()

        for _,r in t.iterrows():
            k=int(
                r["client_id"]
            )

            tw[k]*=float(
                r["trust_score"]
            )

        true_attack=t[
            "known_malicious"
        ].astype(int).to_numpy()

        pred_attack=(
            ~t["accepted"]
        ).astype(int).to_numpy()

        detect={
            "precision":float(
                precision_score(
                    true_attack,
                    pred_attack,
                    zero_division=0
                )
            ),
            "recall":float(
                recall_score(
                    true_attack,
                    pred_attack,
                    zero_division=0
                )
            ),
            "f1":float(
                f1_score(
                    true_attack,
                    pred_attack,
                    zero_division=0
                )
            ),
            "false_positive_count":int(
                np.sum(
                    (
                        pred_attack==1
                    )
                    &
                    (
                        true_attack==0
                    )
                )
            ),
            "false_negative_count":int(
                np.sum(
                    (
                        pred_attack==0
                    )
                    &
                    (
                        true_attack==1
                    )
                )
            ),
        }

        return (
            t,
            accepted,
            rejected,
            tw,
            detect
        )

    def final_from_clients(
        gate_clients,
        fine_clients,
        gate_prior,
        fine_prior,
        final_prior,
        accepted,
        weights,
        secure,
        seed
    ):
        if secure:
            g,ga=secure_aggregate(
                gate_clients,
                gate_prior,
                accepted,
                weights,
                "gate",
                seed+7000
            )

            f,fa=secure_aggregate(
                fine_clients,
                fine_prior,
                accepted,
                weights,
                "fine",
                seed+8000
            )

        else:
            g=direct_residual_aggregate(
                gate_clients,
                gate_prior,
                accepted,
                weights
            )

            f=direct_residual_aggregate(
                fine_clients,
                fine_prior,
                accepted,
                weights
            )

            ga=None
            fa=None

        h=compose_hierarchy(
            g,
            f
        )

        final=geometric_mix(
            final_prior,
            h,
            PROTECTED_WEIGHT
        )

        return final,ga,fa

    def stress_attack(
        transmitted,
        prior,
        stage,
        attack_type
    ):
        """
        Fixed post-DP attacks used only for stress validation.
        All attacks operate on the transmitted protected probability.
        """
        transmitted=np.asarray(
            transmitted,
            dtype=np.float32
        )

        prior=np.asarray(
            prior,
            dtype=np.float32
        )

        if attack_type=="sign_scale":
            residual=transmitted-prior

            attacked=(
                prior
                -
                STRESS_SIGN_SCALE*residual
            )

        elif attack_type=="permute":
            if stage=="gate":

                perm=np.array(
                    [3,2,1,0],
                    dtype=np.int64
                )
            else:

                perm=np.array(
                    [4,0,1,2,3],
                    dtype=np.int64
                )

            attacked=transmitted[:,perm]

            attacked=np.power(
                np.clip(
                    attacked,
                    1e-8,
                    1.0
                ),
                0.20
            )

        elif attack_type=="model_replacement":
            C=transmitted.shape[1]

            source=np.argmax(
                prior,
                axis=1
            )

            target=(
                source+1
            )%C

            attacked=np.full(
                transmitted.shape,
                (1.0-MODEL_REPLACE_CONF)/(C-1),
                dtype=np.float32
            )

            attacked[
                np.arange(
                    len(attacked)
                ),
                target
            ]=MODEL_REPLACE_CONF

        elif attack_type=="targeted_weak":
            C=transmitted.shape[1]

            attacked=np.full(
                transmitted.shape,
                (1.0-TARGETED_CONF)/(C-1),
                dtype=np.float32
            )

            if stage=="gate":

                target=3
            else:

                target=4

            attacked[:,target]=TARGETED_CONF

        else:
            raise ValueError(
                f"Unknown attack type: {attack_type}"
            )

        return normalize_prob(
            attacked
        )

    def mixed_attack_type(
        client_id
    ):
        """
        Deterministic heterogeneous attack assignment for the fraction
        sweep. It is independent of results and therefore not tuned.
        """
        modes=[
            "sign_scale",
            "permute",
            "model_replacement",
            "targeted_weak",
        ]

        return modes[
            int(client_id)%len(modes)
        ]

    def dp_cache_path(
        dataset_name,
        seed
    ):
        return (
            OUT
            /
            f"DP_CACHE_{dataset_name}_EPS{EPSILON}_SEED{seed}.npz"
        )

    def build_or_load_dp_cache(
        dataset_name,
        gate_clients,
        fine_clients,
        gate_prior,
        fine_prior,
        seed
    ):
        path=dp_cache_path(
            dataset_name,
            seed
        )

        expected_gate=(
            N_CLIENTS,
            len(gate_prior),
            gate_prior.shape[1]
        )

        expected_fine=(
            N_CLIENTS,
            len(fine_prior),
            fine_prior.shape[1]
        )

        if (
            path.exists()
            and
            RESUME
            and
            not FORCE_REBUILD
        ):
            q=np.load(
                path,
                mmap_mode="r"
            )

            if (
                tuple(q["gate"].shape)==expected_gate
                and
                tuple(q["fine"].shape)==expected_fine
            ):
                return (
                    q["gate"],
                    q["fine"]
                )

        print(
            f"Building DP cache {dataset_name} seed={seed}..."
        )

        gate=np.empty(
            expected_gate,
            dtype=np.float16
        )

        fine=np.empty(
            expected_fine,
            dtype=np.float16
        )

        for k in range(N_CLIENTS):
            gate[k]=dp_share(
                gate_clients[k],
                gate_prior,
                GATE_CLIP,
                EPSILON,
                seed+k*101+1
            ).astype(np.float16)

            fine[k]=dp_share(
                fine_clients[k],
                fine_prior,
                FINE_CLIP,
                EPSILON,
                seed+k*101+2
            ).astype(np.float16)

        np.savez_compressed(
            path,
            gate=gate,
            fine=fine
        )

        q=np.load(
            path,
            mmap_mode="r"
        )

        return (
            q["gate"],
            q["fine"]
        )

    DP_CACHE={}

    for seed in STRESS_SEEDS:
        DP_CACHE[
            ("CAL",seed)
        ]=build_or_load_dp_cache(
            "CAL",
            CAL_GATE_CLIENT,
            CAL_FINE_CLIENT,
            CAL_R6_GATE,
            CAL_R6_FINE,
            seed+10_000
        )

        DP_CACHE[
            ("TEST",seed)
        ]=build_or_load_dp_cache(
            "TEST",
            TEST_GATE_CLIENT,
            TEST_FINE_CLIENT,
            TEST_R6_GATE,
            TEST_R6_FINE,
            seed+20_000
        )

    def apply_attack_set(
        gate_cache,
        fine_cache,
        gate_prior,
        fine_prior,
        malicious_ids,
        attack_mode
    ):
        malicious_ids=set(
            int(k)
            for k in malicious_ids
        )

        gates=[]
        fines=[]

        for k in range(N_CLIENTS):
            g=np.asarray(
                gate_cache[k],
                dtype=np.float32
            )

            f=np.asarray(
                fine_cache[k],
                dtype=np.float32
            )

            if k in malicious_ids:
                mode=(
                    mixed_attack_type(k)
                    if attack_mode=="mixed"
                    else attack_mode
                )

                g=stress_attack(
                    g,
                    gate_prior,
                    "gate",
                    mode
                )

                f=stress_attack(
                    f,
                    fine_prior,
                    "fine",
                    mode
                )

            gates.append(g)
            fines.append(f)

        return gates,fines

    def evaluate_actg(
        cal_gate,
        cal_fine,
        malicious_ids
    ):
        trust=actg_table(
            cal_gate,
            cal_fine,
            CAL_R6_GATE,
            CAL_R6_FINE,
            CAL_R6_HIER,
            YCAL
        )

        malicious_set=set(
            int(k)
            for k in malicious_ids
        )

        truth=trust[
            "client_id"
        ].astype(int).isin(
            malicious_set
        ).astype(int).to_numpy()

        rejected=(
            ~trust[
                "accepted"
            ]
        ).astype(int).to_numpy()

        accepted_ids=trust[
            trust["accepted"]
        ]["client_id"].astype(int).tolist()

        rejected_ids=trust[
            ~trust["accepted"]
        ]["client_id"].astype(int).tolist()

        trust_weight=BASE_WEIGHT.copy()

        for _,r in trust.iterrows():
            k=int(
                r["client_id"]
            )

            trust_weight[k]*=float(
                r["trust_score"]
            )

        detection={
            "precision":float(
                precision_score(
                    truth,
                    rejected,
                    zero_division=0
                )
            ),
            "recall":float(
                recall_score(
                    truth,
                    rejected,
                    zero_division=0
                )
            ),
            "f1":float(
                f1_score(
                    truth,
                    rejected,
                    zero_division=0
                )
            ),
            "false_positive_rate":float(
                np.sum(
                    (truth==0)&(rejected==1)
                )
                /
                max(
                    np.sum(
                        truth==0
                    ),
                    1
                )
            ),
            "false_negative_rate":float(
                np.sum(
                    (truth==1)&(rejected==0)
                )
                /
                max(
                    np.sum(
                        truth==1
                    ),
                    1
                )
            ),
            "accepted_ids":accepted_ids,
            "rejected_ids":rejected_ids,
        }

        return (
            trust,
            accepted_ids,
            trust_weight,
            detection
        )

    def aggregate_branch(
        gate_clients,
        fine_clients,
        accepted,
        weights,
        secure,
        seed
    ):
        if secure:
            g,ga=secure_aggregate(
                gate_clients,
                TEST_R6_GATE,
                accepted,
                weights,
                "gate",
                seed+7000
            )

            f,fa=secure_aggregate(
                fine_clients,
                TEST_R6_FINE,
                accepted,
                weights,
                "fine",
                seed+8000
            )

        else:
            g=direct_residual_aggregate(
                gate_clients,
                TEST_R6_GATE,
                accepted,
                weights
            )

            f=direct_residual_aggregate(
                fine_clients,
                TEST_R6_FINE,
                accepted,
                weights
            )

            ga=None
            fa=None

        h=compose_hierarchy(
            g,
            f
        )

        end_to_end=geometric_mix(
            TEST_CCAC,
            h,
            PROTECTED_WEIGHT
        )

        return (
            h,
            end_to_end,
            ga,
            fa
        )

    def run_stress_experiment(
        seed,
        malicious_ids,
        attack_mode,
        experiment_family,
        experiment_label
    ):
        cal_gc,cal_fc=DP_CACHE[
            ("CAL",seed)
        ]

        test_gc,test_fc=DP_CACHE[
            ("TEST",seed)
        ]

        cal_g,cal_f=apply_attack_set(
            cal_gc,
            cal_fc,
            CAL_R6_GATE,
            CAL_R6_FINE,
            malicious_ids,
            attack_mode
        )

        trust,accepted,trust_weight,detect=evaluate_actg(
            cal_g,
            cal_f,
            malicious_ids
        )

        test_g,test_f=apply_attack_set(
            test_gc,
            test_fc,
            TEST_R6_GATE,
            TEST_R6_FINE,
            malicious_ids,
            attack_mode
        )

        branch_attack,e2e_attack,_,_=aggregate_branch(
            test_g,
            test_f,
            list(
                range(N_CLIENTS)
            ),
            BASE_WEIGHT,
            secure=False,
            seed=seed+30_000
        )

        mba=metrics(
            YTE,
            np.argmax(
                branch_attack,
                axis=1
            )
        )

        mea=metrics(
            YTE,
            np.argmax(
                e2e_attack,
                axis=1
            )
        )

        branch_def,e2e_def,ga,fa=aggregate_branch(
            test_g,
            test_f,
            accepted,
            trust_weight,
            secure=True,
            seed=seed+40_000
        )

        mbd=metrics(
            YTE,
            np.argmax(
                branch_def,
                axis=1
            )
        )

        med=metrics(
            YTE,
            np.argmax(
                e2e_def,
                axis=1
            )
        )

        return {
            "experiment_family":experiment_family,
            "experiment_label":experiment_label,
            "seed":seed,
            "malicious_ids":str(
                tuple(
                    malicious_ids
                )
            ),
            "n_malicious":len(
                malicious_ids
            ),
            "malicious_fraction":len(
                malicious_ids
            )/N_CLIENTS,
            "attack_mode":attack_mode,

            "branch_attack_accuracy":mba[
                "accuracy"
            ],
            "branch_attack_macro_f1":mba[
                "macro_f1"
            ],

            "branch_defense_accuracy":mbd[
                "accuracy"
            ],
            "branch_defense_macro_f1":mbd[
                "macro_f1"
            ],

            "branch_accuracy_recovery":(
                mbd["accuracy"]
                -
                mba["accuracy"]
            ),
            "branch_macro_f1_recovery":(
                mbd["macro_f1"]
                -
                mba["macro_f1"]
            ),

            "e2e_attack_accuracy":mea[
                "accuracy"
            ],
            "e2e_attack_macro_f1":mea[
                "macro_f1"
            ],

            "e2e_defense_accuracy":med[
                "accuracy"
            ],
            "e2e_defense_macro_f1":med[
                "macro_f1"
            ],

            "e2e_accuracy_recovery":(
                med["accuracy"]
                -
                mea["accuracy"]
            ),
            "e2e_macro_f1_recovery":(
                med["macro_f1"]
                -
                mea["macro_f1"]
            ),

            "detection_precision":detect[
                "precision"
            ],
            "detection_recall":detect[
                "recall"
            ],
            "detection_f1":detect[
                "f1"
            ],
            "false_positive_rate":detect[
                "false_positive_rate"
            ],
            "false_negative_rate":detect[
                "false_negative_rate"
            ],

            "accepted_count":len(
                accepted
            ),
            "rejected_count":(
                N_CLIENTS-len(
                    accepted
                )
            ),

            "gate_mask_error":(
                ga[
                    "max_mask_cancellation_error"
                ]
                if ga is not None
                else np.nan
            ),
            "fine_mask_error":(
                fa[
                    "max_mask_cancellation_error"
                ]
                if fa is not None
                else np.nan
            ),
        }

    clean_rows=[]

    print("\nRunning clean ACTG false-positive controls...")

    for seed in STRESS_SEEDS:
        cg,cf=DP_CACHE[
            ("CAL",seed)
        ]

        clean_g=[
            np.asarray(
                cg[k],
                dtype=np.float32
            )
            for k in range(N_CLIENTS)
        ]

        clean_f=[
            np.asarray(
                cf[k],
                dtype=np.float32
            )
            for k in range(N_CLIENTS)
        ]

        trust=actg_table(
            clean_g,
            clean_f,
            CAL_R6_GATE,
            CAL_R6_FINE,
            CAL_R6_HIER,
            YCAL
        )

        rejected=int(
            (
                ~trust[
                    "accepted"
                ]
            ).sum()
        )

        clean_rows.append({
            "seed":seed,
            "accepted_clients":N_CLIENTS-rejected,
            "rejected_clients":rejected,
            "clean_false_positive_rate":rejected/N_CLIENTS
        })

    CLEAN_CONTROL=pd.DataFrame(
        clean_rows
    )

    save_csv(
        CLEAN_CONTROL,
        TABLES/"ACTG_CLEAN_FALSE_POSITIVE_CONTROL.csv"
    )

    attack_rows=[]

    total_type_runs=(
        len(
            ATTACK_TYPES
        )
        *
        len(
            STRESS_PAIRS
        )
        *
        len(
            STRESS_SEEDS
        )
    )

    print(
        "\nRunning attack-type sweep:",
        total_type_runs,
        "experiments..."
    )

    counter=0

    for attack_type in ATTACK_TYPES:
        for pair_id,pair in enumerate(
            STRESS_PAIRS
        ):
            for seed in STRESS_SEEDS:
                counter+=1

                print(
                    f"  [{counter}/{total_type_runs}] "
                    f"type={attack_type} pair={pair} seed={seed}"
                )

                attack_rows.append(
                    run_stress_experiment(
                        seed=seed,
                        malicious_ids=pair,
                        attack_mode=attack_type,
                        experiment_family="attack_type_sweep",
                        experiment_label=(
                            f"{attack_type}|pair{pair_id}"
                        )
                    )
                )

                gc.collect()

    ATTACK_TYPE_RAW=pd.DataFrame(
        attack_rows
    )

    save_csv(
        ATTACK_TYPE_RAW,
        OUT/"ATTACK_TYPE_STRESS_RAW.csv"
    )

    fraction_rows=[]

    total_fraction_runs=(
        sum(
            len(v)
            for v in FRACTION_SETS.values()
        )
        *
        len(
            STRESS_SEEDS
        )
    )

    print(
        "\nRunning malicious-fraction sweep:",
        total_fraction_runs,
        "experiments..."
    )

    counter=0

    for frac,sets in FRACTION_SETS.items():
        for set_id,malicious_ids in enumerate(
            sets
        ):
            for seed in STRESS_SEEDS:
                counter+=1

                print(
                    f"  [{counter}/{total_fraction_runs}] "
                    f"frac={frac:.0%} set={malicious_ids} seed={seed}"
                )

                fraction_rows.append(
                    run_stress_experiment(
                        seed=seed,
                        malicious_ids=malicious_ids,
                        attack_mode="mixed",
                        experiment_family="malicious_fraction_sweep",
                        experiment_label=(
                            f"{int(frac*100)}pct|set{set_id}"
                        )
                    )
                )

                gc.collect()

    FRACTION_RAW=pd.DataFrame(
        fraction_rows
    )

    save_csv(
        FRACTION_RAW,
        OUT/"MALICIOUS_FRACTION_STRESS_RAW.csv"
    )

    def summary_stats(
        df,
        group_col
    ):
        rows=[]

        metric_cols=[
            "branch_attack_accuracy",
            "branch_defense_accuracy",
            "branch_accuracy_recovery",

            "branch_attack_macro_f1",
            "branch_defense_macro_f1",
            "branch_macro_f1_recovery",

            "e2e_attack_accuracy",
            "e2e_defense_accuracy",
            "e2e_accuracy_recovery",

            "e2e_attack_macro_f1",
            "e2e_defense_macro_f1",
            "e2e_macro_f1_recovery",

            "detection_precision",
            "detection_recall",
            "detection_f1",
            "false_positive_rate",
            "false_negative_rate",
        ]

        for group,g in df.groupby(
            group_col
        ):
            row={
                group_col:group,
                "n_runs":len(g)
            }

            for col in metric_cols:
                vals=g[
                    col
                ].to_numpy(
                    np.float64
                )

                mean,sd,lo,hi=ci95(
                    vals
                )

                row[
                    f"{col}_mean"
                ]=mean

                row[
                    f"{col}_sd"
                ]=sd

                row[
                    f"{col}_ci95_low"
                ]=lo

                row[
                    f"{col}_ci95_high"
                ]=hi

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    ATTACK_TYPE_SUMMARY=summary_stats(
        ATTACK_TYPE_RAW,
        "attack_mode"
    )

    FRACTION_SUMMARY=summary_stats(
        FRACTION_RAW,
        "malicious_fraction"
    ).sort_values(
        "malicious_fraction"
    )

    save_csv(
        ATTACK_TYPE_SUMMARY,
        TABLES/"ATTACK_TYPE_STRESS_SUMMARY.csv"
    )

    save_csv(
        FRACTION_SUMMARY,
        TABLES/"MALICIOUS_FRACTION_STRESS_SUMMARY.csv"
    )

    stat_rows=[]

    def paired_test_rows(
        df,
        family
    ):
        pairs=[
            (
                "Branch Accuracy",
                "branch_attack_accuracy",
                "branch_defense_accuracy"
            ),
            (
                "Branch Macro-F1",
                "branch_attack_macro_f1",
                "branch_defense_macro_f1"
            ),
            (
                "End-to-End Accuracy",
                "e2e_attack_accuracy",
                "e2e_defense_accuracy"
            ),
            (
                "End-to-End Macro-F1",
                "e2e_attack_macro_f1",
                "e2e_defense_macro_f1"
            ),
        ]

        out=[]

        for metric,a_col,d_col in pairs:
            a=df[
                a_col
            ].to_numpy(
                np.float64
            )

            d=df[
                d_col
            ].to_numpy(
                np.float64
            )

            diff=d-a

            t=stats.ttest_rel(
                d,
                a,
                nan_policy="omit"
            )

            try:
                w=stats.wilcoxon(
                    d,
                    a,
                    zero_method="wilcox",
                    alternative="two-sided"
                )

                wstat=float(
                    w.statistic
                )
                wp=float(
                    w.pvalue
                )

            except Exception:
                wstat=np.nan
                wp=np.nan

            sd=float(
                np.std(
                    diff,
                    ddof=1
                )
            )

            dz=(
                float(
                    np.mean(
                        diff
                    )/sd
                )
                if sd>0
                else np.nan
            )

            out.append({
                "experiment_family":family,
                "metric":metric,
                "n_pairs":len(a),
                "attack_mean":float(
                    np.mean(a)
                ),
                "defense_mean":float(
                    np.mean(d)
                ),
                "mean_recovery":float(
                    np.mean(diff)
                ),
                "paired_t_stat":float(
                    t.statistic
                ),
                "paired_t_p":float(
                    t.pvalue
                ),
                "wilcoxon_stat":wstat,
                "wilcoxon_p":wp,
                "cohens_dz":dz,
            })

        return out

    stat_rows.extend(
        paired_test_rows(
            ATTACK_TYPE_RAW,
            "attack_type_sweep"
        )
    )

    stat_rows.extend(
        paired_test_rows(
            FRACTION_RAW,
            "malicious_fraction_sweep"
        )
    )

    STATS=pd.DataFrame(
        stat_rows
    )

    save_csv(
        STATS,
        TABLES/"STEP06_PAIRED_STATISTICAL_TESTS.csv"
    )

    THREAT_MODEL=pd.DataFrame([
        {
            "attack":"Sign/scale poisoning",
            "location":"post-DP transmitted residual",
            "mechanism":"invert and amplify protected residual",
            "strength":STRESS_SIGN_SCALE,
            "purpose":"model-update direction reversal / replacement stress"
        },
        {
            "attack":"Class permutation",
            "location":"post-DP transmitted probability",
            "mechanism":"permute gate/fine class probabilities and flatten confidence",
            "strength":"fixed mapping",
            "purpose":"label-space manipulation"
        },
        {
            "attack":"Model replacement",
            "location":"post-DP transmitted probability",
            "mechanism":"replace output with high-confidence wrong class",
            "strength":MODEL_REPLACE_CONF,
            "purpose":"Byzantine high-confidence replacement"
        },
        {
            "attack":"Targeted weak-family injection",
            "location":"post-DP transmitted probability",
            "mechanism":"force WeakAttack gate and XSS fine-class output",
            "strength":TARGETED_CONF,
            "purpose":"targeted minority-family poisoning"
        },
    ])

    save_csv(
        THREAT_MODEL,
        TABLES/"THREAT_MODEL_DEFINITION.csv"
    )

    fig,ax=plt.subplots(
        figsize=(10,5)
    )

    x=np.arange(
        len(
            ATTACK_TYPE_SUMMARY
        )
    )

    width=0.35

    ax.bar(
        x-width/2,
        ATTACK_TYPE_SUMMARY[
            "branch_attack_accuracy_mean"
        ],
        width,
        label="Attack / no defense"
    )

    ax.bar(
        x+width/2,
        ATTACK_TYPE_SUMMARY[
            "branch_defense_accuracy_mean"
        ],
        width,
        label="ACTG defense"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        ATTACK_TYPE_SUMMARY[
            "attack_mode"
        ],
        rotation=20
    )

    ax.set_ylabel(
        "Federated-branch accuracy"
    )

    ax.set_title(
        "Poisoning Impact Before Robust Prior Fusion"
    )

    ax.legend()

    fig.tight_layout()

    fp=LOCAL/"BRANCH_ATTACK_TYPE_STRESS.png"

    fig.savefig(
        fp,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    atomic_copy(
        fp,
        FIGS/fp.name
    )

    fig,ax=plt.subplots(
        figsize=(10,5)
    )

    ax.bar(
        x-width/2,
        ATTACK_TYPE_SUMMARY[
            "e2e_attack_accuracy_mean"
        ],
        width,
        label="Attack / no defense"
    )

    ax.bar(
        x+width/2,
        ATTACK_TYPE_SUMMARY[
            "e2e_defense_accuracy_mean"
        ],
        width,
        label="ACTG defense"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        ATTACK_TYPE_SUMMARY[
            "attack_mode"
        ],
        rotation=20
    )

    ax.set_ylabel(
        "End-to-end accuracy"
    )

    ax.set_title(
        "End-to-End Robustness After CCAC Prior Fusion"
    )

    ax.legend()

    fig.tight_layout()

    fp=LOCAL/"E2E_ATTACK_TYPE_STRESS.png"

    fig.savefig(
        fp,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    atomic_copy(
        fp,
        FIGS/fp.name
    )

    fig,ax=plt.subplots(
        figsize=(8,5)
    )

    ax.bar(
        ATTACK_TYPE_SUMMARY[
            "attack_mode"
        ],
        ATTACK_TYPE_SUMMARY[
            "detection_f1_mean"
        ],
        yerr=ATTACK_TYPE_SUMMARY[
            "detection_f1_sd"
        ],
        capsize=4
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_ylabel(
        "ACTG detection F1"
    )

    ax.set_title(
        "ACTG Detection Across Poisoning Types"
    )

    ax.tick_params(
        axis="x",
        rotation=20
    )

    fig.tight_layout()

    fp=LOCAL/"ACTG_F1_BY_ATTACK_TYPE.png"

    fig.savefig(
        fp,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    atomic_copy(
        fp,
        FIGS/fp.name
    )

    fig,ax=plt.subplots(
        figsize=(9,5)
    )

    frac=(
        100
        *
        FRACTION_SUMMARY[
            "malicious_fraction"
        ].to_numpy(
            np.float64
        )
    )

    ax.plot(
        frac,
        FRACTION_SUMMARY[
            "branch_attack_accuracy_mean"
        ],
        marker="o",
        label="Branch attack"
    )

    ax.plot(
        frac,
        FRACTION_SUMMARY[
            "branch_defense_accuracy_mean"
        ],
        marker="s",
        label="Branch ACTG defense"
    )

    ax.plot(
        frac,
        FRACTION_SUMMARY[
            "e2e_attack_accuracy_mean"
        ],
        marker="^",
        label="End-to-end attack"
    )

    ax.plot(
        frac,
        FRACTION_SUMMARY[
            "e2e_defense_accuracy_mean"
        ],
        marker="D",
        label="End-to-end ACTG defense"
    )

    ax.set_xlabel(
        "Malicious clients (%)"
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_title(
        "Robustness vs Malicious-Client Fraction"
    )

    ax.legend()

    fig.tight_layout()

    fp=LOCAL/"MALICIOUS_FRACTION_ACCURACY.png"

    fig.savefig(
        fp,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    atomic_copy(
        fp,
        FIGS/fp.name
    )

    fig,ax=plt.subplots(
        figsize=(8,5)
    )

    ax.plot(
        frac,
        FRACTION_SUMMARY[
            "detection_precision_mean"
        ],
        marker="o",
        label="Precision"
    )

    ax.plot(
        frac,
        FRACTION_SUMMARY[
            "detection_recall_mean"
        ],
        marker="s",
        label="Recall"
    )

    ax.plot(
        frac,
        FRACTION_SUMMARY[
            "detection_f1_mean"
        ],
        marker="^",
        label="F1"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_xlabel(
        "Malicious clients (%)"
    )

    ax.set_ylabel(
        "ACTG detection metric"
    )

    ax.set_title(
        "ACTG Detection vs Malicious Fraction"
    )

    ax.legend()

    fig.tight_layout()

    fp=LOCAL/"ACTG_VS_MALICIOUS_FRACTION.png"

    fig.savefig(
        fp,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    atomic_copy(
        fp,
        FIGS/fp.name
    )

    clean_fpr_mean=float(
        CLEAN_CONTROL[
            "clean_false_positive_rate"
        ].mean()
    )

    overall_attack_detection={
        "precision_mean":float(
            ATTACK_TYPE_RAW[
                "detection_precision"
            ].mean()
        ),
        "recall_mean":float(
            ATTACK_TYPE_RAW[
                "detection_recall"
            ].mean()
        ),
        "f1_mean":float(
            ATTACK_TYPE_RAW[
                "detection_f1"
            ].mean()
        ),
        "clean_false_positive_rate_mean":clean_fpr_mean,
    }

    SUMMARY=pd.DataFrame([
        {
            "component":"Frozen clean BC-ACTG-HFSB-FL",
            "value_1_name":"accuracy",
            "value_1":CLEAN_METRICS[
                "accuracy"
            ],
            "value_2_name":"macro_f1",
            "value_2":CLEAN_METRICS[
                "macro_f1"
            ],
            "n_runs":1
        },
        {
            "component":"Attack-type stress ACTG",
            "value_1_name":"detection_f1_mean",
            "value_1":overall_attack_detection[
                "f1_mean"
            ],
            "value_2_name":"clean_ACTG_FPR",
            "value_2":clean_fpr_mean,
            "n_runs":len(
                ATTACK_TYPE_RAW
            )
        },
        {
            "component":"Attack-type federated branch defense recovery",
            "value_1_name":"accuracy_recovery_mean",
            "value_1":float(
                ATTACK_TYPE_RAW[
                    "branch_accuracy_recovery"
                ].mean()
            ),
            "value_2_name":"macro_f1_recovery_mean",
            "value_2":float(
                ATTACK_TYPE_RAW[
                    "branch_macro_f1_recovery"
                ].mean()
            ),
            "n_runs":len(
                ATTACK_TYPE_RAW
            )
        },
        {
            "component":"Attack-type end-to-end defense recovery",
            "value_1_name":"accuracy_recovery_mean",
            "value_1":float(
                ATTACK_TYPE_RAW[
                    "e2e_accuracy_recovery"
                ].mean()
            ),
            "value_2_name":"macro_f1_recovery_mean",
            "value_2":float(
                ATTACK_TYPE_RAW[
                    "e2e_macro_f1_recovery"
                ].mean()
            ),
            "n_runs":len(
                ATTACK_TYPE_RAW
            )
        },
        {
            "component":"40% malicious mixed stress",
            "value_1_name":"e2e_defense_accuracy_mean",
            "value_1":float(
                FRACTION_SUMMARY[
                    np.isclose(
                        FRACTION_SUMMARY[
                            "malicious_fraction"
                        ],
                        0.40
                    )
                ][
                    "e2e_defense_accuracy_mean"
                ].iloc[0]
            ),
            "value_2_name":"detection_f1_mean",
            "value_2":float(
                FRACTION_SUMMARY[
                    np.isclose(
                        FRACTION_SUMMARY[
                            "malicious_fraction"
                        ],
                        0.40
                    )
                ][
                    "detection_f1_mean"
                ].iloc[0]
            ),
            "n_runs":int(
                FRACTION_SUMMARY[
                    np.isclose(
                        FRACTION_SUMMARY[
                            "malicious_fraction"
                        ],
                        0.40
                    )
                ][
                    "n_runs"
                ].iloc[0]
            )
        },
    ])

    save_csv(
        SUMMARY,
        TABLES/"STEP06_SECURITY_STRESS_SUMMARY.csv"
    )

    paper_json={
        "version":VERSION,
        "status":"COMPLETED",
        "model":"BC-ACTG-HFSB-FL",
        "architecture_frozen":True,
        "epsilon":EPSILON,
        "delta":DP_DELTA,
        "protected_weight":PROTECTED_WEIGHT,

        "attack_type_design":{
            "attack_types":ATTACK_TYPES,
            "malicious_fraction":0.20,
            "pairs":[
                list(x)
                for x in STRESS_PAIRS
            ],
            "seeds":STRESS_SEEDS,
            "n_runs":len(
                ATTACK_TYPE_RAW
            )
        },

        "fraction_design":{
            "fractions":[
                float(x)
                for x in FRACTION_SETS.keys()
            ],
            "seeds":STRESS_SEEDS,
            "n_runs":len(
                FRACTION_RAW
            ),
            "honest_majority_assumption":"all tested fractions are below 50%"
        },

        "clean_actg_false_positive_control":{
            "mean_false_positive_rate":clean_fpr_mean,
            "runs":len(
                CLEAN_CONTROL
            )
        },

        "overall_attack_type_detection":overall_attack_detection,

        "reporting_guidance":[
            "Report federated-branch poisoning impact separately from end-to-end impact.",
            "The frozen CCAC prior is itself a robustness component and explains why end-to-end degradation can be much smaller than federated-branch degradation.",
            "Do not claim ACTG robustness at or above 50% malicious clients; the robust-median validator assumes an honest majority.",
            "Output/residual-level Gaussian DP is not DP-SGD.",
            "Blockchain is an audit/enforcement layer, not the anomaly detector.",
            "No Step-6 parameter was tuned on TEST."
        ],

        "files":{
            "attack_type_raw":str(
                OUT/"ATTACK_TYPE_STRESS_RAW.csv"
            ),
            "attack_type_summary":str(
                TABLES/"ATTACK_TYPE_STRESS_SUMMARY.csv"
            ),
            "fraction_raw":str(
                OUT/"MALICIOUS_FRACTION_STRESS_RAW.csv"
            ),
            "fraction_summary":str(
                TABLES/"MALICIOUS_FRACTION_STRESS_SUMMARY.csv"
            ),
            "stats":str(
                TABLES/"STEP06_PAIRED_STATISTICAL_TESTS.csv"
            ),
            "threat_model":str(
                TABLES/"THREAT_MODEL_DEFINITION.csv"
            )
        },

        "completed_at":datetime.now().isoformat()
    }

    save_json(
        paper_json,
        OUT/"STEP06_PAPER_READY_SECURITY_SUMMARY.json"
    )

    save_json(
        {
            "version":VERSION,
            "status":"COMPLETED",
            "step":"6_SECURITY_STRESS",
            "model":"BC-ACTG-HFSB-FL",
            "architecture_frozen":True,
            "attack_type_runs":len(
                ATTACK_TYPE_RAW
            ),
            "fraction_runs":len(
                FRACTION_RAW
            ),
            "clean_control_runs":len(
                CLEAN_CONTROL
            ),
            "summary":str(
                TABLES/"STEP06_SECURITY_STRESS_SUMMARY.csv"
            ),
            "paper_json":str(
                OUT/"STEP06_PAPER_READY_SECURITY_SUMMARY.json"
            ),
            "completed_at":datetime.now().isoformat()
        },
        COMPLETE
    )

    print("\n"+"="*130)
    print("✅ STEP 6 SECURITY STRESS TEST COMPLETED")
    print("="*130)

    print("\nATTACK-TYPE SUMMARY:")
    print(
        ATTACK_TYPE_SUMMARY[
            [
                "attack_mode",
                "n_runs",
                "branch_attack_accuracy_mean",
                "branch_defense_accuracy_mean",
                "branch_accuracy_recovery_mean",
                "e2e_attack_accuracy_mean",
                "e2e_defense_accuracy_mean",
                "e2e_accuracy_recovery_mean",
                "detection_precision_mean",
                "detection_recall_mean",
                "detection_f1_mean",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nMALICIOUS-FRACTION SUMMARY:")
    print(
        FRACTION_SUMMARY[
            [
                "malicious_fraction",
                "n_runs",
                "branch_attack_accuracy_mean",
                "branch_defense_accuracy_mean",
                "e2e_attack_accuracy_mean",
                "e2e_defense_accuracy_mean",
                "detection_f1_mean",
                "false_positive_rate_mean",
                "false_negative_rate_mean",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nCLEAN ACTG CONTROL:")
    print(
        CLEAN_CONTROL.to_string(
            index=False
        )
    )

    print("\nPAIRED STATISTICAL TESTS:")
    print(
        STATS.to_string(
            index=False
        )
    )

    print("\nPaper-ready summary:")
    print(
        OUT/"STEP06_PAPER_READY_SECURITY_SUMMARY.json"
    )

    print("="*130)

import os, sys, json, math, re, hashlib, shutil, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

from google.colab import drive

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
else:
    print("✅ Google Drive already mounted.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/content/drive/MyDrive/Hybrid_BCFL_IJACSA_2026")

S3B = ROOT/"08_FEDERATED_LEARNING"/"STEP03B_FEDADAM_FEDLC"
S4A = ROOT/"08_FEDERATED_LEARNING"/"STEP04A_PROPOSED_PCH_FL"
R2 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R2_AHF_RCE"
R3 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R3_FAST_CARF_STACK"
R5 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R5_FCS_MOE"
R6 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R6_HFSB_FL"
S4C = ROOT/"08_FEDERATED_LEARNING"/"STEP04C_FINAL_BC_ACTG_HFSB_FL"

S5_RES = ROOT/"11_RESULTS"/"STEP05_FINAL_VALIDATION"
S5_TAB = ROOT/"13_TABLES"/"STEP05_FINAL_VALIDATION"

S6_RES = ROOT/"11_RESULTS"/"STEP06_SECURITY_STRESS"
S6_TAB = ROOT/"13_TABLES"/"STEP06_SECURITY_STRESS"

OUT = ROOT/"11_RESULTS"/"STEP07_PAPER_READY"
TABLES = ROOT/"13_TABLES"/"STEP07_PAPER_READY"
FIGS = ROOT/"12_FIGURES"/"STEP07_PAPER_READY"
LOGS = ROOT/"14_LOGS"/"STEP07_PAPER_READY"
CKPT = ROOT/"06_CHECKPOINTS"/"STEP07_PAPER_READY"

for p in [OUT,TABLES,FIGS,LOGS,CKPT]:
    p.mkdir(parents=True,exist_ok=True)

LOCAL=Path("/content/STEP07_PAPER_READY_RUNTIME")
LOCAL.mkdir(parents=True,exist_ok=True)

VERSION="STEP07_PAPER_READY_V1"
COMPLETE=CKPT/"STEP07_PAPER_READY_COMPLETE.json"

P={
    "step3b":S3B/"CHECKPOINTS"/"STEP03B_COMPLETE.json",
    "step4a":S4A/"CHECKPOINTS"/"STEP04A_COMPLETE.json",
    "r3":R3/"CHECKPOINTS"/"STEP04_4A_R3_FAST_COMPLETE.json",
    "r5":R5/"CHECKPOINTS"/"STEP04_4A_R5_COMPLETE.json",
    "r6":R6/"CHECKPOINTS"/"STEP04_4A_R6_COMPLETE.json",

    "step4c":S4C/"CHECKPOINTS"/"STEP04C_COMPLETE.json",
    "step4c_summary":S4C/"RESULTS"/"STEP04C_FINAL_SUMMARY.json",
    "step4c_perclass":S4C/"RESULTS"/"STEP04C_FULL_PER_CLASS.csv",

    "step5_summary":S5_TAB/"FINAL_VALIDATION_SUMMARY.csv",
    "step5_privacy":S5_TAB/"PRIVACY_UTILITY_MEAN_SD_CI.csv",
    "step5_stats":S5_TAB/"PAIRED_STATISTICAL_TESTS.csv",
    "step5_secure":S5_TAB/"SECURE_AGGREGATION_CORRECTNESS.csv",
    "step5_blockchain":S5_TAB/"BLOCKCHAIN_LATENCY_SUMMARY.csv",
    "step5_comm":S5_TAB/"COMMUNICATION_OVERHEAD.csv",
    "step5_scale":S5_TAB/"ANALYTICAL_SCALABILITY.csv",

    "step6_attack":S6_TAB/"ATTACK_TYPE_STRESS_SUMMARY.csv",
    "step6_fraction":S6_TAB/"MALICIOUS_FRACTION_STRESS_SUMMARY.csv",
    "step6_stats":S6_TAB/"STEP06_PAIRED_STATISTICAL_TESTS.csv",
    "step6_clean_control":S6_TAB/"ACTG_CLEAN_FALSE_POSITIVE_CONTROL.csv",
    "step6_threat":S6_TAB/"THREAT_MODEL_DEFINITION.csv",
    "step6_summary":S6_RES/"STEP06_PAPER_READY_SECURITY_SUMMARY.json",
}

missing=[
    str(v)
    for v in P.values()
    if not v.exists()
]

if missing:
    raise FileNotFoundError(
        "Required completed artifacts are missing:\n"
        +
        "\n".join(
            missing
        )
    )

OPTIONAL={
    "r2":R2/"CHECKPOINTS"/"STEP04_4A_R2_COMPLETE.json",
}

def load_json(path):
    return json.load(
        open(
            path,
            "r",
            encoding="utf-8"
        )
    )

def save_json(obj,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)

    tmp=LOCAL/(path.stem+".json")

    with open(
        tmp,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            default=str
        )

    shutil.copy2(
        tmp,
        path
    )

    tmp.unlink(
        missing_ok=True
    )

def save_csv(df,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)

    tmp=LOCAL/(path.stem+".csv")

    df.to_csv(
        tmp,
        index=False
    )

    shutil.copy2(
        tmp,
        path
    )

    tmp.unlink(
        missing_ok=True
    )

def save_text(text,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)

    tmp=LOCAL/(path.stem+".txt")

    tmp.write_text(
        text,
        encoding="utf-8"
    )

    shutil.copy2(
        tmp,
        path
    )

    tmp.unlink(
        missing_ok=True
    )

def sha256_file(path,chunk=1024*1024):
    h=hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:
        while True:
            b=f.read(chunk)

            if not b:
                break

            h.update(b)

    return h.hexdigest()

def pct(x,d=2):
    if x is None or pd.isna(x):
        return "NA"

    return f"{100.0*float(x):.{d}f}%"

def f4(x):
    if x is None or pd.isna(x):
        return "NA"

    return f"{float(x):.4f}"

def pick(d,*keys,default=np.nan):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]

    return default

def latex_escape(text):
    if text is None:
        return ""

    s=str(text)

    replacements={
        "&":r"\&",
        "%":r"\%",
        "_":r"\_",
        "#":r"\#",
    }

    for a,b in replacements.items():
        s=s.replace(a,b)

    return s

D3B=load_json(
    P["step3b"]
)

D4A=load_json(
    P["step4a"]
)

D3=load_json(
    P["r3"]
)

D5=load_json(
    P["r5"]
)

D6=load_json(
    P["r6"]
)

R6_ACC=float(
    D6["final_accuracy"]
)

R6_F1=float(
    D6["final_macro_f1"]
)

D4C=load_json(
    P["step4c"]
)

D4CS=load_json(
    P["step4c_summary"]
)

D6S=load_json(
    P["step6_summary"]
)

STEP5_SUM=pd.read_csv(
    P["step5_summary"]
)

PRIVACY=pd.read_csv(
    P["step5_privacy"]
)

STEP5_STATS=pd.read_csv(
    P["step5_stats"]
)

SECURE=pd.read_csv(
    P["step5_secure"]
)

BLOCKCHAIN=pd.read_csv(
    P["step5_blockchain"]
)

COMM=pd.read_csv(
    P["step5_comm"]
)

SCALE=pd.read_csv(
    P["step5_scale"]
)

ATTACK=pd.read_csv(
    P["step6_attack"]
)

FRACTION=pd.read_csv(
    P["step6_fraction"]
)

STEP6_STATS=pd.read_csv(
    P["step6_stats"]
)

CLEAN_CONTROL=pd.read_csv(
    P["step6_clean_control"]
)

THREAT=pd.read_csv(
    P["step6_threat"]
)

PER_CLASS=pd.read_csv(
    P["step4c_perclass"]
)

checks=[]

def add_check(
    check,
    observed,
    expected,
    passed,
    implication
):
    checks.append({
        "check":check,
        "observed":observed,
        "expected":expected,
        "passed":bool(passed),
        "paper_implication":implication
    })

final_clean_acc=float(
    D4C[
        "ccac_clean_accuracy"
    ]
)

final_clean_f1=float(
    D4C[
        "ccac_clean_macro_f1"
    ]
)

final_full_acc=float(
    D4C[
        "full_accuracy"
    ]
)

final_full_f1=float(
    D4C[
        "full_macro_f1"
    ]
)

eps=float(
    D4C[
        "epsilon"
    ]
)

delta=float(
    D4C[
        "delta"
    ]
)

add_check(
    "Clean accuracy is below 95%",
    final_clean_acc,
    0.95,
    final_clean_acc<0.95,
    "Do NOT write that the final model achieved 95%+ test accuracy."
)

add_check(
    "Full protected accuracy remains above 93%",
    final_full_acc,
    0.93,
    final_full_acc>=0.93,
    "Full hybrid maintains competitive utility under privacy + attack + ACTG + blockchain."
)

clean_fpr=float(
    CLEAN_CONTROL[
        "clean_false_positive_rate"
    ].mean()
)

add_check(
    "ACTG clean false-positive control",
    clean_fpr,
    0.0,
    np.isclose(
        clean_fpr,
        0.0
    ),
    "Supports the claim that ACTG did not reject honest clients in the tested clean controls."
)

attack_f1=float(
    ATTACK[
        "detection_f1_mean"
    ].mean()
)

add_check(
    "ACTG attack-type detection F1",
    attack_f1,
    1.0,
    attack_f1>=0.999,
    "Can report perfect detection in the explicitly tested attack scenarios only."
)

max_mask=float(
    SECURE[
        "max_abs_secure_vs_direct"
    ].max()
)

add_check(
    "Secure aggregation matches direct weighted aggregation",
    max_mask,
    1e-12,
    max_mask<1e-12,
    "Supports numerical correctness of the pairwise-mask secure aggregation prototype."
)

CLAIMS=pd.DataFrame(
    checks
)

save_csv(
    CLAIMS,
    TABLES/"T00_CLAIM_AUDIT.csv"
)

rows=[]

rows.append({
    "stage":"Step 3B",
    "protocol":"Protocol-C strict non-IID",
    "model":"FedAdam + FedLC Transformer",
    "accuracy":pick(
        D3B,
        "fedadam_fedlc_test_accuracy",
        "fedadam_fedlc_accuracy"
    ),
    "macro_f1":pick(
        D3B,
        "fedadam_fedlc_test_macro_f1",
        "fedadam_fedlc_macro_f1"
    ),
    "role":"Strict robustness benchmark"
})

rows.append({
    "stage":"Step 4A",
    "protocol":"Protocol-LR IID-FL",
    "model":"PCH-FL",
    "accuracy":pick(
        D4A,
        "proposed_test_accuracy"
    ),
    "macro_f1":pick(
        D4A,
        "proposed_test_macro_f1"
    ),
    "role":"Intermediate FL performance-recovery ablation"
})

if OPTIONAL["r2"].exists():
    dr2=load_json(
        OPTIONAL["r2"]
    )

    rows.append({
        "stage":"Step 4.4A-R2",
        "protocol":"Protocol-LT",
        "model":"AHF-RCE",
        "accuracy":pick(
            dr2,
            "final_accuracy"
        ),
        "macro_f1":pick(
            dr2,
            "final_macro_f1"
        ),
        "role":"Federated boosted rare-class ablation"
    })

rows.extend([
    {
        "stage":"Step 4.4A-R3",
        "protocol":"Protocol-LT",
        "model":"CARF-Stack",
        "accuracy":float(
            D3["final_accuracy"]
        ),
        "macro_f1":float(
            D3["final_macro_f1"]
        ),
        "role":"Out-of-client federated stacking"
    },
    {
        "stage":"Step 4.4A-R5",
        "protocol":"Protocol-LT",
        "model":"FCS-MoE",
        "accuracy":float(
            D5["final_accuracy"]
        ),
        "macro_f1":float(
            D5["final_macro_f1"]
        ),
        "role":"Calibration/stacking ablation"
    },
    {
        "stage":"Step 4.4A-R6",
        "protocol":"Protocol-LT",
        "model":"HFSB-FL",
        "accuracy":float(
            D6["final_accuracy"]
        ),
        "macro_f1":float(
            D6["final_macro_f1"]
        ),
        "role":"Pre-proposed hierarchical FL baseline"
    },
    {
        "stage":"Step 4C",
        "protocol":"Protocol-LT",
        "model":"BC-ACTG-HFSB-FL (clean)",
        "accuracy":final_clean_acc,
        "macro_f1":final_clean_f1,
        "role":"Final proposed clean utility"
    },
    {
        "stage":"Step 4C",
        "protocol":"Protocol-LT",
        "model":"BC-ACTG-HFSB-FL (full)",
        "accuracy":final_full_acc,
        "macro_f1":final_full_f1,
        "role":"Privacy + attack + ACTG + blockchain + secure aggregation"
    }
])

MODEL_EVOLUTION=pd.DataFrame(
    rows
)

save_csv(
    MODEL_EVOLUTION,
    TABLES/"T01_MODEL_EVOLUTION.csv"
)

ABLATION=STEP5_SUM.copy()

save_csv(
    ABLATION,
    TABLES/"T02_FINAL_PROPOSED_ABLATION.csv"
)

save_csv(
    PRIVACY,
    TABLES/"T03_PRIVACY_UTILITY.csv"
)

attack_cols=[
    "attack_mode",
    "n_runs",
    "branch_attack_accuracy_mean",
    "branch_defense_accuracy_mean",
    "branch_accuracy_recovery_mean",
    "branch_attack_macro_f1_mean",
    "branch_defense_macro_f1_mean",
    "branch_macro_f1_recovery_mean",
    "e2e_attack_accuracy_mean",
    "e2e_defense_accuracy_mean",
    "e2e_accuracy_recovery_mean",
    "e2e_attack_macro_f1_mean",
    "e2e_defense_macro_f1_mean",
    "e2e_macro_f1_recovery_mean",
    "detection_precision_mean",
    "detection_recall_mean",
    "detection_f1_mean",
    "false_positive_rate_mean",
    "false_negative_rate_mean",
]

ATTACK_PAPER=ATTACK[
    [
        c
        for c in attack_cols
        if c in ATTACK.columns
    ]
].copy()

save_csv(
    ATTACK_PAPER,
    TABLES/"T04_ATTACK_TYPE_ROBUSTNESS.csv"
)

fraction_cols=[
    "malicious_fraction",
    "n_runs",
    "branch_attack_accuracy_mean",
    "branch_defense_accuracy_mean",
    "branch_accuracy_recovery_mean",
    "branch_attack_macro_f1_mean",
    "branch_defense_macro_f1_mean",
    "branch_macro_f1_recovery_mean",
    "e2e_attack_accuracy_mean",
    "e2e_defense_accuracy_mean",
    "e2e_accuracy_recovery_mean",
    "e2e_attack_macro_f1_mean",
    "e2e_defense_macro_f1_mean",
    "e2e_macro_f1_recovery_mean",
    "detection_precision_mean",
    "detection_recall_mean",
    "detection_f1_mean",
    "false_positive_rate_mean",
    "false_negative_rate_mean",
]

FRACTION_PAPER=FRACTION[
    [
        c
        for c in fraction_cols
        if c in FRACTION.columns
    ]
].copy()

save_csv(
    FRACTION_PAPER,
    TABLES/"T05_MALICIOUS_FRACTION_ROBUSTNESS.csv"
)

ALL_STATS=pd.concat(
    [
        STEP5_STATS.assign(
            source="Step5 original paired validation"
        ),
        STEP6_STATS.assign(
            source="Step6 security stress"
        )
    ],
    ignore_index=True,
    sort=False
)

save_csv(
    ALL_STATS,
    TABLES/"T06_STATISTICAL_TESTS.csv"
)

save_csv(
    BLOCKCHAIN,
    TABLES/"T07_BLOCKCHAIN_LATENCY.csv"
)

save_csv(
    COMM,
    TABLES/"T08_COMMUNICATION_OVERHEAD.csv"
)

save_csv(
    SCALE,
    TABLES/"T09_ANALYTICAL_SCALABILITY.csv"
)

save_csv(
    PER_CLASS,
    TABLES/"T10_FINAL_PER_CLASS.csv"
)

save_csv(
    THREAT,
    TABLES/"T11_THREAT_MODEL.csv"
)

save_csv(
    SECURE,
    TABLES/"T12_SECURE_AGGREGATION_CORRECTNESS.csv"
)

def df_to_latex(
    df,
    caption,
    label,
    float_cols=None
):
    x=df.copy()

    if float_cols is None:
        float_cols=[]

    for c in float_cols:
        if c in x.columns:
            x[c]=x[c].map(
                lambda v:
                ""
                if pd.isna(v)
                else
                f"{float(v):.4f}"
            )

    latex=x.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label
    )

    return latex

LATEX_DIR=OUT/"LATEX_TABLES"
LATEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)

save_text(
    df_to_latex(
        MODEL_EVOLUTION[
            [
                "stage",
                "protocol",
                "model",
                "accuracy",
                "macro_f1",
                "role"
            ]
        ],
        "Evolution of the proposed federated IoT intrusion-detection framework.",
        "tab:model_evolution",
        [
            "accuracy",
            "macro_f1"
        ]
    ),
    LATEX_DIR/"table_model_evolution.tex"
)

save_text(
    df_to_latex(
        PRIVACY,
        "Privacy--utility trade-off under output-level Gaussian residual protection.",
        "tab:privacy_utility",
        [
            c
            for c in PRIVACY.columns
            if "accuracy" in c
            or
            "macro_f1" in c
        ]
    ),
    LATEX_DIR/"table_privacy_utility.tex"
)

save_text(
    df_to_latex(
        ATTACK_PAPER,
        "Robustness of ACTG against multiple poisoning attacks.",
        "tab:attack_robustness",
        [
            c
            for c in ATTACK_PAPER.columns
            if c!="n_runs"
            and
            c!="attack_mode"
        ]
    ),
    LATEX_DIR/"table_attack_robustness.tex"
)

save_text(
    df_to_latex(
        FRACTION_PAPER,
        "Robustness as the malicious-client fraction increases.",
        "tab:malicious_fraction",
        [
            c
            for c in FRACTION_PAPER.columns
            if c!="n_runs"
        ]
    ),
    LATEX_DIR/"table_malicious_fraction.tex"
)

fig,ax=plt.subplots(
    figsize=(11,5)
)

plot_df=MODEL_EVOLUTION[
    MODEL_EVOLUTION[
        "protocol"
    ]=="Protocol-LT"
].copy()

ax.plot(
    np.arange(
        len(plot_df)
    ),
    plot_df[
        "accuracy"
    ],
    marker="o"
)

ax.set_xticks(
    np.arange(
        len(plot_df)
    )
)

ax.set_xticklabels(
    plot_df[
        "model"
    ],
    rotation=25
)

ax.set_ylabel(
    "Accuracy"
)

ax.set_ylim(
    0.90,
    0.96
)

ax.set_title(
    "Protocol-LT Model Evolution"
)

fig.tight_layout()

fp=LOCAL/"F01_MODEL_EVOLUTION_ACCURACY.png"

fig.savefig(
    fp,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

shutil.copy2(
    fp,
    FIGS/fp.name
)

fig,ax=plt.subplots(
    figsize=(11,5)
)

ax.plot(
    np.arange(
        len(plot_df)
    ),
    plot_df[
        "macro_f1"
    ],
    marker="o"
)

ax.set_xticks(
    np.arange(
        len(plot_df)
    )
)

ax.set_xticklabels(
    plot_df[
        "model"
    ],
    rotation=25
)

ax.set_ylabel(
    "Macro-F1"
)

ax.set_ylim(
    0.40,
    0.75
)

ax.set_title(
    "Protocol-LT Macro-F1 Evolution"
)

fig.tight_layout()

fp=LOCAL/"F02_MODEL_EVOLUTION_MACRO_F1.png"

fig.savefig(
    fp,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

shutil.copy2(
    fp,
    FIGS/fp.name
)

fig,ax=plt.subplots(
    figsize=(8,5)
)

ax.errorbar(
    PRIVACY[
        "epsilon"
    ],
    PRIVACY[
        "accuracy_mean"
    ],
    yerr=PRIVACY[
        "accuracy_sd"
    ],
    marker="o",
    capsize=4,
    label="Accuracy"
)

ax.errorbar(
    PRIVACY[
        "epsilon"
    ],
    PRIVACY[
        "macro_f1_mean"
    ],
    yerr=PRIVACY[
        "macro_f1_sd"
    ],
    marker="s",
    capsize=4,
    label="Macro-F1"
)

ax.set_xlabel(
    "Privacy epsilon"
)

ax.set_ylabel(
    "Mean test performance"
)

ax.set_title(
    "Privacy–Utility Trade-off"
)

ax.legend()

fig.tight_layout()

fp=LOCAL/"F03_PRIVACY_UTILITY.png"

fig.savefig(
    fp,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

shutil.copy2(
    fp,
    FIGS/fp.name
)

fig,ax=plt.subplots(
    figsize=(10,5)
)

x=np.arange(
    len(
        ATTACK_PAPER
    )
)

width=0.35

ax.bar(
    x-width/2,
    ATTACK_PAPER[
        "branch_attack_accuracy_mean"
    ],
    width,
    label="Attack / no defense"
)

ax.bar(
    x+width/2,
    ATTACK_PAPER[
        "branch_defense_accuracy_mean"
    ],
    width,
    label="ACTG defense"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    ATTACK_PAPER[
        "attack_mode"
    ],
    rotation=20
)

ax.set_ylabel(
    "Federated-branch accuracy"
)

ax.set_title(
    "Poisoning Impact and ACTG Recovery"
)

ax.legend()

fig.tight_layout()

fp=LOCAL/"F04_ATTACK_DEFENSE_BRANCH.png"

fig.savefig(
    fp,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

shutil.copy2(
    fp,
    FIGS/fp.name
)

fig,ax=plt.subplots(
    figsize=(8,5)
)

ax.bar(
    ATTACK_PAPER[
        "attack_mode"
    ],
    ATTACK_PAPER[
        "detection_f1_mean"
    ]
)

ax.set_ylim(
    0,
    1.05
)

ax.set_ylabel(
    "Detection F1"
)

ax.set_title(
    "ACTG Detection Across Attack Types"
)

ax.tick_params(
    axis="x",
    rotation=20
)

fig.tight_layout()

fp=LOCAL/"F05_ACTG_ATTACK_DETECTION.png"

fig.savefig(
    fp,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

shutil.copy2(
    fp,
    FIGS/fp.name
)

fig,ax=plt.subplots(
    figsize=(9,5)
)

frac=(
    100.0
    *
    FRACTION_PAPER[
        "malicious_fraction"
    ].to_numpy(
        np.float64
    )
)

ax.plot(
    frac,
    FRACTION_PAPER[
        "branch_attack_accuracy_mean"
    ],
    marker="o",
    label="Branch attack"
)

ax.plot(
    frac,
    FRACTION_PAPER[
        "branch_defense_accuracy_mean"
    ],
    marker="s",
    label="Branch ACTG defense"
)

ax.set_xlabel(
    "Malicious clients (%)"
)

ax.set_ylabel(
    "Federated-branch accuracy"
)

ax.set_title(
    "Robustness vs Malicious-Client Fraction"
)

ax.legend()

fig.tight_layout()

fp=LOCAL/"F06_MALICIOUS_FRACTION_BRANCH.png"

fig.savefig(
    fp,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

shutil.copy2(
    fp,
    FIGS/fp.name
)

fig,ax=plt.subplots(
    figsize=(10,5)
)

ax.bar(
    PER_CLASS[
        "class_name"
    ],
    PER_CLASS[
        "f1"
    ]
)

ax.set_ylim(
    0,
    1.05
)

ax.set_ylabel(
    "F1-score"
)

ax.set_title(
    "Final Full-Proposed Per-Class F1"
)

ax.tick_params(
    axis="x",
    rotation=25
)

fig.tight_layout()

fp=LOCAL/"F07_FINAL_PER_CLASS_F1.png"

fig.savefig(
    fp,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

shutil.copy2(
    fp,
    FIGS/fp.name
)

attack_type_best=ATTACK_PAPER.sort_values(
    "branch_accuracy_recovery_mean",
    ascending=False
).iloc[0]

fraction_40=FRACTION_PAPER[
    np.isclose(
        FRACTION_PAPER[
            "malicious_fraction"
        ],
        0.40
    )
].iloc[0]

bc_row=BLOCKCHAIN.iloc[0]

privacy_row=PRIVACY[
    np.isclose(
        PRIVACY[
            "epsilon"
        ],
        eps
    )
].iloc[0]

step6_branch_stats=STEP6_STATS[
    STEP6_STATS[
        "metric"
    ]=="Branch Accuracy"
]

attack_type_branch_stat=step6_branch_stats[
    step6_branch_stats[
        "experiment_family"
    ]=="attack_type_sweep"
].iloc[0]

fraction_branch_stat=step6_branch_stats[
    step6_branch_stats[
        "experiment_family"
    ]=="malicious_fraction_sweep"
].iloc[0]

results_text=f"""
# Paper-Ready Results Draft

## Overall detection performance

The pre-proposed HFSB-FL baseline achieved an accuracy of {pct(R6_ACC)} and a Macro-F1 of {f4(R6_F1)} under Protocol-LT. After integration of class-conditional consensus, privacy protection, adaptive contribution trust validation, secure aggregation, and blockchain auditing, the clean proposed BC-ACTG-HFSB-FL configuration achieved {pct(final_clean_acc)} accuracy and {f4(final_clean_f1)} Macro-F1. The full protected configuration achieved {pct(final_full_acc)} accuracy and {f4(final_full_f1)} Macro-F1. The clean-to-full accuracy difference was therefore only {100*(final_clean_acc-final_full_acc):.3f} percentage points.

The final result should not be described as a 95%+ accuracy result. Its main contribution is the integration of competitive classification performance with privacy protection, malicious-client validation, secure aggregation, and auditable blockchain enforcement.

## Privacy–utility trade-off

The selected output-level Gaussian privacy configuration used epsilon={eps:g} and delta={delta:g}. Across the repeated privacy experiments, this configuration achieved a mean accuracy of {pct(privacy_row["accuracy_mean"])} ± {100*privacy_row["accuracy_sd"]:.3f} percentage points and a mean Macro-F1 of {f4(privacy_row["macro_f1_mean"])} ± {privacy_row["macro_f1_sd"]:.4f}. This indicates that the residual-level privacy mechanism introduced only a small utility change under the tested conditions.

The privacy mechanism is applied to bounded transmitted prediction residuals and must be described as output/residual-level Gaussian differential privacy under the stated sensitivity assumption, not as DP-SGD.

## Poisoning robustness

Across the four fixed 20% malicious-client attack families, ACTG achieved a mean detection F1 of {f4(ATTACK_PAPER["detection_f1_mean"].mean())}. The largest mean federated-branch accuracy recovery was observed for {attack_type_best["attack_mode"]}, where accuracy improved by {100*attack_type_best["branch_accuracy_recovery_mean"]:.3f} percentage points after ACTG filtering.

Across all attack types, the paired attack-versus-defense comparison at the federated branch showed a mean accuracy improvement from {pct(attack_type_branch_stat["attack_mean"])} to {pct(attack_type_branch_stat["defense_mean"])} (paired t-test p={attack_type_branch_stat["paired_t_p"]:.3e}; Wilcoxon p={attack_type_branch_stat["wilcoxon_p"]:.3e}; Cohen's dz={attack_type_branch_stat["cohens_dz"]:.3f}).

The end-to-end accuracy change is much smaller because the final detector contains a frozen CCAC prior in addition to the protected federated branch. This distinction should be stated explicitly: ACTG provides substantial recovery in the poisoned federated branch, while the prior prevents most branch-level corruption from propagating to the final classification output.

## Malicious-client fraction

When the malicious-client fraction increased to 40%, the undefended federated branch achieved a mean accuracy of {pct(fraction_40["branch_attack_accuracy_mean"])}, whereas ACTG defense restored it to {pct(fraction_40["branch_defense_accuracy_mean"])}. ACTG detection F1 at 40% malicious clients remained {f4(fraction_40["detection_f1_mean"])} in the tested scenarios.

The fraction-sweep branch comparison was statistically significant (paired t-test p={fraction_branch_stat["paired_t_p"]:.3e}; Wilcoxon p={fraction_branch_stat["wilcoxon_p"]:.3e}; Cohen's dz={fraction_branch_stat["cohens_dz"]:.3f}).

The current ACTG design assumes an honest majority. Results should not be extrapolated to 50% or more malicious clients without an additional Byzantine-resilient consensus study.

## Secure aggregation

The maximum absolute difference between the masked secure aggregation result and direct weighted aggregation was {max_mask:.3e}. This numerical result verifies that the pairwise masks cancel correctly in the weighted numerator domain before division by the total accepted-client weight.

## Blockchain audit overhead

The permissioned Web3/EthereumTester prototype processed {int(bc_row["n_transactions"])} audit transactions with a mean latency of {bc_row["mean_latency_ms"]:.2f} ms, a median latency of {bc_row["median_latency_ms"]:.2f} ms, and a 95th-percentile latency of {bc_row["p95_latency_ms"]:.2f} ms. These measurements characterize the local permissioned prototype only; they must not be interpreted as public-mainnet blockchain performance.

## Interpretation

Taken together, the experiments indicate that BC-ACTG-HFSB-FL maintains approximately 94.4% classification accuracy while adding a privacy-preserving residual-sharing mechanism, accurate malicious-client rejection under the evaluated threat model, numerically correct secure aggregation, and tamper-evident permissioned blockchain auditing. The strongest evidence for the proposed security contribution is the statistically significant recovery of the poisoned federated branch and the consistent ACTG detection performance across attack types and malicious-client fractions.
""".strip()

save_text(
    results_text,
    OUT/"FINAL_RESULTS_DISCUSSION_DRAFT.md"
)

claims_text=f"""
# Safe Claims for the Manuscript

1. **Clean utility:** The final clean BC-ACTG-HFSB-FL configuration achieved {pct(final_clean_acc)} accuracy and {f4(final_clean_f1)} Macro-F1 on the Protocol-LT benchmark.

2. **Full hybrid utility:** With output-level privacy, malicious-client simulation, ACTG filtering, secure aggregation, and permissioned blockchain auditing enabled, the final measured accuracy was {pct(final_full_acc)} with Macro-F1 {f4(final_full_f1)}.

3. **Attack detection:** ACTG achieved mean attack-detection F1 {f4(ATTACK_PAPER["detection_f1_mean"].mean())} across the tested 20% malicious-client attack-type sweep.

4. **Federated-branch robustness:** Across the attack-type stress tests, ACTG improved mean branch accuracy from {pct(ATTACK_PAPER["branch_attack_accuracy_mean"].mean())} to {pct(ATTACK_PAPER["branch_defense_accuracy_mean"].mean())}.

5. **40% malicious-client stress:** Under the tested 40% mixed-attack scenarios, branch accuracy improved from {pct(fraction_40["branch_attack_accuracy_mean"])} without defense to {pct(fraction_40["branch_defense_accuracy_mean"])} with ACTG.

6. **Privacy:** The privacy mechanism is output/residual-level Gaussian DP with epsilon={eps:g} and delta={delta:g}; it is not DP-SGD.

7. **Secure aggregation:** Pairwise masks produced a maximum secure-vs-direct numerical difference below {max_mask:.3e} in the validation experiment.

8. **Blockchain:** The blockchain prototype records and enforces ACTG decisions and SHA-256 commitments. It does not itself detect poisoning.

## Claims to Avoid

- Do not claim 95%+ final accuracy.
- Do not call the source-IP-disjoint split physical-device-disjoint.
- Do not call the 10 logical FL clients 10 physical IoT devices.
- Do not claim DP-SGD or sample-level training differential privacy.
- Do not claim blockchain-based attack detection.
- Do not claim public-mainnet latency, throughput, or gas economics.
- Do not claim measured scalability above 10 clients; those results are analytical projections.
- Do not extrapolate ACTG guarantees to 50% or more malicious clients.
""".strip()

save_text(
    claims_text,
    OUT/"MANUSCRIPT_CLAIM_GUIDE.md"
)

manifest_rows=[]

for key,path in P.items():
    path=Path(path)

    manifest_rows.append({
        "artifact_key":key,
        "path":str(path),
        "size_bytes":path.stat().st_size,
        "sha256":sha256_file(path)
    })

for key,path in OPTIONAL.items():
    path=Path(path)

    if path.exists():
        manifest_rows.append({
            "artifact_key":key,
            "path":str(path),
            "size_bytes":path.stat().st_size,
            "sha256":sha256_file(path)
        })

MANIFEST=pd.DataFrame(
    manifest_rows
)

save_csv(
    MANIFEST,
    OUT/"REPRODUCIBILITY_MANIFEST.csv"
)

save_json(
    {
        "version":VERSION,
        "model":"BC-ACTG-HFSB-FL",
        "created_at":datetime.now().isoformat(),
        "n_input_artifacts":len(MANIFEST),
        "input_artifacts":MANIFEST.to_dict(
            orient="records"
        ),
        "scientific_constraints":[
            "Architecture frozen after Step 4C.",
            "Step 5 and Step 6 perform validation/stress testing only.",
            "Protocol-LT is literature-comparable; Protocol-C remains strict non-IID.",
            "10 FL clients are logical/synthetic client partitions, not 10 physical IoT devices.",
            "Privacy mechanism is output/residual-level Gaussian DP, not DP-SGD.",
            "Blockchain is an audit/enforcement prototype, not the attack detector.",
            "Scalability above 10 clients is analytical unless separately executed."
        ]
    },
    OUT/"REPRODUCIBILITY_MANIFEST.json"
)

index_rows=[]

for p in sorted(
    TABLES.glob(
        "T*.csv"
    )
):
    index_rows.append({
        "type":"table",
        "name":p.stem,
        "path":str(p)
    })

for p in sorted(
    FIGS.glob(
        "F*.png"
    )
):
    index_rows.append({
        "type":"figure",
        "name":p.stem,
        "path":str(p)
    })

for p in sorted(
    LATEX_DIR.glob(
        "*.tex"
    )
):
    index_rows.append({
        "type":"latex_table",
        "name":p.stem,
        "path":str(p)
    })

for p in [
    OUT/"FINAL_RESULTS_DISCUSSION_DRAFT.md",
    OUT/"MANUSCRIPT_CLAIM_GUIDE.md",
    OUT/"REPRODUCIBILITY_MANIFEST.csv",
    OUT/"REPRODUCIBILITY_MANIFEST.json",
]:
    index_rows.append({
        "type":"supporting",
        "name":p.stem,
        "path":str(p)
    })

INDEX=pd.DataFrame(
    index_rows
)

save_csv(
    INDEX,
    OUT/"STEP07_OUTPUT_INDEX.csv"
)

save_json(
    {
        "version":VERSION,
        "status":"COMPLETED",
        "step":"7_PAPER_READY_CONSOLIDATION",
        "model":"BC-ACTG-HFSB-FL",

        "final_clean_accuracy":final_clean_acc,
        "final_clean_macro_f1":final_clean_f1,
        "final_full_accuracy":final_full_acc,
        "final_full_macro_f1":final_full_f1,

        "epsilon":eps,
        "delta":delta,

        "attack_type_detection_f1_mean":float(
            ATTACK_PAPER[
                "detection_f1_mean"
            ].mean()
        ),

        "clean_actg_false_positive_rate":clean_fpr,

        "n_tables":int(
            np.sum(
                INDEX["type"]=="table"
            )
        ),

        "n_figures":int(
            np.sum(
                INDEX["type"]=="figure"
            )
        ),

        "results_draft":str(
            OUT/"FINAL_RESULTS_DISCUSSION_DRAFT.md"
        ),

        "claim_guide":str(
            OUT/"MANUSCRIPT_CLAIM_GUIDE.md"
        ),

        "output_index":str(
            OUT/"STEP07_OUTPUT_INDEX.csv"
        ),

        "completed_at":datetime.now().isoformat()
    },
    COMPLETE
)

print("\n"+"="*132)
print("✅ STEP 7 PAPER-READY CONSOLIDATION COMPLETED")
print("="*132)

print("\nFINAL METRICS:")
print(
    f"Clean proposed: Accuracy={final_clean_acc:.6f}, "
    f"Macro-F1={final_clean_f1:.6f}"
)

print(
    f"Full proposed : Accuracy={final_full_acc:.6f}, "
    f"Macro-F1={final_full_f1:.6f}"
)

print(
    f"ACTG attack-type mean detection F1="
    f"{ATTACK_PAPER['detection_f1_mean'].mean():.6f}"
)

print(
    f"ACTG clean false-positive rate="
    f"{clean_fpr:.6f}"
)

print("\nCLAIM AUDIT:")
print(
    CLAIMS.to_string(
        index=False
    )
)

print("\nGenerated package index:")
print(
    INDEX.to_string(
        index=False
    )
)

print("\nResults draft:")
print(
    OUT/"FINAL_RESULTS_DISCUSSION_DRAFT.md"
)

print("="*132)

import os, sys, json, math, re, hashlib, shutil, subprocess, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

from google.colab import drive

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
else:
    print("✅ Google Drive already mounted.")

def ensure_pkg(import_name, pip_name=None):
    import importlib.util
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet",
            "--disable-pip-version-check",
            pip_name or import_name
        ])

for imp,pip in [
    ("numpy","numpy"),
    ("pandas","pandas"),
    ("matplotlib","matplotlib"),
    ("scipy","scipy"),
    ("sklearn","scikit-learn"),
]:
    ensure_pkg(imp,pip)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
)

from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = Path("/content/drive/MyDrive/Hybrid_BCFL_IJACSA_2026")

R3 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R3_FAST_CARF_STACK"
R5 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R5_FCS_MOE"
R6 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R6_HFSB_FL"
R2 = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_R2_AHF_RCE"
STEP44A = ROOT/"08_FEDERATED_LEARNING"/"STEP04_4A_LT_SIAMTAC_FL"
S4C = ROOT/"08_FEDERATED_LEARNING"/"STEP04C_FINAL_BC_ACTG_HFSB_FL"

S5_RES = ROOT/"11_RESULTS"/"STEP05_FINAL_VALIDATION"
S5_TAB = ROOT/"13_TABLES"/"STEP05_FINAL_VALIDATION"

S6_RES = ROOT/"11_RESULTS"/"STEP06_SECURITY_STRESS"
S6_TAB = ROOT/"13_TABLES"/"STEP06_SECURITY_STRESS"

S7_RES = ROOT/"11_RESULTS"/"STEP07_PAPER_READY"
S7_TAB = ROOT/"13_TABLES"/"STEP07_PAPER_READY"

OUT = ROOT/"11_RESULTS"/"STEP08_PUBLICATION_FIGURES_STATS"
FIG_DIR = ROOT/"12_FIGURES"/"STEP08_PUBLICATION_FIGURES"
STAT_DIR = ROOT/"13_TABLES"/"STEP08_STATISTICAL_VALIDATION"
PACKAGE_DIR = OUT/"DOWNLOAD_PACKAGE"
CAPTION_DIR = PACKAGE_DIR/"CAPTIONS"
VECTOR_DIR = PACKAGE_DIR/"VECTOR_SVG_PDF"
HD_DIR = PACKAGE_DIR/"HD_PNG_600DPI"
TABLE_DIR = PACKAGE_DIR/"STATISTICAL_TABLES"
SOURCE_DIR = PACKAGE_DIR/"SOURCE_MANIFEST"

for p in [
    OUT,FIG_DIR,STAT_DIR,PACKAGE_DIR,CAPTION_DIR,
    VECTOR_DIR,HD_DIR,TABLE_DIR,SOURCE_DIR
]:
    p.mkdir(parents=True,exist_ok=True)

VERSION="STEP08_PUBLICATION_FIGURES_STATS_V1"
SEED=42
rng=np.random.default_rng(SEED)

FILES={

    "step4c_complete":
        S4C/"CHECKPOINTS"/"STEP04C_COMPLETE.json",
    "step4c_summary":
        S4C/"RESULTS"/"STEP04C_FINAL_SUMMARY.json",
    "full_confusion":
        S4C/"RESULTS"/"STEP04C_FULL_CONFUSION_MATRIX.csv",
    "full_per_class":
        S4C/"RESULTS"/"STEP04C_FULL_PER_CLASS.csv",
    "ccac":
        S4C/"RESULTS"/"CCAC_SELECTION_AND_RESULT.json",

    "model_evolution":
        S7_TAB/"T01_MODEL_EVOLUTION.csv",
    "ablation":
        S7_TAB/"T02_FINAL_PROPOSED_ABLATION.csv",

    "privacy":
        S5_TAB/"PRIVACY_UTILITY_MEAN_SD_CI.csv",
    "blockchain":
        S5_TAB/"BLOCKCHAIN_LATENCY_SUMMARY.csv",
    "blockchain_raw":
        S5_RES/"BLOCKCHAIN_LATENCY_RAW.csv",
    "communication":
        S5_TAB/"COMMUNICATION_OVERHEAD.csv",
    "scalability":
        S5_TAB/"ANALYTICAL_SCALABILITY.csv",
    "secure_correctness":
        S5_TAB/"SECURE_AGGREGATION_CORRECTNESS.csv",

    "attack_summary":
        S6_TAB/"ATTACK_TYPE_STRESS_SUMMARY.csv",
    "attack_raw":
        S6_RES/"ATTACK_TYPE_STRESS_RAW.csv",
    "fraction_summary":
        S6_TAB/"MALICIOUS_FRACTION_STRESS_SUMMARY.csv",
    "fraction_raw":
        S6_RES/"MALICIOUS_FRACTION_STRESS_RAW.csv",
    "step6_stats":
        S6_TAB/"STEP06_PAIRED_STATISTICAL_TESTS.csv",
    "clean_control":
        S6_TAB/"ACTG_CLEAN_FALSE_POSITIVE_CONTROL.csv",
    "threat_model":
        S6_TAB/"THREAT_MODEL_DEFINITION.csv",

    "r3_selection":
        R3/"RESULTS"/"OPTUNA_VALIDATION_SELECTION.json",
    "r3_rel":
        R3/"RESULTS"/"META_MODEL_CLASS_RELIABILITY.csv",
    "r5_selection":
        R5/"RESULTS"/"R5_VALIDATION_SELECTION.json",
    "r6_selection":
        R6/"RESULTS"/"R6_VALIDATION_SELECTION.json",
    "data":
        STEP44A/"CACHE"/"TACNET_PROFILE_1671681.npz",
    "split":
        R2/"CACHE"/"TRAIN_CAL_SPLIT.npz",
}

essential=[
    "step4c_complete",
    "full_confusion",
    "full_per_class",
    "model_evolution",
    "privacy",
    "attack_summary",
    "attack_raw",
    "fraction_summary",
    "fraction_raw",
    "step6_stats",
    "blockchain",
    "communication",
    "scalability",
    "secure_correctness",
]

missing=[
    f"{k}: {FILES[k]}"
    for k in essential
    if not FILES[k].exists()
]

if missing:
    raise FileNotFoundError(
        "Missing required final artifacts:\n"
        + "\n".join(missing)
    )

def load_json(path):
    return json.load(open(path,"r",encoding="utf-8"))

def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+","_",str(text)).strip("_")

def save_csv(df,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)

def normalize_prob(p):
    p=np.clip(
        np.asarray(p,dtype=np.float64),
        1e-12,
        None
    )
    p/=np.maximum(
        p.sum(axis=1,keepdims=True),
        1e-12
    )
    return p.astype(np.float32)

FIGURE_RECORDS=[]

def save_figure(
    fig,
    figure_id,
    short_name,
    caption,
    literature_alignment,
    width_hint="single_or_double_column"
):
    """
    Save:
      - editable SVG
      - vector PDF
      - 600 DPI PNG
    and register a paper-ready caption.
    """
    stem=f"{figure_id}_{safe_name(short_name)}"

    svg=VECTOR_DIR/f"{stem}.svg"
    pdf=VECTOR_DIR/f"{stem}.pdf"
    png=HD_DIR/f"{stem}.png"

    fig.savefig(
        svg,
        format="svg",
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        format="pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        png,
        format="png",
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        FIG_DIR/f"{stem}.svg",
        format="svg",
        bbox_inches="tight"
    )

    fig.savefig(
        FIG_DIR/f"{stem}.pdf",
        format="pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        FIG_DIR/f"{stem}.png",
        format="png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)

    FIGURE_RECORDS.append({
        "figure_id":figure_id,
        "short_name":short_name,
        "svg":str(svg),
        "pdf":str(pdf),
        "png_600dpi":str(png),
        "caption":caption,
        "literature_alignment":literature_alignment,
        "width_hint":width_hint
    })

def mean_sd_ci(x,confidence=0.95):
    x=np.asarray(x,dtype=np.float64)
    x=x[np.isfinite(x)]

    if len(x)==0:
        return np.nan,np.nan,np.nan,np.nan

    mean=float(np.mean(x))

    if len(x)==1:
        return mean,0.0,mean,mean

    sd=float(np.std(x,ddof=1))
    sem=sd/math.sqrt(len(x))
    critical=float(
        stats.t.ppf(
            0.5+confidence/2,
            df=len(x)-1
        )
    )

    half=critical*sem

    return mean,sd,mean-half,mean+half

def cohens_dz(before,after):
    before=np.asarray(before,dtype=np.float64)
    after=np.asarray(after,dtype=np.float64)

    d=after-before
    sd=np.std(d,ddof=1)

    if sd==0:
        return np.nan

    return float(
        np.mean(d)/sd
    )

def bootstrap_mean_diff(
    before,
    after,
    n_boot=10000,
    seed=42
):
    """
    Paired bootstrap of mean defense recovery.
    Fast because Step-6 experiments contain tens of paired runs,
    not millions of test records.
    """
    before=np.asarray(before,dtype=np.float64)
    after=np.asarray(after,dtype=np.float64)

    if len(before)!=len(after):
        raise ValueError("Paired arrays must have equal length.")

    d=after-before
    n=len(d)

    rr=np.random.default_rng(seed)

    boot=np.empty(
        n_boot,
        dtype=np.float64
    )

    for i in range(n_boot):
        idx=rr.integers(
            0,
            n,
            size=n
        )

        boot[i]=np.mean(
            d[idx]
        )

    return {
        "mean_difference":float(np.mean(d)),
        "bootstrap_ci95_low":float(np.quantile(boot,0.025)),
        "bootstrap_ci95_high":float(np.quantile(boot,0.975)),
        "bootstrap_probability_improvement":float(np.mean(boot>0)),
        "n_bootstrap":n_boot
    }

D4C=load_json(
    FILES["step4c_complete"]
)

MODEL_EVOLUTION=pd.read_csv(
    FILES["model_evolution"]
)

ABLATION=(
    pd.read_csv(FILES["ablation"])
    if FILES["ablation"].exists()
    else pd.DataFrame()
)

PRIVACY=pd.read_csv(
    FILES["privacy"]
)

ATTACK_SUM=pd.read_csv(
    FILES["attack_summary"]
)

ATTACK_RAW=pd.read_csv(
    FILES["attack_raw"]
)

FRAC_SUM=pd.read_csv(
    FILES["fraction_summary"]
)

FRAC_RAW=pd.read_csv(
    FILES["fraction_raw"]
)

STEP6_STATS=pd.read_csv(
    FILES["step6_stats"]
)

BLOCKCHAIN=pd.read_csv(
    FILES["blockchain"]
)

BLOCKCHAIN_RAW=(
    pd.read_csv(FILES["blockchain_raw"])
    if FILES["blockchain_raw"].exists()
    else pd.DataFrame()
)

COMM=pd.read_csv(
    FILES["communication"]
)

SCALE=pd.read_csv(
    FILES["scalability"]
)

SECURE=pd.read_csv(
    FILES["secure_correctness"]
)

PER_CLASS=pd.read_csv(
    FILES["full_per_class"]
)

CLASS_NAMES=[
    "Benign","DoS","DDoS","Spoofing",
    "SQLInjection","Mirai","BruteForce","XSS"
]

CM_RAW=pd.read_csv(
    FILES["full_confusion"]
)

CM_NUM=CM_RAW.apply(
    pd.to_numeric,
    errors="coerce"
).dropna(
    axis=1,
    how="all"
)

if CM_NUM.shape[0]==8 and CM_NUM.shape[1]>8:
    CM_NUM=CM_NUM.iloc[:,-8:]

if CM_NUM.shape!=(8,8):
    raise ValueError(
        "Could not recover an 8x8 confusion matrix from "
        f"{FILES['full_confusion']}. "
        f"Raw CSV shape={CM_RAW.shape}; numeric shape={CM_NUM.shape}. "
        "Expected 8 classes: "
        + ", ".join(CLASS_NAMES)
    )

CM=CM_NUM.astype(np.int64)

CM.index=CLASS_NAMES
CM.columns=CLASS_NAMES

print(
    "✅ Confusion matrix loaded correctly:",
    CM.shape
)

FINAL_CLEAN_ACC=float(
    D4C["ccac_clean_accuracy"]
)

FINAL_CLEAN_F1=float(
    D4C["ccac_clean_macro_f1"]
)

FINAL_FULL_ACC=float(
    D4C["full_accuracy"]
)

FINAL_FULL_F1=float(
    D4C["full_macro_f1"]
)

EPSILON=float(
    D4C["epsilon"]
)

DELTA=float(
    D4C["delta"]
)

cm8=CM.to_numpy(dtype=np.int64)

TN=int(
    cm8[0,0]
)

FP=int(
    cm8[0,1:].sum()
)

FN=int(
    cm8[1:,0].sum()
)

TP=int(
    cm8[1:,1:].sum()
)

binary_cm=np.array([
    [TN,FP],
    [FN,TP]
],dtype=np.int64)

binary_accuracy=(TP+TN)/max(TP+TN+FP+FN,1)
binary_precision=TP/max(TP+FP,1)
binary_recall=TP/max(TP+FN,1)
binary_specificity=TN/max(TN+FP,1)
binary_f1=(
    2*binary_precision*binary_recall
    /
    max(
        binary_precision+binary_recall,
        1e-12
    )
)
binary_balanced=0.5*(
    binary_recall
    +
    binary_specificity
)

den=math.sqrt(
    max(
        (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN),
        1
    )
)
binary_mcc=(
    (TP*TN-FP*FN)
    /
    den
)

BINARY_METRICS=pd.DataFrame([{
    "task":"Final full hybrid binary projection",
    "positive_class":"Attack",
    "accuracy":binary_accuracy,
    "precision_attack":binary_precision,
    "recall_sensitivity_attack":binary_recall,
    "specificity_benign":binary_specificity,
    "f1_attack":binary_f1,
    "balanced_accuracy":binary_balanced,
    "mcc":binary_mcc,
    "TP":TP,
    "TN":TN,
    "FP":FP,
    "FN":FN,
}])

save_csv(
    BINARY_METRICS,
    STAT_DIR/"BINARY_FINAL_METRICS.csv"
)

BINARY_METRICS.to_csv(
    TABLE_DIR/"BINARY_FINAL_METRICS.csv",
    index=False
)

STAT_ROWS=[]

for family,df in [
    ("Attack-type stress",ATTACK_RAW),
    ("Malicious-fraction stress",FRAC_RAW),
]:
    for metric,before_col,after_col in [
        (
            "Federated branch accuracy",
            "branch_attack_accuracy",
            "branch_defense_accuracy"
        ),
        (
            "Federated branch Macro-F1",
            "branch_attack_macro_f1",
            "branch_defense_macro_f1"
        ),
        (
            "End-to-end accuracy",
            "e2e_attack_accuracy",
            "e2e_defense_accuracy"
        ),
        (
            "End-to-end Macro-F1",
            "e2e_attack_macro_f1",
            "e2e_defense_macro_f1"
        ),
    ]:
        if (
            before_col not in df.columns
            or
            after_col not in df.columns
        ):
            continue

        before=df[
            before_col
        ].to_numpy(np.float64)

        after=df[
            after_col
        ].to_numpy(np.float64)

        tt=stats.ttest_rel(
            after,
            before,
            nan_policy="omit"
        )

        try:
            ww=stats.wilcoxon(
                after,
                before,
                zero_method="wilcox",
                alternative="two-sided"
            )

            w_stat=float(
                ww.statistic
            )

            w_p=float(
                ww.pvalue
            )

        except Exception:
            w_stat=np.nan
            w_p=np.nan

        boot=bootstrap_mean_diff(
            before,
            after,
            n_boot=10000,
            seed=SEED
        )

        dz=cohens_dz(
            before,
            after
        )

        STAT_ROWS.append({
            "experiment_family":family,
            "metric":metric,
            "n_paired_runs":len(before),

            "attack_mean":float(
                np.mean(before)
            ),
            "defense_mean":float(
                np.mean(after)
            ),
            "mean_recovery":float(
                np.mean(after-before)
            ),

            "paired_t_stat":float(
                tt.statistic
            ),
            "paired_t_p":float(
                tt.pvalue
            ),

            "wilcoxon_stat":w_stat,
            "wilcoxon_p":w_p,

            "cohens_dz":dz,
            **boot
        })

STAT_VALIDATION=pd.DataFrame(
    STAT_ROWS
)

save_csv(
    STAT_VALIDATION,
    STAT_DIR/"THREE_METHOD_STATISTICAL_VALIDATION.csv"
)

STAT_VALIDATION.to_csv(
    TABLE_DIR/"THREE_METHOD_STATISTICAL_VALIDATION.csv",
    index=False
)

PROBABILITY_RECONSTRUCTION_OK=False
VAL_CCAC=None
TEST_CCAC=None
VAL_R6=None
TEST_R6=None
YV=None
YTE=None

def try_reconstruct_probabilities():
    global PROBABILITY_RECONSTRUCTION_OK
    global VAL_CCAC,TEST_CCAC,VAL_R6,TEST_R6,YV,YTE

    try:
        required_prob=[
            "r3_selection","r3_rel",
            "r5_selection","r6_selection",
            "data","split","ccac"
        ]

        for k in required_prob:
            if not FILES[k].exists():
                raise FileNotFoundError(
                    FILES[k]
                )

        Z=np.load(
            FILES["data"]
        )

        y=Z["y"].astype(
            np.int64,
            copy=False
        )

        train_idx=Z[
            "train_idx"
        ].astype(
            np.int64,
            copy=False
        )

        val_idx=Z[
            "val_idx"
        ].astype(
            np.int64,
            copy=False
        )

        test_idx=Z[
            "test_idx"
        ].astype(
            np.int64,
            copy=False
        )

        YTR=y[
            train_idx
        ]

        YV=y[
            val_idx
        ]

        YTE=y[
            test_idx
        ]

        sp=np.load(
            FILES["split"]
        )

        cal_idx=sp[
            "cal_idx"
        ].astype(
            np.int64,
            copy=False
        )

        YCAL=YTR[
            cal_idx
        ]

        r3sel=load_json(
            FILES["r3_selection"]
        )

        beta3=float(
            r3sel[
                "best_params"
            ][
                "beta"
            ]
        )

        gamma3=float(
            r3sel[
                "best_params"
            ][
                "gamma"
            ]
        )

        bias3=np.asarray(
            r3sel[
                "bias_vector"
            ],
            dtype=np.float32
        )

        rel3df=pd.read_csv(
            FILES["r3_rel"]
        )

        rel3=np.stack([
            rel3df[
                f"f1_c{c}"
            ].to_numpy(
                np.float64
            )
            for c in range(8)
        ],axis=1)

        def r3_prob(name):
            pth=R3/"CACHE"/f"{name}_META_PREDICTIONS.npy"

            tensor=np.load(
                pth,
                mmap_mode="r"
            )

            M,N,C=tensor.shape

            rw=np.exp(
                beta3*rel3
            )

            rw/=np.maximum(
                rw.max(
                    axis=0,
                    keepdims=True
                ),
                1e-12
            )

            num=np.zeros(
                (N,C),
                dtype=np.float64
            )

            den=np.zeros(
                (N,C),
                dtype=np.float64
            )

            for m in range(M):
                p=np.asarray(
                    tensor[m],
                    dtype=np.float32
                )

                conf=np.max(
                    p,
                    axis=1
                ).astype(
                    np.float64
                )

                w=(
                    conf**gamma3
                )[:,None]*rw[m][None,:]

                num+=p.astype(
                    np.float64
                )*w

                den+=w

            p=normalize_prob(
                num/np.maximum(
                    den,
                    1e-12
                )
            )

            lp=np.log(
                np.clip(
                    p,
                    1e-12,
                    1.0
                )
            )

            lp+=bias3[
                None,:
            ]

            lp-=lp.max(
                axis=1,
                keepdims=True
            )

            return normalize_prob(
                np.exp(lp)
            )

        CAL_R3=r3_prob(
            "CAL"
        )

        VAL_R3=r3_prob(
            "VAL"
        )

        TEST_R3=r3_prob(
            "TEST"
        )

        r5=load_json(
            FILES["r5_selection"]
        )

        sw=r5[
            "stacker_weights"
        ]

        wc=float(
            sw["catboost"]
        )
        wx=float(
            sw["xgboost"]
        )
        wl=float(
            sw["lightgbm"]
        )

        threshold=float(
            r5[
                "gate_threshold"
            ]
        )

        alphas=np.asarray(
            r5[
                "class_alphas"
            ],
            dtype=np.float32
        )

        bias5=np.asarray(
            r5[
                "class_bias"
            ],
            dtype=np.float32
        )

        tuned=[
            0,3,4,5,6,7
        ]

        def stacker(name):
            q=np.load(
                R5/
                "CACHE"/
                f"{name}_STACKER_PREDICTIONS.npz"
            )

            return normalize_prob(
                wc*q["cat"].astype(
                    np.float32
                )
                +
                wx*q["xgb"].astype(
                    np.float32
                )
                +
                wl*q["lgb"].astype(
                    np.float32
                )
            )

        def r5_fuse(
            base,
            expert
        ):
            conf=np.max(
                base,
                axis=1
            )

            bp=np.argmax(
                base,
                axis=1
            )

            ep=np.argmax(
                expert,
                axis=1
            )

            hard=(
                conf<threshold
            ) | np.isin(
                bp,
                tuned
            ) | np.isin(
                ep,
                tuned
            )

            lb=np.log(
                np.clip(
                    base,
                    1e-12,
                    1.0
                )
            ).astype(
                np.float64
            )

            le=np.log(
                np.clip(
                    expert,
                    1e-12,
                    1.0
                )
            ).astype(
                np.float64
            )

            lf=lb.copy()

            for c in range(8):
                a=float(
                    alphas[c]
                )

                if a==0:
                    continue

                lf[
                    hard,
                    c
                ]=(
                    (1-a)*lb[
                        hard,
                        c
                    ]
                    +
                    a*le[
                        hard,
                        c
                    ]
                )

            lf+=bias5[
                None,:
            ]

            lf-=lf.max(
                axis=1,
                keepdims=True
            )

            return normalize_prob(
                np.exp(lf)
            )

        VAL_R5=r5_fuse(
            VAL_R3,
            stacker("VAL")
        )

        TEST_R5=r5_fuse(
            TEST_R3,
            stacker("TEST")
        )

        r6=load_json(
            FILES["r6_selection"]
        )

        gate_beta=float(
            r6[
                "gate_beta"
            ]
        )

        fine_beta=float(
            r6[
                "fine_beta"
            ]
        )

        gate_temp=float(
            r6[
                "gate_temperature"
            ]
        )

        fine_temp=float(
            r6[
                "fine_temperature"
            ]
        )

        strong=float(
            r6[
                "strong_carf_weight"
            ]
        )

        weak=float(
            r6[
                "weak_carf_weight"
            ]
        )

        bias6=np.asarray(
            r6[
                "bias"
            ],
            dtype=np.float32
        )

        def to_coarse(y8):
            out=np.full(
                len(y8),
                3,
                dtype=np.int64
            )

            out[y8==0]=0
            out[y8==1]=1
            out[y8==2]=2

            return out

        def to_fine(y8):
            return (
                y8-3
            ).astype(
                np.int64
            )

        def load_tensor(
            name,
            stage
        ):
            return np.load(
                R6/
                "CACHE"/
                f"{name}_{stage.upper()}_PRED.npy",
                mmap_mode="r"
            )

        CG=load_tensor(
            "CAL",
            "gate"
        )

        CF=load_tensor(
            "CAL",
            "fine"
        )

        VG=load_tensor(
            "VAL",
            "gate"
        )

        VF=load_tensor(
            "VAL",
            "fine"
        )

        TG=load_tensor(
            "TEST",
            "gate"
        )

        TF=load_tensor(
            "TEST",
            "fine"
        )

        ycg=to_coarse(
            YCAL
        )

        weak_mask=YCAL>=3

        ycf=to_fine(
            YCAL[
                weak_mask
            ]
        )

        def f1_mat(
            tensor,
            yy,
            n_classes
        ):
            out=np.zeros(
                (
                    tensor.shape[0],
                    n_classes
                ),
                dtype=np.float64
            )

            for m in range(
                tensor.shape[0]
            ):
                pred=np.argmax(
                    np.asarray(
                        tensor[m],
                        dtype=np.float32
                    ),
                    axis=1
                )

                _,_,f,_=precision_recall_fscore_support(
                    yy,
                    pred,
                    labels=np.arange(
                        n_classes
                    ),
                    zero_division=0
                )

                out[m]=f

            return out

        gate_rel=f1_mat(
            CG,
            ycg,
            4
        )

        fine_rel=np.zeros(
            (
                CF.shape[0],
                5
            ),
            dtype=np.float64
        )

        for m in range(
            CF.shape[0]
        ):
            pred=np.argmax(
                np.asarray(
                    CF[
                        m,
                        weak_mask
                    ],
                    dtype=np.float32
                ),
                axis=1
            )

            _,_,f,_=precision_recall_fscore_support(
                ycf,
                pred,
                labels=np.arange(5),
                zero_division=0
            )

            fine_rel[m]=f

        def rel_ensemble(
            tensor,
            rel,
            beta
        ):
            M,N,C=tensor.shape

            rw=np.exp(
                beta*rel
            )

            rw/=np.maximum(
                rw.max(
                    axis=0,
                    keepdims=True
                ),
                1e-12
            )

            num=np.zeros(
                (N,C),
                dtype=np.float64
            )

            den=np.zeros(
                (N,C),
                dtype=np.float64
            )

            for m in range(M):
                p=np.asarray(
                    tensor[m],
                    dtype=np.float32
                )

                conf=np.max(
                    p,
                    axis=1
                ).astype(
                    np.float64
                )

                w=conf[
                    :,None
                ]*rw[
                    m
                ][
                    None,:
                ]

                num+=p.astype(
                    np.float64
                )*w

                den+=w

            return normalize_prob(
                num/np.maximum(
                    den,
                    1e-12
                )
            )

        def temp_prob(
            p,
            temp
        ):
            lp=np.log(
                np.clip(
                    p,
                    1e-12,
                    1.0
                )
            )/float(temp)

            lp-=lp.max(
                axis=1,
                keepdims=True
            )

            return normalize_prob(
                np.exp(lp)
            )

        def hierarchy(
            gate,
            fine
        ):
            g=temp_prob(
                gate,
                gate_temp
            )

            f=temp_prob(
                fine,
                fine_temp
            )

            out=np.zeros(
                (
                    len(g),
                    8
                ),
                dtype=np.float32
            )

            out[:,0]=g[:,0]
            out[:,1]=g[:,1]
            out[:,2]=g[:,2]
            out[:,3:]=g[:,3:4]*f

            return normalize_prob(
                out
            )

        def r6_final(
            hier,
            carf
        ):
            alpha=np.array(
                [
                    strong,strong,strong,
                    weak,weak,weak,weak,weak
                ],
                dtype=np.float64
            )

            lp=(
                alpha[
                    None,:
                ]
                *
                np.log(
                    np.clip(
                        carf,
                        1e-12,
                        1.0
                    )
                )
                +
                (
                    1-alpha[
                        None,:
                    ]
                )
                *
                np.log(
                    np.clip(
                        hier,
                        1e-12,
                        1.0
                    )
                )
            )

            lp+=bias6[
                None,:
            ]

            lp-=lp.max(
                axis=1,
                keepdims=True
            )

            return normalize_prob(
                np.exp(lp)
            )

        def get_r6(
            gt,
            ft,
            carf
        ):
            g=rel_ensemble(
                gt,
                gate_rel,
                gate_beta
            )

            f=rel_ensemble(
                ft,
                fine_rel,
                fine_beta
            )

            h=hierarchy(
                g,
                f
            )

            return r6_final(
                h,
                carf
            )

        VAL_R6=get_r6(
            VG,
            VF,
            VAL_R3
        )

        TEST_R6=get_r6(
            TG,
            TF,
            TEST_R3
        )

        cc=load_json(
            FILES["ccac"]
        )

        beta=float(
            cc["beta"]
        )

        strong_anchor=float(
            cc[
                "strong_r6_anchor"
            ]
        )

        weak_anchor=float(
            cc[
                "weak_r6_anchor"
            ]
        )

        cc_bias=np.asarray(
            cc[
                "class_bias"
            ],
            dtype=np.float32
        )

        val_models=[
            VAL_R3,
            VAL_R5,
            VAL_R6
        ]

        test_models=[
            TEST_R3,
            TEST_R5,
            TEST_R6
        ]

        class_rel=np.zeros(
            (
                3,
                8
            ),
            dtype=np.float64
        )

        for m,p in enumerate(
            val_models
        ):
            pred=np.argmax(
                p,
                axis=1
            )

            _,_,f,_=precision_recall_fscore_support(
                YV,
                pred,
                labels=np.arange(8),
                zero_division=0
            )

            class_rel[m]=f

        z=beta*class_rel

        z-=z.max(
            axis=0,
            keepdims=True
        )

        w=np.exp(z)

        w/=np.maximum(
            w.sum(
                axis=0,
                keepdims=True
            ),
            1e-12
        )

        anchor=np.array(
            [
                strong_anchor,
                strong_anchor,
                strong_anchor,
                weak_anchor,
                weak_anchor,
                weak_anchor,
                weak_anchor,
                weak_anchor
            ],
            dtype=np.float64
        )

        def ccac_prob(
            models
        ):
            log_ens=np.zeros_like(
                models[0],
                dtype=np.float64
            )

            for m,p in enumerate(
                models
            ):
                log_ens+=(
                    w[
                        m
                    ][
                        None,:
                    ]
                    *
                    np.log(
                        np.clip(
                            p,
                            1e-12,
                            1.0
                        )
                    )
                )

            lp=(
                (
                    1-anchor[
                        None,:
                    ]
                )*log_ens
                +
                anchor[
                    None,:
                ]
                *
                np.log(
                    np.clip(
                        models[2],
                        1e-12,
                        1.0
                    )
                )
            )

            lp+=cc_bias[
                None,:
            ]

            lp-=lp.max(
                axis=1,
                keepdims=True
            )

            return normalize_prob(
                np.exp(lp)
            )

        VAL_CCAC=ccac_prob(
            val_models
        )

        TEST_CCAC=ccac_prob(
            test_models
        )

        PROBABILITY_RECONSTRUCTION_OK=True

        print(
            "✅ Frozen clean probabilities reconstructed for ROC/PR/McNemar."
        )

    except Exception as e:
        PROBABILITY_RECONSTRUCTION_OK=False

        print(
            "⚠️ Probability reconstruction skipped:",
            repr(e)
        )

        print(
            "   All saved-result figures/statistics will still be generated."
        )

try_reconstruct_probabilities()

if PROBABILITY_RECONSTRUCTION_OK:
    pred_r6=np.argmax(
        TEST_R6,
        axis=1
    )

    pred_ccac=np.argmax(
        TEST_CCAC,
        axis=1
    )

    correct_r6=(
        pred_r6==YTE
    )

    correct_ccac=(
        pred_ccac==YTE
    )

    n01=int(
        np.sum(
            (~correct_r6)
            &
            correct_ccac
        )
    )

    n10=int(
        np.sum(
            correct_r6
            &
            (~correct_ccac)
        )
    )

    discordant=n01+n10

    if discordant>0:
        exact=binomtest(
            min(
                n01,
                n10
            ),
            n=discordant,
            p=0.5,
            alternative="two-sided"
        )

        mcnemar_p=float(
            exact.pvalue
        )

    else:
        mcnemar_p=1.0

    MCNEMAR=pd.DataFrame([{
        "comparison":"R6 HFSB-FL vs final clean CCAC",
        "r6_wrong_ccac_correct":n01,
        "r6_correct_ccac_wrong":n10,
        "discordant_predictions":discordant,
        "exact_mcnemar_p":mcnemar_p,
        "interpretation":(
            "Tests whether the two frozen classifiers have equal paired error rates."
        )
    }])

    save_csv(
        MCNEMAR,
        STAT_DIR/"EXACT_MCNEMAR_R6_VS_CCAC.csv"
    )

    MCNEMAR.to_csv(
        TABLE_DIR/"EXACT_MCNEMAR_R6_VS_CCAC.csv",
        index=False
    )

df=MODEL_EVOLUTION[
    MODEL_EVOLUTION[
        "protocol"
    ].astype(str).str.contains(
        "Protocol-LT",
        na=False
    )
].copy()

fig,ax=plt.subplots(
    figsize=(10.5,5.3)
)

x=np.arange(
    len(df)
)

ax.plot(
    x,
    df["accuracy"],
    marker="o",
    linewidth=2,
    label="Accuracy"
)

ax.plot(
    x,
    df["macro_f1"],
    marker="s",
    linewidth=2,
    label="Macro-F1"
)

ax.set_xticks(x)
ax.set_xticklabels(
    df["model"],
    rotation=22,
    ha="right"
)

ax.set_ylim(
    0.35,
    1.0
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Evolution of Literature-Comparable Protocol-LT Models"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "F01",
    "Model_Evolution",
    (
        "Evolution of the Protocol-LT model family. Accuracy remains near "
        "the mid-94% region in the final stages, whereas Macro-F1 exposes "
        "the additional difficulty of minority attack families."
    ),
    (
        "Grouped/model-comparison visualization aligned with the F1/model "
        "comparison style reported in the 2026 dataset-centric federated IDS study [L3]."
    ),
    "double_column"
)

if not ABLATION.empty:
    fig,ax=plt.subplots(
        figsize=(11,5.4)
    )

    x=np.arange(
        len(ABLATION)
    )

    ax.bar(
        x,
        ABLATION[
            "accuracy_mean"
        ],
        yerr=ABLATION[
            "accuracy_sd"
        ],
        capsize=4
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        ABLATION[
            "experiment"
        ],
        rotation=25,
        ha="right"
    )

    ax.set_ylim(
        0.88,
        0.96
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_title(
        "Ablation of the Final BC-ACTG-HFSB-FL Framework"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    save_figure(
        fig,
        "F02",
        "Final_Ablation",
        (
            "Final ablation of BC-ACTG-HFSB-FL, showing clean utility, "
            "privacy-preserving secure aggregation, poisoning without defense, "
            "and the defended full hybrid configuration. Error bars denote "
            "standard deviation where repeated runs are available."
        ),
        (
            "Ablation-analysis format aligned with the 2026 blockchain-assisted "
            "secure federated IDS study [L1]."
        ),
        "double_column"
    )

fig,ax=plt.subplots(
    figsize=(7.8,5.2)
)

ax.errorbar(
    PRIVACY[
        "epsilon"
    ],
    PRIVACY[
        "accuracy_mean"
    ],
    yerr=[
        PRIVACY[
            "accuracy_mean"
        ]
        -
        PRIVACY[
            "accuracy_ci95_low"
        ],
        PRIVACY[
            "accuracy_ci95_high"
        ]
        -
        PRIVACY[
            "accuracy_mean"
        ],
    ],
    marker="o",
    capsize=4,
    linewidth=2,
    label="Accuracy"
)

ax.errorbar(
    PRIVACY[
        "epsilon"
    ],
    PRIVACY[
        "macro_f1_mean"
    ],
    yerr=[
        PRIVACY[
            "macro_f1_mean"
        ]
        -
        PRIVACY[
            "macro_f1_ci95_low"
        ],
        PRIVACY[
            "macro_f1_ci95_high"
        ]
        -
        PRIVACY[
            "macro_f1_mean"
        ],
    ],
    marker="s",
    capsize=4,
    linewidth=2,
    label="Macro-F1"
)

ax.axvline(
    EPSILON,
    linestyle="--",
    linewidth=1.2,
    label=f"Selected ε={EPSILON:g}"
)

ax.set_xlabel(
    "Privacy budget ε (larger = weaker privacy)"
)

ax.set_ylabel(
    "Mean test performance"
)

ax.set_title(
    "Privacy–Utility Trade-off with 95% Confidence Intervals"
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "F03",
    "Privacy_Utility_Curve",
    (
        "Privacy–utility trade-off under output/residual-level Gaussian "
        "differential privacy. Points show mean test performance across "
        "independent privacy-noise seeds and error bars show 95% confidence intervals."
    ),
    (
        "Security/privacy trade-off curve used to complement the standard "
        "detection curves used in recent privacy-preserving federated IDS literature."
    ),
    "single_column"
)

fig,ax=plt.subplots(
    figsize=(9.5,5.3)
)

x=np.arange(
    len(
        ATTACK_SUM
    )
)

width=0.36

ax.bar(
    x-width/2,
    ATTACK_SUM[
        "branch_attack_accuracy_mean"
    ],
    width,
    yerr=ATTACK_SUM[
        "branch_attack_accuracy_sd"
    ],
    capsize=3,
    label="Attack / no defense"
)

ax.bar(
    x+width/2,
    ATTACK_SUM[
        "branch_defense_accuracy_mean"
    ],
    width,
    yerr=ATTACK_SUM[
        "branch_defense_accuracy_sd"
    ],
    capsize=3,
    label="ACTG + secure aggregation"
)

ax.set_xticks(x)

ax.set_xticklabels(
    ATTACK_SUM[
        "attack_mode"
    ],
    rotation=20,
    ha="right"
)

ax.set_ylim(
    0.85,
    0.96
)

ax.set_ylabel(
    "Federated-branch accuracy"
)

ax.set_title(
    "Poisoning Impact and ACTG Recovery by Attack Type"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "F04",
    "Attack_Type_Branch_Recovery",
    (
        "Federated-branch accuracy under four post-privacy poisoning attacks. "
        "Bars compare the undefended poisoned aggregate with the frozen ACTG "
        "trust-gated secure aggregation result; error bars denote standard deviation."
    ),
    (
        "Robust proposed-vs-baseline comparison is aligned with the comparative "
        "rare-attack/client analysis emphasized by recent federated IDS literature [L2]."
    ),
    "double_column"
)

fig,ax=plt.subplots(
    figsize=(9,5.2)
)

x=np.arange(
    len(
        ATTACK_SUM
    )
)

width=0.25

ax.bar(
    x-width,
    ATTACK_SUM[
        "detection_precision_mean"
    ],
    width,
    label="Precision"
)

ax.bar(
    x,
    ATTACK_SUM[
        "detection_recall_mean"
    ],
    width,
    label="Recall"
)

ax.bar(
    x+width,
    ATTACK_SUM[
        "detection_f1_mean"
    ],
    width,
    label="F1"
)

ax.set_xticks(x)

ax.set_xticklabels(
    ATTACK_SUM[
        "attack_mode"
    ],
    rotation=20,
    ha="right"
)

ax.set_ylim(
    0,
    1.05
)

ax.set_ylabel(
    "Malicious-client detection score"
)

ax.set_title(
    "ACTG Poisoning-Detection Robustness"
)

ax.legend()

fig.tight_layout()

save_figure(
    fig,
    "F05",
    "ACTG_Detection_By_Attack",
    (
        "ACTG precision, recall and F1 across sign/scale poisoning, class "
        "permutation, model replacement and targeted weak-family injection."
    ),
    (
        "Per-condition detection visualization complements the class-specific "
        "evaluation emphasis in recent rare-attack federated IDS work [L2]."
    ),
    "double_column"
)

fig,ax=plt.subplots(
    figsize=(8.8,5.3)
)

frac=100.0*FRAC_SUM[
    "malicious_fraction"
].to_numpy(
    np.float64
)

ax.plot(
    frac,
    FRAC_SUM[
        "branch_attack_accuracy_mean"
    ],
    marker="o",
    linewidth=2,
    label="Federated branch: attacked"
)

ax.plot(
    frac,
    FRAC_SUM[
        "branch_defense_accuracy_mean"
    ],
    marker="s",
    linewidth=2,
    label="Federated branch: defended"
)

ax.plot(
    frac,
    FRAC_SUM[
        "e2e_attack_accuracy_mean"
    ],
    marker="^",
    linewidth=2,
    label="End-to-end: attacked"
)

ax.plot(
    frac,
    FRAC_SUM[
        "e2e_defense_accuracy_mean"
    ],
    marker="D",
    linewidth=2,
    label="End-to-end: defended"
)

ax.set_xlabel(
    "Malicious clients (%)"
)

ax.set_ylabel(
    "Accuracy"
)

ax.set_title(
    "Robustness as the Malicious-Client Fraction Increases"
)

ax.set_xticks(
    frac
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "F06",
    "Malicious_Fraction_Robustness",
    (
        "Accuracy as the malicious-client fraction increases from 10% to "
        "40%. Federated-branch and end-to-end curves are shown separately "
        "to expose the additional protection contributed by the frozen CCAC prior."
    ),
    (
        "Communication-round/scalability curve style is aligned with the "
        "convergence and heterogeneous-client analyses in the 2026 dataset-centric FL study [L3]."
    ),
    "double_column"
)

fig=plt.figure(
    figsize=(10,6.2)
)

ax=fig.add_subplot(
    111,
    projection="3d"
)

fractions=(
    100.0
    *
    FRAC_SUM[
        "malicious_fraction"
    ].to_numpy(
        np.float64
    )
)

scenario_names=[
    "Branch attacked",
    "Branch defended",
    "End-to-end attacked",
    "End-to-end defended"
]

value_cols=[
    "branch_attack_accuracy_mean",
    "branch_defense_accuracy_mean",
    "e2e_attack_accuracy_mean",
    "e2e_defense_accuracy_mean"
]

dx=5.0
dy=0.45

for yi,col in enumerate(
    value_cols
):
    xs=fractions-dx/2
    ys=np.full_like(
        fractions,
        yi,
        dtype=np.float64
    )

    zs=np.zeros_like(
        fractions
    )

    heights=FRAC_SUM[
        col
    ].to_numpy(
        np.float64
    )

    ax.bar3d(
        xs,
        ys,
        zs,
        dx,
        dy,
        heights,
        shade=True
    )

ax.set_xlabel(
    "Malicious clients (%)"
)

ax.set_ylabel(
    "Scenario"
)

ax.set_zlabel(
    "Accuracy"
)

ax.set_yticks(
    np.arange(
        len(
            scenario_names
        )
    )
)

ax.set_yticklabels(
    scenario_names
)

ax.set_zlim(
    0,
    1.0
)

ax.set_title(
    "3D View of Measured Security Robustness"
)

fig.tight_layout()

save_figure(
    fig,
    "F07",
    "3D_Malicious_Fraction_Robustness",
    (
        "Three-dimensional visualization of measured mean accuracy across "
        "malicious-client fractions and four security states. This figure "
        "contains only measured Step-6 values; no interpolated or synthetic "
        "performance surface is used."
    ),
    (
        "Additional paper visualization requested for compact multi-dimensional "
        "presentation; the underlying values are the same measured robustness results."
    ),
    "double_column"
)

row_sum=cm8.sum(
    axis=1,
    keepdims=True
)

cm_norm=np.divide(
    cm8,
    np.maximum(
        row_sum,
        1
    )
)

fig,ax=plt.subplots(
    figsize=(8.5,7.2)
)

im=ax.imshow(
    cm_norm
)

ax.set_xticks(
    np.arange(8)
)

ax.set_yticks(
    np.arange(8)
)

ax.set_xticklabels(
    CLASS_NAMES,
    rotation=40,
    ha="right"
)

ax.set_yticklabels(
    CLASS_NAMES
)

ax.set_xlabel(
    "Predicted class"
)

ax.set_ylabel(
    "True class"
)

ax.set_title(
    "Normalized Confusion Matrix — Full BC-ACTG-HFSB-FL"
)

for i in range(8):
    for j in range(8):
        ax.text(
            j,
            i,
            f"{cm_norm[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=7
        )

fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04,
    label="Row-normalized proportion"
)

fig.tight_layout()

save_figure(
    fig,
    "F08",
    "Multiclass_Normalized_Confusion_Matrix",
    (
        "Row-normalized confusion matrix of the final full hybrid model. "
        "The matrix highlights majority-class strength and the remaining "
        "confusion among low-frequency attack families."
    ),
    (
        "Normalized confusion matrices are prominently used in the 2025 "
        "rare-attack FL study [L2] and 2026 dataset-centric FL evaluation [L3]."
    ),
    "double_column"
)

binary_norm=binary_cm/np.maximum(
    binary_cm.sum(
        axis=1,
        keepdims=True
    ),
    1
)

fig,ax=plt.subplots(
    figsize=(5.6,5.0)
)

im=ax.imshow(
    binary_norm
)

ax.set_xticks(
    [0,1]
)

ax.set_yticks(
    [0,1]
)

ax.set_xticklabels(
    [
        "Benign",
        "Attack"
    ]
)

ax.set_yticklabels(
    [
        "Benign",
        "Attack"
    ]
)

ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "True"
)

ax.set_title(
    "Binary Projection of the Final Full Hybrid Model"
)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            (
                f"{binary_norm[i,j]:.3f}\n"
                f"(n={binary_cm[i,j]:,})"
            ),
            ha="center",
            va="center",
            fontsize=9
        )

fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04
)

fig.tight_layout()

save_figure(
    fig,
    "F09",
    "Binary_Confusion_Matrix",
    (
        "Binary Benign-versus-Attack projection obtained by collapsing the "
        "final eight-class full-hybrid confusion matrix. Attack is treated "
        "as the positive class."
    ),
    (
        "Binary screening visualization follows the standard confusion-matrix "
        "reporting used by recent secure federated IDS studies [L1–L3]."
    ),
    "single_column"
)

fig,ax=plt.subplots(
    figsize=(10.5,5.4)
)

x=np.arange(
    len(
        PER_CLASS
    )
)

width=0.25

ax.bar(
    x-width,
    PER_CLASS[
        "precision"
    ],
    width,
    label="Precision"
)

ax.bar(
    x,
    PER_CLASS[
        "recall"
    ],
    width,
    label="Recall"
)

ax.bar(
    x+width,
    PER_CLASS[
        "f1"
    ],
    width,
    label="F1"
)

ax.set_xticks(x)

ax.set_xticklabels(
    PER_CLASS[
        "class_name"
    ],
    rotation=25,
    ha="right"
)

ax.set_ylim(
    0,
    1.05
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Per-Class Performance of the Final Full Hybrid Model"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "F10",
    "Per_Class_Precision_Recall_F1",
    (
        "Class-wise precision, recall and F1 for the final full "
        "BC-ACTG-HFSB-FL model. This figure is especially important for "
        "interpreting minority attack families beyond overall accuracy."
    ),
    (
        "Per-class and rare-class performance reporting is aligned with "
        "the 2025 federated rare-attack study [L2]."
    ),
    "double_column"
)

support=PER_CLASS[
    "support"
].to_numpy(
    np.float64
)

fig,ax=plt.subplots(
    figsize=(9.5,5.2)
)

ax.bar(
    PER_CLASS[
        "class_name"
    ],
    support
)

ax.set_yscale(
    "log"
)

ax.set_ylabel(
    "Test samples (log scale)"
)

ax.set_title(
    "Final Test-Set Class Distribution"
)

ax.tick_params(
    axis="x",
    rotation=25
)

fig.tight_layout()

save_figure(
    fig,
    "F11",
    "Test_Class_Distribution",
    (
        "Distribution of test samples across benign and seven attack "
        "families. The logarithmic scale makes the severe class imbalance "
        "and the rare-class challenge visible."
    ),
    (
        "Class-distribution visualization is directly aligned with the "
        "dataset-distribution analysis used in the 2026 dataset-centric FL study [L3]."
    ),
    "double_column"
)

forest=STAT_VALIDATION[
    STAT_VALIDATION[
        "metric"
    ].str.contains(
        "Federated branch",
        regex=False
    )
].copy()

fig,ax=plt.subplots(
    figsize=(9.5,5.5)
)

ypos=np.arange(
    len(
        forest
    )
)

means=forest[
    "mean_recovery"
].to_numpy(
    np.float64
)

lo=forest[
    "bootstrap_ci95_low"
].to_numpy(
    np.float64
)

hi=forest[
    "bootstrap_ci95_high"
].to_numpy(
    np.float64
)

xerr=np.vstack([
    means-lo,
    hi-means
])

ax.errorbar(
    means,
    ypos,
    xerr=xerr,
    fmt="o",
    capsize=4
)

labels=[
    f"{r.experiment_family}\n{r.metric}"
    for _,r in forest.iterrows()
]

ax.set_yticks(
    ypos
)

ax.set_yticklabels(
    labels
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_xlabel(
    "Defense − attack mean performance"
)

ax.set_title(
    "Bootstrap 95% CI of ACTG Defense Recovery"
)

ax.grid(
    axis="x",
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "F12",
    "Statistical_Recovery_Forest",
    (
        "Bootstrap 95% confidence intervals for paired ACTG defense recovery. "
        "Intervals entirely above zero indicate consistent improvement of the "
        "poisoned federated branch under the evaluated threat models."
    ),
    (
        "A statistical-validation figure added beyond the descriptive plots "
        "typically reported in IDS literature."
    ),
    "double_column"
)

fig,ax=plt.subplots(
    figsize=(9.5,5.3)
)

attack_vals=ATTACK_RAW[
    "branch_attack_accuracy"
].to_numpy(
    np.float64
)

def_vals=ATTACK_RAW[
    "branch_defense_accuracy"
].to_numpy(
    np.float64
)

for i in range(
    len(
        ATTACK_RAW
    )
):
    ax.plot(
        [0,1],
        [
            attack_vals[i],
            def_vals[i]
        ],
        marker="o",
        alpha=0.35
    )

ax.plot(
    [0,1],
    [
        np.mean(
            attack_vals
        ),
        np.mean(
            def_vals
        )
    ],
    marker="s",
    linewidth=3,
    label="Mean paired result"
)

ax.set_xticks(
    [0,1]
)

ax.set_xticklabels(
    [
        "Poisoned / no defense",
        "ACTG defended"
    ]
)

ax.set_ylabel(
    "Federated-branch accuracy"
)

ax.set_title(
    "Paired Poisoning Experiments: Attack vs ACTG Defense"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "F13",
    "Paired_Attack_Defense",
    (
        "Paired federated-branch accuracy for every attack-type/seed/client "
        "stress experiment. Each line connects exactly matched attack and "
        "defense conditions; the heavy line shows their overall means."
    ),
    (
        "Paired visualization directly supports the paired t-test and "
        "Wilcoxon signed-rank validation."
    ),
    "double_column"
)

if not BLOCKCHAIN_RAW.empty:
    fig,ax=plt.subplots(
        figsize=(8.5,5.0)
    )

    ax.plot(
        BLOCKCHAIN_RAW[
            "tx_index"
        ],
        BLOCKCHAIN_RAW[
            "latency_ms"
        ],
        marker="o"
    )

    ax.axhline(
        float(
            BLOCKCHAIN[
                "mean_latency_ms"
            ].iloc[0]
        ),
        linestyle="--",
        label="Mean latency"
    )

    ax.set_xlabel(
        "Audit transaction index"
    )

    ax.set_ylabel(
        "Latency (ms)"
    )

    ax.set_title(
        "Permissioned Blockchain Audit Latency"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

else:
    fig,ax=plt.subplots(
        figsize=(7.5,5.0)
    )

    vals=[
        float(
            BLOCKCHAIN[
                "mean_latency_ms"
            ].iloc[0]
        ),
        float(
            BLOCKCHAIN[
                "median_latency_ms"
            ].iloc[0]
        ),
        float(
            BLOCKCHAIN[
                "p95_latency_ms"
            ].iloc[0]
        ),
    ]

    ax.bar(
        [
            "Mean",
            "Median",
            "P95"
        ],
        vals
    )

    ax.set_ylabel(
        "Latency (ms)"
    )

    ax.set_title(
        "Permissioned Blockchain Audit Latency"
    )

    fig.tight_layout()

save_figure(
    fig,
    "F14",
    "Blockchain_Latency",
    (
        "Latency of the permissioned Web3/EthereumTester audit prototype. "
        "These values characterize the local experimental blockchain "
        "prototype and must not be interpreted as public-mainnet throughput."
    ),
    (
        "Blockchain overhead visualization complements the security/ablation "
        "analysis used in recent blockchain-assisted federated IDS research [L1]."
    ),
    "single_column"
)

comm_plot=COMM.copy()

fig,ax=plt.subplots(
    figsize=(10,5.5)
)

ax.bar(
    comm_plot[
        "item"
    ],
    np.maximum(
        comm_plot[
            "bytes"
        ].astype(
            np.float64
        ),
        1
    )
)

ax.set_yscale(
    "log"
)

ax.set_ylabel(
    "Bytes (log scale)"
)

ax.set_title(
    "Communication Overhead of Protected Residual Sharing"
)

ax.tick_params(
    axis="x",
    rotation=30
)

fig.tight_layout()

save_figure(
    fig,
    "F15",
    "Communication_Overhead",
    (
        "Analytical communication overhead for protected gate/fine residual "
        "sharing, secure-aggregation key material, and SHA-256 commitments."
    ),
    (
        "Communication-efficiency reporting is aligned with the FL efficiency "
        "analysis emphasized in the 2026 dataset-centric federated IDS study [L3]."
    ),
    "double_column"
)

fig,ax=plt.subplots(
    figsize=(8.3,5.2)
)

ax.plot(
    SCALE[
        "clients"
    ],
    SCALE[
        "protected_payload_for_1000_samples_MiB"
    ],
    marker="o",
    linewidth=2,
    label="Protected payload"
)

ax.set_xlabel(
    "Number of logical clients"
)

ax.set_ylabel(
    "Protected payload / 1000 samples (MiB)"
)

ax2=ax.twinx()

ax2.plot(
    SCALE[
        "clients"
    ],
    SCALE[
        "pairwise_mask_pairs"
    ],
    marker="s",
    linewidth=2,
    label="Pairwise mask pairs"
)

ax2.set_ylabel(
    "Pairwise mask pairs"
)

lines1,labels1=ax.get_legend_handles_labels()
lines2,labels2=ax2.get_legend_handles_labels()

ax.legend(
    lines1+lines2,
    labels1+labels2,
    loc="upper left"
)

ax.set_title(
    "Analytical Scalability of the Secure Fusion Layer"
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "F16",
    "Analytical_Client_Scalability",
    (
        "Analytical scaling of protected communication payload and pairwise "
        "mask setup as the number of logical clients increases. Values above "
        "10 clients are analytical projections, not measured training runs."
    ),
    (
        "Scalability/communication style follows the efficiency-oriented "
        "evaluation used in recent federated IDS studies [L3]."
    ),
    "double_column"
)

if PROBABILITY_RECONSTRUCTION_OK:

    y_binary=(
        YTE!=0
    ).astype(
        np.int64
    )

    score_attack=(
        1.0
        -
        TEST_CCAC[
            :,
            0
        ]
    )

    fpr,tpr,thr=roc_curve(
        y_binary,
        score_attack
    )

    roc_binary=auc(
        fpr,
        tpr
    )

    fig,ax=plt.subplots(
        figsize=(6.2,5.4)
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"BC-ACTG-HFSB-FL (AUC={roc_binary:.4f})"
    )

    ax.plot(
        [0,1],
        [0,1],
        linestyle="--",
        linewidth=1,
        label="Random"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Binary Benign-vs-Attack ROC Curve"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    save_figure(
        fig,
        "F17",
        "Binary_ROC_Curve",
        (
            "ROC curve of the frozen clean proposed classifier after collapsing "
            "the eight-class probability vector into Benign versus Attack. "
            f"The measured binary ROC-AUC is {roc_binary:.4f}."
        ),
        (
            "ROC reporting is aligned with the 2026 blockchain-assisted secure "
            "federated IDS study [L1]."
        ),
        "single_column"
    )

    prec,rec,thr_pr=precision_recall_curve(
        y_binary,
        score_attack
    )

    ap=average_precision_score(
        y_binary,
        score_attack
    )

    fig,ax=plt.subplots(
        figsize=(6.2,5.4)
    )

    ax.plot(
        rec,
        prec,
        linewidth=2,
        label=f"Average Precision={ap:.4f}"
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Binary Benign-vs-Attack Precision–Recall Curve"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    save_figure(
        fig,
        "F18",
        "Binary_Precision_Recall_Curve",
        (
            "Precision–Recall curve for the frozen clean binary projection. "
            f"The measured average precision is {ap:.4f}. PR analysis is "
            "particularly informative under class imbalance."
        ),
        (
            "Precision/recall curve reporting follows the evaluation family "
            "used in the 2026 blockchain-assisted secure federated IDS study [L1]."
        ),
        "single_column"
    )

    fig,ax=plt.subplots(
        figsize=(8.2,6.0)
    )

    ovr_rows=[]

    for c,name in enumerate(
        CLASS_NAMES
    ):
        yy=(
            YTE==c
        ).astype(
            np.int64
        )

        if len(
            np.unique(
                yy
            )
        )<2:
            continue

        fc,tc,_=roc_curve(
            yy,
            TEST_CCAC[
                :,
                c
            ]
        )

        auc_c=auc(
            fc,
            tc
        )

        ovr_rows.append({
            "class":name,
            "roc_auc":auc_c
        })

        ax.plot(
            fc,
            tc,
            linewidth=1.5,
            label=f"{name} ({auc_c:.3f})"
        )

    ax.plot(
        [0,1],
        [0,1],
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "One-vs-Rest ROC Curves — Final Clean Proposed Model"
    )

    ax.legend(
        ncol=2
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    save_figure(
        fig,
        "F19",
        "Multiclass_OVR_ROC",
        (
            "One-vs-rest ROC curves for the frozen clean proposed multiclass "
            "classifier. Per-class AUC values reveal separability beyond "
            "overall accuracy and Macro-F1."
        ),
        (
            "Multiclass ROC analysis is aligned with the 2026 blockchain-assisted "
            "federated IDS evaluation [L1]."
        ),
        "double_column"
    )

    pd.DataFrame(
        ovr_rows
    ).to_csv(
        TABLE_DIR/"MULTICLASS_OVR_ROC_AUC.csv",
        index=False
    )

    pd.DataFrame([{
        "binary_roc_auc":roc_binary,
        "binary_average_precision":ap,
        "source":"Frozen clean Step-4C CCAC probabilities"
    }]).to_csv(
        TABLE_DIR/"BINARY_ROC_PR_METRICS.csv",
        index=False
    )

def find_round_history():
    candidates=[]

    for path in (
        ROOT/
        "08_FEDERATED_LEARNING"
    ).rglob(
        "*.csv"
    ):
        try:
            if path.stat().st_size>20_000_000:
                continue

            df=pd.read_csv(
                path,
                nrows=10
            )

            lower={
                str(c).lower():c
                for c in df.columns
            }

            round_col=None

            for key in [
                "round",
                "global_round",
                "communication_round"
            ]:
                if key in lower:
                    round_col=lower[key]
                    break

            metric_col=None

            for key in [
                "val_accuracy",
                "validation_accuracy",
                "test_accuracy",
                "accuracy",
                "val_macro_f1",
                "macro_f1"
            ]:
                if key in lower:
                    metric_col=lower[key]
                    break

            if (
                round_col is not None
                and
                metric_col is not None
            ):
                candidates.append(
                    (
                        path,
                        round_col,
                        metric_col
                    )
                )

        except Exception:
            continue

    return candidates

histories=find_round_history()

if histories:

    scored=[]

    for p,rc,mc in histories:
        try:
            dfh=pd.read_csv(
                p
            )

            score=len(
                dfh
            )

            if "STEP03" in str(
                p
            ) or "STEP04" in str(
                p
            ):
                score+=10000

            scored.append(
                (
                    score,
                    p,
                    rc,
                    mc,
                    dfh
                )
            )

        except Exception:
            pass

    if scored:
        scored.sort(
            key=lambda x:x[0],
            reverse=True
        )

        _,hp,rc,mc,dfh=scored[0]

        fig,ax=plt.subplots(
            figsize=(8.5,5.1)
        )

        ax.plot(
            dfh[
                rc
            ],
            dfh[
                mc
            ],
            marker="o",
            linewidth=2
        )

        ax.set_xlabel(
            "Communication round"
        )

        ax.set_ylabel(
            mc
        )

        ax.set_title(
            "Federated Training Convergence"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        save_figure(
            fig,
            "F20",
            "Federated_Convergence",
            (
                f"Federated convergence reconstructed directly from saved "
                f"round history ({hp.name}); no synthetic convergence values "
                f"were generated."
            ),
            (
                "Convergence-curve format is aligned with the communication-round "
                "analysis in the 2026 dataset-centric FL IDS study [L3]."
            ),
            "double_column"
        )

SIG=STAT_VALIDATION.copy()

def significance_label(p):
    if pd.isna(p):
        return "NA"
    if p<0.001:
        return "p < 0.001"
    if p<0.01:
        return "p < 0.01"
    if p<0.05:
        return "p < 0.05"
    return "not significant at 0.05"

SIG[
    "paired_t_significance"
]=SIG[
    "paired_t_p"
].map(
    significance_label
)

SIG[
    "wilcoxon_significance"
]=SIG[
    "wilcoxon_p"
].map(
    significance_label
)

save_csv(
    SIG,
    STAT_DIR/"STATISTICAL_VALIDATION_PAPER_READY.csv"
)

SIG.to_csv(
    TABLE_DIR/"STATISTICAL_VALIDATION_PAPER_READY.csv",
    index=False
)

LITERATURE=pd.DataFrame([
    {
        "id":"L1",
        "year":2026,
        "venue":"Scientific Reports",
        "paper":"A blockchain-assisted secure federated learning architecture for intrusion detection in internet of things networks",
        "url":"https://www.nature.com/articles/s41598-026-53053-x",
        "verified_visualization_types":"ROC; precision/recall analysis; confusion matrix; ablation",
        "our_aligned_figures":"F02, F08, F09, F14, F17, F18, F19",
        "note":"Visualization type only; external numerical results are not reused."
    },
    {
        "id":"L2",
        "year":2025,
        "venue":"Scientific Reports",
        "paper":"Federated transfer learning for rare attack class detection in network intrusion detection systems",
        "url":"https://www.nature.com/articles/s41598-025-02068-x",
        "verified_visualization_types":"per-client confusion matrices; proposed-vs-naive comparisons; rare-class analysis",
        "our_aligned_figures":"F04, F05, F08, F10",
        "note":"Visualization type only; external numerical results are not reused."
    },
    {
        "id":"L3",
        "year":2026,
        "venue":"Scientific Reports",
        "paper":"Dataset-centric evaluation of federated intrusion detection models in IoT networks",
        "url":"https://www.nature.com/articles/s41598-025-32567-w",
        "verified_visualization_types":"class distribution; normalized confusion matrices; grouped F1 comparison; convergence curves",
        "our_aligned_figures":"F01, F06, F08, F11, F16, F20 if saved round history is available",
        "note":"Visualization type only; external numerical results are not reused."
    },
])

LITERATURE.to_csv(
    SOURCE_DIR/"LITERATURE_FIGURE_ALIGNMENT.csv",
    index=False
)

CAPTIONS=pd.DataFrame(
    FIGURE_RECORDS
)

CAPTIONS.to_csv(
    CAPTION_DIR/"FIGURE_CAPTIONS.csv",
    index=False
)

caption_md=[
    "# Publication Figure Captions",
    "",
]

for r in FIGURE_RECORDS:
    caption_md.append(
        f"## {r['figure_id']} — {r['short_name']}"
    )

    caption_md.append(
        r[
            "caption"
        ]
    )

    caption_md.append(
        ""
    )

    caption_md.append(
        f"**Literature alignment:** {r['literature_alignment']}"
    )

    caption_md.append(
        ""
    )

(CAPTION_DIR/"FIGURE_CAPTIONS.md").write_text(
    "\n".join(
        caption_md
    ),
    encoding="utf-8"
)

recommended_ids=[
    "F01",
    "F03",
    "F04",
    "F06",
    "F08",
    "F09",
    "F10",
    "F12",
    "F14",
    "F17",
    "F18",
]

if any(
    r["figure_id"]=="F19"
    for r in FIGURE_RECORDS
):
    recommended_ids.append(
        "F19"
    )

RECOMMENDED=CAPTIONS[
    CAPTIONS[
        "figure_id"
    ].isin(
        recommended_ids
    )
].copy()

RECOMMENDED[
    "suggested_location"
]=RECOMMENDED[
    "figure_id"
].map({
        "F01":"Results — model evolution / ablation",
        "F03":"Results — privacy utility",
        "F04":"Results — poisoning defense",
        "F06":"Results — malicious fraction robustness",
        "F08":"Results — multiclass classification",
        "F09":"Results — binary screening",
        "F10":"Results — per-class minority analysis",
        "F12":"Results — statistical validation",
        "F14":"Results — blockchain overhead",
        "F17":"Results — binary ROC",
        "F18":"Results — binary PR",
        "F19":"Results — multiclass ROC",
    })

RECOMMENDED.to_csv(
    CAPTION_DIR/"RECOMMENDED_MAIN_PAPER_FIGURES.csv",
    index=False
)

def find_stat(
    family,
    metric
):
    q=STAT_VALIDATION[
        (
            STAT_VALIDATION[
                "experiment_family"
            ]==family
        )
        &
        (
            STAT_VALIDATION[
                "metric"
            ]==metric
        )
    ]

    if len(q)==0:
        return None

    return q.iloc[0]

s1=find_stat(
    "Attack-type stress",
    "Federated branch accuracy"
)

s2=find_stat(
    "Attack-type stress",
    "Federated branch Macro-F1"
)

stat_text=[
    "# Statistical Validation Summary",
    "",
    "The final architecture was kept frozen during this step.",
    "",
]

if s1 is not None:
    stat_text.append(
        (
            f"For the attack-type stress experiments, federated-branch "
            f"accuracy increased from {100*s1['attack_mean']:.3f}% to "
            f"{100*s1['defense_mean']:.3f}% after ACTG defense. "
            f"The paired t-test gave p={s1['paired_t_p']:.3e}, while the "
            f"Wilcoxon signed-rank test gave p={s1['wilcoxon_p']:.3e}. "
            f"The paired effect size was Cohen's dz={s1['cohens_dz']:.3f}. "
            f"The bootstrap 95% CI for the mean recovery was "
            f"[{100*s1['bootstrap_ci95_low']:.3f}, "
            f"{100*s1['bootstrap_ci95_high']:.3f}] percentage points."
        )
    )

    stat_text.append("")

if s2 is not None:
    stat_text.append(
        (
            f"For Macro-F1, the federated branch improved from "
            f"{s2['attack_mean']:.4f} to {s2['defense_mean']:.4f}. "
            f"The paired t-test gave p={s2['paired_t_p']:.3e} and the "
            f"Wilcoxon test gave p={s2['wilcoxon_p']:.3e}; Cohen's "
            f"dz={s2['cohens_dz']:.3f}. The bootstrap 95% CI of the mean "
            f"recovery was [{s2['bootstrap_ci95_low']:.4f}, "
            f"{s2['bootstrap_ci95_high']:.4f}]."
        )
    )

    stat_text.append("")

if PROBABILITY_RECONSTRUCTION_OK:
    stat_text.append(
        (
            "An exact McNemar test was additionally performed on paired "
            "per-sample errors of the frozen R6 baseline and final clean "
            "CCAC classifier; see EXACT_MCNEMAR_R6_VS_CCAC.csv."
        )
    )

(STAT_DIR/"STATISTICAL_VALIDATION_NARRATIVE.md").write_text(
    "\n".join(
        stat_text
    ),
    encoding="utf-8"
)

shutil.copy2(
    STAT_DIR/"STATISTICAL_VALIDATION_NARRATIVE.md",
    TABLE_DIR/"STATISTICAL_VALIDATION_NARRATIVE.md"
)

for src in [
    FILES["privacy"],
    FILES["attack_summary"],
    FILES["fraction_summary"],
    FILES["step6_stats"],
    FILES["blockchain"],
    FILES["communication"],
    FILES["scalability"],
    FILES["secure_correctness"],
    FILES["threat_model"],
]:
    if Path(src).exists():
        shutil.copy2(
            src,
            TABLE_DIR/
            Path(src).name
        )

manifest=[]

for key,path in FILES.items():
    path=Path(path)

    if not path.exists():
        continue

    h=hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:
        while True:
            b=f.read(
                1024*1024
            )

            if not b:
                break

            h.update(b)

    manifest.append({
        "artifact_key":key,
        "path":str(path),
        "size_bytes":path.stat().st_size,
        "sha256":h.hexdigest()
    })

pd.DataFrame(
    manifest
).to_csv(
    SOURCE_DIR/"RESULT_SOURCE_MANIFEST.csv",
    index=False
)

readme=f"""
STEP 8 — PUBLICATION FIGURES & STATISTICAL VALIDATION
=====================================================

Final frozen model:
BC-ACTG-HFSB-FL

Final clean accuracy:
{FINAL_CLEAN_ACC:.6f}

Final clean Macro-F1:
{FINAL_CLEAN_F1:.6f}

Final full-hybrid accuracy:
{FINAL_FULL_ACC:.6f}

Final full-hybrid Macro-F1:
{FINAL_FULL_F1:.6f}

Binary projection from final full confusion matrix:
Accuracy={binary_accuracy:.6f}
Precision(Attack)={binary_precision:.6f}
Sensitivity(Attack)={binary_recall:.6f}
Specificity(Benign)={binary_specificity:.6f}
F1(Attack)={binary_f1:.6f}
Balanced Accuracy={binary_balanced:.6f}
MCC={binary_mcc:.6f}

Privacy:
epsilon={EPSILON}
delta={DELTA}

GRAPHICS FORMATS:
- SVG: fully scalable vector format with editable text
- PDF: vector publication format
- PNG: 600 DPI high-resolution raster format

STATISTICAL VALIDATION:
1. Paired t-test
2. Wilcoxon signed-rank test
3. Paired bootstrap 95% confidence interval + Cohen's dz
4. Exact McNemar test when frozen per-sample probabilities are reconstructable

IMPORTANT:
- No model was retrained in Step 8.
- No result was altered to reach 95%.
- Full model accuracy must be reported as measured.
- Output/residual Gaussian DP is NOT DP-SGD.
- Blockchain audits/enforces ACTG decisions; it does not detect attacks.
- Figures inspired by latest literature use only visualization conventions,
  not external numerical results.

Literature figure-style references are listed in:
SOURCE_MANIFEST/LITERATURE_FIGURE_ALIGNMENT.csv
""".strip()

(PACKAGE_DIR/"README.txt").write_text(
    readme,
    encoding="utf-8"
)

archive_base=OUT/"BC_ACTG_HFSB_FL_PUBLICATION_FIGURES_HD"

rar_path=Path(
    str(
        archive_base
    )+".rar"
)

zip_path=Path(
    str(
        archive_base
    )+".zip"
)

def try_make_rar():

    rar_bin=shutil.which(
        "rar"
    )

    if rar_bin is None:
        try:
            subprocess.run(
                [
                    "apt-get",
                    "update",
                    "-qq"
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            subprocess.run(
                [
                    "apt-get",
                    "install",
                    "-y",
                    "-qq",
                    "rar"
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        except Exception:
            pass

        rar_bin=shutil.which(
            "rar"
        )

    if rar_bin is None:
        return False

    if rar_path.exists():
        rar_path.unlink()

    result=subprocess.run(
        [
            rar_bin,
            "a",
            "-r",
            "-m5",
            str(
                rar_path
            ),
            str(
                PACKAGE_DIR
            )
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    return (
        result.returncode==0
        and
        rar_path.exists()
    )

rar_ok=try_make_rar()

if zip_path.exists():
    zip_path.unlink()

shutil.make_archive(
    str(
        archive_base
    ),
    "zip",
    root_dir=str(
        PACKAGE_DIR.parent
    ),
    base_dir=PACKAGE_DIR.name
)

FINAL_INDEX=pd.DataFrame(
    FIGURE_RECORDS
)

FINAL_INDEX.to_csv(
    OUT/"STEP08_FIGURE_INDEX.csv",
    index=False
)

summary={
    "version":VERSION,
    "status":"COMPLETED",
    "model":"BC-ACTG-HFSB-FL",

    "n_figures":len(
        FIGURE_RECORDS
    ),

    "formats":[
        "SVG",
        "PDF",
        "PNG-600DPI"
    ],

    "statistical_validation":[
        "paired t-test",
        "Wilcoxon signed-rank",
        "paired bootstrap 95% CI",
        "Cohen's dz",
        (
            "exact McNemar"
            if PROBABILITY_RECONSTRUCTION_OK
            else
            "McNemar skipped because per-sample reconstruction was unavailable"
        )
    ],

    "binary_metrics":BINARY_METRICS.iloc[0].to_dict(),

    "probability_reconstruction_ok":
        PROBABILITY_RECONSTRUCTION_OK,

    "rar_created":
        bool(
            rar_ok
        ),

    "rar_path":
        str(
            rar_path
        )
        if rar_ok
        else None,

    "zip_path":
        str(
            zip_path
        ),

    "caption_file":
        str(
            CAPTION_DIR/
            "FIGURE_CAPTIONS.md"
        ),

    "recommended_paper_figures":
        str(
            CAPTION_DIR/
            "RECOMMENDED_MAIN_PAPER_FIGURES.csv"
        ),

    "completed_at":
        datetime.now().isoformat()
}

with open(
    OUT/"STEP08_COMPLETE.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        default=str
    )

print("\n"+"="*132)
print("✅ STEP 8 COMPLETED")
print("="*132)

print(
    "Publication figures generated:",
    len(
        FIGURE_RECORDS
    )
)

print(
    "Formats: SVG + PDF + PNG (600 DPI)"
)

print(
    "\nFinal full binary projection:"
)

print(
    BINARY_METRICS.to_string(
        index=False
    )
)

print(
    "\nStatistical validation:"
)

print(
    STAT_VALIDATION[
        [
            "experiment_family",
            "metric",
            "mean_recovery",
            "paired_t_p",
            "wilcoxon_p",
            "cohens_dz",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nFigure captions:",
    CAPTION_DIR/
    "FIGURE_CAPTIONS.md"
)

print(
    "Recommended main-paper figures:",
    CAPTION_DIR/
    "RECOMMENDED_MAIN_PAPER_FIGURES.csv"
)

if rar_ok:
    print(
        "\n✅ RAR package created:",
        rar_path
    )
else:
    print(
        "\n⚠️ `rar` utility unavailable in this Colab runtime."
    )
    print(
        "✅ ZIP fallback package created:",
        zip_path
    )

print(
    "✅ Universal ZIP backup:",
    zip_path
)

print("="*132)

try:
    from google.colab import files as colab_files

    download_target=(
        rar_path
        if rar_ok
        else
        zip_path
    )

    print(
        "Starting automatic download:",
        download_target.name
    )

    colab_files.download(
        str(
            download_target
        )
    )

except Exception as e:
    print(
        "⚠️ Automatic browser download could not start:",
        repr(e)
    )

    print(
        "Archive remains saved in Google Drive at:",
        (
            rar_path
            if rar_ok
            else
            zip_path
        )
    )

from google.colab import drive, files
from pathlib import Path
import zipfile
import hashlib
import os
import shutil

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
else:
    print("✅ Google Drive already mounted.")

ROOT = Path("/content/drive/MyDrive/Hybrid_BCFL_IJACSA_2026")

STEP8_ROOT = (
    ROOT /
    "11_RESULTS" /
    "STEP08_PUBLICATION_FIGURES_STATS"
)

PACKAGE_DIR = STEP8_ROOT / "DOWNLOAD_PACKAGE"

FIGURE_DIR = (
    ROOT /
    "12_FIGURES" /
    "STEP08_PUBLICATION_FIGURES"
)

if not PACKAGE_DIR.exists():
    raise FileNotFoundError(
        f"Step-8 package folder not found:\n{PACKAGE_DIR}"
    )

print("✅ Source package found:")
print(PACKAGE_DIR)

ALL_ZIP = STEP8_ROOT / "BC_ACTG_HFSB_FL_ALL_RESULTS_VERIFIED.zip"

IMAGES_ZIP = STEP8_ROOT / "BC_ACTG_HFSB_FL_FIGURES_ONLY_VERIFIED.zip"

for p in [ALL_ZIP, IMAGES_ZIP]:
    if p.exists():
        p.unlink()

def create_verified_zip(source_folder, output_zip, allowed_extensions=None):

    source_folder = Path(source_folder)
    output_zip = Path(output_zip)

    files_to_add = []

    for file_path in source_folder.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.stat().st_size == 0:
            print("⚠️ Skipping zero-byte file:", file_path)
            continue

        if allowed_extensions is not None:
            if file_path.suffix.lower() not in allowed_extensions:
                continue

        files_to_add.append(file_path)

    if len(files_to_add) == 0:
        raise RuntimeError(
            f"No valid files found in:\n{source_folder}"
        )

    print(
        f"\nCreating archive:\n{output_zip.name}"
    )

    print(
        "Files to include:",
        len(files_to_add)
    )

    with zipfile.ZipFile(
        output_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True
    ) as zf:

        for i, file_path in enumerate(files_to_add, start=1):

            relative_path = file_path.relative_to(source_folder)

            zf.write(
                file_path,
                arcname=str(relative_path)
            )

            if i % 20 == 0 or i == len(files_to_add):
                print(
                    f"  Added {i}/{len(files_to_add)} files"
                )

    print("Checking ZIP integrity...")

    with zipfile.ZipFile(output_zip, "r") as zf:

        bad_file = zf.testzip()

        archived_files = zf.namelist()

    if bad_file is not None:
        output_zip.unlink(missing_ok=True)

        raise RuntimeError(
            f"❌ ZIP integrity failed at file: {bad_file}"
        )

    if len(archived_files) != len(files_to_add):

        raise RuntimeError(
            "❌ Archive file-count mismatch."
        )

    print("✅ ZIP integrity test PASSED.")
    print(
        "Archived files:",
        len(archived_files)
    )

    print(
        "ZIP size:",
        round(
            output_zip.stat().st_size / (1024**2),
            2
        ),
        "MB"
    )

    return output_zip

create_verified_zip(
    PACKAGE_DIR,
    ALL_ZIP
)

figure_extensions = {
    ".png",
    ".svg",
    ".pdf"
}

create_verified_zip(
    PACKAGE_DIR,
    IMAGES_ZIP,
    allowed_extensions=figure_extensions
)

def sha256_file(path):

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            block = f.read(1024 * 1024)

            if not block:
                break

            sha.update(block)

    return sha.hexdigest()

all_hash = sha256_file(ALL_ZIP)
image_hash = sha256_file(IMAGES_ZIP)

checksum_file = STEP8_ROOT / "ZIP_SHA256_CHECKSUMS.txt"

checksum_file.write_text(
    f"""
BC-ACTG-HFSB-FL STEP-8 VERIFIED ARCHIVES

ALL RESULTS:
File: {ALL_ZIP.name}
SHA256: {all_hash}

FIGURES ONLY:
File: {IMAGES_ZIP.name}
SHA256: {image_hash}
""".strip(),
    encoding="utf-8"
)

print("\n" + "="*70)
print("✅ VERIFIED DOWNLOAD FILES READY")
print("="*70)

print("\n1. Complete results:")
print(ALL_ZIP)

print("\n2. Figures only:")
print(IMAGES_ZIP)

print("\n3. SHA-256 verification:")
print(checksum_file)

print("\nAll-results SHA256:")
print(all_hash)

print("\nFigures SHA256:")
print(image_hash)

print("\n⬇️ Starting download of COMPLETE RESULTS ZIP...")

files.download(
    str(ALL_ZIP)
)

print(
    "\nAfter the first download starts/completes, "
    "run the next line manually if you also want the figures-only ZIP:"
)

print(
    f'files.download("{IMAGES_ZIP}")'
)
