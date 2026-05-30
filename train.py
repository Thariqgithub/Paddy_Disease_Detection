# ============================================================
#  Rice / Paddy Leaf Disease Detection — YOLOv8 Training
#  Author  : Thariq
#  Model   : YOLOv8s  (swap to yolov8m.pt for better accuracy)
#  Dataset : Roboflow  →  rice-disease-detection-zwaa8-3ycgi
# ============================================================

import os
import sys
import time
import shutil
from pathlib import Path

# ── Dependency check ────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed.")
    print("  Run:  pip install ultralytics")
    sys.exit(1)

try:
    from roboflow import Roboflow
except ImportError:
    print("[ERROR] roboflow not installed.")
    print("  Run:  pip install roboflow")
    sys.exit(1)

# ============================================================
#  CONFIGURATION  ← Edit these values
# ============================================================

ROBOFLOW_API_KEY  = yHJa2mGEAVjWvN4hYOi5   # 👈 Paste your Roboflow API key
WORKSPACE         = "ms-workspace-7urmz"
PROJECT_NAME      = "rice-disease-detection-zwaa8-3ycgi"
DATASET_VERSION   = 1
DATASET_FORMAT    = "yolov8"

# Model
BASE_MODEL        = "yolov8s.pt"   # yolov8n / yolov8s / yolov8m / yolov8l
EPOCHS            = 100
IMAGE_SIZE        = 640
BATCH_SIZE        = 8              # 8 → CPU safe  |  16–32 → GPU
DEVICE            = "cpu"          # "cpu"  or  0  (GPU index)
PATIENCE          = 20             # Early stopping patience
LEARNING_RATE     = 0.01
WORKERS           = 2              # Data loader workers (keep low on Windows)

# Output
PROJECT_DIR       = "runs/detect"
RUN_NAME          = "leaf_disease_v1"

# ============================================================
#  STEP 1 — Download Dataset from Roboflow
# ============================================================

def download_dataset():
    print("\n" + "="*55)
    print("  STEP 1 : Downloading Dataset from Roboflow")
    print("="*55)

    if ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        print("\n[ERROR] You forgot to set your Roboflow API key!")
        print("  1. Go to  https://roboflow.com  → Login")
        print("  2. Click Profile → Settings → Copy API Key")
        print("  3. Paste it into ROBOFLOW_API_KEY above\n")
        sys.exit(1)

    # Skip download if dataset already exists
    dataset_path = Path("dataset")
    data_yaml    = dataset_path / "data.yaml"

    if data_yaml.exists():
        print(f"  [INFO] Dataset already found at '{dataset_path}' — skipping download.")
        return str(dataset_path)

    try:
        rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT_NAME)
        version = project.version(DATASET_VERSION)
        dataset = version.download(DATASET_FORMAT)
        print(f"  [OK] Dataset downloaded to: {dataset.location}")
        return dataset.location

    except Exception as e:
        print(f"\n[ERROR] Roboflow download failed: {e}")
        print("  Check your API key and internet connection.")
        sys.exit(1)


# ============================================================
#  STEP 2 — Validate data.yaml
# ============================================================

def validate_yaml(dataset_location: str) -> str:
    print("\n" + "="*55)
    print("  STEP 2 : Validating data.yaml")
    print("="*55)

    import yaml

    yaml_path = Path(dataset_location) / "data.yaml"
    if not yaml_path.exists():
        # Try root-level dataset folder
        yaml_path = Path("dataset") / "data.yaml"

    if not yaml_path.exists():
        print(f"  [ERROR] data.yaml not found at: {yaml_path}")
        sys.exit(1)

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"  Classes   : {cfg.get('nc', '?')}")
    print(f"  Names     : {cfg.get('names', [])}")
    print(f"  Train path: {cfg.get('train', '?')}")
    print(f"  Val path  : {cfg.get('val', '?')}")
    print(f"  YAML      : {yaml_path}")
    print("  [OK] data.yaml looks good.")

    return str(yaml_path)


# ============================================================
#  STEP 3 — Train
# ============================================================

def train(yaml_path: str):
    print("\n" + "="*55)
    print("  STEP 3 : Training YOLOv8 Model")
    print("="*55)
    print(f"  Base model  : {BASE_MODEL}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Image size  : {IMAGE_SIZE}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Device      : {DEVICE}")
    print(f"  Data yaml   : {yaml_path}")
    print()

    model = YOLO(BASE_MODEL)   # Downloads pretrained weights automatically

    start = time.time()

    results = model.train(
        data       = yaml_path,
        epochs     = EPOCHS,
        imgsz      = IMAGE_SIZE,
        batch      = BATCH_SIZE,
        device     = DEVICE,
        patience   = PATIENCE,
        lr0        = LEARNING_RATE,
        workers    = WORKERS,
        project    = PROJECT_DIR,
        name       = RUN_NAME,
        exist_ok   = True,       # Overwrite run folder if exists
        pretrained = True,
        verbose    = True,

        # Augmentation (good defaults for leaf disease)
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        degrees    = 10.0,
        translate  = 0.1,
        scale      = 0.5,
        flipud     = 0.2,
        fliplr     = 0.5,
        mosaic     = 1.0,
    )

    elapsed = time.time() - start
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n  [OK] Training complete in {int(hours)}h {int(mins)}m {int(secs)}s")

    return model


# ============================================================
#  STEP 4 — Validate
# ============================================================

def validate(model):
    print("\n" + "="*55)
    print("  STEP 4 : Validating Model")
    print("="*55)

    metrics = model.val()

    map50    = metrics.box.map50
    map5095  = metrics.box.map
    precision = metrics.box.mp
    recall    = metrics.box.mr

    print(f"\n  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  mAP@0.5    : {map50:.4f}")
    print(f"  mAP@0.5:95 : {map5095:.4f}")

    return metrics


# ============================================================
#  STEP 5 — Copy best.pt to root for Live_detect.py
# ============================================================

def copy_best_weights():
    print("\n" + "="*55)
    print("  STEP 5 : Copying best.pt to project root")
    print("="*55)

    best_src = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    best_dst = Path("best.pt")

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"  [OK] best.pt copied to: {best_dst.resolve()}")
    else:
        print(f"  [WARN] best.pt not found at {best_src}")
        print("  Check the runs/detect folder manually.")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Rice Leaf Disease Detection — YOLOv8 Training     ║")
    print("╚══════════════════════════════════════════════════════╝")

    # 1. Download
    dataset_location = download_dataset()

    # 2. Validate YAML
    yaml_path = validate_yaml(dataset_location)

    # 3. Train
    model = train(yaml_path)

    # 4. Validate metrics
    validate(model)

    # 5. Copy best.pt to root
    copy_best_weights()

    print("\n" + "="*55)
    print("  ✅  ALL DONE!")
    print("  Run:  python Live_detect.py")
    print("="*55 + "\n")