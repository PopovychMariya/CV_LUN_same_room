from pathlib import Path

# Root dataset directory
ROOT = Path(__file__).resolve().parents[0]

# Dataset
DATASET_FOLDER_PATHS = {
	"train_folder_path": ROOT / "dataset" / "train",
	"test_folder_path": ROOT / "dataset" / "test",
}

DATASET_ANNOTATIONS = {
	"train_annotation_path": ROOT / "dataset" / "97.json",
	"test_annotation_path": ROOT / "dataset" / "test_dataset.json",
}

TRAIN_LABELS = ROOT / "dataset" / "train_labels.csv"

# Detected keypoints
DETECTED_KEYPOINTS = {
    "train": ROOT / "keypoints" / "train",
    "test": ROOT / "keypoints" / "test"
}

# Archives
ARCHIVES_PATH = ROOT/ "archives"