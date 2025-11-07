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

# Saving models weights
MODELS_PATH = ROOT / "models" 

# Archives
ARCHIVES_PATH = ROOT/ "archives"

if __name__ == "__main__":
	# Collect all directories to ensure they exist
	to_make = [
		DATASET_FOLDER_PATHS["train_folder_path"],
		DATASET_FOLDER_PATHS["test_folder_path"],
		DETECTED_KEYPOINTS["train"],
		DETECTED_KEYPOINTS["test"],
		MODELS_PATH,
		ARCHIVES_PATH,
	]

	for p in to_make:
		p.mkdir(parents=True, exist_ok=True)
		print(f"[OK] ensured: {p}")