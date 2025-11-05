import json
import csv


def majority_label(answers, id_map={'0': 0, '1': 1}):
	votes = []
	for a in (answers or []):
		for obj in (a.get('answer') or []):
			k = str(obj.get('id'))
			if k in id_map:
				votes.append(id_map[k])
	return 1 if sum(votes) >= len(votes) / 2 else 0


def save_labels(ann_path, out_csv):
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    results = data["data"]["results"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "label"])
        for r in results:
            tid = r["taskId"]
            label = majority_label(r.get("answers"))
            writer.writerow([tid, label])


if __name__ == "__main__":
    from path_config import DATASET_ANNOTATIONS, TRAIN_LABELS
    
    save_labels(DATASET_ANNOTATIONS["train_annotation_path"], TRAIN_LABELS)
    print(f"✅ CSV saved to {TRAIN_LABELS}")
    