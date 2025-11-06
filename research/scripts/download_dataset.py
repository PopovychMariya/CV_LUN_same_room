from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, os
from urllib.request import Request, urlopen
from tqdm.auto import tqdm

from path_config import DATASET_FOLDER_PATHS, DATASET_ANNOTATIONS

CHUNK    = 1 << 15
WORKERS  = 24


def _iter_pairs(ann_path: Path):
    data = json.loads(ann_path.read_text(encoding="utf-8"))["data"]["results"]
    for row in data:
        tid = str(row["taskId"])
        img1 = row["representativeData"]["image1"]["imageUrl"]
        img2 = row["representativeData"]["image2"]["imageUrl"]
        yield tid, img1, img2


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    backoff = 1
    tmp = dest.with_suffix(dest.suffix + ".part")
    while True:
        try:
            req = Request(url, headers={"User-Agent": "curl/8"})
            with urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return
        except Exception:
            try:
                tmp.exists() and tmp.unlink()
            except Exception:
                pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 32)


def download_dataset(ann_path: Path, out_root: Path):
    jobs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for tid, url1, url2 in _iter_pairs(ann_path):
            folder = out_root / tid
            jobs.append(ex.submit(_download, url1, folder / "image1.jpg"))
            jobs.append(ex.submit(_download, url2, folder / "image2.jpg"))
        with tqdm(total=len(jobs), desc="Downloading images", dynamic_ncols=True, leave=False) as pbar:
            for fut in as_completed(jobs):
                fut.result()
                pbar.update(1)


if __name__ == "__main__":
    from path_config import DATASET_FOLDER_PATHS, DATASET_ANNOTATIONS
    
    train_out = DATASET_FOLDER_PATHS["train_folder_path"]
    test_out  = DATASET_FOLDER_PATHS["test_folder_path"]

    train_ann = DATASET_ANNOTATIONS["train_annotation_path"]
    test_ann  = DATASET_ANNOTATIONS["test_annotation_path"]

    print("Downloading training dataset...")
    download_dataset(train_ann, train_out)

    print("Downloading test dataset...")
    download_dataset(test_ann, test_out)

    print("✅ All downloads completed successfully.")
