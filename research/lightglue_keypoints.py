from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import match_pair, numpy_image_to_torch


class LGMatcher:
	def __init__(self, device=None):
		self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
		self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
		self.matcher = LightGlue(features="superpoint", mp=True).eval().to(self.device)

	@torch.inference_mode()
	def FindMatches(self, img1: np.ndarray, img2: np.ndarray):
		def prepare(img: np.ndarray) -> torch.Tensor:
			arr = img.astype(np.float32)
			if arr.max() <= 1.0:
				arr *= 255.0
			arr = np.clip(arr, 0.0, 255.0)
			tensor = numpy_image_to_torch(arr)
			return tensor.to(self.device)

		im0 = prepare(img1)
		im1 = prepare(img2)

		f0, f1, out = match_pair(
			self.extractor,
			self.matcher,
			im0,
			im1,
			device=self.device,
		)
		m = out["matches"]
		s = out["scores"]
		if m is None or m.numel() == 0:
			return (np.zeros((0, 2), np.float32),
			        np.zeros((0, 2), np.float32),
			        np.zeros((0,), np.float32))

		k0 = f0["keypoints"][m[:, 0]].detach().cpu().numpy().astype(np.float32)
		k1 = f1["keypoints"][m[:, 1]].detach().cpu().numpy().astype(np.float32)
		conf = s.detach().cpu().numpy().astype(np.float32)
		return k0, k1, conf


def save_keypoints(loader, path: Path, lg: LGMatcher):
	path = Path(path)
	os.makedirs(path, exist_ok=True)
	for batch in tqdm(loader, desc="Loading batches"):
		imgs1 = batch["image1"]
		imgs2 = batch["image2"]
		task_ids = batch["task_id"]
		for idx, (img1, img2) in enumerate(zip(imgs1, imgs2)):
			img1_np = np.array(img1)
			img2_np = np.array(img2)
			kp1, kp2, conf = lg.FindMatches(img1_np, img2_np)
			H1, W1 = img1_np.shape[:2]
			H2, W2 = img2_np.shape[:2]
			out_fp = Path(path) / task_ids[idx]
			os.makedirs(out_fp, exist_ok=True)
			np.savez_compressed(
				out_fp / "lightglue_matches.npz",
				kp1=kp1,
				kp2=kp2,
				conf=conf,
				H1=np.int32(H1), W1=np.int32(W1),
				H2=np.int32(H2), W2=np.int32(W2),
			)


if __name__ == "__main__":
    from dataloader import make_loader
    from path_config import (
        DATASET_FOLDER_PATHS,
        DATASET_ANNOTATIONS,
        DETECTED_KEYPOINTS,
    )

    print("Preparing dataloaders...")
    train_ds, train_loader = make_loader(
        annotation_path=DATASET_ANNOTATIONS["train_annotation_path"],
        dataset_path=DATASET_FOLDER_PATHS["train_folder_path"],
        mode="train", batch_size=64, num_workers=8, shuffle=False,
    )
    test_ds, test_loader = make_loader(
        annotation_path=DATASET_ANNOTATIONS["test_annotation_path"],
        dataset_path=DATASET_FOLDER_PATHS["test_folder_path"],
        mode="test", batch_size=64, num_workers=8, shuffle=False
    )

    print("Loading LightGlue weights...")
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    lg = LGMatcher(device=dev)

    print("Extracting keypoints for train set...")
    save_keypoints(train_loader, DETECTED_KEYPOINTS["train"], lg)
    print("Extracting keypoints for test set...")
    save_keypoints(test_loader, DETECTED_KEYPOINTS["test"], lg)
    print("✅ All keypoints detected and saved successfully.")
