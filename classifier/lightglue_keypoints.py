import numpy as np

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
