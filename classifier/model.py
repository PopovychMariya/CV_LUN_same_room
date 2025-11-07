import torch, numpy as np
from lightglue_keypoints import LGMatcher
from keypoints_grid import keypoints_to_grid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_clf = None
_lg  = None

def _lazy_init():
	global _clf, _lg
	if _clf is None:
		_clf = torch.jit.load("models/same_room_binary.ts", map_location=DEVICE).eval()
	if _lg is None:
		_lg = LGMatcher(device=DEVICE)

def warmup():
	_lazy_init()
	if DEVICE.type == "cuda":
		t = torch.zeros(1, 7, 32, 32, device=DEVICE)
		_ = _clf(t, t)

@torch.inference_mode()
def inference(img1_pil, img2_pil) -> int:
	_lazy_init()
	img1_np = np.array(img1_pil)
	img2_np = np.array(img2_pil)

	kp1, kp2, conf = _lg.FindMatches(img1_np, img2_np)
	H1, W1 = img1_np.shape[:2]
	H2, W2 = img2_np.shape[:2]
	grid = keypoints_to_grid(kp1, kp2, conf, H1, W1, H2, W2, G=32)  # [14,32,32]

	t = torch.from_numpy(grid).float().unsqueeze(0).to(DEVICE, non_blocking=True)
	A7, B7 = t[:, :7], t[:, 7:]
	return int(_clf(A7, B7).item())


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # pair 1
    img1a = np.array(Image.open("test0/image1.jpg"))
    img1b = np.array(Image.open("test0/image2.jpg"))
    res1 = inference(img1a, img1b)
    print("Pair test0:", res1)

    # pair 2
    img2a = np.array(Image.open("test1/image1.jpg"))
    img2b = np.array(Image.open("test1/image2.jpg"))
    res2 = inference(img2a, img2b)
    print("Pair test1:", res2)