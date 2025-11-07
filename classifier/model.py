# model.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from lightglue_keypoints import LGMatcher
from keypoints_grid import rasterize_matches

# ---------------- config ----------------
GRID = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_CANDIDATES = [
    "models/cross_att1_model_best.pt",
]

# Globals (lazy singletons)
_matcher: Optional[LGMatcher] = None
_head: Optional[torch.nn.Module] = None
_thr: float = 0.50

# --------------- loading ----------------
def _ckpt_path() -> Path:
    for p in CKPT_CANDIDATES:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"Checkpoint not found. Tried: {', '.join(CKPT_CANDIDATES)}")


def _build_head(in_channels: int) -> torch.nn.Module:
    """
    Replace the import below with your actual model class/factory.
    Must accept `in_channels` and produce a module that maps [B, C=14 or config.C, G, G] -> logits/prob.
    """
    # Example (adjust to your project):
    # from cross_att1_model import CrossAtt1
    # return CrossAtt1(in_channels=in_channels)
    raise RuntimeError("Provide your head constructor in _build_head(in_channels).")


def _load_head() -> torch.nn.Module:
    ckpt = torch.load(str(_ckpt_path()), map_location=DEVICE)
    in_ch = int(ckpt.get("config", {}).get("in_channels") or 14)
    head = _build_head(in_ch).to(DEVICE)
    head.load_state_dict(ckpt["model_state"], strict=True)
    head.eval()

    global _thr
    _thr = float(ckpt.get("best_thr", 0.50))
    return head


def _ensure_loaded() -> Tuple[LGMatcher, torch.nn.Module]:
    global _matcher, _head
    if _matcher is None:
        _matcher = LGMatcher(device=DEVICE)
    if _head is None:
        _head = _load_head()
    return _matcher, _head

# --------------- helpers ----------------
def _np_rgb(img: Image.Image) -> np.ndarray:
    if not isinstance(img, Image.Image):
        raise TypeError("inference expects PIL.Image inputs")
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


@torch.inference_mode()
def _head_prob(grid: np.ndarray, head: torch.nn.Module) -> float:
    x = torch.from_numpy(grid).unsqueeze(0).to(DEVICE)  # [1, C, G, G]
    out = head(x)
    if isinstance(out, (tuple, list)):
        out = out[-1]
    if out.ndim == 0:
        prob = torch.sigmoid(out).item()
    elif out.ndim == 1 and out.shape[0] in (1,):
        prob = torch.sigmoid(out[0]).item()
    elif out.ndim == 2 and out.shape[-1] == 1:
        prob = torch.sigmoid(out[:, 0]).item()
    elif out.ndim == 2 and out.shape[-1] == 2:
        prob = torch.softmax(out, dim=-1)[:, 1].item()
    else:
        prob = torch.sigmoid(out.mean()).item()
    return float(prob)

# --------------- public API --------------
def warmup() -> None:
    """
    Optional: pre-load matcher + head and do a dry run so first call is fast.
    """
    matcher, head = _ensure_loaded()
    dummy = np.zeros((14, GRID, GRID), np.float32)
    _ = _head_prob(dummy, head)


def inference(img1: Image.Image, img2: Image.Image) -> int:
    """
    Returns 1 if same room, 0 otherwise. Color preserved (no grayscale).
    """
    matcher, head = _ensure_loaded()

    a = _np_rgb(img1)
    b = _np_rgb(img2)

    k0, k1, conf = matcher.FindMatches(a, b)  # np arrays
    H1, W1 = a.shape[:2]
    H2, W2 = b.shape[:2]

    grid = rasterize_matches(k0, k1, conf, H1, W1, H2, W2, G=GRID).astype(np.float32, copy=False)
    prob = _head_prob(grid, head)
    return 1 if prob >= _thr else 0