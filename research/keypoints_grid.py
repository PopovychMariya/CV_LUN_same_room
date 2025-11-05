import numpy as np
import cv2

_EPS = 1e-12


def ransac_inliers(kp1, kp2, ransac_thresh=4.0):
    if len(kp1) < 4:
        return np.zeros(len(kp1), bool)
    H, mask = cv2.findHomography(kp1, kp2, cv2.RANSAC, ransac_thresh)
    return np.zeros(len(kp1), bool) if mask is None else mask.ravel().astype(bool)


def rasterize_matches(kp1, kp2, conf, H1, W1, H2, W2, G=32):
    kp1, kp2, conf = [np.asarray(x, np.float32) for x in [kp1, kp2, conf]]
    grid = np.zeros((14, G, G), np.float32)

    N = kp1.shape[0]
    if N==0:  return grid

    # ransac inliers
    inliers = ransac_inliers(kp1, kp2)

    # normalized coordinates in [0,1]
    x1 = kp1[:, 0] / max(W1, 1); y1 = kp1[:, 1] / max(H1, 1)
    x2 = kp2[:, 0] / max(W2, 1); y2 = kp2[:, 1] / max(H2, 1)

    # motion in normalized space
    dx = x2 - x1
    dy = y2 - y1
    ang = np.arctan2(dy, dx + _EPS)
    cos_t, sin_t = np.cos(ang), np.sin(ang)

    # map to grid indices (row=y, col=x)
    r1 = y1 * G - 0.5
    c1 = x1 * G - 0.5
    r2 = y2 * G - 0.5
    c2 = x2 * G - 0.5
    ri1 = np.clip(np.round(r1).astype(int), 0, G-1)
    ci1 = np.clip(np.round(c1).astype(int), 0, G-1)
    ri2 = np.clip(np.round(r2).astype(int), 0, G-1)
    ci2 = np.clip(np.round(c2).astype(int), 0, G-1)

    # accumulate per keypoint with confidence weights
    for k in range(N):
        w = float(conf[k])
        ii, jj = ri1[k], ci1[k]
        grid[0, ii, jj] += 1.0                  # count of kps
        grid[1, ii, jj] += w                    # confidence sum
        grid[2, ii, jj] += dx[k]                # motion x (norm)
        grid[3, ii, jj] += dy[k]                # motion y (norm)
        grid[4, ii, jj] += cos_t[k]
        grid[5, ii, jj] += sin_t[k]
        grid[6, ii, jj] += float(inliers[k])

        i2, j2 = ri2[k], ci2[k]
        grid[7,  i2, j2] += 1.0
        grid[8,  i2, j2] += conf[k]
        grid[9,  i2, j2] += -dx[k]              # opposite direction
        grid[10, i2, j2] += -dy[k]
        grid[11, i2, j2] += -cos_t[k]
        grid[12, i2, j2] += -sin_t[k]
        grid[13, i2, j2] += float(inliers[k])

    # counts
    c1 = grid[0]
    c2 = grid[7]

    # masks for non-empty cells
    m1 = c1 > 0
    m2 = c2 > 0

    # normalize channels 1..6 by c1, where m1 is True
    np.divide(grid[1:7], c1, out=grid[1:7], where=m1)
    grid[1:7][:, ~m1] = 0

    # normalize channels 8..13 by c2, where m2 is True
    np.divide(grid[8:14], c2, out=grid[8:14], where=m2)
    grid[8:14][:, ~m2] = 0
    
    return grid


if __name__ == "__main__":
    from tqdm import tqdm
    from path_config import OMNIGLUE_KEYPOINTS

    G = 32  # grid size

    for split_name in ["train", "test"]:
        base_dir = OMNIGLUE_KEYPOINTS[split_name]
        npz_files = list(base_dir.rglob("matches.npz"))

        print(f"Processing {len(npz_files)} files for split '{split_name}'...")

        for npz_path in tqdm(npz_files, desc=f"{split_name} split"):
            data = np.load(npz_path)
            kp1, kp2, conf = data["kp1"], data["kp2"], data["conf"]
            H1, W1, H2, W2 = map(int, [data["H1"], data["W1"], data["H2"], data["W2"]])

            grid = rasterize_matches(kp1, kp2, conf, H1, W1, H2, W2, G=G)
            out_fp = npz_path.parent / f"grid_G{G}.npy"
            np.save(out_fp, grid.astype(np.float16))

    print("✅ All grids generated and saved successfully.")
