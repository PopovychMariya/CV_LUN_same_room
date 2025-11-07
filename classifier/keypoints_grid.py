import numpy as np
import cv2

_EPS = 1e-6

def ransac_inliers(kp1, kp2, ransac_thresh=4.0):
	# kp1, kp2: float32 [N,2] in pixels
	if len(kp1) < 4:
		return np.zeros(len(kp1), dtype=bool)
	H, mask = cv2.findHomography(kp1, kp2, cv2.RANSAC, ransac_thresh)
	return np.zeros(len(kp1), dtype=bool) if mask is None else mask.ravel().astype(bool)


def keypoints_to_grid(kp1, kp2, conf, H1, W1, H2, W2, G=32, max_kp_cap=1024):
	"""
	Returns grid[14, G, G] with confidence-weighted means.

	Channels (source 0..6):
	  0: count
	  1: sum(conf)
	  2: mean(dx) weighted by conf
	  3: mean(dy) weighted by conf
	  4: mean(cosθ) weighted by conf
	  5: mean(sinθ) weighted by conf
	  6: mean(inlier) weighted by conf  (≈ inlier ratio under conf weights)

	Channels (target 7..13) mirror source, with motion/opposites for target bins.
	"""
	kp1, kp2, conf = [np.asarray(x, np.float32) for x in (kp1, kp2, conf)]
	grid = np.zeros((14, G, G), np.float32)

	N = kp1.shape[0]
	if N == 0:
		return grid

	# Inliers by homography RANSAC in pixel space
	inliers = ransac_inliers(kp1, kp2)

	# Normalized coordinates in [0,1]
	x1 = kp1[:, 0] / max(W1, 1); y1 = kp1[:, 1] / max(H1, 1)
	x2 = kp2[:, 0] / max(W2, 1); y2 = kp2[:, 1] / max(H2, 1)

	# Motion in normalized space and orientation unit vectors
	dx = x2 - x1
	dy = y2 - y1
	ang = np.arctan2(dy, dx + _EPS)
	cos_t, sin_t = np.cos(ang), np.sin(ang)

	# Map to grid indices (row=y, col=x), rounding like your original
	r1 = y1 * G - 0.5; c1 = x1 * G - 0.5
	r2 = y2 * G - 0.5; c2 = x2 * G - 0.5
	ri1 = np.clip(np.round(r1).astype(int), 0, G - 1)
	ci1 = np.clip(np.round(c1).astype(int), 0, G - 1)
	ri2 = np.clip(np.round(r2).astype(int), 0, G - 1)
	ci2 = np.clip(np.round(c2).astype(int), 0, G - 1)

	# Confidence weights (float scalar per kp)
	w = conf.astype(np.float32)

	# Accumulate (source bins)
	for k in range(N):
		ii, jj = ri1[k], ci1[k]
		wk = float(w[k]); inl = float(inliers[k])

		grid[0, ii, jj] += 1.0                # count
		grid[1, ii, jj] += wk                 # sum(conf)
		grid[2, ii, jj] += wk * dx[k]
		grid[3, ii, jj] += wk * dy[k]
		grid[4, ii, jj] += wk * cos_t[k]
		grid[5, ii, jj] += wk * sin_t[k]
		grid[6, ii, jj] += wk * inl           # sum(conf*inlier)

		# Accumulate (target bins) with opposite motion/orientation
		i2, j2 = ri2[k], ci2[k]
		grid[7,  i2, j2] += 1.0
		grid[8,  i2, j2] += wk
		grid[9,  i2, j2] += wk * (-dx[k])
		grid[10, i2, j2] += wk * (-dy[k])
		grid[11, i2, j2] += wk * (-cos_t[k])
		grid[12, i2, j2] += wk * (-sin_t[k])
		grid[13, i2, j2] += wk * inl

	# Masks for valid averaging by sum of confidences (not by count)
	w1 = grid[1]     # sum(conf) source
	w2 = grid[8]     # sum(conf) target
	m1 = w1 > 0
	m2 = w2 > 0

	# Turn weighted sums into weighted means
	np.divide(grid[2:7], w1, out=grid[2:7], where=m1)     # dx,dy,cos,sin, inlier_ratio
	grid[2:7][:, ~m1] = 0

	np.divide(grid[9:14], w2, out=grid[9:14], where=m2)   # mirrored for target
	grid[9:14][:, ~m2] = 0

	# Optional: cap/scale counts to keep them tame (your original intent), but fix the mask.
	mc1 = grid[0] > 0
	mc2 = grid[7] > 0
	np.divide(grid[0], float(max_kp_cap), out=grid[0], where=mc1)
	np.divide(grid[7], float(max_kp_cap), out=grid[7], where=mc2)

	return grid
