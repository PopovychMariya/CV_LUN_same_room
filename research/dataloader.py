import json
from pathlib import Path
from typing import Optional, Sequence, Dict, Any
from PIL import Image, ImageFile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

ImageFile.LOAD_TRUNCATED_IMAGES = False


def majority_label(answers, id_map={'0': 0, '1': 1}):
	votes = []
	for a in (answers or []):
		for obj in (a.get('answer') or []):
			k = str(obj.get('id'))
			if k in id_map:
				votes.append(id_map[k])
	if not votes:
		return None
	return 1 if sum(votes) >= len(votes) / 2 else 0


class RoomPairsOG(Dataset):
	def __init__(self, annotation_path: Path, dataset_path: Path, mode: str = 'train', max_side: int = 1024):
		self.mode = mode
		self.dataset_path = dataset_path
		self.max_side = max_side

		raw = json.loads(annotation_path.read_text(encoding='utf-8'))['data']['results']

		self.data = []
		self.bad = []
		for row in raw:
			task_id = row['taskId']
			folder = self.dataset_path / task_id
			p1 = folder / 'image1.jpg'
			p2 = folder / 'image2.jpg'
			try:
				im1 = Image.open(p1).convert('RGB'); im1.load()
				im2 = Image.open(p2).convert('RGB'); im2.load()
				self.data.append(row)
			except Exception as e:
				self.bad.append({'task_id': task_id, 'error': f'{type(e).__name__}: {e}'})
				continue

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		row = self.data[idx]
		task_id = row['taskId']
		folder = self.dataset_path / task_id


		img1_np = np.array(Image.open(folder / 'image1.jpg').convert('RGB'))
		img2_np = np.array(Image.open(folder / 'image2.jpg').convert('RGB'))

		item = {
			'task_id': task_id,
			'image1': img1_np,
			'image2': img2_np,
		}
		if self.mode != 'test':
			item['label'] = majority_label(row.get('answers'))
		return item


def collate_pairs_og(batch):
	out = {
		'task_id': [b['task_id'] for b in batch],
		'image1': [b['image1'] for b in batch],
		'image2': [b['image2'] for b in batch],
	}
	if 'label' in batch[0]:
		out['label'] = torch.tensor([b['label'] for b in batch], dtype=torch.long)
	return out


def make_loader(annotation_path, dataset_path, mode='train', batch_size=16, num_workers=8,
                   shuffle=False, max_side=1024, splits: Optional[Sequence[float]] = None):
	ds = RoomPairsOG(Path(annotation_path), Path(dataset_path), mode=mode, max_side=max_side)

	if splits:
		n = len(ds)
		sizes = [int(r * n) for r in splits]
		sizes[-1] = n - sum(sizes[:-1])
		subsets = random_split(ds, sizes, generator=torch.Generator().manual_seed(42))
		loaders = [
			DataLoader(s, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_pairs_og)
			for s in subsets
		]
		return ds, loaders

	loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_pairs_og)
	return ds, loader