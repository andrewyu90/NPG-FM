"""MVTec-AD and VisA datasets. Images are resized to `image_size`, center-cropped to `center_size`
and normalised to [-1, 1]. Each item is (image, mask, class_index)."""
import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

SPLITS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'splits')

CLASSES = {
    'mvtec': ['capsule', 'bottle', 'grid', 'leather', 'metal_nut', 'tile', 'transistor', 'zipper',
              'cable', 'carpet', 'hazelnut', 'pill', 'screw', 'toothbrush', 'wood'],
    'visa': ['candle', 'cashew', 'fryum', 'macaroni2', 'pcb2', 'pcb4', 'capsules', 'chewinggum',
             'macaroni1', 'pcb1', 'pcb3', 'pipe_fryum'],
}


class ADDataset(Dataset):
    def __init__(self, dataset, root, split='test', category='all', image_size=288, center_size=256, augment=False):
        df = pd.read_csv(os.path.join(SPLITS, f'{dataset}-split.csv'))
        df = df[df['split'] == split]
        if category != 'all':
            df = df[df['object'] == category]
        if split == 'train':  # normal images only
            df = df[df['label'] == 'normal']
        if len(df) == 0:
            raise FileNotFoundError(f'no {split} images for {dataset}/{category}')
        self.dataset, self.augment = dataset, augment
        self.image_size, self.center_size = image_size, center_size
        self.items = []
        for _, r in df.iterrows():
            mask = None if r['label'] == 'normal' else os.path.join(root, r['mask'])
            self.items.append((os.path.join(root, r['image']), mask, CLASSES[dataset].index(r['object'])))
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def _load(self, i):
        if i not in self.cache:
            path, mask_path, _ = self.items[i]
            s = self.image_size
            img = np.array(Image.open(path).convert('RGB').resize((s, s)), dtype=np.uint8)
            mask = np.zeros((s, s), np.uint8) if mask_path is None else \
                (np.array(Image.open(mask_path).convert('L').resize((s, s))) > 0).astype(np.uint8)
            self.cache[i] = (img, mask)
        return self.cache[i]

    def __getitem__(self, i):
        img, mask = self._load(i)
        o_y = o_x = (self.image_size - self.center_size) // 2
        if self.augment and random.random() < 0.5:  # small random translation before the center crop
            r = (self.image_size - self.center_size) // 8
            o_y, o_x = o_y + random.randint(-r, r), o_x + random.randint(-r, r)
        c = self.center_size
        img, mask = img[o_y:o_y + c, o_x:o_x + c], mask[o_y:o_y + c, o_x:o_x + c]
        if self.augment and self.dataset == 'visa' and random.random() < 0.5:
            img, mask = img[::-1, ::-1], mask[::-1, ::-1]
        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 127.5 - 1
        return x, torch.from_numpy(mask.astype(np.float32)), self.items[i][2]


def to_uint8(x):
    """(3,H,W) tensor in [-1,1] -> (H,W,3) uint8 array."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
