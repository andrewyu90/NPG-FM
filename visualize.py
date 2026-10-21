"""Visualise NPG-FM defect detection results.

Dataset mode (with ground truth): one figure per class with rows of
  input | reconstruction | ground truth | anomaly map | overlay + predicted defect contour
  python visualize.py --dataset mvtec --data-dir ./data/mvtec --category bottle --num 6

Image mode (any image of a known class, no ground truth):
  python visualize.py --images path/a.png path/b.png --category bottle
"""
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from npgfm import NPGFM, ADDataset, CLASSES
from npgfm.data import to_uint8
from npgfm.metrics import precision_recall_curve


def best_threshold(masks, maps):
    """Pixel threshold maximising F1 over the whole test set of the class."""
    p, r, th = precision_recall_curve(masks.ravel().astype(int), maps.ravel())
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return th[np.argmax(f1[:-1])]


def plot_rows(rows, vmax, thr, title, path):
    """rows: list of dict(image, recon, amap, mask or None, score)."""
    has_gt = rows[0]['mask'] is not None
    cols = ['Input', 'Reconstruction'] + (['Ground truth'] if has_gt else []) + ['Anomaly map', 'Detection']
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.6 * len(cols), 2.6 * len(rows)), squeeze=False)
    for i, r in enumerate(rows):
        panels = [r['image'], r['recon']] + ([r['mask']] if has_gt else [])
        for j, img in enumerate(panels):
            axes[i, j].imshow(img, cmap='gray' if img.ndim == 2 else None, vmin=0, vmax=img.max() or 1)
        axes[i, len(panels)].imshow(r['amap'], cmap='jet', vmin=0, vmax=vmax)
        ax = axes[i, len(panels) + 1]
        ax.imshow(r['image'])
        ax.imshow(np.ma.masked_less(r['amap'], thr), cmap='jet', alpha=0.5, vmin=0, vmax=vmax)
        ax.contour(r['amap'] > thr, levels=[0.5], colors='red', linewidths=1.2)
        if has_gt:
            ax.contour(r['mask'], levels=[0.5], colors='lime', linewidths=1.0, linestyles='--')
        axes[i, 0].set_ylabel(f"score {r['score']:.3f}", fontsize=9)
        for j, c in enumerate(cols):
            axes[i, j].set_xticks([]), axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(c, fontsize=10)
    fig.suptitle(title + ('   (red: NPG-FM detection, green dashed: ground truth)' if has_gt else '   (red: NPG-FM detection)'),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', path)


@torch.no_grad()
def visualize_dataset(model, a):
    cats = sorted(CLASSES[a.dataset]) if a.category == 'all' else [a.category]
    for c in cats:
        ds = ADDataset(a.dataset, a.data_dir, 'test', c)
        xs, masks, maps, recons, scores = [], [], [], [], []
        for x, m, y in DataLoader(ds, batch_size=8, num_workers=4):
            out = model(x.cuda(), y.cuda(), a.steps)
            xs.append(x), masks.append(m.numpy()), maps.append(out['anomaly_map'].cpu().numpy())
            recons.append(out['reconstruction'].cpu()), scores.append(out['score'].cpu().numpy())
        xs, recons = torch.cat(xs), torch.cat(recons)
        masks, maps, scores = np.concatenate(masks), np.concatenate(maps), np.concatenate(scores)
        thr, vmax = best_threshold(masks, maps), np.percentile(maps, 99.9)
        # evenly spread defective samples plus one normal sample
        bad, good = np.flatnonzero(masks.reshape(len(masks), -1).max(1) > 0), np.flatnonzero(masks.reshape(len(masks), -1).max(1) == 0)
        pick = list(bad[np.linspace(0, len(bad) - 1, min(a.num - 1, len(bad))).astype(int)]) + list(good[:1])
        rows = [dict(image=to_uint8(xs[i]), recon=to_uint8(recons[i]), amap=maps[i], mask=masks[i], score=scores[i]) for i in pick]
        plot_rows(rows, vmax, thr, f'NPG-FM  {a.dataset}/{c}', os.path.join(a.out_dir, f'{a.dataset}_{c}.png'))


@torch.no_grad()
def calibrate(model, a, n=16):
    """Threshold = highest anomaly-map value on n normal training images of the class."""
    ds = ADDataset(a.dataset, a.data_dir, 'train', a.category)
    x, _, y = next(iter(DataLoader(ds, batch_size=n, shuffle=True)))
    return model(x.cuda(), y.cuda(), a.steps)['anomaly_map'].max().item()


@torch.no_grad()
def visualize_images(model, a):
    cls = CLASSES[a.dataset].index(a.category)
    thr = a.threshold if a.threshold is not None else calibrate(model, a)
    print(f'detection threshold {thr:.3f}')
    rows = []
    for path in a.images:
        img = Image.open(path).convert('RGB').resize((288, 288))
        x = torch.from_numpy(np.array(img)[16:272, 16:272].copy()).permute(2, 0, 1).float()[None] / 127.5 - 1
        out = model(x.cuda(), torch.tensor([cls]).cuda(), a.steps)
        rows.append(dict(image=to_uint8(x[0]), recon=to_uint8(out['reconstruction'][0]),
                         amap=out['anomaly_map'][0].cpu().numpy(), mask=None, score=out['score'].item()))
    plot_rows(rows, 2 * thr, thr, f'NPG-FM  {a.category}', os.path.join(a.out_dir, f'images_{a.category}.png'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='mvtec', choices=['mvtec', 'visa'])
    p.add_argument('--data-dir', default='./data/mvtec')
    p.add_argument('--ckpt', default='checkpoints/npgfm_mvtec.pt')
    p.add_argument('--category', default='all', help="class name ('all' for every class in dataset mode)")
    p.add_argument('--images', nargs='*', help='image files to analyse instead of the test set')
    p.add_argument('--num', type=int, default=6, help='rows per class in dataset mode')
    p.add_argument('--steps', type=int, default=None)
    p.add_argument('--threshold', type=float, default=None,
                   help='image mode: detection threshold (default: calibrated on normal training images in --data-dir)')
    p.add_argument('--out-dir', default='vis')
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(0)
    model = NPGFM.from_pretrained(a.ckpt).cuda()
    if a.images:
        assert a.category != 'all', '--category is required with --images'
        visualize_images(model, a)
    else:
        visualize_dataset(model, a)


if __name__ == '__main__':
    main()
