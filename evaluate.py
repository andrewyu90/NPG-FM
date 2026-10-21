"""Evaluate NPG-FM on MVTec-AD / VisA and report per-class and mean metrics.

python evaluate.py --dataset mvtec --data-dir ./data/mvtec --ckpt checkpoints/npgfm_mvtec.pt
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from npgfm import NPGFM, ADDataset, CLASSES, compute_metrics


@torch.no_grad()
def run_category(model, dataset, data_dir, category, steps, batch_size=8, device='cuda'):
    ds = ADDataset(dataset, data_dir, 'test', category)
    maps, masks = [], []
    for x, m, y in DataLoader(ds, batch_size=batch_size, num_workers=4):
        maps.append(model(x.to(device), y.to(device), steps)['anomaly_map'].cpu().numpy())
        masks.append(m.numpy())
    return np.concatenate(masks), np.concatenate(maps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='mvtec', choices=['mvtec', 'visa'])
    p.add_argument('--data-dir', default='./data/mvtec')
    p.add_argument('--ckpt', default='checkpoints/npgfm_mvtec.pt')
    p.add_argument('--category', default='all')
    p.add_argument('--steps', type=int, default=None, help='ODE steps (default: value stored in the checkpoint)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None, help='optional JSON file for the results')
    a = p.parse_args()

    torch.manual_seed(a.seed)
    model = NPGFM.from_pretrained(a.ckpt).cuda()
    cats = sorted(CLASSES[a.dataset]) if a.category == 'all' else [a.category]
    res = {}
    for c in cats:
        masks, maps = run_category(model, a.dataset, a.data_dir, c, a.steps)
        res[c] = {k: round(100 * v, 2) for k, v in compute_metrics(masks, maps).items()}
        print(f'{c:12s}', '  '.join(f'{k} {v:5.1f}' for k, v in res[c].items()), flush=True)
    res['mean'] = {k: round(float(np.mean([res[c][k] for c in cats])), 2) for k in res[cats[0]]}
    print(f'{"mean":12s}', '  '.join(f'{k} {v:5.1f}' for k, v in res['mean'].items()))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
