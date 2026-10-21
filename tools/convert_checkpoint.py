"""Convert a legacy training checkpoint ({'model': unet_state_dict, 'args': ...}) into an NPG-FM checkpoint
(U-Net weights + class prototypes + config). Prototypes are built from the normal training images.

python tools/convert_checkpoint.py --src model.pt --data-dir ./data/mvtec --out checkpoints/npgfm_mvtec.pt
"""
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npgfm import NPGFM, ADDataset, CLASSES

p = argparse.ArgumentParser()
p.add_argument('--src', required=True)
p.add_argument('--dataset', default='mvtec', choices=['mvtec', 'visa'])
p.add_argument('--data-dir', default='./data/mvtec')
p.add_argument('--prototypes', default=None, help='optional precomputed {class_id: (K,4,32,32)} file')
p.add_argument('--steps', type=int, default=10)
p.add_argument('--out', required=True)
a = p.parse_args()

model = NPGFM(num_classes=len(CLASSES[a.dataset]), steps=a.steps).cuda()
model.unet.load_state_dict(torch.load(a.src, map_location='cpu', weights_only=False)['model'])
if a.prototypes:
    for c, v in torch.load(a.prototypes).items():
        model.prototypes[c] = v.cuda()
else:
    model.build_prototypes(DataLoader(ADDataset(a.dataset, a.data_dir, 'train'), batch_size=32, num_workers=4))
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
model.save(a.out)
print('saved', a.out)
