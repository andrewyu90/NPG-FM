"""Train NPG-FM (one model for all classes). Single GPU:  python train.py --data-dir ./data/mvtec
Multi GPU:  torchrun --nproc_per_node 2 train.py --data-dir ./data/mvtec
"""
import argparse
import math
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from npgfm import NPGFM, ADDataset, CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='mvtec', choices=['mvtec', 'visa'])
    p.add_argument('--data-dir', default='./data/mvtec')
    p.add_argument('--category', default='all', help="train on one class only (default: all classes)")
    p.add_argument('--size', default='L', choices=['S', 'M', 'L', 'XL'])
    p.add_argument('--epochs', type=int, default=800)
    p.add_argument('--batch-size', type=int, default=32, help='per GPU')
    p.add_argument('--accum', type=int, default=2, help='gradient accumulation steps')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup-epochs', type=int, default=10)
    p.add_argument('--steps', type=int, default=10, help='flow steps N')
    p.add_argument('--num-prototypes', type=int, default=5)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--out-dir', default='runs/npgfm')
    a = p.parse_args()

    ddp = 'RANK' in os.environ
    if ddp:
        dist.init_process_group('nccl')
    rank = dist.get_rank() if ddp else 0
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.manual_seed(1000 + rank)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    model = NPGFM(num_classes=len(CLASSES[a.dataset]), size=a.size, steps=a.steps,
                  num_prototypes=a.num_prototypes).cuda()
    data = ADDataset(a.dataset, a.data_dir, 'train', a.category, augment=True)
    if rank == 0:
        os.makedirs(a.out_dir, exist_ok=True)
        print(f'{len(data)} training images, U-Net params: {sum(p.numel() for p in model.unet.parameters()) / 1e6:.1f}M')
    # normal prototypes per class (k-means in latent space), computed once and stored in the checkpoint
    model.build_prototypes(DataLoader(ADDataset(a.dataset, a.data_dir, 'train', a.category), batch_size=32, num_workers=4))

    unet = DDP(model.unet, device_ids=[device]) if ddp else model.unet
    sampler = DistributedSampler(data) if ddp else None
    loader = DataLoader(data, batch_size=a.batch_size, shuffle=sampler is None, sampler=sampler,
                        num_workers=4, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.unet.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * 1.5  # warmup + cosine spanning 1.5x the run, so the lr ends at ~25% of its peak
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / a.warmup_epochs if e < a.warmup_epochs else
                                              0.5 * (1 + math.cos(math.pi * (e - a.warmup_epochs) / (total - a.warmup_epochs))))

    unet.train()
    for epoch in range(1, a.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)
        t0, running = time.time(), 0.
        for i, (x, _, y) in enumerate(loader):
            z = model.encode(x.cuda(), sample=True)
            loss = model.flow.loss(unet, z, y.cuda()) / a.accum
            loss.backward()
            if (i + 1) % a.accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            running += loss.item() * a.accum
        sched.step()
        if rank == 0:
            print(f'epoch {epoch:4d}  loss {running / (i + 1):.4f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s', flush=True)
            if epoch % a.save_every == 0 or epoch == a.epochs:
                model.save(os.path.join(a.out_dir, 'last.pt'))
    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
