"""NPG-FM: Normal-Prototype-Guided Flow Matching for multi-class anomaly detection.

Pipeline for an image x of class c:
  z      = E(x)                         frozen SD-VAE encoder (latent mean * 0.18215)
  z_in   = z + lambda * p_c             prototype routing (p_c: a normal prototype of class c)
  z_hat  = ODE_v(z_in, c)               deviation flow integrated over N steps by the U-Net v_theta
  x_hat  = D(z_hat), x_rec = D(z)       decoder
  A      = sqrt(A_img * A_lat)          anomaly map (image and latent discrepancies)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

from .unet import UNet, UNET_CONFIGS
from .flow import DeviationFlow

LATENT_SCALE = 0.18215


def _gauss_blur(m, sigma):
    """Separable Gaussian blur with reflect padding (same as scipy.ndimage.gaussian_filter, truncate=4)."""
    r = int(4 * sigma + 0.5)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=m.device, dtype=m.dtype) / sigma) ** 2)
    k = k / k.sum()
    m = F.conv2d(F.pad(m, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(m, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1))


def anomaly_map(x_rec, x_hat, z, z_hat):
    """Pixel anomaly map in [0, 1] of shape (B, H, W)."""
    a_img = (x_hat - x_rec).abs().amax(1, keepdim=True).clamp(0, 0.4) * 2.5
    a_img = _gauss_blur(a_img.float(), 3.0)
    a_lat = (z_hat - z).abs().mean(1, keepdim=True).clamp(0, 0.2) * 5
    a_lat = F.interpolate(_gauss_blur(a_lat.float(), 1.0), size=a_img.shape[-2:], mode='bilinear', align_corners=False)
    return torch.sqrt(a_img * a_lat)[:, 0]


class NPGFM(nn.Module):
    def __init__(self, num_classes=15, size='L', steps=10, num_prototypes=5, proto_weight=0.1,
                 latent_size=32, vae='stabilityai/sd-vae-ft-ema'):
        super().__init__()
        self.config = dict(num_classes=num_classes, size=size, steps=steps, num_prototypes=num_prototypes,
                           proto_weight=proto_weight, latent_size=latent_size, vae=vae)
        self.vae = AutoencoderKL.from_pretrained(vae).eval().requires_grad_(False)
        self.unet = UNet(num_classes=num_classes, **UNET_CONFIGS[size])
        self.flow = DeviationFlow(steps)
        self.register_buffer('prototypes', torch.zeros(num_classes, num_prototypes, 4, latent_size, latent_size))

    # ---------------------------------------------------------------- building blocks
    @torch.no_grad()
    def encode(self, x, sample=False):
        d = self.vae.encode(x).latent_dist
        return (d.sample() if sample else d.mean) * LATENT_SCALE

    @torch.no_grad()
    def decode(self, z):
        return self.vae.decode(z / LATENT_SCALE).sample

    def route(self, z, cls):
        """Add a randomly chosen normal prototype of the image's class to its latent."""
        idx = torch.randint(0, self.prototypes.shape[1], (z.shape[0],), device=z.device)
        return z + self.config['proto_weight'] * self.prototypes[cls, idx]

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def forward(self, x, cls, steps=None):
        """x: (B,3,256,256) in [-1,1]; cls: (B,) class index. Returns anomaly map, image score and reconstructions."""
        z = self.encode(x)
        flow = self.flow if steps in (None, self.flow.steps) else DeviationFlow(steps)  # schedule is defined per N
        z_hat = flow.sample(self.unet, self.route(z, cls), cls)
        x_hat, x_rec = self.decode(z_hat), self.decode(z)
        amap = anomaly_map(x_rec, x_hat, z, z_hat)
        return dict(anomaly_map=amap, score=amap.flatten(1).amax(1), reconstruction=x_hat, autoencoded=x_rec)

    # ---------------------------------------------------------------- prototypes
    @torch.no_grad()
    def build_prototypes(self, loader, device='cuda'):
        """Per class: k-means on the latents of normal training images; keep the real latent nearest to each centroid."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        feats, labels = [], []
        for x, _, y in loader:
            feats.append(self.encode(x.to(device)).flatten(1).cpu())
            labels.append(y)
        feats, labels = torch.cat(feats).numpy(), torch.cat(labels).numpy()
        K, shape = self.prototypes.shape[1], self.prototypes.shape[2:]
        for c in sorted(set(labels.tolist())):
            f = feats[labels == c]
            fs = StandardScaler().fit_transform(f)
            km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(fs)
            nearest = ((fs[:, None] - km.cluster_centers_[None]) ** 2).sum(-1).argmin(0)
            self.prototypes[c] = torch.from_numpy(f[nearest]).view(K, *shape).to(self.prototypes.device)

    # ---------------------------------------------------------------- checkpoint I/O
    def save(self, path):
        torch.save(dict(config=self.config, unet=self.unet.state_dict(), prototypes=self.prototypes.cpu()), path)

    @classmethod
    def from_pretrained(cls, path, map_location='cpu', **overrides):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**{**ckpt['config'], **overrides})
        model.unet.load_state_dict(ckpt['unet'])
        model.prototypes.copy_(ckpt['prototypes'])
        return model.eval()
