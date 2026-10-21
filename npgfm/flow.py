"""Deviation flow: the time schedule, the training target and the deterministic ODE sampler of NPG-FM.

With a_t = alpha_bar(t) (cosine schedule over T steps):
  forward   x_t   = sqrt(a_t) * x_0 + sqrt(1 - a_t) * eps
  target    v*_t  = eps - (1 - sqrt(a_t)) / sqrt(1 - a_t) * x_0         (direction of deviation)
  sampler   x0_hat = x_t - sqrt(1 - a_t) * v_theta(x_t, t)
            x_{t-1} = x0_hat + sqrt(1 - a_{t-1}) * v_theta(x_t, t)       (x0_hat at the last step)
"""
import math
import torch


class DeviationFlow:
    def __init__(self, steps=10):
        self.steps = steps
        alpha_bar = lambda s: math.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = [min(1 - alpha_bar((i + 1) / steps) / alpha_bar(i / steps), 0.999) for i in range(steps)]
        a = torch.cumprod(1 - torch.tensor(betas, dtype=torch.float64), 0)
        self.sqrt_a = a.sqrt().float()
        self.sqrt_1ma = (1 - a).sqrt().float()
        self.coeff = ((1 - a.sqrt()) / (1 - a).sqrt()).float()

    def _at(self, arr, t, x):
        return arr.to(x.device)[t].view(-1, *([1] * (x.dim() - 1)))

    def loss(self, model, x0, cls):
        """Training objective: MSE between v_theta(x_t, t) and the deviation direction."""
        t = torch.randint(0, self.steps, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)
        x_t = self._at(self.sqrt_a, t, x0) * x0 + self._at(self.sqrt_1ma, t, x0) * eps
        target = eps - self._at(self.coeff, t, x0) * x0
        return ((model(x_t, t, cls) - target) ** 2).mean()

    @torch.no_grad()
    def sample(self, model, x, cls, return_path=False):
        """Integrate from t = steps-1 down to 0 starting at x (the routed latent of the test image)."""
        path = [x]
        for i in reversed(range(self.steps)):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            v = model(x, t, cls)
            x = x - self._at(self.sqrt_1ma, t, x) * v
            if i > 0:
                x = x + self._at(self.sqrt_1ma, t - 1, x) * v
            path.append(x)
        return (x, path) if return_path else x
