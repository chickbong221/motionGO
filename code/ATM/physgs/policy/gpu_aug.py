"""
ATM's BC training augmentation on the GPU, for a whole batch.

ATM (atm/dataloader/base_dataset.py) augments each item in the loader workers with probability aug_prob:
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3) with one random draw (factors and op order) per
item, shared by its views and frames, then ImgViewDiffTranslationAug (replicate-pad 4 px, one random 256 crop per
view). At 256 px that costs ~0.24 s per item on a CPU core (16 cores cannot feed the H100); here the same
distribution is applied on the GPU with per-item random parameters (~3 ms per item), and the images reach the model
as float [0, 255] without the uint8 round trip.
"""
import torch
from torchvision.transforms._functional_tensor import _hsv2rgb, _rgb2hsv

from atm.dataloader.utils import ImgViewDiffTranslationAug


def _gray(x):  # torchvision rgb_to_grayscale for float images
    return (0.2989 * x[..., 0, :, :] + 0.587 * x[..., 1, :, :] + 0.114 * x[..., 2, :, :]).unsqueeze(-3)


def _blend(a, b, r):  # torchvision _blend with a per-item ratio
    return (r * a + (1.0 - r) * b).clamp(0, 1.0)


def jitter(x, order, b, c, s, h):
    """
    torchvision ColorJitter.forward with per-item parameters: x (n, ..., 3, H, W) in [0, 1]; order (n, 4) a permutation
    of 0..3 (brightness, contrast, saturation, hue) per item; b, c, s, h (n,) factors.
    """
    shape = (len(x),) + (1,) * (x.dim() - 1)
    b, c, s = b.view(shape), c.view(shape), s.view(shape)
    h = h.view(shape[:-1])  # broadcast over the hue plane (n, ..., H, W)
    x = x.clone()
    for pos in range(4):
        for op in range(4):
            m = order[:, pos] == op
            if not m.any():
                continue
            y = x[m]
            if op == 0:
                y = _blend(y, torch.zeros_like(y), b[m])
            elif op == 1:
                y = _blend(y, _gray(y).mean(dim=(-3, -2, -1), keepdim=True), c[m])
            elif op == 2:
                y = _blend(y, _gray(y), s[m])
            else:
                hh, ss, vv = _rgb2hsv(y).unbind(dim=-3)
                y = _hsv2rgb(torch.stack(((hh + h[m]) % 1.0, ss, vv), dim=-3))
            x[m] = y
    return x


class GPUAugment:
    def __init__(self, img_size, prob, strength=0.3, translation=8, chunk=32):
        self.prob, self.k, self.chunk = prob, strength, chunk
        self.translate = ImgViewDiffTranslationAug(input_shape=tuple(img_size), translation=translation,
                                                   augment_track=False)

    def __call__(self, obs):
        """obs (B, V, T, 3, H, W) uint8 on the GPU -> float in [0, 255], each item augmented with probability prob."""
        x = obs.float() / 255.0
        sel = torch.nonzero(torch.rand(len(x)) < self.prob).flatten().to(x.device)
        for i in range(0, len(sel), self.chunk):  # chunks bound the temporary memory (~6 GB per 32 items at 256)
            x[sel[i:i + self.chunk]] = self.augment(x[sel[i:i + self.chunk]])
        return x * 255.0

    def augment(self, x):
        n, dev = len(x), x.device
        order = torch.stack([torch.randperm(4) for _ in range(n)]).to(dev)
        b, c, s = (torch.empty(n).uniform_(1 - self.k, 1 + self.k).to(dev) for _ in range(3))
        h = torch.empty(n).uniform_(-self.k, self.k).to(dev)
        x = jitter(x, order, b, c, s, h)
        v = x.shape[1]
        x = self.translate((x.flatten(0, 1), None))[0]  # one crop per (item, view), as ATM per view
        return x.unflatten(0, (n, v))
