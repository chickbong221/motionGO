"""
Encode every (frame, object) of a packed dataset with a frozen object encoder -> <data>/tokens.npy (F, O, D).

    python -m physgs.engine.precompute_tokens --data data/packed/libero_spatial --ckpt runs/encoder_libero_spatial/last.pt
"""
import argparse
import os

import numpy as np
import torch

from physgs.data.format import MIN_POINTS, Packed
from physgs.model.object_encoder import ObjectAutoencoder


class HistoryBatches(torch.utils.data.Dataset):
    """Batch i: the (frame, object) items s..s+batch and their K-frame Gaussian histories, gathered in loader workers
    so that reading the packed Gaussians overlaps the encoder (each takes about half of the time)."""

    def __init__(self, root, batch, K):
        self.root, self.batch, self.K = root, batch, K
        self.d = None
        self.npts = np.asarray(Packed(root).npts)
        self.f_idx, self.o_idx = np.nonzero(self.npts >= MIN_POINTS)

    def __len__(self):
        return (len(self.f_idx) + self.batch - 1) // self.batch

    def __getitem__(self, i):
        if self.d is None:  # opened in each worker (memory maps)
            self.d = Packed(self.root)
        d = self.d
        npts = self.npts
        s = i * self.batch
        f, o = self.f_idx[s:s + self.batch], self.o_idx[s:s + self.batch]
        # same rule as physgs.data.dataset.history_indices, vectorised
        start = d.ep_start[d.frame_episode[f]]
        hist, cur = [f], f.copy()
        for j in range(1, self.K):
            c = f - j
            ok = c >= start
            ok &= npts[np.where(ok, c, 0), o] >= MIN_POINTS
            cur = np.where(ok, c, cur)
            hist.append(cur)
        hist = np.stack(hist[::-1], 1)                                                   # (n, K)
        g_hist = np.stack([np.asarray(d.gauss[hist[:, k], o], np.float32) for k in range(self.K)], 1)
        return torch.from_numpy(f), torch.from_numpy(o), torch.from_numpy(g_hist)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4, help="loader processes gathering the Gaussian histories")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    model = ObjectAutoencoder(**ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["model"])
    d = Packed(args.data)
    F, O = d.npts.shape
    D = 16 + ck["cfg"]["latent_dim"]
    # float32, computed without autocast: the online evaluation encodes in float32 too (until 2026-10-23: bf16 / float16)
    out = np.lib.format.open_memmap(os.path.join(args.data, "tokens.npy"), mode="w+", dtype=np.float32, shape=(F, O, D))
    out[:] = 0
    batches = HistoryBatches(args.data, args.batch, model.num_frames)
    loader = torch.utils.data.DataLoader(batches, batch_size=None, shuffle=False, num_workers=args.workers,
                                         pin_memory=device == "cuda", prefetch_factor=2 if args.workers else None)
    for f, o, g_hist in loader:
        tok = model.tokens(g_hist.to(device, non_blocking=True))
        out[f.numpy(), o.numpy()] = tok.float().cpu().numpy()
    out.flush()
    print("tokens", out.shape, "->", os.path.join(args.data, "tokens.npy"))


if __name__ == "__main__":
    main()
