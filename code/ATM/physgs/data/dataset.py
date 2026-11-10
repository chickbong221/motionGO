"""
Training datasets over packed episode directories (physgs.data.format.pack).

ObjectFrameDataset: (object, t) samples for the object encoder: Gaussians at t-1 and t + labels.
TokenSeqDataset:    fixed-length windows of precomputed object tokens + labels for the RSSM.
Positions in TokenSeqDataset are expressed relative to the episode's scene centre.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from physgs.data.format import MIN_POINTS, Packed


def history_indices(d, f, o, K):
    """
    Frames t-K+1 .. t of object o for the encoder (oldest first). A frame in which the object is not visible, or
    that lies before the episode start, is replaced by the nearest newer frame of the window. The online pipelines
    (engine.eval_policy*) apply the same rule.
    """
    start = d.ep_start[d.frame_episode[f]]
    idx, cur = [f], f
    for j in range(1, K):
        c = f - j
        if c >= start and d.npts[c, o] >= MIN_POINTS:
            cur = c
        idx.append(cur)
    return idx[::-1]


class ObjectFrameDataset(Dataset):
    def __init__(self, root, episodes=None, jitter=0.0, frames=2):
        self.d = Packed(root)
        self.jitter = jitter
        self.frames = frames
        eps = np.arange(self.d.num_episodes) if episodes is None else np.asarray(episodes)
        frames = np.concatenate([np.arange(self.d.ep_start[e] + 1, self.d.ep_start[e + 1]) for e in eps])
        vis = np.asarray(self.d.npts[frames] >= MIN_POINTS)                    # (F, O)
        f_idx, o_idx = np.nonzero(vis)
        self.items = np.stack([frames[f_idx], o_idx], 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, o = self.items[i]
        e = self.d.frame_episode[f]
        hist = history_indices(self.d, f, o, self.frames)
        if self.d.gauss_obj is not None:  # one contiguous read of the object's track (same values)
            base = self.d.track_start[e, o] - self.d.ep_start[e]
            lo, hi = base + min(hist), base + max(hist) + 1
            span = np.asarray(self.d.gauss_obj[lo:hi], np.float32)
            g_hist = torch.from_numpy(span[np.asarray(hist) + base - lo])                            # (K, P, 7)
        else:
            g_hist = torch.from_numpy(np.stack([np.asarray(self.d.gauss[h, o], np.float32) for h in hist]))  # (K, P, 7)
        if self.jitter > 0:  # small xyz noise as augmentation
            g_hist[..., :3] += torch.randn_like(g_hist[..., :3]) * self.jitter
        return dict(
            g_hist=g_hist,
            pos=torch.from_numpy(np.asarray(self.d.pos[f, o], np.float32)),
            vel=torch.from_numpy(np.asarray(self.d.vel[f, o], np.float32)),
            angvel=torch.from_numpy(np.asarray(self.d.angvel[f, o], np.float32)),
            contact=torch.tensor(float(self.d.contact[f, o])),
            props=torch.from_numpy(self.d.props[e, o].astype(np.float32)),
        )


class TokenSeqDataset(Dataset):
    """
    Windows of length `seq_len` (stride `step`) over episodes; needs tokens.npy written by precompute_tokens.
    With aug_shift > 0, a window is translated horizontally by up to aug_shift metres (with probability aug_prob) by
    moving the scene centre: token and label positions shift together (ATM's random-shift augmentation, in 3D).
    """

    def __init__(self, root, seq_len=32, step=8, episodes=None, aug_shift=0.0, aug_prob=0.9):
        self.d = Packed(root)
        self.tokens = np.load(f"{root}/tokens.npy", mmap_mode="r")             # (F, O, D)
        self.seq_len = seq_len
        self.aug_shift, self.aug_prob = aug_shift, aug_prob
        eps = np.arange(self.d.num_episodes) if episodes is None else np.asarray(episodes)
        self.items = []
        for e in eps:
            s, t = self.d.ep_start[e], self.d.ep_start[e + 1]
            if t - s < 2:
                continue
            last = max(s, t - seq_len)
            starts = list(range(s, last + 1, step)) or [s]
            self.items += [(e, a, min(a + seq_len, t)) for a in starts]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e, a, b = self.items[i]
        L = self.seq_len
        pad = L - (b - a)
        c = np.asarray(self.d.center[e], np.float32)
        if self.aug_shift > 0 and float(torch.rand(())) < self.aug_prob:
            xy = (torch.rand(2).numpy() * 2 - 1) * self.aug_shift
            c = c - np.array([xy[0], xy[1], 0.0], np.float32)

        def take(arr, fill=np.nan, shift=None):
            x = np.asarray(arr[a:b], np.float32)
            if shift is not None:
                x = x - shift
            if pad:
                x = np.concatenate([x, np.full((pad,) + x.shape[1:], fill, np.float32)], 0)
            return torch.from_numpy(x)

        tok = np.asarray(self.tokens[a:b], np.float32).copy()
        tok[..., 0:3] -= c
        vis = np.asarray(self.d.npts[a:b] >= MIN_POINTS)
        tok[~vis] = 0.0
        if pad:
            tok = np.concatenate([tok, np.zeros((pad,) + tok.shape[1:], np.float32)], 0)
            vis = np.concatenate([vis, np.zeros((pad,) + vis.shape[1:], bool)], 0)
        return dict(
            tokens=torch.from_numpy(tok), visible=torch.from_numpy(vis),
            present=torch.from_numpy(self.d.present[e]), is_robot=torch.from_numpy(self.d.is_robot[e]),
            task=torch.from_numpy(self.d.task_emb[e]),
            pos=take(self.d.pos, shift=c), vel=take(self.d.vel), angvel=take(self.d.angvel),
            contact=take(self.d.contact), props=torch.from_numpy(self.d.props[e].astype(np.float32)),
        )
