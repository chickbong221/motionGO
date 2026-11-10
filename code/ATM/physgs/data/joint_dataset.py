"""
Windows of per-object Gaussian sets + labels for the joint world model (physgs.model.joint_wm), read from a packed
dataset (physgs.data.format). Positions (Gaussians and CoM labels) are expressed relative to the episode's scene
centre (metres; the model converts to its own units). Each item carries the episode's frame rate (fps) and length
scale L (physgs.model.joint_wm.length_scale of the episode's first frame, the frame that also fixes the scene centre).
With aug_shift > 0 a window is translated horizontally by up to aug_shift x L (Gaussians and labels together); jitter
is xyz noise in units of L.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from physgs.data.format import MIN_POINTS, Packed
from physgs.model.joint_wm import length_scale


def episode_length_scale(d, e, num_points):
    """Length scale L of episode e of a Packed dataset, from its first frame (as the scene centre)."""
    s = d.ep_start[e]
    g = torch.from_numpy(np.array(d.gauss[s, :, :num_points], np.float32))[None]
    vis = torch.from_numpy(np.asarray(d.npts[s]) >= MIN_POINTS)[None] & torch.from_numpy(d.present[e])[None]
    return float(length_scale(g, vis, torch.from_numpy(d.is_robot[e])[None])[0])


class GaussSeqDataset(Dataset):
    def __init__(self, root, seq_len=32, step=8, episodes=None, num_points=512, aug_shift=0.0, aug_prob=0.9, jitter=0.0):
        self.d = Packed(root)
        self.seq_len, self.num_points = seq_len, min(num_points, self.d.gauss.shape[2])
        self.aug_shift, self.aug_prob, self.jitter = aug_shift, aug_prob, jitter
        self.fps = np.array([float(e.get("fps", 20.0)) for e in self.d.meta["episodes"]], np.float32)
        eps = np.arange(self.d.num_episodes) if episodes is None else np.asarray(episodes)
        self.L = {int(e): episode_length_scale(self.d, e, self.num_points) for e in eps}
        self.items = []
        for e in eps:
            s, t = self.d.ep_start[e], self.d.ep_start[e + 1]
            if t - s < 2:
                continue
            last = max(s, t - seq_len)
            self.items += [(e, a, min(a + seq_len, t)) for a in (list(range(s, last + 1, step)) or [s])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e, a, b = self.items[i]
        pad = self.seq_len - (b - a)
        L = self.L[int(e)]
        c = np.asarray(self.d.center[e], np.float32)
        if self.aug_shift > 0 and float(torch.rand(())) < self.aug_prob:
            xy = (torch.rand(2).numpy() * 2 - 1) * self.aug_shift * L
            c = c - np.array([xy[0], xy[1], 0.0], np.float32)

        def take(arr, shift=None):
            x = np.array(arr[a:b], np.float32)
            if shift is not None:
                x = x - shift
            if pad:
                x = np.concatenate([x, np.full((pad,) + x.shape[1:], np.nan, np.float32)], 0)
            return torch.from_numpy(x)

        g = np.array(self.d.gauss[a:b, :, :self.num_points], np.float32)         # (T, O, P, 7), FPS-order prefix
        g[..., :3] -= c
        if self.jitter > 0:
            g[..., :3] += np.random.randn(*g[..., :3].shape).astype(np.float32) * self.jitter * L
        vis = np.asarray(self.d.npts[a:b]) >= MIN_POINTS
        if pad:
            g = np.concatenate([g, np.zeros((pad,) + g.shape[1:], np.float32)], 0)
            vis = np.concatenate([vis, np.zeros((pad,) + vis.shape[1:], bool)], 0)
        # float16 as stored in the pack (centred, jittered): half the host-to-GPU transfer; the model converts
        return dict(gauss=torch.from_numpy(g.astype(np.float16)), visible=torch.from_numpy(vis),
                    present=torch.from_numpy(self.d.present[e]), is_robot=torch.from_numpy(self.d.is_robot[e]),
                    task=torch.from_numpy(self.d.task_emb[e]), fps=torch.tensor(self.fps[e]), L=torch.tensor(L),
                    pos=take(self.d.pos, shift=c), vel=take(self.d.vel),
                    props=torch.from_numpy(self.d.props[e].astype(np.float32)))
