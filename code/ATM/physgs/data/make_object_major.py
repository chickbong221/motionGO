"""
Object-major copy of a packed dataset's Gaussians, for fast encoder training from network storage.

The pack stores gauss as (frames, objects, P, 7), so an encoder sample (one object over its last K frames) is K
small reads strided by a whole frame row; on NFS that makes training I/O-bound. This writes
    gauss_obj.npy    (rows, P, 7)  float16: every present (episode, object slot) as one contiguous track of frames
    track_start.npy  (episodes, O) int64:   first row of that track (-1 = slot not present)
so row(e, o, f) = track_start[e, o] + f - ep_start[e], and a sample is a single contiguous read. One sequential pass
(read one episode block, write its tracks). Packed and ObjectFrameDataset use the copy when it exists.

    python -m physgs.data.make_object_major --data data/packed/physai1080_v2
"""
import argparse
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    load = lambda n, mmap="r": np.load(os.path.join(a.data, f"{n}.npy"), mmap_mode=mmap)
    gauss, present, ep_start = load("gauss"), load("present", None), load("ep_start", None)
    n_eps, O = present.shape
    lengths = np.diff(ep_start)
    track_start = np.full((n_eps, O), -1, np.int64)
    rows = 0
    for e in range(n_eps):
        for o in np.nonzero(present[e])[0]:
            track_start[e, o] = rows
            rows += lengths[e]
    tmp = os.path.join(a.data, "gauss_obj.npy.tmp")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=gauss.dtype, shape=(rows,) + gauss.shape[2:])
    t0 = time.time()
    for e in range(n_eps):
        block = np.asarray(gauss[ep_start[e]:ep_start[e + 1]])              # (T, O, P, 7), one contiguous read
        for o in np.nonzero(present[e])[0]:
            out[track_start[e, o]:track_start[e, o] + lengths[e]] = block[:, o]
        if e % 500 == 0:
            print(f"{e}/{n_eps} episodes, {time.time() - t0:.0f} s", flush=True)
    out.flush()
    del out
    os.replace(tmp, os.path.join(a.data, "gauss_obj.npy"))
    np.save(os.path.join(a.data, "track_start.npy"), track_start)
    print(f"gauss_obj: {rows:,} rows ({rows * np.prod(gauss.shape[2:]) * gauss.dtype.itemsize / 1e9:.1f} GB) "
          f"for {int((track_start >= 0).sum()):,} tracks in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
