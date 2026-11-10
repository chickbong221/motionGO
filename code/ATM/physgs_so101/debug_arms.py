"""debug_arms.py: why is right_arm never lifted? SAM 3.1 'robot arm' tracks on one episode (ids, persistence, image x),
and registered-depth coverage in each track's mask."""
import json
import sys

import numpy as np

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM/physgs_so101")
import lift_so101 as L  # noqa: E402

name, ep, ckpt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
root = f"/dev/shm/data/so101_v2/{name}"
info, eps, _ = L.load_meta(root)
row = eps.loc[ep]
n30 = int(row["length"])
top = L.decode_episode(root, row, "observation.images.top", n30)
q = L.decode_episode(root, row, "observation.images.top_depth", n30, depth=True)
calib = json.load(open(f"/dev/shm/physgo/so101/calib/{name}.json"))
Kt = np.array(calib["K_top"], np.float32)
dinfo = info["features"]["observation.images.top_depth"]["info"]
seg = L.Segmenter(ckpt)
tr = seg.track(top, L.ARM_PROMPT)
ids = {}
for t, d in enumerate(tr):
    for oid, (m, p) in d.items():
        ids.setdefault(oid, []).append((t, np.nonzero(m)[1].mean(), m.sum(), p))
for oid, v in sorted(ids.items(), key=lambda kv: -len(kv[1])):
    a = np.array(v)
    print(f"id {oid}: {len(v)} frames ({len(v) / len(tr):.0%}), mean x {a[:, 1].mean():.0f} px, area median {np.median(a[:, 2]):.0f} px, prob {a[:, 3].mean():.2f}")
left, right = L.split_arms(tr)
for nm, M in (("left", left), ("right", right)):
    cov = []
    for t in range(0, len(top), 20):
        if M[t].any():
            d = L.register_depth(L.depth_from_codes(q[t], dinfo), calib["depth_reg"], Kt, top.shape[1:3])
            cov.append(np.isfinite(d[M[t]]).mean())
    print(f"{nm}: mask in {M.any((1, 2)).mean():.0%} of frames; registered depth on mask pixels {np.mean(cov) if cov else float('nan'):.0%}")
