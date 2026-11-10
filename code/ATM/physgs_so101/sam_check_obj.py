"""
sam_check_obj.py (2026-11-07): per-frame SAM 3.1 mask area of one object over a whole top-camera episode, for several
phrases: which phrase finds the object, in which frames, and how best_instance picks. Saves frames with the masks.
  python sam_check_obj.py <dataset dir> <episode> <sam3 ckpt> <out prefix> "<phrase 1>" "<phrase 2>" ...
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM/physgs_so101")
import lift_so101 as L  # noqa: E402

root, ep, ckpt, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
phrases = sys.argv[5:]
info, eps, _ = L.load_meta(root)
top = L.decode_episode(root, eps.loc[ep], "observation.images.top", int(eps.loc[ep, "length"]))
T = len(top)
seg = L.Segmenter(ckpt)
for ph in phrases:
    if ph.startswith("<"):         # "<phrase": prompt at the last frame, propagate backward
        ph = ph[1:]
        tr = seg.track(top, ph, frame=T - 1, direction="backward")
        ph = "BACKWARD " + ph
    elif ph.startswith("="):       # "=phrase": lift_so101.track_both_ends (what lift_so101_v2 uses)
        ph = ph[1:]
        tr, npass = L.track_both_ends(seg, top, ph)
        ph = f"BOTH ENDS ({npass} passes) " + ph
    else:
        tr = seg.track(top, ph)
    ids = {}
    for t, d in enumerate(tr):
        for oid, (mm, pr) in d.items():
            ids.setdefault(oid, []).append((t, int(mm.sum()), pr))
    print(f"phrase '{ph}': {len(ids)} track ids, frames with any instance {np.mean([bool(d) for d in tr]):.0%}")
    for oid, v in ids.items():
        ts = [x[0] for x in v]
        print(f"   id {oid}: {len(v)} frames ({min(ts)}..{max(ts)}), area median {np.median([x[1] for x in v]):.0f} px, prob median {np.median([x[2] for x in v]):.2f}")
    m = L.best_instance(tr)
    if m is None:
        print("   best_instance: None")
        continue
    a = m.reshape(T, -1).sum(1)
    print(f"   best_instance: frames with mask {np.mean(a > 0):.0%}, area median {np.median(a[a > 0]) if (a > 0).any() else 0:.0f} px; "
          f"first/last visible {np.argmax(a > 0)}/{T - 1 - np.argmax(a[::-1] > 0)}")
    rows = []
    for t in np.linspace(0, T - 1, 6).round().astype(int):
        im = top[t].copy()
        im[m[t]] = (0.4 * im[m[t]] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
        rows.append(cv2.resize(im, (480, 270)))
    cv2.imwrite(f"{out}_{ph.replace(' ', '_')}.jpg", np.concatenate(rows, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
