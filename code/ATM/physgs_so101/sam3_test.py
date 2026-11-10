"""sam3_test.py: SAM 3.1 loads offline and tracks text prompts on one SO-101 top video (15 fps); writes an overlay strip."""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM/physgs_so101")
from lift_so101 import Segmenter, decode_episode, load_meta, split_arms  # noqa: E402

root = "/dev/shm/data/so101_v2/so101-left_arm_task_red_cube_out_of_pot_into_the_orange_hollow_cube"
info, eps, tasks = load_meta(root)
row = eps.loc[0]
top = decode_episode(root, row, "observation.images.top", int(row["length"]))
wr = decode_episode(root, row, "observation.images.wrist", int(row["length"]))
print("frames", top.shape, wr.shape)
t0 = time.time()
seg = Segmenter(os.environ.get("SAM3_CKPT"))
print(f"SAM 3.1 built in {time.time() - t0:.1f} s")
res = {}
for name, frames, prompt in (("top", top, "red cube"), ("top", top, "steel pot"), ("top", top, "orange hollow cube"),
                             ("top", top, "robot arm"), ("wrist", wr, "red cube")):
    t0 = time.time()
    tr = seg.track(frames, prompt)
    n_ids = len({k for d in tr for k in d})
    vis = np.mean([len(d) > 0 for d in tr])
    print(f"{name:5s} '{prompt}': {time.time() - t0:.1f} s, ids {n_ids}, frames with a mask {vis:.0%}, "
          f"mean instances/frame {np.mean([len(d) for d in tr]):.2f}")
    res[(name, prompt)] = tr
# overlay strip: 6 instants of the top video, objects coloured
cols = {"red cube": (255, 0, 0), "steel pot": (0, 255, 255), "orange hollow cube": (255, 160, 0)}
left, right = split_arms(res[("top", "robot arm")])
T = len(top)
tiles = []
for t in np.linspace(0, T - 1, 6).round().astype(int):
    im = top[t].copy().astype(np.float32)
    for p, c in cols.items():
        for m, _ in res[("top", p)][t].values():
            im[m] = 0.5 * im[m] + 0.5 * np.array(c)
    im[left[t]] = 0.5 * im[left[t]] + 0.5 * np.array((0, 255, 0))
    im[right[t]] = 0.5 * im[right[t]] + 0.5 * np.array((255, 0, 255))
    w = wr[t].copy().astype(np.float32)
    for m, _ in res[("wrist", "red cube")][t].values():
        w[m] = 0.5 * w[m] + 0.5 * np.array((255, 0, 0))
    tiles.append(np.concatenate([cv2.resize(im.astype(np.uint8), (320, 180)), cv2.resize(w.astype(np.uint8), (240, 180))], 1))
cv2.imwrite("/dev/shm/sam3_test.jpg", np.concatenate(tiles, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
print("saved /dev/shm/sam3_test.jpg")
