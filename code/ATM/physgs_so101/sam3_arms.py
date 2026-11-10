"""
sam3_prompts.py: which text phrase does SAM 3.1 detect best for each SO-101 object? For each dataset, 16 top frames and
16 wrist frames spread over 3 episodes; per phrase: share of frames with a detection and the mean best probability
(output threshold 0.3 to see weak detections). Writes one overlay per dataset with the best phrase per object.
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM/physgs_so101")
import lift_so101 as L  # noqa: E402

CANDS = {
    d: {"left arm": ["teal robot arm", "blue robot arm", "cyan robot arm", "robot arm"],
        "right arm": ["white robot arm", "white robot", "robot arm on the right"]}
    for d in ("so101-left_arm_task_red_cube_out_of_pot_into_the_orange_hollow_cube",
              "so101_left_arm_open_drawer_pick_red_cube_stack_on_blue_cube",
              "so101-left_arm_task_drawer_placing_red_squre_block")}


class Seg(L.Segmenter):
    def track_thresh(self, frames, prompt, thr=0.3):
        from PIL import Image
        sid = self._start([Image.fromarray(f) for f in frames])
        out = [dict() for _ in frames]
        try:
            self.p.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=prompt,
                                       output_prob_thresh=thr))
            for r in self.p.handle_stream_request(dict(type="propagate_in_video", session_id=sid, output_prob_thresh=thr)):
                o = r["outputs"]
                probs = np.asarray(o.get("out_probs", np.ones(len(o["out_obj_ids"]))))
                for i, oid in enumerate(np.asarray(o["out_obj_ids"]).tolist()):
                    m = np.asarray(o["out_binary_masks"][i]).astype(bool)
                    if m.any():
                        out[r["frame_index"]][oid] = (m, float(probs[i]))
        finally:
            self.p.handle_request(dict(type="close_session", session_id=sid))
        return out


def clip(root, eps, key, n=16):
    frames = []
    ids = eps.index[np.linspace(0, len(eps) - 1, 3).round().astype(int)].tolist()
    for e in ids:
        row = eps.loc[e]
        f = L.decode_episode(root, row, key, int(row["length"]))
        frames += list(f[np.linspace(0, len(f) - 1, n // 3 + 1).round().astype(int)])
    return np.stack(frames[:n])


seg = Seg(sys.argv[1])
for ds, objs in CANDS.items():
    root = f"/dev/shm/data/so101_v2/{ds}"
    info, eps, _ = L.load_meta(root)
    top, wr = clip(root, eps, "observation.images.top"), clip(root, eps, "observation.images.wrist")
    print(f"\n=== {ds}")
    best = {}
    for obj, phrases in objs.items():
        rows = []
        for ph in phrases:
            for view, fr in (("top", top),):
                t0 = time.time()
                tr = seg.track_thresh(fr, ph)
                det = np.mean([len(d) > 0 for d in tr])
                pb = np.mean([max(p for _, p in d.values()) for d in tr if d]) if det > 0 else 0.0
                rows.append((obj, ph, view, det, pb))
                print(f"  {obj:12s} '{ph}' {view:5s}: detected in {det:.0%} of frames, mean best prob {pb:.2f} ({time.time() - t0:.0f} s)", flush=True)
        for view in ("top",):
            r = max((x for x in rows if x[2] == view), key=lambda x: (x[3] * x[4], x[3]))
            best[(obj, view)] = r[1]
    print("  BEST:", {f"{o}/{v}": p for (o, v), p in best.items()})
    tiles = []
    for t in range(0, len(top), 3):
        im, w = top[t].astype(np.float32), wr[t].astype(np.float32)
        for k, obj in enumerate(objs):
            c = np.array([(255, 0, 0), (0, 255, 255), (255, 0, 255)][k])
            for view, img, frames in (("top", im, top),):
                d = seg.track_thresh(frames[t:t + 1], best[(obj, view)])[0]
                for m, _ in d.values():
                    img[m] = 0.5 * img[m] + 0.5 * c
        tiles.append(np.concatenate([cv2.resize(im.astype(np.uint8), (320, 180)), cv2.resize(w.astype(np.uint8), (240, 180))], 1))
    cv2.imwrite(f"/dev/shm/sam3_arms_{ds[:30]}.jpg", np.concatenate(tiles, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
