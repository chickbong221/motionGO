"""
live_lift.py (2026-11-10): the SO-101 lift (lift_so101_v2.py + postfilter_arm.py + postfilter_track.py) one frame at a
time, from the top RGB, the top sensor depth and the joint state of that frame only (no look-ahead). Same functions and
constants as the offline lift wherever they need no future frame:
  depth      sensor depth coded with the recordings' LeRobot log quantization and decoded (lift_so101.depth_from_codes),
             registered to the RGB camera (lift_so101.register_depth_gpu, the per-dataset fit), flying pixels removed
             (lift_so101_v2.edge_free)
  gripper    kinematic surface points of the fixed and the moving jaw (so101_kin, kin_v5 calibration), footprint z / f of
             the top camera; colour = the dataset's median gripper colour (offline: measured over the whole episode)
  objects    SAM 3.1 masks of the newest frame (live_sam.LiveSegmenter, forward only), the most confident instance per
             phrase, lift_so101_v2.to_points with the depth gate, points within 8 mm of the posed robot dropped,
             farthest-point order and resampling to 1024 (fp16, as stored offline)
Offline steps that read the whole episode, here causal (statistics of the frames so far):
  consistency  (lift) a frame whose RMS radius is over 2 x the object's median radius is unseen; median over the
               visible frames so far. The single-frame jump rule (needs frame t + 1) is covered by the track rule
  arm          (postfilter_arm) unseen when > 50% of the points lie within 2 x min(median radius, gripper radius / 2)
               of the arm links (base .. wrist); medians over the frames so far
  track        (postfilter_track) the offline rule keeps the longest physically consistent chain of sightings over the
               episode. Live: the same chain lengths, computed online (best[t] = 1 + max best[s] over earlier kept-or-
               not sightings s consistent with t), and frame t is kept when it ends a longest chain so far
"""
import json

import numpy as np
import torch
from scipy.spatial import cKDTree

import live_common as C
import lift_so101 as L
import lift_so101_v2 as L2
from physgs.data.lift import resample_many
from so101_kin import ARM_LINKS, GRIPPER_LINKS, SO101

NUM_POINTS, MIN_POINTS = L2.NUM_POINTS, L2.MIN_POINTS


def code_depth(depth_mm, info):
    """Sensor depth in mm (uint16, 0 = no reading, LeRobot RealSense read_latest_depth) -> 12-bit codes, as LeRobot
    wrote the recorded top_depth (lerobot.datasets.depth_utils.quantize_depth, log mode)."""
    from lerobot.datasets.depth_utils import quantize_depth
    return quantize_depth(np.asarray(depth_mm).squeeze(), depth_min=info["video.depth_min"], depth_max=info["video.depth_max"],
                          shift=info["video.shift"], use_log=info["video.use_log"], video_backend=None, input_unit="mm")


class LiveLifter:
    def __init__(self, name, seg, meta, dev="cuda", edge_rel=0.03, k_arm=2.0):
        self.name, self.seg, self.dev, self.edge_rel, self.k_arm = name, seg, dev, edge_rel, k_arm
        m = meta[name]
        self.depth_info = dict(m["depth_info"])     # depth_unit, video.depth_min / depth_max / shift / use_log
        self.grip_col = np.array(m["gripper_colour"], np.float64)
        self.names = ["gripper"] + [p[0] for p in m["prompts"]]
        kc = json.load(open(C.asset("so101/calib/kin_v5.json")))
        calib = json.load(open(C.asset(f"so101/calib/{name}.json")))
        dk = kc["datasets"][name]
        self.Kt = np.array([[dk["f"], 0, dk["cx"]], [0, dk["f"], dk["cy"]], [0, 0, 1]])
        self.Kreg = np.array(calib["K_top"], np.float64)
        self.reg = calib["depth_reg"]
        self.T_bt = np.array(dk["T_base_to_cam"])
        self.T_tb = np.linalg.inv(self.T_bt)
        urdf = C.asset("so101/robot/so101_new_calib_camera.urdf")
        self.kin = SO101(urdf, device=dev, points_per_link=6000)          # lift_so101_v2.Lifter
        self.kin_arm = SO101(urdf, device="cpu", points_per_link=1500)    # postfilter_arm
        self.kin_trk = SO101(urdf, device="cpu", points_per_link=1000)    # postfilter_track
        self.s = torch.tensor(kc["joint_scale"], dtype=torch.float64, device=dev)
        self.o = torch.tensor(kc["joint_offset"], dtype=torch.float64, device=dev)
        self.s_c, self.o_c = self.s.cpu(), self.o.cpu()
        self.arm_links = tuple(l for l in ARM_LINKS if l not in GRIPPER_LINKS)
        self.reset()

    def reset(self, seed=0):
        """New episode (the SAM sessions start at the next step)."""
        self.t, self.seed = 0, int(seed)
        O = len(self.names)
        self.rad_vis = [[] for _ in range(O)]      # consistency: radii of visible frames
        self.rad_arm = [[] for _ in range(O)]      # arm filter: radii of frames visible after the consistency rule
        self.r_grip = []                           # gripper token RMS radius per frame
        self.trk = [dict(c=[], cum=[], best=[], rad=[]) for _ in range(O)]   # track rule: sightings so far
        self.cum, self.G_prev = 0.0, None          # cumulative path of the fastest gripper surface point
        self.stats = dict(dropped_consistency=np.zeros(O, int), dropped_arm=np.zeros(O, int), dropped_track=np.zeros(O, int))

    @torch.no_grad()
    def step(self, top, depth_codes, state):
        """top (H, W, 3) uint8 RGB, depth_codes (Hd, Wd) uint16 (code_depth), state (6,) LeRobot joint values ->
        gauss (O, 1024, 7) float32 (fp16-rounded), npts (O,) int, timings."""
        import time
        t0 = time.perf_counter()
        dev, T_bt = self.dev, self.T_bt
        masks = self.seg.start(top) if self.t == 0 else self.seg.step(top)
        t_sam = time.perf_counter() - t0
        raw = L.depth_from_codes(np.asarray(depth_codes), self.depth_info)
        sensor = L.register_depth_gpu(raw[None], self.reg, self.Kreg, top.shape[:2], device=dev)[0]
        valid = L2.edge_free(sensor, self.edge_rel)
        S = np.asarray(state, np.float64)[None]
        ang = torch.tensor(S, dtype=torch.float64, device=dev) * self.s + self.o
        G = self.kin.points(ang, GRIPPER_LINKS)[0][0].float()
        rob = self.kin.points(ang, ARM_LINKS + L2.CAM_LINKS)[0][0][::4].float()
        O = len(self.names)
        sets, owner = [], []
        npts = np.zeros(O, np.int64)
        zc = (G.double().cpu().numpy() @ T_bt[:3, :3].T + T_bt[:3, 3])[:, 2]
        g7 = torch.cat([G, torch.tensor(np.tile(self.grip_col, (len(G), 1)), dtype=torch.float32, device=dev),
                        torch.tensor(zc / self.Kt[0, 0], dtype=torch.float32, device=dev)[:, None]], 1)
        sets.append(g7); owner.append(0); npts[0] = len(g7)
        self.last = dict(masks=[None] * (O - 1), gripper=G.cpu().numpy())     # for run_live --scene check images
        for k in range(1, O):
            d = masks[k - 1]
            if not d:
                continue
            mk = max(d.values(), key=lambda mp: mp[1])[0]      # lift_so101.best_instance: the most confident instance
            self.last["masks"][k - 1] = mk
            pts = L2.to_points(sensor, mk, self.Kt, top, self.T_tb, dev, valid, depth_gate=True)
            if pts.shape[0] >= MIN_POINTS:
                # points within 8 mm of the posed robot are dropped (lift_so101_v2); only robot points inside the
                # object's box grown by 8 mm can be that close, so the distance matrix is built on those (same result)
                lo_, hi_ = pts[:, :3].min(0).values - 0.008, pts[:, :3].max(0).values + 0.008
                rb = rob[((rob >= lo_) & (rob <= hi_)).all(1)]
                if rb.shape[0]:
                    near = torch.cdist(pts[:, :3], rb).min(1).values < 0.008
                    pts = pts[~near]
            npts[k] = pts.shape[0]
            if pts.shape[0] >= MIN_POINTS:
                sets.append(pts); owner.append(k)
        t_pts = time.perf_counter()
        sets = [L2.fps_order(s_) for s_ in sets]
        gen = torch.Generator(device=dev).manual_seed(self.seed * 100003 + self.t)
        gauss = np.zeros((O, NUM_POINTS, 7), np.float16)
        for k, g_ in zip(owner, resample_many(sets, NUM_POINTS, [gen] * len(sets))):
            gauss[k] = g_.cpu().numpy().astype(np.float16)
        t_end = time.perf_counter()
        t_lift = t_end - t0 - t_sam
        gauss = gauss.astype(np.float32)
        self._filters(gauss, npts, S[0])
        self.t += 1
        return gauss, npts, dict(sam=t_sam, lift=t_lift, filters=time.perf_counter() - t_end,
                                 lift_points=t_pts - t0 - t_sam, lift_fps=t_end - t_pts)

    def scene_image(self, top):
        """The top image with the last step's object masks (tinted) and the posed gripper model projected with the
        dataset's camera calibration (red dots): the dots must lie on the real gripper, or the rig differs."""
        import cv2
        img = top.copy()
        cols = [(255, 160, 0), (0, 200, 255), (255, 0, 255), (0, 255, 0)]
        for k, mk in enumerate(self.last["masks"]):
            if mk is not None:
                img[mk] = (0.5 * img[mk] + 0.5 * np.array(cols[k % 4])).astype(np.uint8)
        P = self.last["gripper"].astype(np.float64) @ self.T_bt[:3, :3].T + self.T_bt[:3, 3]
        u = self.Kt[0, 0] * P[:, 0] / P[:, 2] + self.Kt[0, 2]
        v = self.Kt[1, 1] * P[:, 1] / P[:, 2] + self.Kt[1, 2]
        for x, y in zip(u[::20], v[::20]):
            cv2.circle(img, (int(round(x)), int(round(y))), 1, (255, 0, 0), -1)
        for k, nm in enumerate(self.names[1:]):
            cv2.putText(img, f"{nm}: {'seen' if self.last['masks'][k] is not None else 'NOT SEEN'}", (8, 18 + 16 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, cols[k % 4], 1)
        return img

    # ------------------------------------------------------------------------------------------------ causal filters
    def _drop(self, gauss, npts, k, why):
        gauss[k] = 0
        npts[k] = 0
        self.stats[why][k] += 1

    def _filters(self, gauss, npts, state):
        P = gauss[:, :512, :3]
        Gt = P[0]
        self.r_grip.append(float(np.sqrt(((Gt - Gt.mean(0)) ** 2).sum(-1).mean())))
        # gripper path (postfilter_track: max displacement of the gripper surface points between frames)
        ang = torch.tensor(state[None], dtype=torch.float64) * self.s_c + self.o_c
        Gk = self.kin_trk.points(ang, GRIPPER_LINKS)[0][0].numpy()
        if self.G_prev is not None:
            self.cum += float(np.linalg.norm(Gk - self.G_prev, axis=-1).max())
        self.G_prev = Gk
        A = None
        for k in range(1, len(self.names)):
            if npts[k] < MIN_POINTS:
                continue
            c = P[k].mean(0)
            rad = float(np.sqrt(((P[k] - c) ** 2).sum(-1).mean()))
            # 1. consistency (lift): radius over 2 x the median radius of the visible frames so far (offline: >= 3)
            self.rad_vis[k].append(rad)
            if len(self.rad_vis[k]) >= 3 and rad > 2 * float(np.median(self.rad_vis[k])):
                self._drop(gauss, npts, k, "dropped_consistency")
                continue
            # 2. arm (postfilter_arm): over half of the points near the arm links (not the gripper)
            self.rad_arm[k].append(rad)
            if len(self.rad_arm[k]) >= 3:
                if A is None:
                    A = cKDTree(self.kin_arm.points(ang, self.arm_links)[0][0].numpy())
                rmed = min(float(np.median(self.rad_arm[k])), float(np.median(self.r_grip)) / self.k_arm)
                dd = A.query(P[k], distance_upper_bound=self.k_arm * rmed)[0]
                if np.mean(np.isfinite(dd)) > 0.5:
                    self._drop(gauss, npts, k, "dropped_arm")
                    continue
            # 3. track (postfilter_track): online longest consistent chain; keep t when it ends a longest chain
            tr = self.trk[k]
            tr["rad"].append(rad)
            tol = 2 * float(np.median(tr["rad"]))
            b = 1
            if tr["c"]:
                d = c[None] - np.array(tr["c"])
                dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + np.maximum(d[:, 2], 0.0) ** 2)
                ok = dist <= 2 * tol + 2 * (self.cum - np.array(tr["cum"]))
                if ok.any():
                    b = int(np.array(tr["best"])[ok].max()) + 1
            longest = max(tr["best"]) if tr["best"] else 0
            tr["c"].append(c); tr["cum"].append(self.cum); tr["best"].append(b)
            if b < longest:
                self._drop(gauss, npts, k, "dropped_track")
