"""
lift_so101_v2.py (2026-11-06): SO-101 episodes -> PhysGO episode files, the LIBERO-like version
(the gripper finger positions are what matters). Differences from lift_so101.py (v1):
  frame      the robot base frame (URDF base_link: z up, gravity -z), the same in every dataset although the top camera
             stands differently in each; calibrate_kin_v5.py gives base -> top camera per dataset
  gripper    the only robot token, as in LIBERO's packed data (gripper + task objects): surface points of the SO-101
             model's fixed jaw (gripper_link) and moving jaw, posed by forward kinematics of the recorded joint states
             (LeRobot so_follower convention: arm joints in degrees + fitted offsets, gripper RANGE_0_100 over the URDF jaw
             limits). Complete in every frame, never occluded. Colour: the arm's median observed colour; footprint as
             if seen by the top camera (z / f)
  wrist      pose = FK(gripper_link) @ X (calibrate_wrist.py), not Depth Anything 3's estimate; wrist depth = DA3
             (single view) x a per-frame scale from the top sensor points projected into the wrist camera
  hidden     (2026-11-07) an object hidden at the start (red cube inside the pot) was never detected by the forward
             pass prompted at frame 0: lift_so101.track_both_ends adds a backward pass prompted at the last frame
  objects    SAM 3.1 masks (best phrase per object and view) x registered sensor depth (top) and scaled DA3 depth
             (wrist); points on the posed robot model (within 8 mm) are dropped (a held cube's mask can include finger
             pixels); a wrist object point set is kept only when it agrees with the top view of that object (median
             distance below the object's radius) or the object is unseen from the top
  storage    1024 Gaussians per object in farthest-point order (training reads 512)
  edges      (2026-11-07) flying pixels removed: a pixel within 1 px of a depth step larger than --edge_rel x its depth
             (3x3 max - min of the registered depth) is not lifted; these pixels lie between an object and what is
             behind it and would pull the object's centroid
  depth gate (2026-11-07) per object and frame, pixels farther than one object width (mask extent x depth / f) from
             the mask's median depth are dropped (background depth inside small masks)
  wrist      (2026-11-07) off by default (--wrist none): scaled DA3 wrist depth was 24-29 mm from the top sensor
             (eval_episode.py, 3 unseen episodes) against 3-5 mm for top sensor points, so it would add error, not detail
Checks in the per-episode summary: table height in the base frame (should be ~0: the base stands on the table),
visibility per object, wrist frames used, the per-frame DA3 scale spread.
  python lift_so101_v2.py --data_root <dir> --datasets a,b --out <dir> --calib_dir <dir> --kin kin_v5.json
                          --wrist wrist.json --emb_dir <dir> [--worker i --nworkers n] [--episodes ...]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lift_so101 as L  # noqa: E402
from physgs.data.format import save_episode  # noqa: E402
from physgs.data.lift import resample_many  # noqa: E402
from physgs.model.layers import farthest_point_sample  # noqa: E402
from so101_kin import ARM_LINKS, GRIPPER_LINKS, SO101  # noqa: E402

NUM_POINTS, MIN_POINTS, STRIDE, FPS_IN = 1024, 16, 2, 30
URDF = "/dev/shm/physgo/so101/robot/so101_new_calib_camera.urdf"
CAM_LINKS = ("wrist_camera_mount_link", "wrist_camera_link")


def edge_free(depth, rel):
    """(H, W) bool: valid depth not within 1 px of a depth step larger than rel x depth (flying pixels)."""
    k3 = np.ones((3, 3), np.uint8)
    ok = np.isfinite(depth) & (depth > 0)
    d0 = np.where(ok, depth, 0).astype(np.float32)
    mx = cv2.dilate(d0, k3)
    mn = cv2.erode(np.where(ok, depth, np.inf).astype(np.float32), k3)
    disc = ok & ((mx - mn) > rel * d0)
    return ok & ~(cv2.dilate(disc.astype(np.uint8), k3) > 0)


def to_points(depth, mask, K, rgb, T_cam_to_base, dev, valid=None, depth_gate=False):
    """Masked pixels with depth -> Gaussians [xyz base, rgb, footprint z / f] (N, 7) on dev."""
    m = mask & np.isfinite(depth) & (depth > 0.05)
    v, u = np.nonzero(m if valid is None else m & valid)
    if len(u) == 0:
        return torch.zeros(0, 7, dtype=torch.float32, device=dev)
    z = depth[v, u]
    if depth_gate and len(u) >= MIN_POINTS:
        # (2026-11-07, test A0: the red cube held in the gripper spread 80-126 mm in height for a 25 mm cube) the
        # registered depth is a few pixels off at small objects, so a band of table / finger depth falls inside the
        # mask: keep pixels within one object width of the mask's median depth, the width = the mask's own extent
        # (5-95% of u and v) x depth / f. Relative to each object: no effect on the drawer or the pot.
        zm = np.median(z)
        w = max(np.percentile(u, 95) - np.percentile(u, 5), np.percentile(v, 95) - np.percentile(v, 5)) * zm / K[0, 0]
        keep = np.abs(z - zm) <= w
        u, v, z = u[keep], v[keep], z[keep]
    P = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    P = P @ T_cam_to_base[:3, :3].T + T_cam_to_base[:3, 3]
    col = rgb[v, u].astype(np.float32) / 255
    return torch.tensor(np.concatenate([P, col, (z / K[0, 0])[:, None]], 1), dtype=torch.float32, device=dev)


def fps_order(s):
    return s[farthest_point_sample(s[None, :, :3], s.shape[0])[0]] if s.shape[0] < NUM_POINTS else s


class Lifter:
    def __init__(self, kin_json, wrist_json, sam3_ckpt, dev="cuda", edge_rel=0.03):
        self.dev = dev
        self.edge_rel = edge_rel
        self.kc = json.load(open(kin_json))
        self.use_wrist = wrist_json not in (None, "", "none")
        self.kin = SO101(URDF, device=dev, points_per_link=6000)
        self.s = torch.tensor(self.kc["joint_scale"], dtype=torch.float64, device=dev)
        self.o = torch.tensor(self.kc["joint_offset"], dtype=torch.float64, device=dev)
        self.seg = L.Segmenter(sam3_ckpt)
        if self.use_wrist:
            self.wc = json.load(open(wrist_json))
            self.X = np.array(self.wc["X_gripper_to_wrist"])
            from depth_anything_3.api import DepthAnything3
            self.da3 = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to(dev).eval()

    @torch.no_grad()
    def episode(self, root, info, row, ep, name, prompts, calib, task_emb, out_path):
        t_start, tim = time.time(), {}
        dev = self.dev
        n30 = int(row["length"])
        top = L.decode_episode(root, row, "observation.images.top", n30)
        wrist = L.decode_episode(root, row, "observation.images.wrist", n30) if self.use_wrist else None
        q = L.decode_episode(root, row, "observation.images.top_depth", n30, depth=True)
        T = len(top)
        dinfo = info["features"]["observation.images.top_depth"]["info"]
        dk = self.kc["datasets"][name]
        Kt = np.array([[dk["f"], 0, dk["cx"]], [0, dk["f"], dk["cy"]], [0, 0, 1]])
        Kreg = np.array(calib["K_top"], np.float64)    # the registration was fitted with the DA3 focal: keep it for it
        raw = np.stack([L.depth_from_codes(x, dinfo) for x in q])
        sensor = np.concatenate([L.register_depth_gpu(raw[s:s + 64], calib["depth_reg"], Kreg, top.shape[1:3])
                                 for s in range(0, T, 64)])
        T_bt = np.array(dk["T_base_to_cam"])
        T_tb = np.linalg.inv(T_bt)                        # top camera -> base
        # joint states at the 15 fps frames
        data = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
                          for f in sorted(__import__("glob").glob(f"{root}/data/chunk-*/file-*.parquet"))])
        st = data[data["episode_index"] == ep].sort_values("frame_index")
        S = np.stack(st["observation.state"].values)[::STRIDE][:T]
        A = np.stack(st["action"].values)[::STRIDE][:T].astype(np.float32)
        fk = self.kin.fk(torch.tensor(S, dtype=torch.float64, device=dev) * self.s + self.o)
        tim["decode"] = time.time() - t_start

        # SAM 3.1 object masks, top and wrist
        t0 = time.time()
        tops, srcs, wrs, passes = [], [], [], {}
        for n_, p_top, p_wr in prompts:
            tr, passes[n_] = L.track_both_ends(self.seg, top, p_top)
            m, src = L.best_instance_src(tr)
            tops.append(np.zeros(top.shape[:3], bool) if m is None else m)
            srcs.append(src)
            if self.use_wrist:
                m = L.best_instance(self.seg.track(wrist, p_wr))
                wrs.append(np.zeros(wrist.shape[:3], bool) if m is None else m)
        # a backward-pass mask can drift onto another object while the real one is hidden (test A0: before the grasp the
        # red-block track lay inside the orange hollow cube while the cube was in the pot): a frame known only from the
        # backward pass whose mask lies mostly (> 50%) inside another object's mask is dropped. Forward frames are kept
        # (the red cube genuinely inside the orange cube at the end).
        drift = {}
        for k in range(len(prompts)):
            n_drop = 0
            for t in np.nonzero(srcs[k] == 2)[0]:
                a_ = tops[k][t].sum()
                if a_ and any((tops[k][t] & tops[j][t]).sum() > 0.5 * a_ for j in range(len(prompts)) if j != k):
                    tops[k][t] = False
                    n_drop += 1
            drift[prompts[k][0]] = dict(backward_frames=int((srcs[k] == 2).sum()), dropped_inside_other=n_drop)
        tim["sam"] = time.time() - t0

        # robot model points per frame (base frame) for the gripper token and the robot-point removal
        t0 = time.time()
        grip_pts, rob_pts = [], []
        for t in range(T):
            ang = torch.tensor(S[t:t + 1], dtype=torch.float64, device=dev) * self.s + self.o
            P, _ = self.kin.points(ang, GRIPPER_LINKS)
            grip_pts.append(P[0].float())
            P, _ = self.kin.points(ang, ARM_LINKS + CAM_LINKS)
            rob_pts.append(P[0][::4].float())
        # gripper colour: the observed colour where the posed model's visible surface matches the sensor depth
        cols = []
        for t in range(0, T, max(1, T // 20)):
            Pc = grip_pts[t].double().cpu().numpy() @ T_bt[:3, :3].T + T_bt[:3, 3]
            u = (Kt[0, 0] * Pc[:, 0] / Pc[:, 2] + Kt[0, 2]).round().astype(int)
            v = (Kt[1, 1] * Pc[:, 1] / Pc[:, 2] + Kt[1, 2]).round().astype(int)
            ok = (u >= 0) & (u < top.shape[2]) & (v >= 0) & (v < top.shape[1])
            u, v, zc = u[ok], v[ok], Pc[ok, 2]
            vis = np.abs(sensor[t][v, u] - zc) < 0.01
            if vis.sum() > 20:
                cols.append(np.median(top[t][v[vis], u[vis]], 0))
        grip_col = (np.median(cols, 0) / 255) if cols else np.array([0.1, 0.7, 0.8])
        tim["robot"] = time.time() - t0

        # wrist: pose from kinematics, DA3 depth with a per-frame scale from the top sensor (off by default)
        t0 = time.time()
        wr_depth, wr_c2b, wr_scale = [], [], np.full(T, np.nan)
        if self.use_wrist:
            fw = self.wc["f_wrist"]
            Hd, Wd = self.wc["wrist_hw"]
            T_bg = fk["gripper_link"].cpu().numpy()
            for t in range(T):
                p = self.da3.inference([wrist[t]], process_res=644)
                d = p.depth[0].astype(np.float64)
                c2b = T_bg[t] @ self.X                             # wrist camera -> base
                b2c = np.linalg.inv(c2b)
                Ps = L.unproject_np(sensor[t], Kt)
                Ps = Ps[np.isfinite(Ps[..., 2])]
                Ps = Ps[np.random.default_rng(t).choice(len(Ps), min(len(Ps), 30000), replace=False)]
                Pb = Ps @ T_tb[:3, :3].T + T_tb[:3, 3]
                Pw = Pb @ b2c[:3, :3].T + b2c[:3, 3]
                Pw = Pw[Pw[:, 2] > 0.02]
                u = (fw * Pw[:, 0] / Pw[:, 2] + Wd / 2).round().astype(int)
                v = (fw * Pw[:, 1] / Pw[:, 2] + Hd / 2).round().astype(int)
                ok = (u >= 0) & (u < d.shape[1]) & (v >= 0) & (v < d.shape[0])
                if ok.sum() > 200:
                    zbuf = np.full(d.shape, np.inf)
                    np.minimum.at(zbuf, (v[ok], u[ok]), Pw[ok, 2])
                    sel = np.isfinite(zbuf) & (d > 0)
                    if sel.sum() > 100:
                        wr_scale[t] = np.median(zbuf[sel] / d[sel])
                wr_depth.append(d)
                wr_c2b.append(c2b)
            med = np.nanmedian(wr_scale) if np.isfinite(wr_scale).any() else 1.0
            wr_scale = np.where(np.isfinite(wr_scale), wr_scale, med)
        tim["wrist"] = time.time() - t0

        # lift
        t0 = time.time()
        O = 1 + len(prompts)
        names = ["gripper"] + [n_ for n_, _, _ in prompts]
        gauss = np.zeros((T, O, NUM_POINTS, 7), np.float16)
        npts = np.zeros((T, O), np.int32)
        wrist_used = np.zeros((T, O), bool)
        gens = [torch.Generator(device=dev).manual_seed(int(ep) * 100003 + t) for t in range(T)]
        Kw = np.array([[fw, 0, Wd / 2], [0, fw, Hd / 2], [0, 0, 1]]) if self.use_wrist else None
        edge_drop = np.zeros(len(prompts))
        edge_tot = np.zeros(len(prompts))
        for t in range(T):
            valid = edge_free(sensor[t], self.edge_rel)
            sets, owner = [], []
            # gripper token: kinematic surface points, footprint as seen from the top camera
            G = grip_pts[t]
            zc = (G.double().cpu().numpy() @ T_bt[:3, :3].T + T_bt[:3, 3])[:, 2]
            g7 = torch.cat([G, torch.tensor(np.tile(grip_col, (len(G), 1)), dtype=torch.float32, device=dev),
                            torch.tensor(zc / Kt[0, 0], dtype=torch.float32, device=dev)[:, None]], 1)
            sets.append(g7); owner.append(0); npts[t, 0] = len(g7)
            if self.use_wrist:
                wr_rgb = cv2.resize(wrist[t], (Wd, Hd), interpolation=cv2.INTER_AREA)
            for k in range(len(prompts)):
                mk = tops[k][t] & np.isfinite(sensor[t]) & (sensor[t] > 0.05)
                edge_tot[k] += mk.sum()
                edge_drop[k] += (mk & ~valid).sum()
                tp = to_points(sensor[t], tops[k][t], Kt, top[t], T_tb, dev, valid, depth_gate=True)
                if self.use_wrist:
                    wm = cv2.resize(wrs[k][t].astype(np.uint8), (Wd, Hd), interpolation=cv2.INTER_NEAREST) > 0
                    wp = to_points(wr_depth[t] * wr_scale[t], wm, Kw, wr_rgb, wr_c2b[t], dev)
                else:
                    wp = tp[:0]
                if wp.shape[0] >= MIN_POINTS and tp.shape[0] >= MIN_POINTS:
                    rad = (tp[:, :3] - tp[:, :3].mean(0)).norm(dim=-1).pow(2).mean().sqrt()
                    dd = torch.cdist(wp[::max(1, wp.shape[0] // 2000), :3], tp[::max(1, tp.shape[0] // 4000), :3]).min(1).values
                    if dd.median() > rad:
                        wp = wp[:0]
                pts = torch.cat([tp, wp], 0) if wp.shape[0] >= MIN_POINTS else tp
                wrist_used[t, k + 1] = wp.shape[0] >= MIN_POINTS
                if pts.shape[0] >= MIN_POINTS:   # drop points on the posed robot (finger pixels inside a held object's mask)
                    near = torch.cdist(pts[:, :3], rob_pts[t]).min(1).values < 0.008
                    pts = pts[~near]
                npts[t, k + 1] = pts.shape[0]
                if pts.shape[0] >= MIN_POINTS:
                    sets.append(pts); owner.append(k + 1)
            sets = [fps_order(s_) for s_ in sets]
            for k, g_ in zip(owner, resample_many(sets, NUM_POINTS, [gens[t]] * len(sets))):
                gauss[t, k] = g_.cpu().numpy().astype(np.float16)
        # consistency per object (2026-11-07, test episode B0: before the blue cube came into view SAM's "blue block"
        # mask lay on the light-blue drawer, radius 116 mm for a 2 cm cube): a frame whose point set is more than twice
        # the object's median RMS radius (another object) or whose centroid jumps more than 2 median radii away from
        # both neighbours while the neighbours agree (one wrong frame) counts as unseen. Scale = the object's own size.
        dropped = {}
        for k in range(1, O):
            vis = npts[:, k] >= MIN_POINTS
            if vis.sum() < 3:
                continue
            P = gauss[:, k, :512, :3].astype(np.float32)
            c = P.mean(1)
            rad = np.sqrt(((P - c[:, None]) ** 2).sum(-1).mean(1))
            rmed = float(np.median(rad[vis]))
            bad = vis & (rad > 2 * rmed)
            ok = vis & ~bad
            d = lambda a_, b_: np.linalg.norm(a_ - b_, axis=1)
            mid = ok[1:-1] & ok[:-2] & ok[2:]
            jump = mid & (np.minimum(d(c[1:-1], c[:-2]), d(c[1:-1], c[2:])) > 2 * rmed) & (d(c[2:], c[:-2]) < rmed)
            bad[1:-1] |= jump
            gauss[bad, k] = 0
            npts[bad, k] = 0
            dropped[names[k]] = int(bad.sum())
        tim["lift"] = time.time() - t0

        # check: table height in the base frame (the base stands on the table -> ~0)
        excl = np.any(np.stack([m[0] for m in tops]), 0)
        Ps = L.unproject_np(np.where(excl, np.nan, sensor[0]), Kt)
        Ps = Ps[np.isfinite(Ps[..., 2])] @ T_tb[:3, :3].T + T_tb[:3, 3]
        zt = float(np.median(Ps[(Ps[:, 2] > -0.05) & (Ps[:, 2] < 0.05), 2])) if len(Ps) else float("nan")
        nan3 = np.full((T, O, 3), np.nan, np.float32)
        save_episode(out_path, gauss=gauss, npts=npts, pos=nan3, vel=nan3.copy(), angvel=nan3.copy(),
                     contact=np.full((T, O), np.nan, np.float32), props=np.full((O, 3), np.nan, np.float32),
                     is_robot=np.array([True] + [False] * (O - 1)), obj_names=np.array(names), task_emb=task_emb,
                     fps=np.float32(FPS_IN / STRIDE), source=np.array("so101"), actions=A, state=S.astype(np.float32))
        cam = dict(K_top=Kt, T_base_to_top=T_bt, gripper_colour=grip_col, edge_rel=self.edge_rel)
        if self.use_wrist:
            cam.update(wrist_c2b=np.stack(wr_c2b), K_wrist=Kw, wrist_scale=wr_scale, wrist_used=wrist_used)
        np.savez_compressed(out_path.replace(".npz", "_cam.npz"), **cam)
        extra = dict(wrist_obj_frac=round(float(wrist_used[:, 1:].mean()), 3),
                     wrist_scale_med=round(float(np.median(wr_scale)), 3),
                     wrist_scale_spread=round(float(np.std(wr_scale) / np.mean(wr_scale)), 3)) if self.use_wrist else {}
        return dict(ep=int(ep), T=T, objects=names, table_z_mm=round(zt * 1000, 1),
                    visible=dict(zip(names, np.round((npts >= MIN_POINTS).mean(0), 3).tolist())),
                    sam_passes=passes, backward=drift, inconsistent_frames=dropped, edge_dropped=dict(zip(names[1:], np.round(edge_drop / np.maximum(edge_tot, 1), 3).tolist())), **extra,
                    seconds=round(time.time() - t_start, 1), stages={k: round(v, 1) for k, v in tim.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--kin", required=True)
    ap.add_argument("--wrist", default="none", help="wrist calibration json, or none (default: top sensor only)")
    ap.add_argument("--edge_rel", type=float, default=0.03)
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--sam3_ckpt", default=None)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    args = ap.parse_args()
    jobs, meta = [], {}
    for name in args.datasets.split(","):
        root = f"{args.data_root.rstrip('/')}/{name}"
        info, eps, _ = L.load_meta(root)
        meta[name] = (root, info, eps, json.load(open(f"{args.calib_dir}/{name}.json")),
                      np.load(f"{args.emb_dir}/{name}.npy").astype(np.float32))
        ids = eps.index.tolist() if args.episodes == "all" else [int(x) for x in args.episodes.split(",")]
        jobs += [(int(eps.loc[e, "length"]), name, int(e)) for e in ids]
    jobs = sorted(jobs, key=lambda j: (-j[0], j[1], j[2]))[args.worker::args.nworkers]
    print(f"worker {args.worker}/{args.nworkers}: {len(jobs)} episodes", flush=True)
    lf = Lifter(args.kin, args.wrist, args.sam3_ckpt, edge_rel=args.edge_rel)
    for _, name, e in jobs:
        out = f"{args.out}/{name}/episode_{e:04d}.npz"
        if os.path.exists(out):
            continue
        root, info, eps, calib, emb = meta[name]
        # (2026-11-07) a SAM 3.1 session holds per-frame state of the whole video: with two workers on one 80 GB GPU a
        # neighbour's long episode (48.7 GB) ran another worker out of memory. Free the cache and retry up to twice,
        # then leave the episode for the next pass (finished episodes are skipped) instead of losing the worker.
        for attempt in range(3):
            try:
                r = lf.episode(root, info, eps.loc[e], e, name, L.PROMPTS[name], calib, emb, out)
                print(json.dumps(dict(dataset=name[:40], **r)), flush=True)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"OOM_RETRY {name[:40]} ep {e} attempt {attempt + 1}", flush=True)
                time.sleep(120)
            except Exception as ex:  # noqa: BLE001  any other failure: log it, keep the worker for the other episodes
                import traceback
                traceback.print_exc()
                print(f"EPISODE_FAILED {name[:40]} ep {e}: {type(ex).__name__}: {ex}", flush=True)
                torch.cuda.empty_cache()
                break
        else:
            print(f"OOM_SKIPPED {name[:40]} ep {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
