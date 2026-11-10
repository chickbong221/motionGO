"""
Convert re-rendered LIBERO demos (scripts/render_libero_multiview.py output) into episode files.

Tokens: every task object, plus two robot tokens (gripper = effector, arm). Labels come from the
simulator: object CoM position, finite-difference linear / angular velocity, contact, mass and
friction; the gripper token gets the end-effector position and velocity.

    python -m physgs.data.libero_process --src data/physgs_libero/libero_spatial --out data/libero_episodes/libero_spatial
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from physgs.data.format import save_episode
from physgs.data.lift import lift_objects

ROBOT_INSTANCES = {"PandaGripper0": "gripper", "Panda0": "arm"}
DT = 1.0 / 20


def quat_angvel(q, dt):
    """q: (T, 4) MuJoCo (w, x, y, z) world quaternions -> (T, 3) world angular velocity (rad/s)."""
    w = np.zeros((len(q), 3), np.float32)
    for t in range(len(q) - 1):
        a, b = q[t], q[t + 1]
        if np.dot(a, b) < 0:
            b = -b
        # dq = b * conj(a), world frame
        aw, ax, ay, az = a[0], -a[1], -a[2], -a[3]
        bw, bx, by, bz = b
        dw = bw * aw - bx * ax - by * ay - bz * az
        dx = bw * ax + bx * aw + by * az - bz * ay
        dy = bw * ay - bx * az + by * aw + bz * ax
        dz = bw * az + bx * ay - by * ax + bz * aw
        v = np.array([dx, dy, dz])
        s = np.linalg.norm(v)
        angle = 2 * np.arctan2(s, dw)
        w[t] = v / s * angle / dt if s > 1e-9 else 0.0
    w[-1] = w[-2] if len(q) > 1 else 0.0
    return w


def convert(path, out_path, views, num_points=512, device="cpu"):
    with h5py.File(path, "r") as f:
        root = f["root"]
        instances = json.loads(f.attrs["instances"])
        T = root["actions"].shape[0]
        rgb = np.stack([root[v]["video"][0].transpose(0, 2, 3, 1) for v in views], 1)    # (T, V, H, W, 3)
        depth = np.stack([root[v]["depth"][:].astype(np.float32) for v in views], 1)       # (T, V, H, W)
        seg = np.stack([root[v]["seg"][:].astype(np.int64) for v in views], 1)
        K = np.stack([root[v]["K"][:] for v in views], 0)
        c2w = np.stack([root[v]["cam2world"][:] for v in views], 1)                        # (T, V, 4, 4)
        objs = root["objects"]
        order = json.loads(objs.attrs["order"])
        contact_all = objs["contact"][:]
        names, seg_ids, pos, vel, angvel, contact, props, is_robot = [], [], [], [], [], [], [], []
        for k, n in enumerate(order):
            g = objs[n]
            names.append(n); seg_ids.append(int(g.attrs["seg_id"])); is_robot.append(False)
            pos.append(g["pos"][:]); vel.append(g["vel"][:]); angvel.append(quat_angvel(g["quat"][:], DT))
            contact.append(contact_all[:, k])
            props.append([np.log(max(float(g.attrs["mass"]), 1e-6)), float(g.attrs["friction"][0]), np.nan])
        ee = root["extra_states"]["ee_pos"][:].astype(np.float32)
        for inst, label in ROBOT_INSTANCES.items():
            if inst not in instances:
                continue
            names.append(label); seg_ids.append(instances.index(inst) + 1); is_robot.append(True)
            if label == "gripper":
                pos.append(ee); vel.append(np.gradient(ee, DT, axis=0).astype(np.float32))
            else:
                pos.append(np.full((T, 3), np.nan, np.float32)); vel.append(np.full((T, 3), np.nan, np.float32))
            angvel.append(np.full((T, 3), np.nan, np.float32))
            contact.append(np.full(T, np.nan, np.float32))
            props.append([np.nan] * 3)
        actions = root["actions"][:].astype(np.float32)
        task_emb = root["task_emb_bert"][:].astype(np.float32)

    O = len(names)
    gauss = np.zeros((T, O, num_points, 7), np.float16)
    npts = np.zeros((T, O), np.int32)
    Kt = torch.tensor(K, dtype=torch.float32, device=device)
    for t in range(T):
        g, n_ = lift_objects(torch.from_numpy(rgb[t]).to(device), torch.from_numpy(depth[t]).to(device),
                             torch.from_numpy(seg[t]).to(device), Kt,
                             torch.tensor(c2w[t], dtype=torch.float32, device=device), seg_ids,
                             num_points=num_points, max_depth=5.0)
        gauss[t] = g.cpu().numpy().astype(np.float16)
        npts[t] = n_.numpy()

    save_episode(out_path, gauss=gauss, npts=npts, pos=np.stack(pos, 1).astype(np.float32),
                 vel=np.stack(vel, 1).astype(np.float32), angvel=np.stack(angvel, 1).astype(np.float32),
                 contact=np.stack(contact, 1).astype(np.float32), props=np.array(props, np.float32),
                 is_robot=np.array(is_robot), obj_names=np.array(names), task_emb=task_emb, fps=np.float32(20.0),
                 source=np.array("libero"), actions=actions)
    return dict(file=os.path.basename(path), T=T, objects=O, visible_frac=float((npts >= 16).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="render_libero_multiview output for one suite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="agentview,mv_left,mv_right,mv_top")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1, help="split the demos across parallel processes")
    args = ap.parse_args()
    views = args.views.split(",")
    files = sorted(glob.glob(os.path.join(args.src, "*", "demo_*.hdf5")))[args.shard::args.num_shards]
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, f"convert_log_{args.shard}.jsonl"), "a")
    for p in tqdm(files):
        task = os.path.basename(os.path.dirname(p))
        out = os.path.join(args.out, task, os.path.basename(p).replace(".hdf5", ".npz"))
        if os.path.exists(out):
            continue
        log.write(json.dumps(convert(p, out, views, device=args.device)) + "\n")
        log.flush()


if __name__ == "__main__":
    main()
