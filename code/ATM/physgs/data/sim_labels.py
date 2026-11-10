"""
Exact simulator labels for a packed LIBERO dataset (2026-10-23: exact values instead of estimates). Every
frame's saved MuJoCo state (root/states of the render files, which store qpos and qvel) is replayed (no rendering) and
the labels are taken from the simulator at that instant:
    objects            velocity of the body frame origin (the position label is the body frame position) and angular
                       velocity, world frame (until now: finite differences of the stored pose)
    fixtures / parts   velocity of the body's centre of mass (the position label) and angular velocity
    gripper            velocity of the end-effector site (the position label is the site position)
    arm                position = centre of mass of the robot's arm bodies (robot0_*: base, links, hand) and its
                       velocity (until now: no labels)
Fixtures (not part of the saved states) are first moved to the pose the render stored for them (place_fixtures).
Frames whose labels are masked (objects still settling at the start) stay masked. pos / vel / angvel are rewritten in
place after copying the originals to <name>.before_sim.npy. Run on a machine with the LIBERO env (EGL for the env only).

    python -m physgs.data.sim_labels --data data/packed/libero_10256_v2 --render data/physgs_libero256_long/libero_10 --suite libero_10
"""
import argparse
import collections
import json
import os
import shutil

import h5py
import mujoco
import numpy as np


def body_vel(M, D, b, at_com):
    """(angular, linear) world-frame velocity of body b at its frame origin, or at its centre of mass."""
    v = np.zeros(6)
    mujoco.mj_objectVelocity(M, D, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
    w, lin = v[:3].copy(), v[3:].copy()
    if at_com:
        lin = lin + np.cross(w, D.xipos[b] - D.xpos[b])
    return w, lin


def place_fixtures(M, D, parts, stored):
    """
    Fixtures have no state in the saved simulator states: their placement is sampled when an environment is built, so
    a fresh replay environment puts them up to ~2 cm / 1.5 deg away from where the render's environment had them. Move
    every fixture's root body to the pose the render stored for it (CoM position + orientation at frame 0), so the
    replay reproduces the render's scene exactly. stored: {name: (pos (3,), quat (4,) w x y z)} for kind == "fixture".
    """
    for n, p in parts.items():
        if p["kind"] != "fixture" or n not in stored:
            continue
        b = p["body"]
        pos, quat = np.asarray(stored[n][0], np.float64), np.asarray(stored[n][1], np.float64)
        quat = quat / np.linalg.norm(quat)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        xpos = pos - rot.reshape(3, 3) @ M.body_ipos[b]            # frame origin from the stored CoM
        par = M.body_parentid[b]
        pq_inv = np.zeros(4)
        mujoco.mju_negQuat(pq_inv, D.xquat[par])
        local_q, local_p = np.zeros(4), np.zeros(3)
        mujoco.mju_mulQuat(local_q, pq_inv, quat)
        mujoco.mju_rotVecQuat(local_p, xpos - D.xpos[par], pq_inv)
        M.body_quat[b], M.body_pos[b] = local_q, local_p
    mujoco.mj_forward(M, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="packed dataset (pos / vel / angvel rewritten in place)")
    ap.add_argument("--render", required=True, help="the suite's render folder (<task>_demo/demo_<k>.hdf5)")
    ap.add_argument("--suite", required=True)
    a = ap.parse_args()
    from libero import benchmark, get_libero_path
    from libero.envs import OffScreenRenderEnv
    from physgs.data.libero_parts import part_layout

    meta = json.load(open(os.path.join(a.data, "meta.json")))
    ep = np.load(os.path.join(a.data, "ep_start.npy"))
    for n in ("pos", "vel", "angvel"):
        bak = os.path.join(a.data, f"{n}.before_sim.npy")
        if not os.path.exists(bak):
            shutil.copy2(os.path.join(a.data, f"{n}.npy"), bak)
    arr = {n: np.load(os.path.join(a.data, f"{n}.npy"), mmap_mode="r+") for n in ("pos", "vel", "angvel")}
    by_task = collections.defaultdict(list)
    for e, m in enumerate(meta["episodes"]):
        by_task[os.path.basename(os.path.dirname(m["file"]))].append(e)
    suite = benchmark.get_benchmark_dict()[a.suite]()
    stats = collections.defaultdict(list)
    for task_dir, eps in sorted(by_task.items()):
        t = suite.get_task(suite.get_task_names().index(task_dir[:-len("_demo")]))
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                                 camera_heights=16, camera_widths=16)
        env.reset()
        sim = env.sim
        M, D = sim.model._model, sim.data._data
        parts = {p["name"]: p for p in part_layout(env, True)}
        site = env.env.robots[0].eef_site_id
        arm = [b for b in range(M.nbody) if (M.body(b).name or "").startswith("robot0_")]
        mass = M.body_mass[arm]
        for e in eps:
            m = meta["episodes"][e]
            names, s0, s1 = m["obj_names"], ep[e], ep[e + 1]
            with h5py.File(os.path.join(a.render, task_dir, os.path.basename(m["file"]).replace(".npz", ".hdf5")), "r") as h:
                states = h["root/states"][:]
                o = h["root/objects"]
                stored = {n: (o[n]["pos"][0], o[n]["quat"][0]) for n in parts if parts[n]["kind"] == "fixture" and n in o}
            assert len(states) == s1 - s0, (task_dir, m["file"], len(states), s1 - s0)
            sim.set_state_from_flattened(states[0])
            place_fixtures(M, D, parts, stored)
            T, O = len(states), len(names)
            pos, vel, ang = np.full((T, O, 3), np.nan), np.full((T, O, 3), np.nan), np.full((T, O, 3), np.nan)
            for i, s in enumerate(states):
                sim.set_state_from_flattened(s)
                sim.forward()
                for k, n in enumerate(names):
                    if n == "gripper":
                        v = np.zeros(6)
                        mujoco.mj_objectVelocity(M, D, mujoco.mjtObj.mjOBJ_SITE, site, v, 0)
                        pos[i, k], vel[i, k] = D.site_xpos[site], v[3:]
                    elif n == "arm":
                        lin = np.zeros(3)
                        for b, w_ in zip(arm, mass):
                            lin += w_ * body_vel(M, D, b, True)[1]
                        pos[i, k] = (D.xipos[arm] * mass[:, None]).sum(0) / mass.sum()
                        vel[i, k] = lin / mass.sum()
                    else:
                        p = parts[n]
                        obj = p["kind"] == "object"
                        w, lin = body_vel(M, D, p["body"], at_com=not obj)
                        pos[i, k] = D.xpos[p["body"]] if obj else D.xipos[p["body"]]
                        vel[i, k], ang[i, k] = lin, w
            old = np.asarray(arr["pos"][s0:s1, :O])
            for k, n in enumerate(names):
                keep = np.isfinite(old[:, k]).all(-1) if n != "arm" else np.ones(T, bool)  # settle-masked frames stay NaN
                if n != "arm":  # check that the replay reproduces the stored position labels exactly
                    stats["pos_err_" + ("robot" if n == "gripper" else "token")].append(np.abs(pos[keep, k] - old[keep, k]).max(initial=0))
                    stats["vel_change"].append(np.nanmedian(np.linalg.norm(vel[keep, k] - np.asarray(arr["vel"][s0:s1, k])[keep], axis=-1)) if keep.any() else 0)
                m_ = np.where(keep)[0] + s0
                arr["pos"][m_, k] = pos[keep, k]
                arr["vel"][m_, k] = vel[keep, k]
                if n not in ("gripper", "arm"):
                    arr["angvel"][m_, k] = ang[keep, k]
            stats["episodes"].append(e)
        env.close()
        print(f"{task_dir}: {len(eps)} episodes; max |replayed - stored position| so far: tokens "
              f"{max(stats['pos_err_token']):.2e} m, gripper {max(stats['pos_err_robot']):.2e} m", flush=True)
    for v in arr.values():
        v.flush()
    print(f"exact labels written for {len(stats['episodes'])} episodes; median |exact - old| velocity per token "
          f"{np.nanmedian(stats['vel_change']):.4f} m/s; max position mismatch: tokens {max(stats['pos_err_token']):.2e} m, "
          f"gripper {max(stats['pos_err_robot']):.2e} m")


if __name__ == "__main__":
    main()
