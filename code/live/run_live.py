"""
run_live.py (2026-11-10): PhysGO (smolVLA + world-model forecast) or the plain smolVLA on the SO-101, live.
Per 15 fps step: read the top RGB-D camera, the wrist camera and the joint state -> (PhysGO) lift the frame to per-object
Gaussians (live_lift.LiveLifter, SAM 3.1 forward only) -> world-model forecast -> policy -> one joint target, sent to
the follower arm. No physics labels anywhere: the inputs are the ones the datasets recorded.

Chunked policies (2026-11-11; the v3 runs, smolVLA's default setting: 50 actions at 30 fps) run as LeRobot runs them and
as eval_policy_so101.chunk_exec_eval scored them: when the action queue is empty, a planning frame (cameras + state ->
full lift -> forecast -> one chunk); its first --exec_steps targets (default: all 50) are sent at 30 Hz; the robot holds
its last target while the next chunk is computed. PhysGO's world model filters every 15 fps frame (dt 1/15 s, as in
training): between planning frames a background thread lifts the gripper from the joint state alone (kinematics) with
every object unseen (live_lift.LiveLifter.gripper_step), since SAM (~0.3 s per frame) cannot run at 15 fps; objects are
seen at the planning frames. Offline this cost 0.06 deg against a lift at every frame (sim_chunk.py "objsparse").

Sources
  robot    LeRobot SO-101 follower (so_follower, arm joints in degrees as recorded) with the RealSense top camera
           (RGB + depth, 640 x 360, depth not aligned: the lift registers it with the fitted calibration) and the wrist
           camera (640 x 480). Same rig and scene as the hungho77 datasets: the per-dataset camera calibration is reused.
  replay   a recorded episode played through the same code (decoded frames, its depth converted to the millimetres a
           RealSense delivers, its states); actions are not sent but scored against the recorded ones and against the
           policy run on the offline lift's forecasts with the same noise (the cost of going live)

  robot:   python run_live.py --task B --policy physgo [--go_home] [--dry_run] [--steps 600] [--max_rel 20]
           (port, calibration id, RealSense serial and wrist camera are found by devices.py; override with --port,
           --robot_id, --top_serial, --wrist_cam)
  replay:  python run_live.py --task B --policy physgo --source replay --episodes 1,2 [--data /dev/shm/data/so101_v2]
Outputs in --out (default ./live_runs/<time>_<task>_<policy>_<source>): steps*.npz (actions, states, timings,
visibility), summary.json.
"""
import argparse
import glob
import json
import os
import signal
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_common as C  # noqa: E402


# ------------------------------------------------------------------------------------------------------------ sources
class RobotSource:
    def __init__(self, args, calibration_dir=None):
        from pathlib import Path

        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        cams = {"top": RealSenseCameraConfig(serial_number_or_name=args.top_serial, fps=30, width=640, height=360,
                                             use_depth=True)}
        if args.wrist_cam.startswith("realsense:"):
            cams["wrist"] = RealSenseCameraConfig(serial_number_or_name=args.wrist_cam.split(":", 1)[1], fps=30,
                                                  width=640, height=480)
        else:
            w = args.wrist_cam
            cams["wrist"] = OpenCVCameraConfig(index_or_path=int(w) if w.isdigit() else Path(w), fps=30, width=640,
                                               height=480)
        cfg = SO101FollowerConfig(port=args.port, id=args.robot_id, calibration_dir=calibration_dir, cameras=cams,
                                  use_degrees=True, max_relative_target=args.max_rel if args.max_rel > 0 else None,
                                  disable_torque_on_disconnect=not args.keep_torque)
        self.robot = SO101Follower(cfg)
        self.robot.connect()
        self.dry = args.dry_run
        self.T = args.steps

    def get(self, t):
        o = self.robot.get_observation()
        state = np.array([o[f"{j}.pos"] for j in C.JOINTS], np.float64)
        return dict(top=np.asarray(o["top"]), wrist=np.asarray(o["wrist"]),
                    depth_mm=np.asarray(o["top_depth"]).squeeze(), state=state)

    def get_state(self, t):
        """The joint state alone (motor bus, no cameras): chunked mode, world-model frames between planning frames."""
        r = self.robot
        pos = r.bus.sync_read("Present_Position", num_retry=r.config.num_read_retries)
        return np.array([pos[j] for j in C.JOINTS], np.float64)

    def send(self, t, action):
        if not self.dry:
            return self.robot.send_action({f"{j}.pos": float(a) for j, a in zip(C.JOINTS, action)})

    def go_home(self, target, seconds=4.0):
        """Move slowly (linear in joint space) to the median first state of the recorded episodes."""
        s0 = self.get(0)["state"]
        n = int(seconds * C.FPS)
        for i in range(1, n + 1):
            self.send(0, s0 + (np.asarray(target) - s0) * i / n)
            time.sleep(1.0 / C.FPS)

    def close(self):
        self.robot.disconnect()


class ReplaySource:
    """A recorded episode through the live code; the offline lift / forecast / policy inputs for comparison."""

    def __init__(self, name, ep, data_root, fc_rel="so101/forecast_rel4lab"):
        import pandas as pd
        import lift_so101 as L
        root = f"{data_root}/{name}"
        info, eps, _ = L.load_meta(root)
        row = eps.loc[ep]
        n30 = int(row["length"])
        self.top = L.decode_episode(root, row, "observation.images.top", n30)
        self.wrist = L.decode_episode(root, row, "observation.images.wrist", n30)
        q = L.decode_episode(root, row, "observation.images.top_depth", n30, depth=True)
        di = info["features"]["observation.images.top_depth"]["info"]
        # what a RealSense hands over: uint16 millimetres, 0 = no reading
        self.depth_mm = np.stack([np.nan_to_num(np.rint(L.depth_from_codes(x, di) * 1000), nan=0).astype(np.uint16) for x in q])
        self.q = q
        d = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
                       for f in sorted(glob.glob(f"{root}/data/chunk-*/file-*.parquet"))])
        st = d[d["episode_index"] == ep].sort_values("frame_index")
        self.S = np.stack(st["observation.state"].values)[::2].astype(np.float64)
        self.A30 = np.stack(st["action"].values).astype(np.float64)    # chunked mode: scored at 30 fps
        self.A = self.A30[::2]
        lf = C.asset(f"so101/lifted_v2/{name}/episode_{ep:04d}.npz")
        self.lift = np.load(lf) if os.path.exists(lf) else None
        fcf = C.asset(f"{fc_rel}/{name}_demo/demo_{ep}.npy")      # the policy's training forecasts (offline lift)
        self.fc_pre = np.load(fcf) if os.path.exists(fcf) else None
        Ts = [len(self.top), len(self.wrist), len(self.depth_mm), len(self.S)]
        if self.lift is not None:
            Ts.append(len(self.lift["npts"]))
        if self.fc_pre is not None:
            Ts.append(len(self.fc_pre))
        self.T = min(Ts)

    def get(self, t):
        return dict(top=self.top[t], wrist=self.wrist[t], depth_mm=self.depth_mm[t], state=self.S[t])

    def get_state(self, t):
        return self.S[t]

    def send(self, t, action):
        return None

    def close(self):
        pass


# ---------------------------------------------------------------------------------------------------------- episode
def run_episode(src, lifter, pol, args, ep_seed, log_extra=None):
    """One episode; returns per-step records."""
    pol.reset()
    if lifter is not None:
        lifter.reset(seed=ep_seed)
    recs = dict(action=[], state=[], npts=[], t_sam=[], t_lift=[], t_lift_points=[], t_lift_fps=[], t_filt=[], t_wm=[],
                t_pol=[], t_step=[], overrun=[])
    extra = dict(a_off=[], fc_err=[], cen_err=[], vis_live=[], vis_off=[], codes_diff=[])
    period = 1.0 / C.FPS
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    t_next = time.perf_counter()
    for t in range(src.T):
        if stop["flag"]:
            print("stopped (Ctrl-C)", flush=True)
            break
        t0 = time.perf_counter()
        ob = src.get(t)
        fc, tl = None, dict(sam=0.0, lift=0.0, filters=0.0, lift_points=0.0, lift_fps=0.0)
        if lifter is not None:
            from live_lift import code_depth
            codes = code_depth(ob["depth_mm"], lifter.depth_info)
            gauss, npts, tl = lifter.step(ob["top"], codes, ob["state"])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fc = pol.forecast(gauss, npts)
            torch.cuda.synchronize()
            t_wm = time.perf_counter() - t1
            recs["npts"].append(npts.copy())
            if args.scene_every and t % args.scene_every == 0 and args.out_dir:
                import cv2
                cv2.imwrite(f"{args.out_dir}/scene_{ep_seed}_{t:04d}.png", cv2.cvtColor(lifter.scene_image(ob["top"]), cv2.COLOR_RGB2BGR))
        else:
            t_wm = 0.0
        t1 = time.perf_counter()
        seed = ep_seed * 100003 + t
        a = pol.act(ob["top"], ob["wrist"], ob["state"], fc, seed)
        torch.cuda.synchronize()
        t_pol = time.perf_counter() - t1
        src.send(t, a)
        recs["action"].append(a); recs["state"].append(ob["state"])
        recs["t_sam"].append(tl["sam"]); recs["t_lift"].append(tl["lift"]); recs["t_filt"].append(tl["filters"])
        recs["t_lift_points"].append(tl["lift_points"]); recs["t_lift_fps"].append(tl["lift_fps"])
        recs["t_wm"].append(t_wm); recs["t_pol"].append(t_pol); recs["t_step"].append(time.perf_counter() - t0)
        if isinstance(src, ReplaySource):
            if pol.use_fc and src.fc_pre is not None:     # same policy and noise on the offline lift's forecast
                extra["a_off"].append(pol.act(ob["top"], ob["wrist"], ob["state"], src.fc_pre[t], seed))
                extra["fc_err"].append(float(np.abs(fc - src.fc_pre[t]).mean()))
            if lifter is not None and src.lift is not None:
                no = src.lift["npts"][t]
                go = src.lift["gauss"][t].astype(np.float32)
                vl, vo = npts >= 16, no >= 16
                both = vl & vo
                ce = np.full(len(npts), np.nan)
                ce[both] = np.linalg.norm(gauss[both, :, :3].mean(1) - go[both, :, :3].mean(1), axis=1)
                extra["cen_err"].append(ce); extra["vis_live"].append(vl); extra["vis_off"].append(vo)
                extra["codes_diff"].append(float(np.mean(codes != src.q[t])))
        else:
            # robot: hold the 15 fps rate when the step is faster; otherwise run as fast as the model allows
            t_next += period
            dt = t_next - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
                recs["overrun"].append(0.0)
            else:
                recs["overrun"].append(-dt)
                t_next = time.perf_counter()
        if t % 25 == 0:
            msg = (f"t {t:4d} step {1000 * recs['t_step'][-1]:6.0f} ms (sam {1000 * tl['sam']:5.0f} lift "
                   f"{1000 * (tl['lift'] + tl['filters']):4.0f} wm {1000 * t_wm:4.0f} policy {1000 * t_pol:4.0f})")
            if lifter is not None:
                msg += " visible " + "".join("x" if n >= 16 else "." for n in recs["npts"][-1])
            print(msg, flush=True)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return recs, extra


def summarize(recs, extra, src, lo_hi=None, demo_sd=None):
    s = {}
    for k in ("t_sam", "t_lift", "t_lift_points", "t_lift_fps", "t_filt", "t_wm", "t_pol", "t_step"):
        v = np.array(recs[k][1:]) * 1000     # the first step includes warm-up
        if len(v):
            s[f"{k}_ms_median"] = round(float(np.median(v)), 1)
            s[f"{k}_ms_p90"] = round(float(np.percentile(v, 90)), 1)
    s["steps"] = len(recs["action"])
    if recs["overrun"]:
        s["rate_hz"] = round(1.0 / max(np.median(recs["t_step"][1:]), 1e-6), 2)
    if isinstance(src, ReplaySource):
        A = np.array(recs["action"])
        R = src.A[:len(A)]
        err = np.abs(A - R)
        s.update(arm_deg=float(err[:, :5].mean()), grip=float(err[:, 5].mean()))
        if lo_hi is not None and demo_sd is not None:
            half = (lo_hi[1] - lo_hi[0]) / 2
            s["mse_z"] = float(np.mean(((A - R) / half / demo_sd) ** 2))
        if extra["a_off"]:
            Ao = np.array(extra["a_off"])
            eo = np.abs(Ao - R)
            s.update(offline_lift_arm_deg=float(eo[:, :5].mean()), offline_lift_grip=float(eo[:, 5].mean()),
                     live_vs_offline_action_arm_deg=float(np.abs(A - Ao)[:, :5].mean()),
                     forecast_abs_err_mean=float(np.mean(extra["fc_err"])))
            if demo_sd is not None:
                s["offline_lift_mse_z"] = float(np.mean(((Ao - R) / ((lo_hi[1] - lo_hi[0]) / 2) / demo_sd) ** 2))
        if extra["cen_err"]:
            ce, vl, vo = np.array(extra["cen_err"]), np.array(extra["vis_live"]), np.array(extra["vis_off"])
            s["objects"] = {}
            for k in range(1, ce.shape[1]):
                s["objects"][str(k)] = dict(vis_live=round(float(vl[:, k].mean()), 3), vis_offline=round(float(vo[:, k].mean()), 3),
                                            vis_agree=round(float((vl[:, k] == vo[:, k]).mean()), 3),
                                            centroid_mm_median=round(float(np.nanmedian(ce[:, k]) * 1000), 1) if np.isfinite(ce[:, k]).any() else None,
                                            centroid_mm_p90=round(float(np.nanpercentile(ce[:, k], 90) * 1000), 1) if np.isfinite(ce[:, k]).any() else None)
            s["depth_codes_changed_frac"] = round(float(np.mean(extra["codes_diff"])), 5)
    return s


# ------------------------------------------------------------------------------------------- chunked mode (2026-11-11)
class WMFeeder:
    """The world-model frames between planning frames: gripper from the joint state (LiveLifter.gripper_step), objects
    unseen, one filter step each. On the robot a background thread does the work so the 30 Hz action stream never waits
    for it; in replay it runs inline. drain() returns once every frame handed over so far is filtered (the planning
    frame's full lift must come after them). Lifter and forecaster are used by one thread at a time: the worker between
    planning frames, the main thread (after drain) at them."""

    def __init__(self, lifter, pol, threaded):
        import queue
        import threading
        self.lifter, self.pol, self.threaded = lifter, pol, threaded
        self.times, self.err = [], None
        if threaded:
            self.q = queue.Queue()
            self.th = threading.Thread(target=self._run, daemon=True)
            self.th.start()

    def _do(self, state):
        t0 = time.perf_counter()
        g, n, _ = self.lifter.gripper_step(state)
        self.pol.observe(g, n)     # filter update only: no forecast is read at this frame
        torch.cuda.synchronize()
        self.times.append(time.perf_counter() - t0)

    def _run(self):
        while True:
            s = self.q.get()
            try:
                if s is not None and self.err is None:
                    self._do(s)
            except Exception as e:   # re-raised in the main thread by drain()
                self.err = e
            finally:
                self.q.task_done()
            if s is None:
                break

    def put(self, state):
        if self.threaded:
            self.q.put(np.array(state, np.float64))
        else:
            self._do(state)

    def drain(self):
        if self.threaded:
            self.q.join()
        if self.err is not None:
            raise self.err

    def close(self):
        if self.threaded:
            self.q.put(None)
            self.th.join(timeout=10)


def run_episode_chunked(src, lifter, pol, args, ep_seed):
    """One episode of a chunked policy at 30 Hz (pol.rate actions per 15 fps frame); returns per-tick records."""
    pol.reset()
    if lifter is not None:
        lifter.reset(seed=ep_seed)
    ex = int(args.exec_steps or pol.chunk)
    assert 0 < ex <= pol.chunk and ex % pol.rate == 0, "--exec_steps: a multiple of the action rate, at most the chunk"
    robot = not isinstance(src, ReplaySource)
    feeder = WMFeeder(lifter, pol, threaded=robot) if lifter is not None else None
    recs = dict(action=[], plan_tick=[], plan_state=[], npts=[], t_drain=[], t_sam=[], t_lift=[], t_filt=[], t_wm=[],
                t_pol=[], t_plan=[], overrun=[])
    extra = dict(chunk=[], chunk_off=[], fc_err=[], cen_err=[], vis_live=[], vis_off=[])
    period = 1.0 / (C.FPS * pol.rate)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    todo = []
    t_next = time.perf_counter()
    try:
        for i in range(src.T * pol.rate):
            if stop["flag"]:
                print("stopped (Ctrl-C)", flush=True)
                break
            t = i // pol.rate                       # 15 fps frame
            if not todo:                            # planning frame (ex is a multiple of rate: always a 15 fps frame)
                t0 = time.perf_counter()
                if feeder is not None:
                    feeder.drain()
                t_drain = time.perf_counter() - t0
                ob = src.get(t)
                fc, tl, t_wm = None, dict(sam=0.0, lift=0.0, filters=0.0), 0.0
                if lifter is not None:
                    from live_lift import code_depth
                    codes = code_depth(ob["depth_mm"], lifter.depth_info)
                    gauss, npts, tl = lifter.step(ob["top"], codes, ob["state"])
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    fc = pol.forecast(gauss, npts)
                    torch.cuda.synchronize()
                    t_wm = time.perf_counter() - t1
                    recs["npts"].append(npts.copy())
                    if args.scene_every and len(recs["plan_tick"]) % args.scene_every == 0 and args.out_dir:
                        import cv2
                        cv2.imwrite(f"{args.out_dir}/scene_{ep_seed}_{t:04d}.png",
                                    cv2.cvtColor(lifter.scene_image(ob["top"]), cv2.COLOR_RGB2BGR))
                t1 = time.perf_counter()
                seed = ep_seed * 100003 + t
                ch = pol.plan(ob["top"], ob["wrist"], ob["state"], fc, seed)
                torch.cuda.synchronize()
                t_pol = time.perf_counter() - t1
                todo = list(ch[:ex])
                recs["plan_tick"].append(i); recs["plan_state"].append(ob["state"])
                recs["t_drain"].append(t_drain); recs["t_sam"].append(tl["sam"])
                recs["t_lift"].append(tl["lift"]); recs["t_filt"].append(tl["filters"]); recs["t_wm"].append(t_wm)
                recs["t_pol"].append(t_pol); recs["t_plan"].append(time.perf_counter() - t0)
                if not robot:
                    extra["chunk"].append(ch)
                    if pol.use_fc and src.fc_pre is not None:   # same policy and noise on the offline lift's forecast
                        extra["chunk_off"].append(pol.plan(ob["top"], ob["wrist"], ob["state"], src.fc_pre[t], seed))
                        extra["fc_err"].append(float(np.abs(fc - src.fc_pre[t]).mean()))
                    if lifter is not None and src.lift is not None:
                        no = src.lift["npts"][t]
                        go = src.lift["gauss"][t].astype(np.float32)
                        vl, vo = npts >= 16, no >= 16
                        both = vl & vo
                        ce = np.full(len(npts), np.nan)
                        ce[both] = np.linalg.norm(gauss[both, :, :3].mean(1) - go[both, :, :3].mean(1), axis=1)
                        extra["cen_err"].append(ce); extra["vis_live"].append(vl); extra["vis_off"].append(vo)
                msg = (f"t {t:4d} plan {len(recs['plan_tick']):3d}: {1000 * recs['t_plan'][-1]:5.0f} ms (wait wm "
                       f"{1000 * t_drain:4.0f} sam {1000 * tl['sam']:4.0f} lift {1000 * (tl['lift'] + tl['filters']):4.0f} "
                       f"wm {1000 * t_wm:3.0f} policy {1000 * t_pol:4.0f})")
                if lifter is not None:
                    msg += " visible " + "".join("x" if n >= 16 else "." for n in recs["npts"][-1])
                    if feeder.times:
                        msg += f"  wm frames {1000 * np.median(feeder.times[-ex // pol.rate:]):3.0f} ms"
                print(msg, flush=True)
            elif feeder is not None and i % pol.rate == 0:   # a 15 fps frame between planning frames
                feeder.put(src.get_state(t))
            a = todo.pop(0)
            src.send(t, a)
            recs["action"].append(a)
            if robot:   # 30 Hz; after a planning pause the stream continues from now (no catch-up burst)
                t_next += period
                dt = t_next - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                    recs["overrun"].append(0.0)
                else:
                    recs["overrun"].append(-dt)
                    t_next = time.perf_counter()
    finally:
        if feeder is not None:
            feeder.close()
        signal.signal(signal.SIGINT, signal.default_int_handler)
    recs["t_wm_frame"] = list(feeder.times) if feeder is not None else []
    return recs, extra


def summarize_chunked(recs, extra, src, rate, half):
    s = {}
    for k in ("t_drain", "t_sam", "t_lift", "t_filt", "t_wm", "t_pol", "t_plan", "t_wm_frame"):
        v = np.array(recs[k][1:]) * 1000     # the first plan includes warm-up
        if len(v):
            s[f"{k}_ms_median"] = round(float(np.median(v)), 1)
            s[f"{k}_ms_p90"] = round(float(np.percentile(v, 90)), 1)
    s["ticks"], s["plans"] = len(recs["action"]), len(recs["plan_tick"])
    if recs["overrun"]:
        ov = np.array(recs["overrun"])
        s["late_ticks_frac"] = round(float(np.mean(ov > 0.005)), 4)     # > 5 ms behind the 30 Hz clock
        s["late_ms_p90_between_plans"] = round(float(np.percentile(np.delete(ov, recs["plan_tick"]) * 1000, 90)), 1) \
            if len(ov) > len(recs["plan_tick"]) else None
    if isinstance(src, ReplaySource):
        A = np.array(recs["action"])
        R = src.A30[:len(A)]
        n = min(len(A), len(R))
        A, R = A[:n], R[:n]
        err = np.abs(A - R)
        s.update(arm_deg=float(err[:, :5].mean()), grip=float(err[:, 5].mean()),
                 jitter_deg=float(np.abs(np.diff(A, axis=0))[:, :5].mean()),
                 rec_jitter_deg=float(np.abs(np.diff(R, axis=0))[:, :5].mean()),
                 arm_deg_scaled=float((err / half)[:, :5].mean()))
        if extra["chunk_off"]:      # the same executed stream from the offline lift's forecasts
            Ao = np.full_like(A, np.nan)
            for j, i0 in enumerate(recs["plan_tick"]):
                m = min(len(extra["chunk_off"][j]), n - i0, (recs["plan_tick"][j + 1] - i0) if j + 1 < len(recs["plan_tick"]) else n - i0)
                Ao[i0:i0 + m] = extra["chunk_off"][j][:m]
            ok = np.isfinite(Ao).all(1)
            eo = np.abs(Ao[ok] - R[ok])
            s.update(offline_lift_arm_deg=float(eo[:, :5].mean()), offline_lift_grip=float(eo[:, 5].mean()),
                     live_vs_offline_action_arm_deg=float(np.abs(A[ok] - Ao[ok])[:, :5].mean()),
                     forecast_abs_err_mean=float(np.mean(extra["fc_err"])))
        if extra["cen_err"]:
            ce, vl, vo = np.array(extra["cen_err"]), np.array(extra["vis_live"]), np.array(extra["vis_off"])
            s["objects_at_plans"] = {}
            for k in range(1, ce.shape[1]):
                s["objects_at_plans"][str(k)] = dict(
                    vis_live=round(float(vl[:, k].mean()), 3), vis_offline=round(float(vo[:, k].mean()), 3),
                    vis_agree=round(float((vl[:, k] == vo[:, k]).mean()), 3),
                    centroid_mm_median=round(float(np.nanmedian(ce[:, k]) * 1000), 1) if np.isfinite(ce[:, k]).any() else None)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="A (cube out of pot), B (drawer + stack), C (drawer placing) or a dataset name")
    ap.add_argument("--policy", default="physgo", choices=sorted(C.POLICIES))
    ap.add_argument("--ckpt", default=None, help="checkpoint name in the run (default: the chosen one, live_common.POLICIES)")
    ap.add_argument("--source", default="robot", choices=("robot", "replay"))
    ap.add_argument("--num_steps", type=int, default=0, help="flow-matching steps (0: the trained 10; 5 is faster)")
    ap.add_argument("--exec_steps", type=int, default=0,
                    help="chunked policies: 30 fps targets executed per chunk before the next plan (0: the whole chunk, "
                         "as LeRobot and the offline selection; must be even)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=1800, help="SAM video capacity (steps per episode)")
    ap.add_argument("--sam_ckpt", default=None, help="SAM 3.1 checkpoint file (default: HF facebook/sam3.1, gated)")
    ap.add_argument("--mem_frac", type=float, default=0.0, help="cap this process's GPU memory (shared GPU)")
    ap.add_argument("--new_det_thresh", type=float, default=None,
                    help="SAM 3.1 score to start a new track (default 0.65); lower finds objects that appear later sooner")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scene_every", type=int, default=0,
                    help="save the scene check image every N steps, chunked: every N plans (0: robot warm-up only)")
    # robot
    ap.add_argument("--port", default="auto", help="follower arm serial port (auto: the only /dev/ttyACM* or ttyUSB*)")
    ap.add_argument("--robot_id", default="auto", help="LeRobot calibration id of the follower (auto: the only file)")
    ap.add_argument("--top_serial", default="auto", help="RealSense serial of the top camera (auto: the only RealSense)")
    ap.add_argument("--wrist_cam", default="auto", help="OpenCV index / path, or realsense:<serial> (auto: first USB camera)")
    ap.add_argument("--steps", type=int, default=600, help="steps per episode (15 fps; recorded episodes 23-35 s)")
    ap.add_argument("--max_rel", type=float, default=20.0, help="LeRobot max_relative_target per step (0: off)")
    ap.add_argument("--go_home", action="store_true", help="first move slowly to the median start pose of the recordings")
    ap.add_argument("--dry_run", action="store_true", help="run everything, send nothing to the motors")
    ap.add_argument("--keep_torque", action="store_true", help="keep motor torque on at exit (LeRobot default: off)")
    # replay
    ap.add_argument("--episodes", default="", help="replay: comma list of episode ids")
    ap.add_argument("--replay_steps", type=int, default=0, help="replay: first N steps only (0: the whole episode)")
    ap.add_argument("--data", default="/dev/shm/data/so101_v2")
    args = ap.parse_args()
    calib_dir = None
    if args.source == "robot":   # find the arm, its calibration and the cameras first (seconds, before the models load)
        import devices
        calib_dir = devices.resolve(args)
    name = C.dataset_name(args.task)
    meta = C.load_meta()
    if args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    out = args.out or f"live_runs/{time.strftime('%m%d_%H%M%S')}_{meta[name]['letter']}_{args.policy}_{args.source}"
    os.makedirs(out, exist_ok=True)
    args.out_dir = out

    from live_policy import LivePolicy
    pol = LivePolicy(args.policy, name, meta, ckpt=args.ckpt, num_steps=args.num_steps)
    lifter = None
    if pol.use_fc:
        from live_lift import LiveLifter
        from live_sam import LiveSegmenter
        seg = LiveSegmenter([p[1] for p in meta[name]["prompts"]], ckpt=args.sam_ckpt, max_frames=args.max_frames,
                            new_det_thresh=args.new_det_thresh)
        lifter = LiveLifter(name, seg, meta)
    print(f"task {meta[name]['letter']} '{meta[name]['task']}', policy {args.policy} {pol.ckpt}, forecast {pol.use_fc}"
          f" (world model {pol.wm}), images {pol.img} px, "
          + (f"chunks of {pol.chunk} at {C.FPS * pol.rate} Hz, {args.exec_steps or pol.chunk} executed per plan"
             if pol.chunked else "one action per 15 fps step"), flush=True)
    episode = run_episode_chunked if pol.chunked else run_episode

    results = {}
    if args.source == "robot":
        src = RobotSource(args, calibration_dir=calib_dir)
        try:
            if args.go_home:
                src.go_home(meta[name]["start_state"])
            # warm-up: one full step on a live frame, nothing sent (the first SAM / policy call sets up its kernels,
            # ~3 s); the episode then starts from a reset state
            ob = src.get(0)
            fc = None
            if lifter is not None:
                from live_lift import code_depth
                lifter.reset(seed=args.seed)
                g_, n_, _ = lifter.step(ob["top"], code_depth(ob["depth_mm"], lifter.depth_info), ob["state"])
                pol.reset()
                fc = pol.forecast(g_, n_)
                print("warm-up: objects seen " + " ".join(f"{nm}:{'yes' if n >= 16 else 'NO'}" for nm, n in zip(lifter.names, n_)), flush=True)
                import cv2
                cv2.imwrite(f"{out}/scene_check.png", cv2.cvtColor(lifter.scene_image(ob["top"]), cv2.COLOR_RGB2BGR))
                print(f"scene check image: {out}/scene_check.png (red dots = gripper model; must lie on the real gripper)", flush=True)
                if pol.chunked:   # the between-plans frame path too (its kernels), then the episode resets
                    g_, n_, _ = lifter.gripper_step(src.get_state(0))
                    pol.observe(g_, n_)
            if pol.chunked:
                pol.plan(ob["top"], ob["wrist"], ob["state"], fc, 0)
            else:
                pol.act(ob["top"], ob["wrist"], ob["state"], fc, 0)
            input("Ready: scene set up as in the recordings. Press ENTER to start the episode (Ctrl-C stops it)...")
            recs, extra = episode(src, lifter, pol, args, args.seed)
        finally:
            src.close()
        if pol.chunked:
            half = (np.array(pol.hi) - np.array(pol.lo)) / 2
            results["robot"] = summarize_chunked(recs, extra, src, pol.rate, half)
        else:
            results["robot"] = summarize(recs, extra, src)
        np.savez_compressed(f"{out}/steps.npz", **{k: np.array(v) for k, v in recs.items() if len(v)})
    else:
        import h5py
        nrm = json.load(open(C.asset("so101/policy_h5/norm.json")))
        lo_hi = (np.array(nrm["min"]), np.array(nrm["max"]))
        dfiles = sorted(glob.glob(C.asset("so101/policy_h5/*_demo/bc_train_5/*.hdf5")))
        # mse_z as eval_policy_so101 (z-score with the 15 demo actions' std); skipped where the demo files are absent
        sd = (np.concatenate([h5py.File(f, "r")["root/actions"][:] for f in dfiles]).std(0) + 1e-8) if dfiles else None
        fd = str(pol.cfg.forecast_dir)     # the policy's training forecasts, in the assets tree
        fc_rel = os.path.relpath(fd, "/dev/shm/physgo") if fd.startswith("/dev/shm/physgo/") else fd
        for ep in [int(e) for e in args.episodes.split(",") if e]:
            src = ReplaySource(name, ep, args.data, fc_rel)
            if args.replay_steps > 0:
                src.T = min(src.T, args.replay_steps)
            print(f"--- replay episode {ep}: {src.T} steps", flush=True)
            recs, extra = episode(src, lifter, pol, args, ep)
            if pol.chunked:
                results[str(ep)] = summarize_chunked(recs, extra, src, pol.rate, (lo_hi[1] - lo_hi[0]) / 2)
                extra = {k: v for k, v in extra.items() if k not in ("chunk", "chunk_off")} | \
                        {k: np.stack(v) for k, v in extra.items() if k in ("chunk", "chunk_off") and v}
            else:
                results[str(ep)] = summarize(recs, extra, src, lo_hi, sd)
            print(json.dumps(results[str(ep)]), flush=True)
            np.savez_compressed(f"{out}/steps_ep{ep}.npz", **{k: np.array(v) for k, v in recs.items() if len(v)},
                                **{f"x_{k}": np.array(v) for k, v in extra.items() if len(v)})
            json.dump(dict(task=name, policy=args.policy, ckpt=pol.ckpt, results=results), open(f"{out}/summary.json", "w"), indent=1)
    if lifter is not None:
        results["filter_drops"] = {k: v.tolist() for k, v in lifter.stats.items()}
    json.dump(dict(task=name, policy=args.policy, ckpt=pol.ckpt, args=vars(args), results=results),
              open(f"{out}/summary.json", "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
