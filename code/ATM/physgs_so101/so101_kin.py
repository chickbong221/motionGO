"""
so101_kin.py (2026-11-06): SO-101 forward kinematics from the public URDF (TheRobotStudio/SO-ARM100,
Simulation/SO101/so101_new_calib_camera.urdf, Apache-2.0) and surface points of its meshes, in torch (differentiable,
batched) so camera / joint-unit calibration can be fitted by gradient descent.
  joint value v (LeRobot so_follower "<joint>.pos") -> angle = scale_j * v + offset_j (rad); the conversion is fitted
  (calibrate_kin.py), starting from degrees (scale pi/180, offset 0)
URDF convention: T_child = T_parent @ T_origin(xyz, rpy) @ R(axis, angle); rpy = Rz(yaw) Ry(pitch) Rx(roll).
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")   # dataset order
GRIPPER_LINKS = ("gripper_link", "moving_jaw_so101_v1_link")       # the gripper token: fixed jaw + moving jaw
ARM_LINKS = ("base_link", "shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link") + GRIPPER_LINKS


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rpy(*rpy)
    T[:3, 3] = xyz
    return T


def _f(s, n=3):
    return [float(x) for x in (s or " ".join(["0"] * n)).split()]


class SO101:
    def __init__(self, urdf, device="cpu", points_per_link=3000, seed=0):
        import trimesh
        root = ET.parse(urdf).getroot()
        base = os.path.dirname(urdf)
        self.device = device
        self.joints = {}                       # child link -> (parent, T_origin (4,4), axis (3), name, type)
        for j in root.findall("joint"):
            o = j.find("origin")
            a = j.find("axis")
            self.joints[j.find("child").get("link")] = (
                j.find("parent").get("link"),
                torch.tensor(_T(_f(o.get("xyz")), _f(o.get("rpy"))) if o is not None else np.eye(4), dtype=torch.float64, device=device),
                torch.tensor(_f(a.get("xyz")) if a is not None else [0, 0, 1], dtype=torch.float64, device=device),
                j.get("name"), j.get("type"))
        self.limits = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
                       for j in root.findall("joint") if j.find("limit") is not None}
        # surface points of every link's visual meshes, in the link frame (area-weighted, fixed seed)
        rng = np.random.default_rng(seed)
        self.link_pts = {}
        for l in root.findall("link"):
            pts, areas, meshes = [], [], []
            for v in l.findall("visual"):
                g = v.find("geometry/mesh")
                if g is None:
                    continue
                m = trimesh.load(os.path.join(base, g.get("filename")), force="mesh")
                if g.get("scale"):
                    m.apply_scale(_f(g.get("scale")))
                o = v.find("origin")
                m.apply_transform(_T(_f(o.get("xyz")), _f(o.get("rpy"))) if o is not None else np.eye(4))
                meshes.append(m); areas.append(m.area)
            if not meshes:
                continue
            n = [max(50, int(round(points_per_link * a / sum(areas)))) for a in areas]
            for m, k in zip(meshes, n):
                p, _ = trimesh.sample.sample_surface(m, k, seed=int(rng.integers(1 << 30)))
                pts.append(p)
            self.link_pts[l.get("name")] = torch.tensor(np.concatenate(pts), dtype=torch.float64, device=device)

    def angles(self, values, scale, offset):
        """values (B, 6) dataset joint values -> (B, 6) angles (rad); scale / offset (6,) tensors."""
        return values * scale + offset

    def fk(self, angles):
        """angles (B, 6) in JOINTS order -> {link: (B, 4, 4) link -> base}."""
        B = angles.shape[0]
        q = {n: angles[:, i] for i, n in enumerate(JOINTS)}
        out = {"base_link": torch.eye(4, dtype=angles.dtype, device=angles.device).expand(B, 4, 4)}

        def get(link):
            if link in out:
                return out[link]
            parent, To, axis, name, typ = self.joints[link]
            T = get(parent) @ To.to(angles.dtype)
            if typ in ("revolute", "continuous"):
                T = T @ _axis_angle(axis.to(angles.dtype), q[name])
            out[link] = T
            return T

        for link in self.joints:
            get(link)
        return out

    def points(self, angles, links):
        """Surface points of the given links in the base frame: (B, N, 3) and the link index of every point (N,)."""
        T = self.fk(angles)
        P, idx = [], []
        for i, l in enumerate(links):
            p = self.link_pts[l].to(angles.dtype)
            P.append(torch.einsum("bij,nj->bni", T[l][:, :3, :3], p) + T[l][:, None, :3, 3])
            idx.append(torch.full((p.shape[0],), i, dtype=torch.long, device=p.device))
        return torch.cat(P, 1), torch.cat(idx)


def _axis_angle(axis, theta):
    """Rodrigues: axis (3,), theta (B,) -> (B, 4, 4)."""
    a = axis / axis.norm()
    K = torch.zeros(3, 3, dtype=theta.dtype, device=theta.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -a[2], a[1], a[2], -a[0], -a[1], a[0]
    s, c = torch.sin(theta)[:, None, None], torch.cos(theta)[:, None, None]
    R = torch.eye(3, dtype=theta.dtype, device=theta.device) + s * K + (1 - c) * (K @ K)
    T = torch.eye(4, dtype=theta.dtype, device=theta.device).repeat(theta.shape[0], 1, 1)
    T[:, :3, :3] = R
    return T


def pose_from_vec(v):
    """6-vector (axis-angle rotation, translation) -> (4, 4)."""
    rot = _axis_angle(v[:3] / (v[:3].norm() + 1e-12), (v[:3].norm() + 1e-12)[None])[0]
    T = rot.clone()
    T[:3, 3] = v[3:]
    return T
