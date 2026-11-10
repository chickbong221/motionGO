"""
Token parts of a LIBERO scene: the task objects and, optionally, the fixtures with their moving parts.

LIBERO's instance segmentation gives each object and each fixture one id. A fixture's moving parts (drawers,
doors, knobs, buttons) are separate bodies attached by joints; their pixels come from element (geom)
segmentation, mapped geom -> body. Each part then becomes a token like an object: its own Gaussians (from its
pixels) and its own pose / velocity labels (from its body), so a drawer opening is that token moving.

Order of the parts (= token slots after the robot): task objects (objects_dict order), then for each fixture its
static body followed by its moving parts. Tables and the floor get no token (static support surfaces).
Segmentation values: objects and fixture bodies keep their instance id (index in instances_to_ids + 1); moving
parts get new ids after the instances. With fixtures=False the layout is exactly the original object-only one.
"""
import numpy as np

SUPPORT = ("table", "floor")


def subtree(model, root):
    """Ids of `root` and all bodies below it."""
    out = []
    for b in range(model.nbody):
        p = b
        while p != 0 and p != root:
            p = model.body_parentid[p]
        if p == root:
            out.append(b)
    return out


def body_geoms(model, bodies):
    return np.where(np.isin(model.geom_bodyid, bodies))[0]


def part_layout(env, fixtures=False):
    """List of dict(name, kind, body, bodies, geoms, seg_id) for the scene of `env` (a LIBERO env wrapper)."""
    inner, m = env.env, env.sim.model
    inst = list(inner.model.instances_to_ids.keys())
    sid = lambda n: inst.index(n) + 1 if n in inst else -1
    parts = []
    for n in inner.objects_dict:
        b = inner.obj_body_id[n]
        bodies = subtree(m, b)
        parts.append(dict(name=n, kind="object", body=b, bodies=bodies, geoms=body_geoms(m, bodies), seg_id=sid(n)))
    if not fixtures:
        return parts
    nxt = len(inst) + 1
    for f in inner.fixtures_dict:
        if any(s in f for s in SUPPORT) or f not in inner.obj_body_id:
            continue
        fb = inner.obj_body_id[f]
        fsub = subtree(m, fb)
        moving, used = [], set()
        for j in range(m.njnt):
            jn = m.joint_id2name(j) or ""
            lb = int(m.jnt_bodyid[j])
            if not jn.startswith(f) or lb == fb or lb not in fsub or lb in used:
                continue
            lsub = subtree(m, lb)
            used.update(lsub)
            moving.append(dict(name=jn, kind="fixture_part", body=lb, bodies=lsub, geoms=body_geoms(m, lsub), seg_id=nxt))
            nxt += 1
        base = [b for b in fsub if b not in used]
        parts.append(dict(name=f, kind="fixture", body=fb, bodies=base, geoms=body_geoms(m, base), seg_id=sid(f)))
        parts += moving
    return parts


def part_segmentation(instance_seg, element_seg, parts):
    """Instance segmentation with the pixels of each moving fixture part relabelled to that part's id."""
    out = instance_seg.copy()
    for p in parts:
        if p["kind"] == "fixture_part":
            out[np.isin(element_seg, p["geoms"])] = p["seg_id"]
    return out


def has_moving_parts(parts):
    return any(p["kind"] == "fixture_part" for p in parts)
