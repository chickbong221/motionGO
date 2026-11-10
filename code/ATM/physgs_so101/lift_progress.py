"""lift_progress.py (2026-11-07): progress of lift_v2_run.sh MODE=all from the worker logs: episodes done per dataset,
mean seconds per episode, and flagged episodes (an object visible in < 10% of frames, > 30% of an object's frames
rejected by the consistency filter, or the table more than 15 mm from z = 0 in the base frame).
  python lift_progress.py /dev/shm/physgo/so101/lifted_v2/logs"""
import glob
import json
import sys

rows = []
for f in sorted(glob.glob(f"{sys.argv[1]}/all_*.log")) + sorted(glob.glob(f"{sys.argv[1]}/test3_*.log")):
    for line in open(f, errors="ignore"):
        if line.startswith('{"dataset"'):
            rows.append(json.loads(line))
def task(d):
    return "A" if "pot" in d else ("B" if "open_drawer" in d else "C")


by = {}
for r in rows:
    by.setdefault(task(r["dataset"]), []).append(r)
print(f"{len(rows)}/159 episodes; " + ", ".join(f"{k} {len(by[k])}" for k in sorted(by)) +
      (f"; mean {sum(r['seconds'] for r in rows) / len(rows):.0f} s/episode" if rows else ""))
for r in rows:
    flags = []
    for o, v in r["visible"].items():
        if v < 0.1:
            flags.append(f"{o} visible {v:.0%}")
    for o, n in r.get("inconsistent_frames", {}).items():
        if n > 0.3 * r["T"]:
            flags.append(f"{o} {n} inconsistent frames")
    if abs(r["table_z_mm"]) > 15:
        flags.append(f"table z {r['table_z_mm']} mm")
    if flags:
        print(f"  FLAG {task(r['dataset'])} ep {r['ep']}: " + "; ".join(flags))
