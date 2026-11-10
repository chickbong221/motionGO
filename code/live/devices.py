"""
devices.py: find the SO-101 rig's hardware so run_live.py needs no device arguments (one arm, one RealSense top camera,
one USB wrist camera). Each finder returns the value or raises with the candidates it saw.
  port        the follower arm's serial port: the only /dev/ttyACM* or /dev/ttyUSB* device
  robot_id    the follower's LeRobot calibration (the one the recordings used): the only <id>.json under
              $HF_LEROBOT_CALIBRATION/robots/so_follower (or the older so101_follower folder)
  top_serial  the only connected RealSense
  wrist_cam   the first video capture device that is not a RealSense node
  python devices.py      prints what run_live.py will use
"""
import glob
import os
from pathlib import Path


def find_port():
    ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if len(ports) == 1:
        return ports[0]
    raise RuntimeError(f"expected one arm serial port, found {ports or 'none'}: unplug the leader arm, or pass --port")


def calibration_root():
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
    return Path(HF_LEROBOT_CALIBRATION) / "robots"


def find_robot_id():
    """-> (id, calibration_dir). Never calibrates: a new calibration would change the joint values the policy saw."""
    root = calibration_root()
    files = sorted(glob.glob(str(root / "so_follower" / "*.json"))) + \
        sorted(glob.glob(str(root / "so101_follower" / "*.json")))
    if len(files) == 1:
        f = Path(files[0])
        return f.stem, f.parent
    raise RuntimeError(f"expected one follower calibration file under {root}/so_follower, found "
                       f"{[str(f) for f in files] or 'none'}: pass --robot_id (and copy the recording PC's calibration "
                       f"file there if it is missing)")


def find_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise RuntimeError("pyrealsense2 is missing: pip install pyrealsense2 (install.sh does it)") from e
    serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
    if len(serials) == 1:
        return serials[0]
    raise RuntimeError(f"expected one RealSense, found {serials or 'none'}: pass --top_serial")


def find_wrist_cam(sysfs="/sys/class/video4linux"):
    """The first V4L2 capture node (index 0 of its device) whose name is not a RealSense."""
    found = []
    for d in sorted(glob.glob(f"{sysfs}/video*"), key=lambda p: int(p.rsplit("video", 1)[1])):
        name = Path(d, "name").read_text().strip() if os.path.exists(Path(d, "name")) else ""
        idx = Path(d, "index").read_text().strip() if os.path.exists(Path(d, "index")) else "0"
        found.append((os.path.basename(d), name))
        if "realsense" in name.lower() or idx != "0":
            continue
        return f"/dev/{os.path.basename(d)}"
    raise RuntimeError(f"no USB wrist camera found among {found or 'no video devices'}: pass --wrist_cam")


def resolve(args):
    """Fill the 'auto' device arguments of run_live.py in place and print them."""
    calib_dir = None
    if args.port == "auto":
        args.port = find_port()
    if args.robot_id == "auto":
        args.robot_id, calib_dir = find_robot_id()
    else:
        for sub in ("so_follower", "so101_follower"):
            p = calibration_root() / sub / f"{args.robot_id}.json"
            if p.is_file():
                calib_dir = p.parent
                break
        if calib_dir is None:
            raise RuntimeError(f"no calibration file {args.robot_id}.json under {calibration_root()}")
    if args.top_serial == "auto":
        args.top_serial = find_realsense()
    if args.wrist_cam == "auto":
        args.wrist_cam = find_wrist_cam()
    print(f"arm port {args.port} | calibration {calib_dir}/{args.robot_id}.json | top RealSense {args.top_serial} | "
          f"wrist camera {args.wrist_cam}", flush=True)
    return calib_dir


if __name__ == "__main__":
    for nm, fn in (("port", find_port), ("robot_id", find_robot_id), ("top_serial", find_realsense),
                   ("wrist_cam", find_wrist_cam)):
        try:
            print(f"{nm:10s} {fn()}")
        except Exception as e:  # noqa: BLE001
            print(f"{nm:10s} NOT FOUND: {e}")
