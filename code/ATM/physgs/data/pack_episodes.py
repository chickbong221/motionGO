"""
Pack episode files into a memory-mappable training directory.

    python -m physgs.data.pack_episodes --src data/physai_episodes --out data/packed/physai --max_objects 24
    python -m physgs.data.pack_episodes --src data/libero_episodes/libero_spatial --out data/packed/libero_spatial \
        --max_objects 16 --native_order
"""
import argparse
import glob
import os

from physgs.data.format import pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory (or comma-separated directories) searched recursively for *.npz episodes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_objects", type=int, default=24)
    ap.add_argument("--native_order", action="store_true", help="keep object order (LIBERO; needed for online eval)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_points", type=int, default=0, help="keep the first N stored Gaussians per object (FPS order); 0 = all")
    args = ap.parse_args()
    files = sorted(f for src in args.src.split(",") for f in glob.glob(os.path.join(src, "**", "*.npz"), recursive=True))
    files = [f for f in files if not f.endswith(".tmp.npz")]
    if args.limit:
        files = files[:args.limit]
    print(f"packing {len(files)} episodes -> {args.out}")
    pack(files, args.out, args.max_objects, sort_by_visibility=not args.native_order, num_points=args.num_points or None)


if __name__ == "__main__":
    main()
