"""sam3_compile.py: is SAM 3.1 with compile=True faster, with the same masks? Same episode / phrases, both builds."""
import sys
import time

import numpy as np

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM/physgs_so101")
import lift_so101 as L  # noqa: E402

ckpt = sys.argv[1]
name = "so101-left_arm_task_red_cube_out_of_pot_into_the_orange_hollow_cube"
root = f"/dev/shm/data/so101_v2/{name}"
info, eps, _ = L.load_meta(root)
row = eps.loc[12]
top = L.decode_episode(root, row, "observation.images.top", int(row["length"]))
phrases = ["silver pot", "teal robot arm"]
res = {}
for comp in (False, True):
    from sam3.model_builder import build_sam3_predictor
    seg = L.Segmenter.__new__(L.Segmenter)
    t0 = time.time()
    seg.p = build_sam3_predictor(checkpoint_path=ckpt, version="sam3.1", use_fa3=False, compile=comp, warm_up=False)
    print(f"compile={comp}: built in {time.time() - t0:.0f} s", flush=True)
    for rep in range(2 if comp else 1):            # compiled: the first run includes compilation
        for ph in phrases:
            t0 = time.time()
            m = L.best_instance(seg.track(top, ph))
            print(f"  compile={comp} run {rep} '{ph}': {time.time() - t0:.1f} s", flush=True)
            res[(comp, ph)] = m
    del seg
    import torch
    torch.cuda.empty_cache()
for ph in phrases:
    a, b = res[(False, ph)], res[(True, ph)]
    inter = (a & b).sum((1, 2)); uni = (a | b).sum((1, 2))
    iou = np.where(uni > 0, inter / np.maximum(uni, 1), 1.0)
    print(f"'{ph}': mask IoU compiled vs eager: mean {iou.mean():.4f}, min {iou.min():.4f}, frames differing in presence "
          f"{np.mean(a.any((1, 2)) != b.any((1, 2))):.1%}")
