"""
so101_forecasts.py (2026-11-07): PhysGO forecasts of every SO-101 episode for the policy, with the original code
(physgs.engine.precompute_forecast_joint.forecast_episodes: filter the Gaussians of steps 0..t, imagine 15 steps;
include_now -> 16 entries; JointWorldModel.forecast_features, 143 features per object) and the adapted world model.
Writes <out>/<dataset>_demo/demo_<ep>.npy, float32 (T, 16, O, 143), where physgs.policy.dataset.ForecastBCDataset
looks for the forecast of <policy_h5>/<dataset>_demo/demo_<ep>.hdf5 (T = the same 15 fps steps).
Run with the training env:  python so101_forecasts.py <packed dir> <model best.pt> <out dir>
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/dev/shm/physgo/code_so101/ATM")
from physgs.data.format import Packed  # noqa: E402
from physgs.engine.precompute_forecast_joint import forecast_episodes, load_model  # noqa: E402


def main():
    root, model_path, out = sys.argv[1:4]
    d = Packed(root)
    files = json.load(open(f"{root}/split.json"))["files"]
    model = load_model(model_path, "cuda")
    todo = [i for i, f in enumerate(files)
            if not os.path.exists(f"{out}/{f.split('/')[0]}_demo/demo_{int(f.split('_')[-1][:-4])}.npy")]
    print(f"{len(todo)} of {len(files)} episodes to forecast", flush=True)
    for s in range(0, len(todo), 16):   # as precompute_forecast_joint.main: fp32, batch 16, 512 points
        eps = todo[s:s + 16]
        fc = forecast_episodes(model, d, eps, 15, "cuda", True, 512)
        for i, a in zip(eps, fc):
            name, ep = files[i].split("/")[0], int(files[i].split("_")[-1][:-4])
            os.makedirs(f"{out}/{name}_demo", exist_ok=True)
            assert a.shape[0] == d.ep_start[i + 1] - d.ep_start[i] and np.isfinite(a).all(), (files[i], a.shape)
            np.save(f"{out}/{name}_demo/demo_{ep}.npy", a.astype(np.float32))
        print(f"{s + len(eps)}/{len(todo)}  last {files[eps[-1]]}  shape {fc[-1].shape}", flush=True)
    json.dump(dict(model=model_path, packed=root, horizon=15, include_now=True, episodes=len(files)),
              open(f"{out}/info.json", "w"), indent=1)
    open(f"{out}/done", "w").close()


if __name__ == "__main__":
    main()
