"""
Run one of ATM's own training engines unchanged on a single GPU.

ATM's engines create `Fabric(..., strategy="deepspeed")` for its 4-GPU setup. DeepSpeed is not installed here
and is unnecessary on one GPU, so this launcher swaps in Fabric's default strategy and then executes the
engine module as `__main__` with the remaining command-line arguments (Hydra overrides as usual).

    python -m physgs.baselines.run_atm engine.train_bc --config-name=libero_vilt train_dataset=[...] ...
"""
import runpy
import sys

import lightning.fabric as lf

_fabric_init = lf.Fabric.__init__


def _single_gpu_init(self, *args, **kwargs):
    kwargs["strategy"] = "auto"
    _fabric_init(self, *args, **kwargs)


lf.Fabric.__init__ = _single_gpu_init

# The engines gather rollout results with torch.distributed.all_gather_object, which needs a process group.
# With one process and no group, the gathered list is just this process's own result.
import torch.distributed as dist

_all_gather_object = dist.all_gather_object


def _single_process_all_gather_object(object_list, obj, group=None):
    if not (dist.is_available() and dist.is_initialized()):
        object_list[0] = obj
        return None
    return _all_gather_object(object_list, obj, group)


dist.all_gather_object = _single_process_all_gather_object

# ATM's MetricLogger asserts plain float/int values; newer numpy returns numpy scalars from the rollout
# metrics (e.g. success rates). Convert them, and skip anything non-numeric.
import numpy as np
import torch
from atm.utils import log_utils

_metric_update = log_utils.MetricLogger.update


def _numeric_update(self, **kwargs):
    clean = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray) and v.size == 1:
            v = v.reshape(-1)[0]
        if isinstance(v, np.generic):
            v = v.item()
        if v is None or isinstance(v, (bool, float, int, torch.Tensor)):
            clean[k] = float(v) if isinstance(v, bool) else v
    return _metric_update(self, **clean)


log_utils.MetricLogger.update = _numeric_update

if __name__ == "__main__":
    module = sys.argv.pop(1)
    runpy.run_module(module, run_name="__main__", alter_sys=True)
