"""
tokenize_h5.py (2026-11-07): add the task sentence, tokenized exactly as smolVLA's preprocessor does it
(lerobot.policies.smolvla.processor_smolvla: task + "\\n", the VLM's tokenizer, right padding, max_length 48), to every
SO-101 policy file as extra_states/lang_tokens and extra_states/lang_mask, (T, 48) each (repeated per frame, the layout
of an extra state). Read only by the smolvla backbone (extra_state_keys); the original ViLT never lists them.
Run with env_vla:  python tokenize_h5.py <policy_h5 dir>
"""
import glob
import sys

import h5py
import numpy as np
from transformers import AutoTokenizer

MAXLEN = 48


def main():
    root = sys.argv[1]
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    tok = AutoTokenizer.from_pretrained(SmolVLAConfig().vlm_model_name)
    files = sorted(glob.glob(f"{root}/*_demo/demo_*.hdf5"))
    cache = {}
    for fn in files:
        with h5py.File(fn, "r+") as f:
            task = f.attrs["task"]
            if task not in cache:
                t = tok(task + "\n", padding="max_length", max_length=MAXLEN, truncation=True, padding_side="right",
                        return_tensors="np")
                cache[task] = (t["input_ids"][0].astype(np.float32), t["attention_mask"][0].astype(np.float32))
                print(repr(task), "->", int(cache[task][1].sum()), "tokens", flush=True)
            T = f["root/actions"].shape[0]
            ex = f["root/extra_states"]
            for k, v in (("lang_tokens", cache[task][0]), ("lang_mask", cache[task][1])):
                if k in ex:
                    del ex[k]
                ex.create_dataset(k, data=np.repeat(v[None], T, 0))
    print(len(files), "files tokenized")


if __name__ == "__main__":
    main()
