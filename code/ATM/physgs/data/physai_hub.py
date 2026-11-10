"""
Selective access to nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes (16 TB) on Hugging Face.

The repo stores each modality per scene type as webdataset tar shards, with members named
`<run>/<file>` and runs in lexicographic order across shards. This module
  * lists shards through the HF API,
  * reads tar headers with HTTP range requests (no full download) to index the first run of each shard,
  * maps a set of runs to the shards that contain them,
  * stream-extracts only the wanted members of a shard.
"""
import bisect
import json
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import requests

REPO = "nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/"
MODALITIES = ("videos", "depths", "segmentation", "physics", "cameras")
SCENES = ("ball_mixer", "billiards", "bowling", "dominoes", "objects_falling", "obstruction",
          "rolling_ramp_objects", "rolling_ramp_obstruct", "towers", "wrecking_ball")

_session = requests.Session()


def _token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        path = os.path.expanduser("~/.cache/huggingface/token")  # written by `huggingface-cli login`
        tok = open(path).read().strip() if os.path.exists(path) else None
    return tok


if _token():  # authenticated requests get much higher rate limits
    _session.headers["Authorization"] = f"Bearer {_token()}"


def _get(url, retries=8, **kw):
    for i in range(retries):
        try:
            r = _session.get(url, timeout=60, **kw)
            if r.status_code == 429:  # rate limited: wait as told (or a minute) and retry
                time.sleep(float(r.headers.get("Retry-After", 60)) + 1)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError(f"giving up on {url} after {retries} attempts")


def list_shards(modality, scene):
    """[(path, size_bytes)] sorted by path."""
    items = _get(API + f"{modality}/{scene}").json()
    return sorted((x["path"], x.get("size", 0)) for x in items if x["type"] == "file" and x["path"].endswith(".tar"))


def _parse_header(block):
    name = block[0:100].split(b"\0", 1)[0].decode()
    size = int(block[124:136].split(b"\0", 1)[0].strip() or b"0", 8)
    typeflag = block[156:157]
    prefix = block[345:500].split(b"\0", 1)[0].decode()
    if prefix:
        name = prefix + "/" + name
    return name, size, typeflag


def iter_headers(path, start=0):
    """Yield (member_name, data_offset, size) by hopping between tar headers with range requests."""
    url = BASE + path
    off, long_name = start, None
    while True:
        block = _get(url, headers={"Range": f"bytes={off}-{off + 511}"}).content
        if len(block) < 512 or block == b"\0" * 512:
            return
        name, size, typeflag = _parse_header(block)
        data = off + 512
        nxt = data + ((size + 511) // 512) * 512
        if typeflag == b"L":  # GNU long name: the data holds the real name of the next member
            long_name = _get(url, headers={"Range": f"bytes={data}-{data + size - 1}"}).content.split(b"\0", 1)[0].decode()
        elif typeflag == b"x":  # pax header: look for path=
            pax = _get(url, headers={"Range": f"bytes={data}-{data + size - 1}"}).content.decode(errors="ignore")
            for line in pax.splitlines():
                if " path=" in line:
                    long_name = line.split(" path=", 1)[1]
        else:
            yield (long_name or name), data, size
            long_name = None
        off = nxt


def first_member(path):
    return next(iter_headers(path))[0]


def build_index(scene, modalities=MODALITIES, cache_dir="."):
    """{modality: [[shard_path, first_member], ...]} cached as json; shards must be in run order."""
    cache = os.path.join(cache_dir, f"index_{scene}.json")
    index = json.load(open(cache)) if os.path.exists(cache) else {}
    for m in modalities:
        if m in index:
            continue
        shards = [p for p, _ in list_shards(m, scene)]
        with ThreadPoolExecutor(16) as pool:
            index[m] = [list(x) for x in zip(shards, pool.map(first_member, shards))]
        firsts = [f for _, f in index[m]]
        if firsts != sorted(firsts):
            print(f"[warn] {scene}/{m}: shards are not in lexicographic member order")
        with open(cache, "w") as f:
            json.dump(index, f)
    return index


def member_sizes(scene, modality="depths", cache_dir=".", threads=8):
    """{member_name: size} for every member of a modality, from tar headers only (cached)."""
    cache = os.path.join(cache_dir, f"sizes_{scene}_{modality}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    shards = [p for p, _ in list_shards(modality, scene)]
    with ThreadPoolExecutor(threads) as pool:
        res = pool.map(lambda p: [(n, s) for n, _, s in iter_headers(p)], shards)
    sizes = {n: s for r in res for n, s in r}
    with open(cache, "w") as f:
        json.dump(sizes, f)
    return sizes


def complete_depth_runs(scene, cache_dir=".", min_bytes=10_000_000, min_cams=2):
    """Runs with at least `min_cams` depth videos larger than `min_bytes` (smaller ones are truncated)."""
    per_run = {}
    for n, s in member_sizes(scene, "depths", cache_dir).items():
        if n.endswith(".mkv"):
            per_run.setdefault(run_of(n), []).append(s)
    return {r for r, v in per_run.items() if sum(x > min_bytes for x in v) >= min_cams}


def run_of(member):
    return member.split("/", 1)[0]


def shards_for_runs(entries, runs):
    """Shards (in order) overlapping any run's member interval [run + '/', run + '/\\uffff')."""
    firsts = [f for _, f in entries]
    out = set()
    for r in runs:
        lo, hi = r + "/", r + "/￿"
        i = max(0, bisect.bisect_right(firsts, lo) - 1)
        while i < len(entries) and firsts[i] < hi:
            out.add(i)
            i += 1
    return [entries[i][0] for i in sorted(out)]


def stream_extract(path, out_dir, keep):
    """Stream a shard and write members for which keep(name) is True. Returns the list of written names."""
    written = []
    with _session.get(BASE + path, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|") as tar:
            for m in tar:
                if not m.isfile() or not keep(m.name):
                    continue
                dst = os.path.join(out_dir, m.name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with tar.extractfile(m) as src, open(dst + ".part", "wb") as f:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                os.replace(dst + ".part", dst)
                written.append(m.name)
    return written


def download_file(path, dst, size=None, connections=8, chunk=64 << 20):
    """Download one repo file with parallel range requests (much faster than one stream)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if size is None:
        size = int(_session.head(BASE + path, allow_redirects=True, timeout=60).headers["Content-Length"])
    part = dst + ".part"
    with open(part, "wb") as f:
        f.truncate(size)

    def fetch(start):
        end = min(start + chunk, size) - 1
        data = _get(BASE + path, headers={"Range": f"bytes={start}-{end}"}).content
        with open(part, "r+b") as f:
            f.seek(start)
            f.write(data)

    with ThreadPoolExecutor(connections) as pool:
        list(pool.map(fetch, range(0, size, chunk)))
    os.replace(part, dst)
    return dst


def extract_local(tar_path, out_dir, keep):
    written = []
    with tarfile.open(tar_path, mode="r:") as tar:
        for m in tar:
            if m.isfile() and keep(m.name):
                tar.extract(m, out_dir)
                written.append(m.name)
    return written


def fetch_member_range(path, data_offset, size, dst):
    """Download one member's bytes with a range request."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with _session.get(BASE + path, headers={"Range": f"bytes={data_offset}-{data_offset + size - 1}"},
                      stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dst + ".part", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(dst + ".part", dst)
