"""Convert RoboCasa365 LeRobot v2 datasets into the compact training layout.

Human300 writes `<root>/data`; Target100 writes `<root>/groups/<group>/data` and
reuses the Human300 normalization. Videos are read in place and are not copied.
"""
import argparse
import concurrent.futures
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from recipe_data import compact_rows, compact_to_native

GROUPS = ["atomic_seen", "composite_seen", "composite_unseen"]
GROUP_TASKS = {"atomic_seen": 18, "composite_seen": 16, "composite_unseen": 16}


def select_episodes(root, count):
    # Same algorithm and seed as RoboCasa get_subset_demos_filter_key.
    episodes = [json.loads(s) for s in (Path(root) / "meta/episodes.jsonl").read_text().splitlines()]
    ids = [e["episode_index"] for e in episodes]
    random.Random(0).shuffle(ids)
    assert len(ids) >= count, (root, len(ids))
    return ids[:count]


def convert_task(item):
    number, root, ids, out = item
    ds = Path(root)
    info = json.loads((ds / "meta/info.json").read_text())
    assert info["fps"] == 20
    episodes = {e["episode_index"]: e for e in map(json.loads, (ds / "meta/episodes.jsonl").read_text().splitlines())}
    arrays, index, offset, error = [], [], 0, 0.0
    for eid in ids:
        args = dict(episode_index=eid, episode_chunk=eid // info["chunks_size"])
        df = pd.read_parquet(ds / info["data_path"].format(**args), columns=["observation.state", "action"])
        state0, action0 = np.stack(df["observation.state"]), np.stack(df["action"])
        state, action = compact_rows(state0, action0)
        assert len(state) == episodes[eid]["length"] and state.shape[1] == 19 and action.shape[1] == 15
        assert np.isfinite(state).all() and np.isfinite(action).all()
        error = max(error, float(np.abs(compact_to_native(action) - action0).max()))
        for key, value in info["features"].items():
            if value.get("dtype") == "video":
                assert (ds / info["video_path"].format(video_key=key, **args)).is_file()
        arrays.append(np.concatenate([state, action], axis=1))
        index.append(dict(root=str(ds), array=f"task_{number:03d}.npy", offset=offset, episode_index=eid,
                          length=len(state), prompt=episodes[eid]["tasks"][0],
                          chunks_size=info["chunks_size"], video_template=info["video_path"]))
        offset += len(state)
    assert error < 2e-5, error
    values = np.concatenate(arrays)
    np.save(out / f"task_{number:03d}.npy", values)
    return dict(index=index, minimum=values.min(0).tolist(), maximum=values.max(0).tolist())


def human300(root, datasets, workers):
    out = root / "data"
    out.mkdir(parents=True, exist_ok=True)
    assert len(datasets) == 300
    items = [(i, path, select_episodes(path, 100), out) for i, path in enumerate(datasets)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(convert_task, items))
    low = np.min([r["minimum"] for r in results], 0)
    high = np.max([r["maximum"] for r in results], 0)
    stats = {}
    # Min-max bounds per dimension; rot6d entries keep the identity range [-1, 1].
    for key, columns, rot6d in [("state", slice(0, 19), list(range(3, 9)) + list(range(13, 19))),
                                ("action", slice(19, 34), list(range(3, 9)))]:
        lo, hi = low[columns].copy(), high[columns].copy()
        lo[rot6d], hi[rot6d] = -1.0, 1.0
        stats[key] = dict(min=lo.tolist(), max=hi.tolist(), rot6d_identity_dims=rot6d)
    episodes = [e for r in results for e in r["index"]]
    (out / "normalization.json").write_text(json.dumps(stats, indent=2))
    (out / "index.json").write_text(json.dumps(dict(episodes=episodes, tasks=300, demos=len(episodes))))


def target100(root, datasets, workers):
    normalization = root / "data/normalization.json"
    if not normalization.is_file():
        raise FileNotFoundError("Prepare Human300 first; Target100 reuses its normalization")
    for group in GROUPS:
        folder = root / "groups" / group
        (folder / "data").mkdir(parents=True, exist_ok=True)
        if not (folder / "text_cache").exists():
            (folder / "text_cache").symlink_to(root / "text_cache", target_is_directory=True)
        rows = sorted((r for r in datasets if r["group"] == group), key=lambda r: r["task"])
        assert len(rows) == GROUP_TASKS[group]
        items = [(i, r["root"], select_episodes(r["root"], 500), folder / "data") for i, r in enumerate(rows)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            lists = [r["index"] for r in pool.map(convert_task, items)]
        episodes, ranges, start = [e for es in lists for e in es], [], 0
        # Window ranges per task, used by the task-uniform sampler.
        for es in lists:
            end = start + sum(max(0, e["length"] - 1) for e in es)
            ranges.append([start, end])
            start = end
        (folder / "data/index.json").write_text(json.dumps(dict(episodes=episodes, tasks=len(rows), demos=len(episodes))))
        (folder / "task_ranges.json").write_text(json.dumps(ranges))
        shutil.copy2(normalization, folder / "data/normalization.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=["human300", "target100"], required=True)
    p.add_argument("--datasets", type=Path, required=True,
                   help="human300: JSON list of 300 dataset roots; target100: JSON list of {task, group, root}")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    root = a.root.resolve()
    (root / "text_cache").mkdir(parents=True, exist_ok=True)
    datasets = json.loads(a.datasets.read_text())
    (human300 if a.stage == "human300" else target100)(root, datasets, a.workers)


if __name__ == "__main__":
    main()
