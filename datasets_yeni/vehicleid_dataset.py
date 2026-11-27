"""T2I-VehicleID.

Images and identity lists (original PKU VehicleID, request from the dataset owners)::

    data/VehicleID/image/0000001.jpg ...
    data/VehicleID/train_test_split/train_list.txt          # "<image_id> <vehicle_id>"
    data/VehicleID/train_test_split/test_list_800.txt       # 800 / 1600 / 2400 / 3200 / 6000 / 13164

Captions (this repository)::

    datasets/T2I-VehicleID/train_list.jsonl
    datasets/T2I-VehicleID/test_list_800.jsonl ...

The caption files contain only ``filename`` and ``caption``; the identity is
looked up in the original split list (an optional ``id`` / ``pid`` field in the
JSONL takes precedence).

VehicleID has no separate query/gallery lists. For each identity of the test
list one image is put in the gallery and the remaining images become queries
(``GALLERY_SELECTION: first`` -> first image of that identity in the original
list order; ``random`` -> seeded random choice).
"""
import os
import random
import warnings

from .base_dataset import read_jsonl


def _read_split(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"VehicleID split list not found: {path}. It ships with the original "
            f"VehicleID dataset (train_test_split/).")
    order, lookup = [], {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            img_id, vid = parts[0], int(parts[1])
            lookup[img_id] = vid
            order.append(img_id)
    return order, lookup


def _records(cfg_data, jsonl_name, split_name):
    rows = read_jsonl(os.path.join(cfg_data["CAPTION_DIR"], jsonl_name))
    order, lookup = _read_split(os.path.join(cfg_data["SPLIT_DIR"], split_name))
    img_dir = os.path.join(cfg_data["IMAGE_ROOT"], cfg_data["IMG_DIR"])

    records, unknown = [], 0
    for r in rows:
        name = r["filename"]
        stem = os.path.splitext(os.path.basename(name))[0]
        pid = r.get("pid", r.get("id"))
        if pid is None:
            pid = lookup.get(stem)
        if pid is None:
            unknown += 1
            continue
        records.append({
            "filename": name,
            "img_path": os.path.join(img_dir, name),
            "pid": int(pid),
            "caption": r.get("caption"),
            "_stem": stem,
        })
    if unknown:
        warnings.warn(f"[vehicleid] {unknown} captions in {jsonl_name} have no identity in "
                      f"{split_name} and were skipped.")
    rank = {img_id: i for i, img_id in enumerate(order)}
    return records, rank


def _split_query_gallery(records, rank, selection="first", seed=0):
    # Sort by position in the original list so that "first" is independent of caption-file order.
    records = sorted(records, key=lambda r: rank.get(r["_stem"], len(rank)))
    by_pid = {}
    for r in records:
        by_pid.setdefault(r["pid"], []).append(r)

    rng = random.Random(seed)
    query, gallery = [], []
    for pid in sorted(by_pid):
        items = by_pid[pid]
        g_idx = 0 if selection == "first" else rng.randrange(len(items))
        for i, r in enumerate(items):
            (gallery if i == g_idx else query).append(r)
    return query, gallery


def build_vehicleid(cfg_data):
    """Return (train_records, query_records, gallery_records) for DATA.TEST_SIZE."""
    size = cfg_data["TEST_SIZE"]
    train, _ = _records(cfg_data, cfg_data["TRAIN_JSONL"], cfg_data["TRAIN_SPLIT"])
    test, rank = _records(cfg_data, cfg_data["TEST_JSONL"].format(size=size),
                          cfg_data["TEST_SPLIT"].format(size=size))
    query, gallery = _split_query_gallery(test, rank, cfg_data.get("GALLERY_SELECTION", "first"),
                                          cfg_data.get("SPLIT_SEED", 0))
    for r in train + query + gallery:
        r.pop("_stem", None)
    return train, query, gallery
