"""T2I-VeRi-Wild.

Images (original VERI-Wild, request from the dataset owners)::

    data/VeRi-Wild/images/<vehicle_id>/<image_id>.jpg     e.g. images/29044/298790.jpg

A flat layout (``images/<image_id>.jpg``) is detected automatically.

Captions (this repository)::

    datasets/T2I-VeRi-Wild/Veri_Wild_captions_train.jsonl
    datasets/T2I-VeRi-Wild/Veri_Wild_captions_test_{3000,5000,10000}_query.jsonl
    datasets/T2I-VeRi-Wild/Veri_Wild_captions_test_{3000,5000,10000}_gallery.jsonl

Each JSONL line carries ``filename`` (e.g. ``298790.jpg``) and ``id`` (vehicle identity).
"""
import os

from .base_dataset import read_jsonl


def _resolve(img_dir, vid_str, name):
    candidates = [
        os.path.join(img_dir, vid_str.zfill(5), name),   # official layout: 5-digit identity folders
        os.path.join(img_dir, vid_str, name),
        os.path.join(img_dir, name),                     # flat layout
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]


def _records(cfg_data, jsonl_name):
    rows = read_jsonl(os.path.join(cfg_data["CAPTION_DIR"], jsonl_name))
    img_dir = os.path.join(cfg_data["IMAGE_ROOT"], cfg_data["IMG_DIR"])
    records = []
    for r in rows:
        if "id" not in r:
            raise KeyError(f"VeRi-Wild caption line without 'id': {r['filename']}")
        vid_str = str(r["id"])
        name = r["filename"]
        records.append({
            "filename": name,
            "img_path": _resolve(img_dir, vid_str, name),
            "pid": int(vid_str),
            "caption": r.get("caption"),
        })
    return records


def build_veriwild(cfg_data):
    """Return (train_records, query_records, gallery_records) for DATA.TEST_SIZE."""
    size = cfg_data["TEST_SIZE"]
    return (_records(cfg_data, cfg_data["TRAIN_JSONL"]),
            _records(cfg_data, cfg_data["QUERY_JSONL"].format(size=size)),
            _records(cfg_data, cfg_data["GALLERY_JSONL"].format(size=size)))
