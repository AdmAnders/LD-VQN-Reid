"""T2I-VeRi-776.

Images (original VeRi-776, request from the dataset owners)::

    data/VeRi/image_train/0001_c001_00016450_0.jpg
    data/VeRi/image_query/...
    data/VeRi/image_test/...

Captions (this repository)::

    datasets/T2I-VeRi-776/image_train.jsonl
    datasets/T2I-VeRi-776/image_query.jsonl
    datasets/T2I-VeRi-776/image_test.jsonl

The identity is the first field of the VeRi file name (``0001_c001_...`` -> 1).
"""
import os

from .base_dataset import read_jsonl


def _records(cfg_data, split_key):
    split = cfg_data[split_key]
    rows = read_jsonl(os.path.join(cfg_data["CAPTION_DIR"], split["JSONL"]))
    img_dir = os.path.join(cfg_data["IMAGE_ROOT"], split["IMG_DIR"])
    records = []
    for r in rows:
        name = r["filename"]
        records.append({
            "filename": name,
            "img_path": os.path.join(img_dir, name),
            "pid": int(name.split("_")[0]),
            "caption": r.get("caption"),
        })
    return records


def build_veri776(cfg_data):
    """Return (train_records, query_records, gallery_records)."""
    return (_records(cfg_data, "TRAIN"),
            _records(cfg_data, "QUERY"),
            _records(cfg_data, "GALLERY"))
