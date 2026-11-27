"""Shared image-text dataset used by all three benchmarks.

Each dataset module (veri776 / vehicleid / veriwild) turns its caption JSONL
files into a list of *records*::

    {"filename": str,   # unique image name, used to exclude the query itself at test time
     "img_path": str,   # absolute/relative path of the image on disk
     "pid": int,        # raw vehicle identity
     "caption": str}

This class turns those records into
``(image, input_ids, attention_mask, pid, filename)`` tuples, exactly as in the
original training notebook.
"""
import json
import os
import warnings

from PIL import Image
from torch.utils.data import Dataset


def read_jsonl(path):
    """Read a JSON-lines file; malformed lines and lines without "filename" are skipped."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Caption file not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "filename" in data:
                records.append(data)
    return records


def check_images(records, dataset_name, split, max_report=5):
    """Warn about records whose image file is missing (they are replaced by a black image)."""
    missing = [r["img_path"] for r in records if not os.path.isfile(r["img_path"])]
    if missing:
        warnings.warn(
            f"[{dataset_name}/{split}] {len(missing)} of {len(records)} images not found, "
            f"e.g. {missing[:max_report]}. Missing images are replaced by a black image - "
            f"check DATA.IMAGE_ROOT / DATA.IMG_DIR.")
    return len(missing)


class TextImageReIDDataset(Dataset):
    def __init__(self, records, tokenizer, transform=None, pid_mapping=None, is_train=True,
                 img_size=(336, 336), max_token_len=128, default_caption="A vehicle."):
        self.data_list = records
        self.tokenizer = tokenizer
        self.transform = transform
        self.pid_mapping = pid_mapping
        self.is_train = is_train
        self.img_size = tuple(img_size)
        self.max_token_len = max_token_len
        self.default_caption = default_caption

    def __len__(self):
        return len(self.data_list)

    @property
    def pids(self):
        """Training labels (after relabelling) for every sample, used by the identity sampler."""
        if self.is_train and self.pid_mapping:
            return [self.pid_mapping.get(r["pid"], -1) for r in self.data_list]
        return [r["pid"] for r in self.data_list]

    def __getitem__(self, idx):
        data = self.data_list[idx]
        try:
            image = Image.open(data["img_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", self.img_size)
        if self.transform:
            image = self.transform(image)

        caption = data.get("caption") or self.default_caption
        inputs = self.tokenizer(caption, padding="max_length", truncation=True,
                                max_length=self.max_token_len, return_tensors="pt")

        raw_pid = data["pid"]
        pid = self.pid_mapping.get(raw_pid, -1) if self.is_train and self.pid_mapping else raw_pid
        return (image, inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0),
                pid, data["filename"])
