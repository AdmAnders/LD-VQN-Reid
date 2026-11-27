"""Builds train / query / gallery loaders for veri776, vehicleid and veriwild."""
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import DebertaV2Tokenizer

from utils.seed import make_generator, seed_worker
from .base_dataset import TextImageReIDDataset, check_images
from .samplers import RandomIdentitySampler
from .vehicleid_dataset import build_vehicleid
from .veri776_dataset import build_veri776
from .veriwild_dataset import build_veriwild

BUILDERS = {
    "veri776": build_veri776,
    "vehicleid": build_vehicleid,
    "veriwild": build_veriwild,
}


def build_transforms(cfg_data):
    normalize = transforms.Normalize(cfg_data["PIXEL_MEAN"], cfg_data["PIXEL_STD"])
    train = transforms.Compose([
        transforms.Resize(tuple(cfg_data["TRAIN_RESIZE"])),
        transforms.RandomCrop(tuple(cfg_data["IMG_SIZE"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test = transforms.Compose([
        transforms.Resize(tuple(cfg_data["IMG_SIZE"])),
        transforms.ToTensor(),
        normalize,
    ])
    return train, test


def build_records(cfg):
    name = cfg["DATA"]["DATASET"].lower()
    if name not in BUILDERS:
        raise ValueError(f"Unknown dataset '{name}', choose from {list(BUILDERS)}")
    return BUILDERS[name](cfg["DATA"])


def make_dataloaders(cfg, build_train=True, check_files=True):
    """Return (train_loader or None, query_loader, gallery_loader, num_classes)."""
    d, dl = cfg["DATA"], cfg["DATALOADER"]
    seed = cfg["EXPERIMENT"]["SEED"]
    train_rec, query_rec, gallery_rec = build_records(cfg)

    if check_files:
        if build_train:
            check_images(train_rec, d["DATASET"], "train")
        check_images(query_rec, d["DATASET"], "query")
        check_images(gallery_rec, d["DATASET"], "gallery")

    print(f"[{d['DATASET']}] train={len(train_rec)}  query={len(query_rec)}  gallery={len(gallery_rec)}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(cfg["MODEL"]["TEXT_BACKBONE"])
    train_tf, test_tf = build_transforms(d)
    common = dict(tokenizer=tokenizer, img_size=d["IMG_SIZE"],
                  max_token_len=d["MAX_TOKEN_LEN"], default_caption=d["DEFAULT_CAPTION"])

    unique_pids = sorted({r["pid"] for r in train_rec})
    pid2label = {pid: i for i, pid in enumerate(unique_pids)}
    num_classes = len(unique_pids)

    train_loader = None
    if build_train:
        train_set = TextImageReIDDataset(train_rec, transform=train_tf, pid_mapping=pid2label,
                                         is_train=True, **common)
        sampler = RandomIdentitySampler(train_set.pids, dl["BATCH_SIZE"], dl["NUM_INSTANCES"])
        train_loader = DataLoader(train_set, batch_size=dl["BATCH_SIZE"], sampler=sampler,
                                  num_workers=dl["NUM_WORKERS"], drop_last=True,
                                  worker_init_fn=seed_worker, generator=make_generator(seed))

    def _test_loader(records):
        ds = TextImageReIDDataset(records, transform=test_tf, is_train=False, **common)
        return DataLoader(ds, batch_size=dl["TEST_BATCH_SIZE"], shuffle=False,
                          num_workers=dl.get("TEST_NUM_WORKERS", 0))

    return train_loader, _test_loader(query_rec), _test_loader(gallery_rec), num_classes
