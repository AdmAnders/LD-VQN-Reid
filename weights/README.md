# Pretrained weights

Download the checkpoints from the GitHub release
[`v1.0`](https://github.com/AdmAnders/LD-VQN-Reid/releases/tag/v1.0) and place them here:

| File | Dataset | Epoch | SHA256 |
|---|---|---|---|
| `ldvqn_veri776.pth`   | T2I-VeRi-776  | 60 | `TODO` |
| `ldvqn_vehicleid.pth` | T2I-VehicleID | TODO | `TODO` |
| `ldvqn_veriwild.pth`  | T2I-VeRi-Wild | TODO | `TODO` |

Verify a download with `sha256sum weights/ldvqn_veri776.pth`.

Each file is a plain `state_dict` (`ldvqn_best.pth` written by `train.py`).
Full training checkpoints (`ldvqn_checkpoint.pth`) are also accepted by `test.py`.
