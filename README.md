# LD-VQN-Reid
Beyond Coarse Labels: A Large-Scale Multimodal Benchmark and Linguistically Driven for Vehicle Re-Identification (LD-VQN)
```bash
├── configs/                     # Training and testing configurations (Hyperparameters)
│   ├── veri776_ldvqn.yml
│   ├── vehicleid_ldvqn.yml
│   └── veriwild_ldvqn.yml
├── datasets/                    # Data loaders and 808K JSONL-formatted text data
│   ├── build_dataloader.py
│   ├── veri776_dataset.py
│   ├── vehicleid_dataset.py
│   ├── T2I-VeRi-Wild/           # T2I-VeRi-Wild Captions
│   ├── T2I-VehicleID/           # T2I-VehicleID Captions
│   └── T2I-VeRi-776/            # T2I-VeRi-776 Captions
├── models/                      # LD-VQN architecture and core modules
│   ├── ld_vqn.py                # Main model (combination of MLA and D-TCR)
│   ├── mla.py                   # Multi-Level Adapter
│   ├── d_tcr.py                 # Dynamic Text-Conditioned Routing
│   ├── losses/                  # Ortogonal Disentanglement ve Triplet Loss
│   │   ├── orthogonal_loss.py
│   │   └── xbm_triplet.py
│   └── text_encoders/           # DeBERTa integration
├── scripts/                     # One-click execution scripts
│   ├── train_veri776.sh
│   ├── eval_vehicleid.sh
│   └── extract_attention.sh
├── tools/                       # Visualization tools (for the figures in the article)
│   ├── visualize_attention.py   # Fig. 9: Cross-modal attention maps
│   └── tsne_visualization.py    # Fig. 11: t-SNE clustering visualization
├── weights/                     # Index where pretrained model weights will be placed
├── Dockerfile                   # The perfect container for environment setup.
├── environment.yml              # For Conda 
├── requirements.txt             # For Pip installation
├── train.py                     # Main training loop (Algorithm 1)
├── test.py                      # Evaluation and mAP/Rank calculation
└── README.md              
```

[![Paper](https://img.shields.io/badge/Paper-Anonymous%20Submission-blue.svg)](link_to_pdf)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)

> **Note to Reviewers:** This repository is strictly anonymized for the double-blind review process of TPAMI. The full dataset (417K textual annotations) and pre-trained weights have been provided via anonymous Google Drive links to facilitate full reproducibility.

Official PyTorch implementation of **LD-VQN**, a novel cross-modal architecture for fine-grained Vehicle Re-Identification. We introduce the first large-scale multimodal benchmark featuring over 808,000 unconstrained natural language descriptions to resolve the "semantic bottleneck" in current Vision-Language Models (VLMs).

## 🚀 Highlights
- **417K Multimodal Corpus:** We semantically enriched VeRi-776 and VehicleID datasets with high-density textual descriptions, bridging the gap between macroscopic shapes and instance-level primitives (e.g., custom decals, specific grille geometries).
- **Multi-Lever Adapter (MLA):** Rescues early-stage textural primitives from the vision backbone before they homogenize.
- **Dynamic Text-Conditioned Routing (D-TCR):** Dynamically infers spatial adjacency directly from text, routing specific linguistic tokens to exact geometric coordinates.
- **Orthogonal Disentanglement Loss:** Mathematically forces latent semantic nodes toward a $\pi/2$ separation, strictly preventing attention collapse.

<div align="center">
  <img src="https://github.com/hidayetergn/LD-VQN-Reid/blob/main/results/LD-VQN_Architecture.png" alt="LD-VQN_Architecture">
  <p><em>Figure 1: Overall architecture of the proposed LD-VQN.</em></p>
</div>

---

## 🛠️ Environment Setup & Installation

We provide both Conda and Docker setups to ensure 100% reproducibility.

### Option 1: Conda Environment (Recommended)
```bash
git clone [https://github.com/LDVQN-Paper/LD-VQN.git](https://github.com/LDVQN-Paper/LD-VQN.git)
cd LD-VQN
conda env create -f environment.yml
conda activate ldvqn_env
