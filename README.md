# BDAChat: Integrating Segmentation and Vision-Language Model for Automated and Interpretable Building Damage Assessment

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/your-paper-link)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/WangYong921/BDAChat?style=social)](https://github.com/WangYong921/BDAChat)

This repository contains the official implementation of the paper: **"Integrating segmentation and vision-language model for automated and interpretable building damage assessment from satellite imagery"**.

**BDAChat** is a novel three-stage framework that integrates instance segmentation with a temporal vision-language model (VLM) for automated, object-level, and interpretable assessment of structural assets from satellite imagery.

## 📰 News
*   **[2025.XX.XX]** 🔥 The **OLBDA** dataset and model weights are now available for download!
*   **[2025.XX.XX]** 🚀 Pipeline scripts for spatial grouping and visualization (`sgts.py`, `damage-aware_heatmap.py`) are available.
*   **[2025.XX.XX]** 📄 The paper is available on arXiv.

## 😮 Highlights

*   **Three-Stage Framework:** Synergizes high-precision segmentation, spatiotemporal data pairing, and VLM-based reasoning.
*   **Modified SAM:** A Multi-LoRA fine-tuned Segment Anything Model (SAM) adapted for high-performance building and road extraction in remote sensing imagery.
*   **Interpretable Reasoning:** Unlike black-box models, **BDAChat** (based on Video-LLaVA) provides damage classification, disaster recognition, and *causal explanations* in natural language.
*   **SOTA Performance:** Achieves state-of-the-art results on the xBD dataset (MF1 0.763) and demonstrates robust generalization on the unseen Lahaina wildfire event.

## 🏗️ Framework Overview

The proposed framework consists of three main stages:
1.  **Segmentation:** Building and road extraction using **Modified SAM** with Multi-LoRA.
2.  **Alignment:** Spatial grouping and temporal sorting of building instances.
3.  **Assessment:** Multi-task visual question answering using **BDAChat**.

![Framework](assets/framework_diagram.png) *(Placeholder for Fig 1 from paper)*

## 💾 Model Weights & Dataset Access

We provide the **OLBDA** dataset and pre-trained weights for both **Modified SAM** and **BDAChat** via Baidu Netdisk.

### 📚 Dataset
**OLBDA (Object-Level Building Damage Assessment)**
*   The first bitemporal image dataset for object-level building damage assessment based on remote sensing imagery, integrating multi-hazard generalization.
*   🔗 **Download:** [Baidu Netdisk](https://pan.baidu.com/s/1e6nw7auDnT81GB-u0ezwWQ?pwd=sbqm) (Extraction code: `sbqm`)

### 🤖 Model Weights
| Model | Description | Download Link |
| :--- | :--- | :--- |
| **Modified SAM** | Multi-LoRA fine-tuned SAM for high-precision segmentation | [Baidu Netdisk](https://pan.baidu.com/s/1Y795X0yZsI2mMzLneFQq5A?pwd=7qk7) (Code: `7qk7`) |
| **BDAChat-7B** | Fine-tuned Video-LLaVA for object-level damage assessment | [Baidu Netdisk](https://pan.baidu.com/s/1m1lHnq1a6hlqPYPBPuuMFg?pwd=qedm) (Code: `qedm`) |

## 🛠️ Requirements and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/WangYong921/BDAChat.git
    cd BDAChat
    ```

2.  **Create a conda environment:**
    ```bash
    conda create -n bdachat python=3.9 -y
    conda activate bdachat
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install opencv-python numpy pandas pillow
    # Install dependencies for Video-LLaVA (see BDAChat folder) and SAM (see Modified_SAM folder)
    pip install -r requirements.txt
    ```

## 🚀 Pipeline Usage

The repository is structured to support the full pipeline from raw satellite imagery to fine-grained damage maps.

### 0. Data Preprocessing
Split large satellite orthophotos into manageable blocks for processing.
```bash
python blocks_genegation.py
Input: Large post-disaster image (e.g., hcpost.jpg).
Output: Tiled images (e.g., block_1_post_disaster.png).
1. Segmentation (Stage 1)
Use the Modified SAM to generate building masks and road networks.
Please refer to the Modified_SAM/ directory for training and inference scripts.
Download the Modified SAM weights from the link above and place them in Modified_SAM/checkpoints/.
2. Spatial Grouping & Temporal Sorting (Stage 2)
Extract and pair building instances from pre- and post-disaster images based on segmentation masks.
code
Bash
python sgts.py
Function: Crops building instances, applies expansion (default ratio 0.8), and aligns pre/post pairs.
Key Parameters:
expand_ratio: 0.8 (Optimized for context inclusion).
use_rotated_rect: Handles oriented bounding boxes.
3. BDAChat Inference (Stage 3)
Run the fine-tuned VLM to classify damage and generate reasoning.
Please refer to the BDAChat/ directory for inference scripts.
Download the BDAChat-7B weights from the link above.
The output should be a JSON file (e.g., bdachat-7B_...json) containing textual responses.
4. Visualization & Mapping
Generate interpretable maps for decision support.
Fine-Grained Damage Map:
Projects the VLM classification results back onto the satellite image.
code
Bash
python fine-grained_damagemap.py
Input: Post-disaster images, building masks, and BDAChat JSON responses.
Output: damage_overlay.png (Color-coded buildings: 🟢 No Damage, 🟡 Minor, 🟠 Major, 🔴 Destroyed).
Damage-Aware Heatmap:
Generates a density heatmap of damaged areas to assess disaster impact intensity.
code
Bash
python damage-aware_heatmap.py
Input: Post-disaster images and classification results.
Output: heatmap_overlay.png.
📊 Main Results
Zero-shot performance on xBD Dataset:
Model	Acc	MP	MR	MF1
Qwen2.5-7B	0.546	0.233	0.249	0.231
Video-LLaVA-7B	0.756	0.252	0.266	0.251
TEOChat-7B	0.684	0.351	0.433	0.366
BDAChat-7B (Ours)	0.901	0.808	0.728	0.763
📂 File Structure
code
Code
BDAChat/
├── BDAChat/                   # Code for the VLM (Stage 3)
├── Modified_SAM/              # Code for Segmentation (Stage 1)
├── blocks_genegation.py       # Image tiling script
├── sgts.py                    # Spatial grouping and temporal sorting (Stage 2)
├── fine-grained_damagemap.py  # Visualization: Damage Classification Map
├── damage-aware_heatmap.py    # Visualization: Heatmap
├── LICENSE
└── README.md
👍 Acknowledgement
This project is based on the following amazing open-source projects:
Video-LLaVA
Segment Anything (SAM)
xBD Dataset
✏️ Citation
If you find our work useful in your research, please consider citing:
code
Bibtex
@article{wang2025integrating,
  title={Integrating segmentation and vision-language model for automated and interpretable building damage assessment from satellite imagery},
  author={Wang, Yong and Cui, Jiawei and Zhai, Changhai and Tao, Xigui and Li, Yuhao},
  journal={arXiv preprint},
  year={2025}
}
📜 License
This project is released under the MIT License.
