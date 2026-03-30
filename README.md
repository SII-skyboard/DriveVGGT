# DriveVGGT: Calibration-Constrained Visual Geometry Transformers for Multi-Camera Autonomous Driving


---

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#1-nuscenes-dataset-preparation)
3. [Training](#2-training)
4. [Evaluation](#3-evaluation)
5. [Visualization](#4-visualization)
6. [Project Structure](#project-structure)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DriveVGGT.git
cd DriveVGGT

# Install the package and core dependencies
pip install -e .
pip install -r requirements.txt

# Additional dependencies for visualization
pip install -r requirements_demo.txt
```

**Key dependencies:**

| Package | Version |
|---|---|
| torch | 2.5.1 |
| torchvision | 0.20.1 |
| numpy | 1.26.1 |
| nuscenes-devkit | latest |
| pyquaternion | latest |
| viser | latest (for visualization) |

---

## 1. NuScenes Dataset Preparation

Data preparation is a **two-step** process: first generate aggregated dense point clouds, then export per-scene metadata as JSON.

### Step 1 — Generate Dense Aggregated Point Clouds

`nusc_prepare/dense_pointcloud.py` aggregates LiDAR scans across all frames in a scene to produce a dense static point cloud. It:

- Transforms per-frame LiDAR points to the global coordinate frame.
- Removes points inside the ego vehicle bounding box.
- Removes points inside annotated object bounding boxes (dynamic objects), then re-inserts them in the canonical object frame aggregated across all annotations of that instance.
- Filters points beyond a 100 m radius / ±20 m height from ego.
- Saves one `.npy` file per keyframe (shape `[N, 3]`, global coordinates).

**Configure paths** inside the script before running:

```python
# nusc_prepare/dense_pointcloud.py
data_path = "/PATH/TO/nuscenes"          # nuScenes root (v1.0-trainval)
write_path = "/PATH/TO/aggregate_lidar/" # output directory for .npy files
```

**Run:**

```bash
python nusc_prepare/dense_pointcloud.py
```

The script processes scenes in batches (default: indices 0–399). Adjust the `range(...)` at the bottom to process a subset.

---

### Step 2 — Export Scene Metadata as JSON

`nusc_prepare/nusc_json.py` iterates over all nuScenes scenes and writes one JSON file per scene containing, for every keyframe and every camera:

- Image path
- Camera intrinsics `K` (3×3)
- `cam2global` extrinsic (4×4)
- LiDAR path (points to the `.npy` file from Step 1)
- Ego-to-global transform
- Timestamp

**Configure paths** inside the script before running:

```python
# nusc_prepare/nusc_json.py
data_path  = "/PATH/TO/nuscenes"            # nuScenes root (v1.0-trainval)
json_path  = "/PATH/TO/nuscene_json"        # output directory for JSON files
write_path = "/PATH/TO/aggregate_lidar/"    # must match Step 1 output
```

**Run:**

```bash
python nusc_prepare/nusc_json.py
```

Output: one `scene-XXXX.json` per scene in `nuscene_json/`.

**Expected directory layout after preparation:**

```
/PATH/TO/
├── nuscenes/            # original nuScenes dataset (v1.0-trainval)
│   ├── samples/
│   ├── sweeps/
│   └── v1.0-trainval/
├── aggregate_lidar/     # dense point clouds (.npy), one per keyframe
│   ├── n015-2018-...bin.npy
│   └── ...
└── nuscene_json/        # scene metadata JSON files
    ├── scene-0001.json
    ├── scene-0002.json
    └── ...
```

---

## 2. Training

Training uses PyTorch DDP via `torchrun`. The entry point is `vggt/training/launch.py`.

### Configure Paths

Edit `vggt/training/config/nuscenes_decoder.yaml` and update the following fields:

```yaml
data:
  train:
    dataset:
      dataset_configs:
        - _target_: data.datasets.nuscenes_mv.NuscenesDataset_MultiView
          split: train
          nusc_info: "/PATH/TO/nuscene_json"   # ← update
          len_train: 850
          expand_ratio: 8

  val:
    dataset:
      dataset_configs:
        - _target_: data.datasets.nuscenes_mv.NuscenesDataset_MultiView
          split: train
          nusc_info: "/PATH/TO/nuscene_json"   # ← update
          len_train: 850
          expand_ratio: 8

checkpoint:
  resume_checkpoint_path: /PATH/TO/pretrained/model.pt  # ← update
```

### Launch Training

```bash
# Single node, 4 GPUs
cd vggt/training
torchrun --nproc_per_node=4 launch.py --config nuscenes_decoder.yaml

# Background with logging
nohup torchrun --nproc_per_node=4 launch.py --config nuscenes_decoder.yaml \
    > logs/nuscenes_decoder.log 2>&1 &
```

### Key Hyperparameters (`nuscenes_decoder.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `max_epochs` | 100 | Total training epochs |
| `img_size` | 518 | Input image resolution |
| `max_img_per_gpu` | 30 | Images per GPU (reduce to avoid OOM) |
| `accum_steps` | 1 | Gradient accumulation steps |
| `optim.optimizer.lr` | 1e-5 | Peak learning rate |
| `loss.depth.weight` | 0.1 | Depth loss weight |
| `val_epoch_freq` | 5 | Validation frequency (epochs) |

**OOM tips:** Decrease `max_img_per_gpu` or increase `accum_steps`.

**Learning rate:** Scales with effective batch size (`max_img_per_gpu × num_gpus`). Try `5e-6`, `1e-5`, `5e-5` as starting points.

Checkpoints are saved to `vggt/training/logs/{exp_name}/ckpts/`.

---

## 3. Evaluation

Evaluation scripts are in `vggt/evaluation/nusc_eval/`. They measure **camera pose accuracy** (AUC@3/5/15/30) and **depth estimation quality** (AbsRel, MAE, SqRel, δ<1.25³) on the nuScenes validation split.

Three evaluation modes are available:

| Script | Frames per window | Notes |
|---|---|---|
| `nusc_eval15.py` | 15 | Standard evaluation |
| `nusc_eval25.py` | 25 | Longer temporal window |
| `nusc_eval15_scale.py` | 15 | Scale-aware variant |

### Run Evaluation

```bash
cd vggt/evaluation/nusc_eval

# Evaluate with default model path (set inside script)
python nusc_eval15.py

# Specify model checkpoint and model variant
python nusc_eval15.py \
    --model_path /PATH/TO/checkpoint.pt \
    --model_name decoder
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | (set in script) | Path to model checkpoint |
| `--model_name` | `None` (MultiViewVGGT_v2) | Model variant: `decoder`, `decoder_global`, `fastvggt`, `streamvggt`, etc. |
| `--use_ba` | `False` | Enable bundle adjustment post-processing |
| `--seed` | `0` | Random seed |

### Metrics

**Pose metrics** (relative pose error over all frame pairs):
- **AUC@N**: Area under the accuracy curve up to N° threshold (higher is better)
- **R_ACC@5**: Fraction of pairs with rotation error < 5°
- **T_ACC@5**: Fraction of pairs with translation angular error < 5°

**Depth metrics** (scale-and-shift aligned to LiDAR GT):
- **AbsRel**: Mean absolute relative error
- **MAE**: Mean absolute error (meters)
- **SqRel**: Scale-invariant squared relative error
- **δ<1.25³**: Fraction of predictions within 25% × 25% × 25% of GT

The validation set comprises **150 scenes** from the standard nuScenes val split.

---

## 4. Visualization

`demo_viser_nusc_eval_norm.py` provides an interactive 3D visualization of model predictions using [viser](https://github.com/nerfstudio-project/viser).

### What it shows

- **Point cloud**: 3D points unprojected from predicted depth maps, colored by the corresponding image pixels.
- **Camera frustums**: Predicted camera poses rendered as frustums; click a frustum to snap the viewpoint to that camera.
- **Confidence filtering**: Slider to filter out low-confidence points in real time.

### Configure

Inside `demo_viser_nusc_eval_norm.py`, set the model checkpoint and nuScenes JSON path:

```python
# Path to scene JSON files
nusc_info = "/PATH/TO/nuscene_json/"

# Model checkpoint
model_ckpt = "/PATH/TO/checkpoint.pt"
```

### Run

```bash
python demo_viser_nusc_eval_norm.py \
    --port 8080 \
    --conf_threshold 25.0

# Use precomputed point map instead of depth-unprojected points
python demo_viser_nusc_eval_norm.py --use_point_map

# Run server in background (non-blocking)
python demo_viser_nusc_eval_norm.py --background_mode
```

Then open `http://localhost:8080` in your browser.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--port` | 8080 | Viser server port |
| `--conf_threshold` | 25.0 | Initial percentile of points to filter (0–100) |
| `--use_point_map` | False | Visualize world_points head instead of depth |
| `--background_mode` | False | Run server in background thread |
| `--mask_sky` | False | Apply sky segmentation to filter sky points |

---

## Project Structure

```
DriveVGGT/
├── nusc_prepare/
│   ├── dense_pointcloud.py      # Step 1: aggregate LiDAR point clouds
│   └── nusc_json.py             # Step 2: export scene metadata as JSON
│
├── vggt/
│   ├── training/
│   │   ├── launch.py            # Training entry point
│   │   ├── config/
│   │   │   ├── nuscenes_decoder.yaml   # nuScenes fine-tuning config
│   │   │   ├── default.yaml            # Default (Co3D) config
│   │   │   └── ...
│   │   ├── data/
│   │   │   └── datasets/
│   │   │       └── nuscenes_mv.py      # nuScenes multi-view dataloader
│   │   └── README.md
│   │
│   ├── evaluation/
│   │   └── nusc_eval/
│   │       ├── nusc_eval15.py          # Eval: 15-frame window
│   │       ├── nusc_eval25.py          # Eval: 25-frame window
│   │       └── nusc_eval15_scale.py    # Eval: scale-aware variant
│   │
│   └── models/                  # Model definitions (VGGT variants)
│
├── demo_viser_nusc_eval_norm.py # Interactive 3D visualization
├── requirements.txt             # Core dependencies
├── requirements_demo.txt        # Visualization dependencies
└── pyproject.toml
```

---

## Citation

If you use this project, please consider citing the following paper:

```bibtex
@article{jia2025drivevggt,
  title={DriveVGGT: Visual Geometry Transformer for Autonomous Driving},
  author={Jia, Xiaosong and Liu, Yanhao and You, Junqi and Xia, Renqiu and Hong, Yu and Yan, Junchi},
  journal={arXiv preprint arXiv:2511.22264},
  year={2025}
}
```
