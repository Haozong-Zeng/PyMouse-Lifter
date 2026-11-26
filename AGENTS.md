# Guide for coding assistants (Codex, Claude Code, and similar)

This file tells an AI coding assistant how this repository is organised and how to run
the PyMouse Lifter pipeline for a user. Human users can read it too; the step-by-step
guides are in [docs/](docs/).

## What the pipeline does

PyMouse Lifter turns top-view videos of a single mouse into 3D keypoints and per-frame
behavioral features, using only 2D keypoint annotation:

1. **2D keypoints** per frame: from the provided YOLO pose model, or imported from
   DeepLabCut or SLEAP ([docs/IMPORT_2D_KEYPOINTS.md](docs/IMPORT_2D_KEYPOINTS.md)).
2. **3D lifting**: a Depth Anything model fine-tuned on synthetic mouse images predicts a
   depth map; each 2D keypoint takes the depth at its pixel.
3. **Features**: 74 posture, motion, and image-change features per frame
   ([Feature-Definition.md](Feature-Definition.md)).
4. **Classification**: a classifier trained on the user's own behavior labels.
5. **Synthetic data (optional)**: Blender projects pose a 3D mouse model from lifted
   keypoints and render images with exact depth labels, used to fine-tune the depth model
   for a new arena.

## Repository map

| Path | Content |
| --- | --- |
| `Depth-Anything/run_PyMouseLifter_lift_keypoints.py` | Video → `<video>_2D.npy`, `<video>_3D.npy` (YOLO or imported 2D keypoints) |
| `tools/import_2d_keypoints.py` | DeepLabCut CSV/H5 or SLEAP analysis H5 → `<video>_2D.npy` |
| `PyMouse-Lifter-Classification/pymouselifter_features.py` | Video + 2D + 3D arrays → 74-feature table |
| `PyMouse-Lifter-Classification/3D classification XGB.ipynb` | Classifier training and evaluation used in the paper |
| `Depth-Anything/metric_depth/` | Depth fine-tuning code (ZoeDepth/Depth Anything), dataset lists in `train_test_inputs/` |
| `Depth-Anything/run_PyMouseLifter_offline_demo.py`, `..._online_*.py` | Real-time demos (use a fixed 29-feature model) |
| `Blender/links_to_projects.txt` | Download links of the Blender projects for synthetic data |
| `Feature-Definition.md` | Definition of every feature |
| `tests/` | Small tests (`python tests/test_import_2d_keypoints.py`) |

## Environment

* `Depth-Anything/PyMouseLifter_environment.yml` (conda, Linux, CUDA). It provides PyTorch,
  torchvision, opencv-python, ultralytics, numpy, pandas, and h5py. Install `xgboost`
  with pip if you train classifiers.
* Run scripts in `Depth-Anything/` from that folder: they import `depth_anything`
  from it. The local DINOv2 code is in `Depth-Anything/tourchhub/` (this spelling is
  intentional) and is found automatically.
* Model weights are not in git. Download links:
  `Depth-Anything/metric_depth/checkpoints/links_to_models.txt` (depth models; use the
  "lightened" model for `run_PyMouseLifter_lift_keypoints.py`) and
  `Depth-Anything/other_models/links_to_models.txt` (YOLO pose model, real-time classifier).
  Put depth checkpoints in `Depth-Anything/metric_depth/checkpoints/`.

## Data conventions (do not change silently)

* Keypoint order. The first eight keypoints must be: `nose, head, left_ear, right_ear,
  neck, spine_center, lumbar_spine, tail_base`. The 14-keypoint YOLO model adds paws and
  tail points after these; features use only the first eight.
* `<video>_2D.npy`: float32 `(frames, K, 2)`, video pixels, NaN when missing.
* `<video>_3D.npy`: float32 `(frames, K, 3)`. x and y are pixels of the 518 × 518 depth map
  from its centre (x right, y up); z is the depth-network output × 2.3529 and increases with
  height. The axes are not calibrated to a common physical unit; do not label them mm.
* Frame rate is assumed to be 30 fps in the feature code (`--fps` to change it).
* Depth training labels: `depth_*.png` (uint8, value = round(depth_mm / 2.3529)), with
  `rgb_*.jpg` and optional mouse mask `contour_*.png` in the same folder; list files
  `train_test_inputs/*.txt` have lines `rgb_path depth_path 2.3529`.

## Typical requests and what to do

* "Get 3D keypoints for my videos": follow [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md),
  steps 1–3. Check that the arena and camera resemble a top-view open field; otherwise warn
  the user that the depth model may need new synthetic training data.
* "I have DeepLabCut/SLEAP labels": ask for the file and the names of their body parts,
  build the `--map` argument, then follow [docs/IMPORT_2D_KEYPOINTS.md](docs/IMPORT_2D_KEYPOINTS.md).
* "Classify behaviors": compute features (step 4), then train a classifier on the user's
  labels. Split train/test by animal, not by frame, to avoid leakage (see the notebook).
* "New arena": see the synthetic-data section of [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).
  Blender work can be done by the user in the GUI, or by an assistant through Blender's
  Python API (`blender -b project.blend --python script.py`) or a Blender MCP server.

## Rules for assistants

* Never overwrite users' videos, labels, or existing `_2D.npy`/`_3D.npy` files; write to a
  new output folder.
* Report how many frames and keypoints are missing (NaN) after each step.
* Do not describe lifted coordinates as millimetres, and do not claim accuracy outside the
  validated setting (top-view open-field arenas; ten keypoints of head, trunk, and tail).
* Large files (videos, weights, `.blend`) must not be committed to git.
