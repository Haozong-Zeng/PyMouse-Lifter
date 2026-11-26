# Getting started

Step-by-step guide from a top-view video to 3D keypoints, behavioral features, and a
behavior classifier. Commands assume Linux with an NVIDIA GPU; a CPU also works but is slow.
If you use a coding assistant, point it to [AGENTS.md](../AGENTS.md) first.

## 1. Install

```bash
git clone https://github.com/Haozong-Zeng/PyMouse-Lifter.git
cd PyMouse-Lifter
conda env create -n pymouselifter -f Depth-Anything/PyMouseLifter_environment.yml
conda activate pymouselifter
```

## 2. Download models

* Depth model: the "Lightened model" link in
  `Depth-Anything/metric_depth/checkpoints/links_to_models.txt`. Save it as
  `Depth-Anything/metric_depth/checkpoints/depth_anything_metric_PyMouse_HQ_orbbec_real.pt`,
  or pass its path with `--depth-checkpoint`.
* 2D pose model (optional, if you do not bring your own 2D keypoints): the YOLO link in
  `Depth-Anything/other_models/links_to_models.txt`.

## 3. Lift keypoints to 3D

With the YOLO pose model:

```bash
cd Depth-Anything
python run_PyMouseLifter_lift_keypoints.py --video /path/to/video.mp4 \
    --yolo-checkpoint other_models/yolo11l-pose-topview14keypoints.pt --outdir /path/to/output
```

With 2D keypoints from DeepLabCut or SLEAP, convert them first
([IMPORT_2D_KEYPOINTS.md](IMPORT_2D_KEYPOINTS.md)) and pass `--keypoints-2d video_2D.npy`
instead of `--yolo-checkpoint`.

Output: `video_2D.npy`, `video_3D.npy`, and `video_metadata.json`. The coordinate
convention is described in the script header and in [Feature-Definition.md](../Feature-Definition.md).
Use `--max-frames 300` for a quick test.

## 4. Compute behavioral features

```bash
cd ../PyMouse-Lifter-Classification
python pymouselifter_features.py /path/to/video.mp4 /path/to/output/video_2D.npy \
    /path/to/output/video_3D.npy -o /path/to/output/video_features.csv
```

The table has one row per frame from frame 9 onward and the 74 features of the paper, in
the same column order. Some motion features are empty (NaN) in the first frames and where
keypoints are missing; tree-based classifiers such as XGBoost accept NaN.

## 5. Train a behavior classifier

Label behaviors on a subset of frames (any annotation tool), join the labels to the feature
table by frame index, and train a classifier. The notebook
`PyMouse-Lifter-Classification/3D classification XGB.ipynb` shows the XGBoost
one-versus-rest setup used in the paper. Keep all frames of one animal in the same split so
that test results are not inflated by near-identical neighboring frames.

## 6. New arenas: synthetic training data (optional)

The provided depth model was fine-tuned for top-view open-field arenas. For a substantially
different arena or camera, generate synthetic training data:

1. Download a Blender project from `Blender/links_to_projects.txt` and replace the arena
   with a model of yours (CAD export, or a photogrammetry scan).
2. Drive the mouse model with lifted keypoints from step 3 (pseudo labels) and render RGB
   images, depth maps (PNG value = round(depth_mm / 2.3529)), and mouse masks named
   `rgb_*.jpg`, `depth_*.png`, and `contour_*.png` into
   `Depth-Anything/metric_depth/data/PyMouse_HQ/<scene>/`.
3. List the pairs in `Depth-Anything/metric_depth/train_test_inputs/PyMouse_HQ_train_files_with_gt.txt`
   (one `rgb_path depth_path 2.3529` per line) and fine-tune as described in
   `Depth-Anything/README.md` and `Depth-Anything/metric_depth/README.md`.

The Blender projects contain their own scripts; paths inside them may need to be changed
to your folders. A coding assistant can do this through Blender's Python API or a Blender
MCP server. Check renders visually before training.
