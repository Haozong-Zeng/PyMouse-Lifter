# Using 2D keypoints from DeepLabCut or SLEAP

If you already track your videos with DeepLabCut or SLEAP, you can keep using their
labeling interfaces and only use PyMouse Lifter for the 3D step.

## 1. Body parts needed

The feature code needs eight body parts in this order:

| PyMouse Lifter name | Where |
| --- | --- |
| nose | tip of the snout |
| head | centre of the head between the ears |
| left_ear, right_ear | ear bases |
| neck | base of the skull |
| spine_center | middle of the back |
| lumbar_spine | lower back |
| tail_base | base of the tail |

Missing body parts are allowed (they become NaN), but features that use them will be empty.

## 2. Convert

```bash
python tools/import_2d_keypoints.py video_DLC.csv -o video_2D.npy --min-confidence 0.6 \
  --map nose=snout,head=head,left_ear=leftear,right_ear=rightear,neck=neck,spine_center=midback,lumbar_spine=lowerback,tail_base=tailbase
```

* DeepLabCut: `.csv` or `.h5` output of `analyze_videos`. Multi-animal projects: add
  `--individual <name>`.
* SLEAP: export an analysis file (`File > Export Analysis HDF5`, or
  `sleap-convert --format analysis -o video.analysis.h5 predictions.slp`), then
  `python tools/import_2d_keypoints.py video.analysis.h5 -o video_2D.npy --map ... --track 0`.
* `--min-confidence` turns low-likelihood points into NaN.
* `--extra tail_mid,tail_tip` appends further keypoints after the eight.

The script prints the fraction of valid frames per keypoint and lists the available
names when a mapping does not match.

## 3. Lift and compute features

```bash
cd Depth-Anything
python run_PyMouseLifter_lift_keypoints.py --video video.mp4 --keypoints-2d video_2D.npy --outdir out
cd ../PyMouse-Lifter-Classification
python pymouselifter_features.py video.mp4 ../Depth-Anything/out/video_2D.npy ../Depth-Anything/out/video_3D.npy -o video_features.csv
```

The 2D file must have one row per video frame and coordinates in pixels of the same video.
