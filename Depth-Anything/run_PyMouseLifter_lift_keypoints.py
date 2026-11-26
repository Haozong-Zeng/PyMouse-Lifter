"""Lift 2D mouse keypoints of a top-view video to 3D with the fine-tuned depth model.

2D keypoints come either from a YOLO pose model (`--yolo-checkpoint`) or from a file you
already have (`--keypoints-2d`, e.g. converted from DeepLabCut or SLEAP with
tools/import_2d_keypoints.py). Outputs, next to each other in --outdir:

  <video>_2D.npy  (frames, K, 2) keypoints in video pixels, NaN if missing
  <video>_3D.npy  (frames, K, 3) lifted keypoints, NaN if missing
  <video>_metadata.json

3D convention (the one used for the paper's feature table): x and y are pixels of the
depth map measured from its centre (x to the right, y up); z is the depth-network output
multiplied by 2.3529 and increases with height above the floor. The three axes are not
calibrated to a common physical unit.

Examples:
  python run_PyMouseLifter_lift_keypoints.py --video my.mp4 --yolo-checkpoint other_models/yolo11l-pose-topview14keypoints.pt
  python run_PyMouseLifter_lift_keypoints.py --video my.mp4 --keypoints-2d my_2D.npy
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

ROOT = Path(__file__).resolve().parent
COEFFICIENT = 2.3529
MODEL_CONFIGS = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}


def make_transform(size=518):
    return Compose([
        Resize(width=size, height=size, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


def lift(xy, depth_map, frame_shape):
    """xy: (K, 2) video pixels (NaN allowed). Returns (K, 3) lifted points."""
    frame_h, frame_w = frame_shape[:2]
    depth_h, depth_w = depth_map.shape
    out = np.full((len(xy), 3), np.nan, dtype=np.float32)
    ok = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < frame_w) & (xy[:, 1] >= 0) & (xy[:, 1] < frame_h)
    if ok.any():
        scaled = xy[ok] * np.array([depth_w / frame_w, depth_h / frame_h])
        col = np.clip(scaled[:, 0].astype(np.int64), 0, depth_w - 1)
        row = np.clip(scaled[:, 1].astype(np.int64), 0, depth_h - 1)
        out[ok] = np.column_stack((scaled[:, 0] - depth_w / 2, depth_h / 2 - scaled[:, 1], depth_map[row, col]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', type=Path, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--keypoints-2d', type=Path, help='(frames, K, 2) or (frames, K, 3) array in video pixels')
    src.add_argument('--yolo-checkpoint', type=Path, help='Ultralytics pose model; the first detection per frame is used')
    p.add_argument('--depth-checkpoint', type=Path,
                   default=ROOT / 'metric_depth/checkpoints/depth_anything_metric_PyMouse_HQ_orbbec_real.pt',
                   help='Lightened fine-tuned checkpoint (see metric_depth/checkpoints/links_to_models.txt)')
    p.add_argument('--encoder', choices=list(MODEL_CONFIGS), default='vitl')
    p.add_argument('--outdir', type=Path, default=Path('.'))
    p.add_argument('--max-frames', type=int)
    p.add_argument('--min-confidence', type=float, default=0.0, help='YOLO keypoint confidence threshold')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthAnything(MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(args.depth_checkpoint, map_location='cpu'))
    model = model.to(device).eval()
    if device == 'cuda':
        model = model.half()
    dtype = next(model.parameters()).dtype
    transform = make_transform()

    given = None
    if args.keypoints_2d:
        given = np.load(args.keypoints_2d)[..., :2].astype(np.float32)
    else:
        from ultralytics import YOLO
        yolo = YOLO(str(args.yolo_checkpoint))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open video: {args.video}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n, args.max_frames) if args.max_frames else n
    if given is not None and len(given) < n:
        raise SystemExit(f'--keypoints-2d has {len(given)} frames but the video has {n}')
    all2d, all3d = [], []
    with torch.inference_mode():
        for t in tqdm(range(n), desc=args.video.name):
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = torch.from_numpy(transform({'image': rgb})['image'])[None].to(device=device, dtype=dtype)
            depth = model(x).float().cpu().numpy().squeeze() * COEFFICIENT
            if given is not None:
                xy = given[t]
            else:
                res = yolo.predict(frame, verbose=False)[0]
                if res.keypoints is None or len(res.keypoints) == 0:
                    xy = np.full((yolo.model.kpt_shape[0], 2), np.nan, np.float32)
                else:
                    kp = res.keypoints.data[0].float().cpu().numpy()
                    xy = kp[:, :2].copy()
                    if kp.shape[1] > 2:
                        xy[kp[:, 2] <= args.min_confidence] = np.nan
                    xy[(xy == 0).all(axis=1)] = np.nan
            all2d.append(xy.astype(np.float32))
            all3d.append(lift(xy, depth, frame.shape))
    cap.release()
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    np.save(args.outdir / f'{stem}_2D.npy', np.stack(all2d))
    np.save(args.outdir / f'{stem}_3D.npy', np.stack(all3d))
    meta = dict(video=str(args.video), frames=len(all3d), keypoints=int(all3d[0].shape[0]),
                source=str(args.keypoints_2d or args.yolo_checkpoint), depth_checkpoint=str(args.depth_checkpoint),
                coordinates=dict(x='depth-map pixels from centre, right positive', y='depth-map pixels from centre, up positive',
                                 z=f'depth-network output x {COEFFICIENT}; larger = higher; not calibrated to x/y units'))
    (args.outdir / f'{stem}_metadata.json').write_text(json.dumps(meta, indent=2))
    print(f'Saved {stem}_2D.npy and {stem}_3D.npy ({len(all3d)} frames) to {args.outdir}')


if __name__ == '__main__':
    main()
