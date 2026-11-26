"""Convert DeepLabCut or SLEAP keypoints to a PyMouse Lifter `<video>_2D.npy` array.

Output: float32 array of shape (frames, K, 2) in video pixels, NaN where a keypoint is
missing or below the confidence threshold. The default K = 8 body keypoints, in the order
the feature code expects: nose, head, left_ear, right_ear, neck, spine_center,
lumbar_spine, tail_base. Add `--extra tail_mid,tail_tip` to append more keypoints.

Your labels rarely use these names, so give a mapping from PyMouse Lifter names to yours:
    --map nose=snout,head=headcenter,left_ear=earL,right_ear=earR,neck=neck,
          spine_center=back,lumbar_spine=hip,tail_base=tailbase

Supported inputs:
  * DeepLabCut CSV or H5 (single- or multi-animal; choose with --individual)
  * SLEAP analysis H5 (File > Export Analysis HDF5 or `sleap-convert --format analysis`;
    choose the track with --track)

Example:
    python tools/import_2d_keypoints.py video_DLC.csv -o video_2D.npy --map nose=snout,...
"""
import argparse
from pathlib import Path

import numpy as np

PYMOUSE_KEYPOINTS = ['nose', 'head', 'left_ear', 'right_ear', 'neck',
                     'spine_center', 'lumbar_spine', 'tail_base']


def parse_map(text):
    mapping = {}
    for item in filter(None, (text or '').split(',')):
        key, _, value = item.partition('=')
        if not value:
            raise ValueError(f'Bad mapping item {item!r}; use pymouse_name=your_name')
        mapping[key.strip()] = value.strip()
    return mapping


def read_deeplabcut(path, individual=None):
    """Return (names, xy, confidence) from a DeepLabCut CSV or H5 file."""
    import pandas as pd
    path = Path(path)
    if path.suffix.lower() in ('.h5', '.hdf5'):
        df = pd.read_hdf(path)
    else:
        with open(path) as f:
            header_rows = sum(1 for _, line in zip(range(4), f) if line.split(',')[0] in ('scorer', 'individuals', 'bodyparts', 'coords'))
        df = pd.read_csv(path, header=list(range(header_rows)), index_col=0)
    levels = list(df.columns.names)
    if 'individuals' in levels:
        names = list(dict.fromkeys(df.columns.get_level_values('individuals')))
        individual = individual or names[0]
        if individual not in names:
            raise ValueError(f'Individual {individual!r} not found; available: {names}')
        df = df.xs(individual, axis=1, level='individuals')
    df = df.droplevel('scorer', axis=1) if 'scorer' in df.columns.names else df
    bodyparts = list(dict.fromkeys(df.columns.get_level_values('bodyparts')))
    xy = np.stack([np.stack([df[(b, 'x')].to_numpy(float), df[(b, 'y')].to_numpy(float)], axis=-1)
                   for b in bodyparts], axis=1)
    conf = np.stack([df[(b, 'likelihood')].to_numpy(float) if (b, 'likelihood') in df.columns
                     else np.ones(len(df)) for b in bodyparts], axis=1)
    return bodyparts, xy, conf


def read_sleap(path, track=0):
    """Return (names, xy, confidence) from a SLEAP analysis HDF5 file."""
    import h5py
    with h5py.File(path, 'r') as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f['node_names'][:]]
        tracks = f['tracks'][:]                      # (tracks, 2, nodes, frames)
        scores = f['point_scores'][:] if 'point_scores' in f else None
    if track >= tracks.shape[0]:
        raise ValueError(f'Track {track} not found; file has {tracks.shape[0]} tracks')
    xy = np.transpose(tracks[track], (2, 1, 0))      # (frames, nodes, 2)
    conf = np.transpose(scores[track], (1, 0)) if scores is not None else np.ones(xy.shape[:2])
    return names, xy, conf


def convert(names, xy, conf, mapping, extra=(), min_confidence=0.0):
    wanted = PYMOUSE_KEYPOINTS + list(extra)
    out = np.full((xy.shape[0], len(wanted), 2), np.nan, dtype=np.float32)
    missing = []
    for j, key in enumerate(wanted):
        source = mapping.get(key, key)
        if source not in names:
            missing.append(f'{key} (looked for {source!r})')
            continue
        i = names.index(source)
        good = np.isfinite(xy[:, i]).all(axis=1) & (np.nan_to_num(conf[:, i], nan=0.0) >= min_confidence)
        out[good, j] = xy[good, i]
    return out, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='DeepLabCut .csv/.h5 or SLEAP analysis .h5')
    parser.add_argument('-o', '--output', required=True, help='Output <video>_2D.npy')
    parser.add_argument('--format', choices=['auto', 'deeplabcut', 'sleap'], default='auto')
    parser.add_argument('--map', default='', help='pymouse_name=your_name,... (see above)')
    parser.add_argument('--extra', default='', help='Extra keypoints to append after the eight body keypoints')
    parser.add_argument('--min-confidence', type=float, default=0.0, help='Likelihood/score threshold; lower points become NaN')
    parser.add_argument('--individual', help='DeepLabCut multi-animal: individual to export')
    parser.add_argument('--track', type=int, default=0, help='SLEAP: track index to export')
    args = parser.parse_args()
    fmt = args.format
    if fmt == 'auto':
        fmt = 'deeplabcut' if args.input.lower().endswith('.csv') else None
        if fmt is None:
            import h5py
            with h5py.File(args.input, 'r') as f:
                fmt = 'sleap' if 'tracks' in f and 'node_names' in f else 'deeplabcut'
    names, xy, conf = (read_sleap(args.input, args.track) if fmt == 'sleap'
                       else read_deeplabcut(args.input, args.individual))
    extra = [e.strip() for e in args.extra.split(',') if e.strip()]
    out, missing = convert(names, xy, conf, parse_map(args.map), extra, args.min_confidence)
    if missing:
        print('WARNING: not found, left as NaN:', '; '.join(missing))
        print('Available keypoints:', ', '.join(names))
    np.save(args.output, out)
    valid = np.isfinite(out).all(axis=-1).mean(axis=0)
    print(f'Saved {args.output}: {out.shape}; valid fraction per keypoint:',
          ', '.join(f'{k}={v:.2f}' for k, v in zip(PYMOUSE_KEYPOINTS + extra, valid)))


if __name__ == '__main__':
    main()
