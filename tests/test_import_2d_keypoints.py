"""Run: python tests/test_import_2d_keypoints.py (needs numpy, pandas, h5py)."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools' / 'import_2d_keypoints.py'
NAMES = ['snout', 'headc', 'earL', 'earR', 'neck', 'back', 'hip', 'tailbase', 'tailtip']
MAP = 'nose=snout,head=headc,left_ear=earL,right_ear=earR,spine_center=back,lumbar_spine=hip,tail_base=tailbase'


def run(*args):
    subprocess.run([sys.executable, str(TOOL), *map(str, args)], check=True, capture_output=True, text=True)


def expected(xy, conf, threshold):
    out = xy[:, :8].copy()
    out[conf[:, :8] < threshold] = np.nan
    return out


def test_deeplabcut_csv(tmp, xy, conf):
    rows = ['scorer,' + ','.join(['DLC'] * 3 * len(NAMES)),
            'bodyparts,' + ','.join(n for n in NAMES for _ in range(3)),
            'coords,' + ','.join(['x', 'y', 'likelihood'] * len(NAMES))]
    for t in range(len(xy)):
        rows.append(f'{t},' + ','.join(f'{xy[t, j, 0]},{xy[t, j, 1]},{conf[t, j]}' for j in range(len(NAMES))))
    path = tmp / 'video_DLC.csv'
    path.write_text('\n'.join(rows) + '\n')
    run(path, '-o', tmp / 'dlc_2D.npy', '--map', MAP + ',tail_tip=tailtip', '--min-confidence', 0.5, '--extra', 'tail_tip')
    got = np.load(tmp / 'dlc_2D.npy')
    assert got.shape == (len(xy), 9, 2)
    np.testing.assert_allclose(got[:, :8], expected(xy, conf, 0.5), rtol=1e-6)


def test_sleap(tmp, xy, conf):
    import h5py
    path = tmp / 'video.analysis.h5'
    with h5py.File(path, 'w') as f:
        f['node_names'] = np.array([n.encode() for n in NAMES])
        tracks = np.full((2, 2, len(NAMES), len(xy)), np.nan)
        tracks[1] = np.transpose(xy, (2, 1, 0))
        f['tracks'] = tracks
        scores = np.zeros((2, len(NAMES), len(xy)))
        scores[1] = conf.T
        f['point_scores'] = scores
    run(path, '-o', tmp / 'sleap_2D.npy', '--map', MAP, '--track', 1, '--min-confidence', 0.5)
    got = np.load(tmp / 'sleap_2D.npy')
    np.testing.assert_allclose(got, expected(xy, conf, 0.5), rtol=1e-6)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 720, (50, len(NAMES), 2))
    conf = rng.uniform(0, 1, (50, len(NAMES)))
    with tempfile.TemporaryDirectory() as d:
        test_deeplabcut_csv(Path(d), xy, conf)
        test_sleap(Path(d), xy, conf)
    print('import_2d_keypoints: all tests passed')
