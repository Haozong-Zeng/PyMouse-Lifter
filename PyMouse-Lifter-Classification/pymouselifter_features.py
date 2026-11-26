"""PyMouse Lifter behavioral features (74 per frame).

The functions below are copied from the notebooks that produced the paper's
feature table ("3D classification new XGB.ipynb" and "data_preprocessing.ipynb");
only comments and messages were translated. Definitions of every feature are in
Feature-Definition.md at the repository root.

Command line:
    python pymouselifter_features.py VIDEO.mp4 VIDEO_2D.npy VIDEO_3D.npy -o VIDEO_features.csv

VIDEO_2D.npy: (frames, K, 2) keypoints in video pixels; VIDEO_3D.npy: (frames, K, 3)
lifted keypoints. The first eight keypoints must be, in order: nose, head, left_ear,
right_ear, neck, spine_center, lumbar_spine, tail_base.
"""
import argparse

import cv2
import numpy as np
import pandas as pd

KEYPOINT_NAMES = ['nose', 'head', 'left_ear', 'right_ear', 'neck',
                  'spine_center', 'lumbar_spine', 'tail_base']
IDX_NOSE, IDX_HEAD, IDX_LEFT_EAR, IDX_RIGHT_EAR = 0, 1, 2, 3
IDX_NECK, IDX_SPINE_CENTER, IDX_LUMBAR_SPINE, IDX_TAIL_BASE = 4, 5, 6, 7
FRAMES_TO_DROP = 9  # the paper's feature table starts at frame 9


def rotate_around_z(points, alpha):
    """
    Rotate points around the Z axis in a right-handed coordinate system.
    
    points has shape (..., 3). alpha is the rotation angle in radians;
    positive values rotate counterclockwise when viewed from positive Z
    toward the origin. The output preserves the input shape.
    """
    c, s = np.cos(alpha), np.sin(alpha)
    Rz = np.array([
        [ c, -s,  0],
        [ s,  c,  0],
        [ 0,  0,  1]], dtype=np.float32)
    return points @ Rz.T


def align_tailbase_lumbar(frame_3d):
    """
    Align one frame of 3D keypoints.
    
    Translate tail_base to the origin, then rotate around Z so the XY
    projection of the tail_base-to-lumbar_spine vector aligns with positive Y.
    frame_3d and the returned array have shape (num_keypoints, 3).
    """
    tail_base    = frame_3d[IDX_TAIL_BASE]     # (x_tail, y_tail, z_tail)
    lumbar_spine = frame_3d[IDX_LUMBAR_SPINE]  # (x_lum, y_lum, z_lum)

    # (1) Translate tail_base to the origin.
    shifted_points = frame_3d - tail_base

    # (2) Compute the required rotation angle alpha around the Z axis.
    vec = lumbar_spine - tail_base
    dx, dy = vec[0], vec[1]
    theta = np.arctan2(dy, dx)   # Angle relative to the X axis.
    alpha = np.pi/2 - theta      # Align with positive Y, whose angle is pi/2.

    rotated_points = rotate_around_z(shifted_points, alpha)
    return rotated_points


def compute_segment_angle_2d(p1, p2):
    """
    Return the angle of p1 -> p2 in the XY plane relative to the X axis.
    
    Ignore Z coordinates. The result lies in (-pi, pi].
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)


def extract_features_3d(keypoints_3d, fps=30.0, frames_to_drop=10):
    """
    Extract features from a 3D keypoint sequence.
    
    keypoints_3d has shape (num_frames, num_keypoints, 3), and fps is the
    sampling rate in frames per second. The returned dictionary contains:
      1) Selected 3D keypoint coordinates aligned using tail_base and lumbar_spine.
      2) XY-plane angles for five body segments.
      3) Angular velocities and accelerations from aligned segment angles.
      4) Head/tail-base linear velocities, accelerations, and scalar speeds
         computed in the original coordinates.
      5) The body-polyline length divided by the nose-to-tail-base distance.
      6) Rolling standard deviations of segment angular velocities.
      7) Rolling standard deviations of head and tail-base velocity components.
    
    Rolling windows include the current frame and the preceding frames_to_drop
    frames. Early motion features are masked with NaN.
    """
    num_frames = keypoints_3d.shape[0]
    dt = 1.0 / fps

    # Preallocate feature arrays.
    # (1) All aligned 3D coordinates.
    aligned_points = np.zeros_like(keypoints_3d)   # (num_frames, 10, 3)

    # Define the five body segments:
    # (nose->head), (head->neck), (neck->spine_center), (spine_center->lumbar_spine), (lumbar_spine->tail_base)
    segment_pairs = [
        (IDX_NOSE, IDX_HEAD),
        (IDX_HEAD, IDX_NECK),
        (IDX_NECK, IDX_SPINE_CENTER),
        (IDX_SPINE_CENTER, IDX_LUMBAR_SPINE),
        (IDX_LUMBAR_SPINE, IDX_TAIL_BASE),
    ]
    n_segments = len(segment_pairs)

    # (2) Body-segment angles.
    angles_2d = np.zeros((num_frames, n_segments), dtype=np.float32)

    # (3) Angular velocity and angular acceleration.
    ang_velocity = np.full((num_frames, n_segments), np.nan, dtype=np.float32)
    ang_acc      = np.full((num_frames, n_segments), np.nan, dtype=np.float32)

    # (4) Linear velocity and acceleration for the head and tail base.
    head_vel  = np.full((num_frames, 3), np.nan, dtype=np.float32)
    head_acc  = np.full((num_frames, 3), np.nan, dtype=np.float32)
    tail_vel  = np.full((num_frames, 3), np.nan, dtype=np.float32)
    tail_acc  = np.full((num_frames, 3), np.nan, dtype=np.float32)
    # (4.x) Also collect scalar speeds for the head and tail base.
    head_speed = np.full((num_frames,), np.nan, dtype=np.float32)  
    tail_speed = np.full((num_frames,), np.nan, dtype=np.float32)


    # (5) Bend ratio.
    bend_ratio = np.zeros(num_frames, dtype=np.float32)

    # (6), (7) Rolling standard deviations of angular and head/tail-base velocities.
    segment_ang_velocity_std = np.full((num_frames, n_segments), np.nan, dtype=np.float32)
    head_vel_3d_std          = np.full((num_frames, 3),         np.nan, dtype=np.float32)
    tailbase_vel_3d_std      = np.full((num_frames, 3),         np.nan, dtype=np.float32)

    # Compute features (1), (2), and (5).
    # Body polyline: nose -> head -> neck -> spine_center -> lumbar_spine -> tail_base.
    bend_chain = [IDX_NOSE, IDX_HEAD, IDX_NECK, IDX_SPINE_CENTER, IDX_LUMBAR_SPINE, IDX_TAIL_BASE]

    for t in range(num_frames):
        # (1) Align the keypoints.
        aligned_frame = align_tailbase_lumbar(keypoints_3d[t])
        aligned_points[t] = aligned_frame

        # (2) Angles of the five body segments.
        for i, (idx1, idx2) in enumerate(segment_pairs):
            p1 = aligned_frame[idx1]
            p2 = aligned_frame[idx2]
            angles_2d[t, i] = compute_segment_angle_2d(p1, p2)

        # (5) Compute the bend ratio.
        poly_len = 0.0
        for k in range(len(bend_chain) - 1):
            pA = aligned_frame[bend_chain[k]]
            pB = aligned_frame[bend_chain[k+1]]
            dist_AB = np.linalg.norm(pB - pA)
            poly_len += dist_AB

        nose_pt = aligned_frame[IDX_NOSE]
        tail_pt = aligned_frame[IDX_TAIL_BASE]
        direct_dist = np.linalg.norm(tail_pt - nose_pt) + 1e-8  # Prevent a zero denominator.
        bend_ratio[t] = poly_len / direct_dist

    # (3) Angular velocity and acceleration from aligned angles.
    for t in range(1, num_frames):
        ang_velocity[t] = (angles_2d[t] - angles_2d[t-1]) / dt
    for t in range(2, num_frames):
        ang_acc[t] = (ang_velocity[t] - ang_velocity[t-1]) / dt

    # Mask the first frames_to_drop frames, consistent with the rolling-window features.
    ang_velocity[:frames_to_drop, :] = np.nan
    ang_acc[:frames_to_drop, :]      = np.nan

    # (4) Head/tail-base velocity and acceleration in original coordinates.
    for t in range(1, num_frames):
        head_vel[t] = (keypoints_3d[t, IDX_HEAD] - keypoints_3d[t-1, IDX_HEAD]) / dt
        tail_vel[t] = (keypoints_3d[t, IDX_TAIL_BASE] - keypoints_3d[t-1, IDX_TAIL_BASE]) / dt
        head_speed[t] = np.linalg.norm(head_vel[t])
        tail_speed[t] = np.linalg.norm(tail_vel[t])

    for t in range(2, num_frames):
        head_acc[t] = (head_vel[t] - head_vel[t-1]) / dt
        tail_acc[t] = (tail_vel[t] - tail_vel[t-1]) / dt

    head_vel[:frames_to_drop, :] = np.nan
    head_acc[:frames_to_drop, :] = np.nan
    tail_vel[:frames_to_drop, :] = np.nan
    tail_acc[:frames_to_drop, :] = np.nan
    head_speed[:frames_to_drop] = np.nan
    tail_speed[:frames_to_drop] = np.nan

    # (6), (7) Standard deviations over the preceding frames plus the current frame.
    # Keep early frames as NaN; thereafter use the inclusive window [t - frames_to_drop, t].
    for t in range(frames_to_drop, num_frames):
        # segment_ang_velocity_std
        window_av = ang_velocity[t-frames_to_drop : t+1, :]  # Shape: (frames_to_drop + 1, n_segments).
        segment_ang_velocity_std[t] = np.nanstd(window_av, axis=0)

        # head_vel_3d_std
        window_hv = head_vel[t-frames_to_drop : t+1, :]  # Shape: (frames_to_drop + 1, 3).
        head_vel_3d_std[t] = np.nanstd(window_hv, axis=0)

        # tailbase_vel_3d_std
        window_tv = tail_vel[t-frames_to_drop : t+1, :]  # Shape: (frames_to_drop + 1, 3).
        tailbase_vel_3d_std[t] = np.nanstd(window_tv, axis=0)

    # Export the eight selected aligned keypoints from (1).
    nose_3d          = aligned_points[:, IDX_NOSE, :]
    head_3d          = aligned_points[:, IDX_HEAD, :]
    left_ear_3d      = aligned_points[:, IDX_LEFT_EAR, :]
    right_ear_3d     = aligned_points[:, IDX_RIGHT_EAR, :]
    neck_3d          = aligned_points[:, IDX_NECK, :]
    spine_center_3d  = aligned_points[:, IDX_SPINE_CENTER, :]
    lumbar_spine_3d  = aligned_points[:, IDX_LUMBAR_SPINE, :]
    tail_base_3d     = aligned_points[:, IDX_TAIL_BASE, :]

    # Assemble the output feature dictionary.
    feat_dict = {
        # (1) Aligned coordinates for each keypoint.
        'nose_3d':             nose_3d,
        'head_3d':             head_3d,
        'left_ear_3d':         left_ear_3d,
        'right_ear_3d':        right_ear_3d,
        'neck_3d':             neck_3d,
        'spine_center_3d':     spine_center_3d,
        'lumbar_spine_3d':     lumbar_spine_3d,
        'tail_base_3d':        tail_base_3d,

        # (2), (3), (5)
        'segment_angles':       angles_2d,          # (num_frames, 5)
        'segment_ang_velocity': ang_velocity,    # (num_frames, 5)
        'segment_ang_acc':      ang_acc,         # (num_frames, 5)
        'bend_ratio':           bend_ratio,      # (num_frames, )

        # (4) Head and tail-base linear velocity and acceleration in original coordinates.
        'head_vel_3d':          head_vel,
        'head_acc_3d':          head_acc,
        'tailbase_vel_3d':      tail_vel,
        'tailbase_acc_3d':      tail_acc,
        'head_speed':           head_speed,
        'tail_speed':           tail_speed,

        # (6), (7) Rolling standard-deviation features.
        'segment_ang_velocity_std': segment_ang_velocity_std,  # (num_frames, 5)
        'head_vel_3d_std':          head_vel_3d_std,          # (num_frames, 3)
        'tailbase_vel_3d_std':      tailbase_vel_3d_std,      # (num_frames, 3)
    }

    return feat_dict


def flatten_feat_dict(feat_dict, frames_to_drop=10):
    """
    Flatten feature arrays into one-dimensional columns and trim initial frames.
    
    Parameters
    ----------
    feat_dict : dict
        Feature names mapped to arrays of shape (num_frames,) or
        (num_frames, D).
    frames_to_drop : int, default 10
        Number of initial frames to discard; use 0 to retain all frames.
    
    Returns
    -------
    flat_feat_dict : dict
        Column names mapped to arrays of shape (num_frames - frames_to_drop,).
        Two- and three-component features receive coordinate suffixes such
        as _x, _y, and _z. Other multicomponent features receive numeric
        suffixes. One-dimensional feature names remain unchanged.
    """
    flat_feat_dict = {}
    axis = ['x', 'y', 'z']
    
    for key, arr in feat_dict.items():
        # Convert the feature values to a NumPy array.
        arr = np.asarray(arr)
        
        # Discard the first frames_to_drop frames.
        # Dropping at least the full sequence length produces an empty array.
        # The caller is expected to leave enough frames for downstream processing.
        arr = arr[frames_to_drop:]
        
        if arr.ndim == 1:
            # (num_frames - frames_to_drop,)
            flat_feat_dict[key] = arr
        elif arr.ndim == 2:
            # (num_frames - frames_to_drop, D)
            num_frames, dim = arr.shape
            if dim == 3 or dim ==2:
                for d in range(dim):
                    new_key = f"{key}_{axis[d]}"
                    flat_feat_dict[new_key] = arr[:, d]
            else:
                for d in range(dim):
                    new_key = f"{key}_{d}"
                    flat_feat_dict[new_key] = arr[:, d]
        else:
            raise ValueError(f"Currently only handle up to 2D features, but got '{key}' with shape {arr.shape}.")
    
    return flat_feat_dict


def _interp_1d_nan(x: np.ndarray) -> np.ndarray:
    """
    Fill NaN values of a 1D series x along time:
    - interior NaN: linear interpolation
    - leading/trailing NaN: nearest valid value
    - all NaN: returned unchanged
    """
    x = x.astype(np.float32, copy=True)
    n = x.shape[0]
    idx = np.arange(n)

    valid = np.isfinite(x)
    if valid.sum() == 0:
        return x  # all NaN

    # np.interp holds the end values constant, which fills both ends
    x_filled = np.interp(idx, idx[valid], x[valid]).astype(np.float32)
    return x_filled


def interpolate_kpts_over_time(kpt_2d: np.ndarray) -> np.ndarray:
    """
    kpt_2d: (T, K, 2) keypoint coordinates, may contain NaN.
    Returns an array of the same shape with x and y of each keypoint interpolated over time.
    """
    if kpt_2d.ndim != 3 or kpt_2d.shape[2] != 2:
        raise ValueError(f"kpt_2d must have shape (frames, num_keypoints, 2), got {kpt_2d.shape}")

    T, K, _ = kpt_2d.shape
    out = kpt_2d.astype(np.float32, copy=True)

    for j in range(K):
        out[:, j, 0] = _interp_1d_nan(out[:, j, 0])
        out[:, j, 1] = _interp_1d_nan(out[:, j, 1])

    return out


def compute_pixel_change(prev_gray, curr_gray, mask=None):
    if mask is not None:
        diff_sq = (curr_gray[mask] - prev_gray[mask])**2
        curr_mean = np.mean(curr_gray[mask]) if diff_sq.size > 0 else 0
    else:
        diff_sq = (curr_gray - prev_gray)**2
        curr_mean = np.mean(curr_gray)

    if curr_mean == 0 or diff_sq.size == 0:
        return 0.0
    return float(np.mean(diff_sq) / curr_mean)


def process_video_features_2d_single_mouse(video_path, kpt_2d_path, out_npy_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    kpt_2d_raw = np.load(kpt_2d_path)  # (frames, num_keypoints, 2)
    if kpt_2d_raw.shape[0] != total_frames:
        print(f"[WARNING] Video frames ({total_frames}) and keypoint frames ({kpt_2d_raw.shape[0]}) differ; check alignment.")

    if kpt_2d_raw.shape[1] < 8:
        raise ValueError(f"Fewer than 8 keypoints: (frames, {kpt_2d_raw.shape[1]}, 2).")

    # Fill missing keypoints by interpolation over time first
    kpt_2d = interpolate_kpts_over_time(kpt_2d_raw)

    features_dict = {}
    pixel_change_arr = np.zeros((total_frames,), dtype=np.float32)
    features_dict["pixel_change"] = pixel_change_arr

    for kpt_name in KEYPOINT_NAMES:
        features_dict[f"{kpt_name}_pixel_change"] = np.zeros((total_frames,), dtype=np.float32)

    prev_gray = None
    frame_idx = 0
    radius = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if frame_idx == 0:
            pixel_change_arr[frame_idx] = 0.0
            for kpt_name in KEYPOINT_NAMES:
                features_dict[f"{kpt_name}_pixel_change"][frame_idx] = 0.0
        else:
            pixel_change_arr[frame_idx] = compute_pixel_change(prev_gray, curr_gray, mask=None)

            for i, kpt_name in enumerate(KEYPOINT_NAMES):
                x_p = kpt_2d[frame_idx, i, 0]
                y_p = kpt_2d[frame_idx, i, 1]

                # Keypoint still NaN (never detected): set the feature to 0
                if not (np.isfinite(x_p) and np.isfinite(y_p)):
                    features_dict[f"{kpt_name}_pixel_change"][frame_idx] = 0.0
                    continue

                x1 = max(int(x_p - radius), 0)
                y1 = max(int(y_p - radius), 0)
                x2 = min(int(x_p + radius), frame_w - 1)
                y2 = min(int(y_p + radius), frame_h - 1)

                if x1 < x2 and y1 < y2:
                    mask_p = np.zeros((frame_h, frame_w), dtype=bool)
                    mask_p[y1:y2 + 1, x1:x2 + 1] = True
                    features_dict[f"{kpt_name}_pixel_change"][frame_idx] = compute_pixel_change(
                        prev_gray, curr_gray, mask=mask_p
                    )
                else:
                    features_dict[f"{kpt_name}_pixel_change"][frame_idx] = 0.0

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()

    if out_npy_path is not None:
        np.save(out_npy_path, features_dict, allow_pickle=True)
        print(f"[INFO] Saved features to {out_npy_path}")

    return features_dict


def build_feature_table(video_path, kpt_2d_path, kpt_3d_path, fps=30.0):
    """Return the 74-feature table of one recording, in the classifier's column order.

    Rows start at frame FRAMES_TO_DROP; the column `frame` gives the video frame index.
    """
    keypoints_3d = np.load(kpt_3d_path)[:, :8]
    flat = flatten_feat_dict(extract_features_3d(keypoints_3d, fps=fps), frames_to_drop=FRAMES_TO_DROP)
    pixel = process_video_features_2d_single_mouse(video_path, kpt_2d_path)
    pixel = {k: v[FRAMES_TO_DROP:] for k, v in pixel.items()}
    table = pd.concat([pd.DataFrame(flat), pd.DataFrame(pixel)], axis=1)
    table.insert(0, 'frame', np.arange(FRAMES_TO_DROP, FRAMES_TO_DROP + len(table)))
    return table


def main():
    parser = argparse.ArgumentParser(description='Compute the 74 PyMouse Lifter features of one recording.')
    parser.add_argument('video')
    parser.add_argument('keypoints_2d')
    parser.add_argument('keypoints_3d')
    parser.add_argument('-o', '--output', required=True, help='CSV or .pkl output path')
    parser.add_argument('--fps', type=float, default=30.0)
    args = parser.parse_args()
    table = build_feature_table(args.video, args.keypoints_2d, args.keypoints_3d, fps=args.fps)
    if args.output.endswith('.pkl'):
        table.to_pickle(args.output)
    else:
        table.to_csv(args.output, index=False)
    print(f'{len(table)} frames x {table.shape[1] - 1} features -> {args.output}')


if __name__ == '__main__':
    main()
