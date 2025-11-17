import argparse
import cv2
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from ultralytics import YOLO
import joblib
import open3d as o3d
import threading
import queue
import matplotlib.pyplot as plt
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import matplotlib as mpl

# ----------------- 全局配置部分 -----------------
INOUT_DIR = './assets/PyMouseLifter_demo/input'
OUTPUT_DIR = './assets/PyMouseLifter_demo/output'

FINAL_HEIGHT = 518
FINAL_WIDTH = 518

DATASET = 'nyu'
COEFFICIENT = 2.3529
fps = 30.0
dt = 1.0 / fps

# KEYPOINTS 定义
KEYPOINT_NAMES = [
    'nose',         # 0
    'head',         # 1
    'left_ear',     # 2
    'right_ear',    # 3
    'neck',         # 4
    'spine_center', # 5
    'lumbar_spine', # 6
    'tail_base',    # 7
]
IDX_NOSE          = 0
IDX_HEAD          = 1
IDX_LEFT_EAR      = 2
IDX_RIGHT_EAR     = 3
IDX_NECK          = 4
IDX_SPINE_CENTER  = 5
IDX_LUMBAR_SPINE  = 6
IDX_TAIL_BASE     = 7

# Turbo 颜色映射，离散成 N 色
_cmap = mpl.colormaps['turbo'].resampled(len(KEYPOINT_NAMES))   # ←★ 不再用 get_cmap
KPT_COLORS = [
    tuple(int(255 * c) for c in _cmap(i)[:3][::-1])  # BGR
    for i in range(len(KEYPOINT_NAMES))
]

FEATURE_COLUMNS = [
    'nose_3d_z',
    'pixel_change',
    'nose_3d_y',
    'neck_3d_y',
    'bend_ratio',
    'head_3d_y',
    'spine_center_3d_y',
    'left_ear_3d_y',
    'right_ear_3d_z',
    'right_ear_3d_y',
    'spine_center_3d_z',
    'left_ear_3d_z',
    'segment_angles_0',
    'neck_3d_z',
    'lumbar_spine_3d_z',
    'segment_angles_2',
    'nose_3d_x',
    'head_3d_z',
    'tail_speed',
    'head_speed',
    'lumbar_spine_3d_y',
    'neck_3d_x',
    'segment_angles_1',
    'right_ear_3d_x',
    'left_ear_3d_x',
    'segment_angles_3',
    'head_3d_x',
    'spine_center_3d_x',
    'head_vel_3d_z'
]

def compute_pixel_change(prev_gray, curr_gray, mask=None):
    """
    计算像素变化: mean((curr - prev)^2) / mean(curr)
    """
    if mask is not None:
        diff_sq = (curr_gray[mask] - prev_gray[mask])**2
        curr_mean = np.mean(curr_gray[mask]) if diff_sq.size > 0 else 0
    else:
        diff_sq = (curr_gray - prev_gray)**2
        curr_mean = np.mean(curr_gray)
    if curr_mean == 0 or diff_sq.size == 0:
        return 0.0
    return np.mean(diff_sq) / curr_mean

def rotate_around_z(points, alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    Rz = np.array([
        [ c, -s,  0],
        [ s,  c,  0],
        [ 0,  0,  1]], dtype=np.float32)
    return points @ Rz.T

def align_tailbase_lumbar(frame_3d):
    tail_base    = frame_3d[IDX_TAIL_BASE]
    lumbar_spine = frame_3d[IDX_LUMBAR_SPINE]
    shifted_points = frame_3d - tail_base
    vec = lumbar_spine - tail_base
    dx, dy = vec[0], vec[1]
    theta = np.arctan2(dy, dx)
    alpha = np.pi/2 - theta
    rotated_points = rotate_around_z(shifted_points, alpha)
    return rotated_points

def compute_segment_angle_2d(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)

def compute_features_for_two_Frame(prev_frame_3d, curr_frame_3d, dt):
    """
    计算「prev_frame_3d -> curr_frame_3d」的行为特征
    """
    if prev_frame_3d is None:
        return None

    aligned_curr = align_tailbase_lumbar(curr_frame_3d)
    aligned_prev = align_tailbase_lumbar(prev_frame_3d)

    segment_pairs = [
        (IDX_NOSE, IDX_HEAD),
        (IDX_HEAD, IDX_NECK),
        (IDX_NECK, IDX_SPINE_CENTER),
        (IDX_SPINE_CENTER, IDX_LUMBAR_SPINE),
        (IDX_LUMBAR_SPINE, IDX_TAIL_BASE),
    ]
    n_segments = len(segment_pairs)

    angles_curr = np.zeros((n_segments,), dtype=np.float32)
    angles_prev = np.zeros((n_segments,), dtype=np.float32)
    for i, (idxA, idxB) in enumerate(segment_pairs):
        pA_curr = aligned_curr[idxA]
        pB_curr = aligned_curr[idxB]
        angles_curr[i] = compute_segment_angle_2d(pA_curr, pB_curr)

        pA_prev = aligned_prev[idxA]
        pB_prev = aligned_prev[idxB]
        angles_prev[i] = compute_segment_angle_2d(pA_prev, pB_prev)

    ang_vel = (angles_curr - angles_prev) / dt
    ang_acc = np.full((n_segments,), np.nan, dtype=np.float32)

    head_curr = curr_frame_3d[IDX_HEAD]
    head_prev = prev_frame_3d[IDX_HEAD]
    tail_curr = curr_frame_3d[IDX_TAIL_BASE]
    tail_prev = prev_frame_3d[IDX_TAIL_BASE]

    head_vel  = (head_curr - head_prev) / dt
    tail_vel  = (tail_curr - tail_prev) / dt
    head_speed = np.linalg.norm(head_vel)
    tail_speed = np.linalg.norm(tail_vel)

    head_acc = np.full((3,), np.nan, dtype=np.float32)
    tail_acc = np.full((3,), np.nan, dtype=np.float32)

    bend_chain = [IDX_NOSE, IDX_HEAD, IDX_NECK, IDX_SPINE_CENTER, IDX_LUMBAR_SPINE, IDX_TAIL_BASE]
    poly_len = 0.0
    for k in range(len(bend_chain) - 1):
        pA = aligned_curr[bend_chain[k]]
        pB = aligned_curr[bend_chain[k+1]]
        poly_len += np.linalg.norm(pB - pA)
    nose_pt = aligned_curr[IDX_NOSE]
    tail_pt = aligned_curr[IDX_TAIL_BASE]
    direct_dist = np.linalg.norm(tail_pt - nose_pt) + 1e-8
    bend_ratio = poly_len / direct_dist

    nose_3d          = aligned_curr[IDX_NOSE]
    head_3d          = aligned_curr[IDX_HEAD]
    left_ear_3d      = aligned_curr[IDX_LEFT_EAR]
    right_ear_3d     = aligned_curr[IDX_RIGHT_EAR]
    neck_3d          = aligned_curr[IDX_NECK]
    spine_center_3d  = aligned_curr[IDX_SPINE_CENTER]
    lumbar_spine_3d  = aligned_curr[IDX_LUMBAR_SPINE]
    tail_base_3d     = aligned_curr[IDX_TAIL_BASE]

    feat_dict = {
        'nose_3d':           nose_3d,
        'head_3d':           head_3d,
        'left_ear_3d':       left_ear_3d,
        'right_ear_3d':      right_ear_3d,
        'neck_3d':           neck_3d,
        'spine_center_3d':   spine_center_3d,
        'lumbar_spine_3d':   lumbar_spine_3d,
        'tail_base_3d':      tail_base_3d,
        'segment_angles':    angles_curr,
        'segment_ang_vel':   ang_vel,
        'segment_ang_acc':   ang_acc,
        'head_vel_3d':       head_vel,
        'head_acc_3d':       head_acc,
        'tailbase_vel_3d':   tail_vel,
        'tailbase_acc_3d':   tail_acc,
        'head_speed':        head_speed,
        'tail_speed':        tail_speed,
        'bend_ratio':        bend_ratio
    }
    return feat_dict

def flatten_feat_dict(feat_dict):
    """
    将多维特征展平到一维
    """
    if feat_dict is None:
        return None

    flat_feat = {}
    axis_names = ['x', 'y', 'z']
    for key, value in feat_dict.items():
        arr = np.array(value)
        if arr.ndim == 0:
            flat_feat[key] = arr.item()
        elif arr.ndim == 1:
            length = arr.shape[0]
            if length == 3:
                for i in range(length):
                    new_key = f"{key}_{axis_names[i]}"
                    flat_feat[new_key] = arr[i]
            else:
                for i in range(length):
                    new_key = f"{key}_{i}"
                    flat_feat[new_key] = arr[i]
    return flat_feat

# 定义线程函数：保存原始视频帧
def save_raw_video_worker(frame_queue, video_writer):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        video_writer.write(frame)
    video_writer.release()

# 定义线程函数：写入分类标签
def write_label_worker(label_queue, class_file):
    while True:
        label_info = label_queue.get()
        if label_info is None:
            break
        class_file.write(label_info)
        class_file.flush()
    class_file.close()

# This is just to simulate real-time recording
# Use run_PyMouseLifter_WebCam_demo.py if using a real camera
def image_loader_worker(filenames, preproc_queue, transform):
    for filename in filenames:
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"警告：无法读取图像 {filename}，跳过。")
            continue
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = raw_image.shape[:2]
        transformed = transform({'image': image_rgb})['image']
        transformed_tensor = torch.from_numpy(transformed).unsqueeze(0)
        gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        preproc_queue.put((filename, raw_image, transformed_tensor, gray_image, h, w))
    # 处理完毕，放入终止信号
    preproc_queue.put(None)

def draw_depth_and_kpts(depth_map, kpts_xy, radius=3):
    """
    depth_map : (H, W) or (1, H, W)  float16/32/64
    """
    # ——★ 兼容 float16、(1,H,W)、NaN/Inf ——
    depth_map = np.squeeze(depth_map).astype(np.float32, copy=False)
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    depth_norm = cv2.normalize(depth_map, None, 0, 255,
                               cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    vis = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)

    if kpts_xy is not None:
        for idx, (x, y) in enumerate(kpts_xy.astype(int)):
            color = KPT_COLORS[idx] if idx < len(KPT_COLORS) else KPT_COLORS[-1]
            cv2.circle(vis, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return vis

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=INOUT_DIR)
    parser.add_argument('--outdir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--save_depth_vis', action='store_true', help='同时保存“深度图+关键点”可视化视频')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 确保 batch_size 为偶数且至少为2
    batch_size = max(2, args.batch_size + (args.batch_size % 2))
    
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    encoder = args.encoder

    depth_anything = DepthAnything(model_configs[encoder])
    depth_anything.load_state_dict(
        torch.load('./metric_depth/checkpoints/depth_anything_metric_PyMouse_HQ_orbbec_trans_synthetic.pt')
    )
    depth_anything.half().to(DEVICE)
    depth_anything.eval()

    yolo_path = './YOLO_models/yolo11m-orbbec-pose-real.pt'
    yolo_model = YOLO(yolo_path)
    
    rf_path = './YOLO_models/rf_model_realtime_demo.pkl'
    rf_model = joblib.load(rf_path)

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # 获取图片路径列表
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = [
            os.path.join(args.img_path, filename)
            for filename in os.listdir(args.img_path)
            if not filename.startswith('.')
        ]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)

    # 创建预处理队列，并启动图像加载线程
    preproc_queue = queue.Queue(maxsize=100)
    loader_thread = threading.Thread(target=image_loader_worker, args=(filenames, preproc_queue, transform))
    loader_thread.start()

    # 初始化视频保存线程
    # 这里使用第一幅图的尺寸作为视频尺寸（注意：预处理线程可能还未加载完，需确保至少有一帧）
    # 可以等待队列中第一个数据再启动视频保存线程
    first_item = preproc_queue.get()
    if first_item is None:
        print("没有可处理的图像")
        exit(1)
    # 将第一个预处理数据放回队列
    preproc_queue.put(first_item)
    _, first_raw, _, _, _, _ = first_item
    video_height, video_width = first_raw.shape[:2]
    raw_frame_queue = queue.Queue()
    raw_video_out_path = os.path.join(args.outdir, "raw_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    raw_video_writer = cv2.VideoWriter(raw_video_out_path, fourcc, fps, (video_width, video_height))
    if args.save_depth_vis:
        depth_vis_out_path = os.path.join(args.outdir, "depth_vis.mp4")
        depth_video_writer = cv2.VideoWriter(depth_vis_out_path, fourcc, fps,
                                            (FINAL_WIDTH, FINAL_HEIGHT))
    save_thread = threading.Thread(target=save_raw_video_worker, args=(raw_frame_queue, raw_video_writer))
    save_thread.start()

    # 初始化标签写入线程
    label_queue = queue.Queue()
    class_file_path = os.path.join(args.outdir, "behavior_classification.txt")
    class_file = open(class_file_path, 'w')
    label_thread = threading.Thread(target=write_label_worker, args=(label_queue, class_file))
    label_thread.start()

    fps_list = []
    batch_count = 0
    # 主推理循环：每次从预处理队列中取出 batch_size 个数据（注意末尾可能不足 batch_size 个）
    while True:
        batch = []
        for _ in range(batch_size):
            item = preproc_queue.get()
            if item is None:
                break
            batch.append(item)
        if not batch:
            break  # 没有数据了
        # 若最后一个批次不足 batch_size 个，也可以选择忽略或者处理剩余的数量
        if len(batch) < batch_size:
            print("最后一个批次数据不足，结束推理")
            break

        batch_count += 1
        start_time = time.time()
        raw_images = []
        tensor_list = []
        gray_list = []
        for (filename, raw, tensor, gray, h, w) in batch:
            raw_images.append(raw)
            tensor_list.append(tensor)
            gray_list.append(gray)
        # 拼接 batch tensor 并进行深度推理
        batch_tensor = torch.cat(tensor_list, dim=0).half().to(DEVICE)
        with torch.no_grad():
            depth_batch = depth_anything(batch_tensor)
        # YOLO 批量推理
        yolo_results = yolo_model(raw_images, half=True, max_det=1, verbose=False)
        
        # 对于当前批次内的每一对图像（两个一组）
        for j in range(0, batch_size, 2):
            global_idx1 = batch_count * batch_size - batch_size + j
            global_idx2 = global_idx1 + 1

            gray_prev = gray_list[j]
            gray_curr = gray_list[j+1]
            raw_prev = raw_images[j]
            raw_curr = raw_images[j+1]
            pixel_change_val = compute_pixel_change(gray_prev, gray_curr)

            if len(yolo_results[j].keypoints) > 0:
                keypoints_2d_prev = yolo_results[j].keypoints.data[0].cpu().numpy()
                x2d_prev = keypoints_2d_prev[:, 0] * (FINAL_WIDTH / video_width)
                y2d_prev = keypoints_2d_prev[:, 1] * (FINAL_HEIGHT / video_height)
                depth_prev = depth_batch[j].cpu().numpy()
                pred_z_prev = depth_prev[y2d_prev.astype(np.uint32), x2d_prev.astype(np.uint32)]
                pred_x_prev = x2d_prev - (FINAL_WIDTH / 2)
                pred_y_prev = -(y2d_prev - (FINAL_HEIGHT / 2))
                prev_frame_3d = np.stack((pred_x_prev, pred_y_prev, pred_z_prev), axis=-1)
            else:
                prev_frame_3d = None

            if len(yolo_results[j+1].keypoints) > 0:
                keypoints_2d_curr = yolo_results[j+1].keypoints.data[0].cpu().numpy()
                x2d_curr = keypoints_2d_curr[:, 0] * (FINAL_WIDTH / video_width)
                y2d_curr = keypoints_2d_curr[:, 1] * (FINAL_HEIGHT / video_height)
                depth_curr = depth_batch[j+1].cpu().numpy()
                pred_z_curr = depth_curr[y2d_curr.astype(np.uint64), x2d_curr.astype(np.uint64)]
                pred_x_curr = x2d_curr - (FINAL_WIDTH / 2)
                pred_y_curr = -(y2d_curr - (FINAL_HEIGHT / 2))
                curr_frame_3d = np.stack((pred_x_curr, pred_y_curr, pred_z_curr), axis=-1)
            else:
                curr_frame_3d = None

            if prev_frame_3d is None or curr_frame_3d is None:
                label = "None"
                label_queue.put(f"Frame {global_idx1}: {label}\n")
                label_queue.put(f"Frame {global_idx2}: {label}\n")
                continue

            feat_dict = compute_features_for_two_Frame(prev_frame_3d, curr_frame_3d, dt=dt)
            if feat_dict is None:
                label = "None"
                label_queue.put(f"Frame {global_idx1}: {label}\n")
                label_queue.put(f"Frame {global_idx2}: {label}\n")
                continue
            feat_dict['pixel_change'] = pixel_change_val
            flat_feat = flatten_feat_dict(feat_dict)
            X = []
            for col in FEATURE_COLUMNS:
                if col in flat_feat:
                    X.append(flat_feat[col])
                else:
                    X.append(np.nan)
            X = np.array(X, dtype=np.float32).reshape(1, -1)
            y_pred = rf_model.predict(X)
            label = y_pred[0] if y_pred is not None else "None"

            raw_frame_queue.put(raw_prev)
            raw_frame_queue.put(raw_curr)
            label_queue.put(f"Frame {global_idx1}: {label}\n")
            label_queue.put(f"Frame {global_idx2}: {label}\n")
            print(f"Frame {global_idx1}: {label}")
            print(f"Frame {global_idx2}: {label}")
            # —— 可视化 & 写文件  ——★
            if args.save_depth_vis:
                vis_prev = draw_depth_and_kpts(depth_prev,
                                            np.column_stack((x2d_prev, y2d_prev)))
                vis_curr = draw_depth_and_kpts(depth_curr,
                                            np.column_stack((x2d_curr, y2d_curr)))
                depth_video_writer.write(vis_prev)
                depth_video_writer.write(vis_curr)

        end_time = time.time()
        inference_time = end_time - start_time
        fps_curr = batch_size / inference_time if inference_time > 0 else float('inf')
        print(f"Batch {batch_count}: 平均每张图像推理时间 {(inference_time * 1000 / batch_size):.2f} ms, FPS: {fps_curr:.2f}")
        fps_list.append(fps_curr)

    # 通知视频与标签线程结束
    raw_frame_queue.put(None)
    save_thread.join()
    label_queue.put(None)
    label_thread.join()
    loader_thread.join()

    mean_fps = np.mean(fps_list)
    std_fps = np.std(fps_list)
    print(f"Finish. Average FPS: {mean_fps:.3f} ± {std_fps:.3f}")
