# ------------------------------------------------------------
# 3-D keypoint error evaluation for the MOSeq dataset
# ------------------------------------------------------------
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# --------------------------- 全局设置 --------------------------
FX = FY = 665.1968503937009        # focal lengths (≈ 26 mm、f=4 mm 时等效像素)
FINAL_HEIGHT = 512
FINAL_WIDTH  = 512
COEFFICIENT  = 2.3529               # disparity→mm 缩放

BASE_DIR   = Path('../YOLO_models/dataset/moseq-pose-real')
IMG_DIR    = BASE_DIR / 'images' / 'val2017'
LABEL_DIR  = BASE_DIR / 'labels' / 'val2017'
OUTPUT_DIR = BASE_DIR

# --------------------- 1. 读取 YOLO 标注 -----------------------
def read_yolo_pose_labels(label_path: Path,
                          img_width: int = FINAL_WIDTH,
                          img_height: int = FINAL_HEIGHT):
    """
    读取单张 YOLO keypoints 标注（class-0 一只鼠，10 个关键点）。
    若不存在则返回空列表。
    """
    keypoints = []
    if not label_path.exists():
        return keypoints

    with open(label_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if int(data[0]) != 0:           # 只看 class-0
                continue
            pts = []
            for i in range(5, 5 + 10 * 2, 2):
                rel_x, rel_y = map(float, data[i:i + 2])
                pts.append((rel_x * img_width, rel_y * img_height))
            keypoints.append(pts)

    return keypoints


# -------------------- 2. 像素 → 相机坐标 ----------------------
def project_to_3d(kps_2d, depth_img, fx: float, fy: float):
    """给定 2-D × 深度，转相机坐标系 3-D。"""
    kps_3d = []
    for (u, v) in kps_2d:
        z = float(depth_img[int(v), int(u)])
        x = (u - FINAL_WIDTH / 2)  * z / fx
        y = (v - FINAL_HEIGHT / 2) * z / fy
        kps_3d.append((x, y, z))
    return np.asarray(kps_3d)


# ------------------- 3. 匹配多只老鼠（可选） -------------------
def match_keypoints(pred_groups, gt_groups):
    """
    若同帧检测到多鼠，用匈牙利算法把两组最相似的配成对。
    返回 (pred_idx, gt_idx)。
    """
    from scipy.optimize import linear_sum_assignment

    cost = np.zeros((len(pred_groups), len(gt_groups)))
    for i, p in enumerate(pred_groups):
        for j, g in enumerate(gt_groups):
            cost[i, j] = np.linalg.norm(np.mean(p, axis=0) - np.mean(g, axis=0))

    row, col = linear_sum_assignment(cost)
    # 只取前两只（够用）
    return row[:2], col[:2]


# ----------------- 4. 每点欧氏距离误差 ------------------------
def calc_3d_error(pred_3d, gt_3d):
    return np.linalg.norm(pred_3d - gt_3d, axis=1)      # (10,)


# --------------------------- 主流程 ---------------------------
def process_images():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    image_paths = sorted(IMG_DIR.glob('*.jpg'))

    # 初始化一次 YOLO
    yolo = YOLO('../YOLO_models/yolo11m-moseq-pose-real.pt')

    results = {}
    for img_path in tqdm(image_paths, desc='Processing Images'):
        
        # ---------------- A. RGB ----------------
        rgb = Image.open(img_path).convert('RGB')
        rgb = rgb.resize((FINAL_WIDTH, FINAL_HEIGHT))

        # ---------------- B. YOLO 推理 -----------
        yout = yolo(rgb, verbose=False)
        boxes_xyxy  = yout[0].boxes.xyxy.cpu().numpy()       # (N,4)
        conf_scores = yout[0].boxes.conf.cpu().numpy()       # (N,)
        pred_kps2d  = [kp.cpu().numpy() for kp in yout[0].keypoints.data]

        yolo_conf = float(conf_scores[0]) if len(conf_scores) else None

        # ---------------- C. 深度 ----------------
        stem       = img_path.stem
        pred_depth = cv2.imread(str(img_path).replace('rgb', 'depth').replace('.jpg', '_pred.png'))
        gt_depth   = cv2.imread(str(img_path).replace('rgb', 'depth').replace('.jpg', '_gt.png'))

        if pred_depth.ndim > 2: pred_depth = pred_depth[:, :, 0]
        if gt_depth.ndim   > 2: gt_depth   = gt_depth[:, :, 0]

        # 把 3 通道→单通道
        if pred_depth.ndim > 2: pred_depth = pred_depth[:, :, 0]
        if gt_depth.ndim   > 2: gt_depth   = gt_depth[:, :, 0]

        # 读取 GT keypoints
        label_path  = LABEL_DIR / f'{stem}.txt'
        gt_kps2d    = read_yolo_pose_labels(label_path)

        # ~~~~~~~~~~~~~~~~~~ 掩码逻辑 ~~~~~~~~~~~~~~~~~~
        if len(boxes_xyxy):
            mask_bbox = np.zeros_like(gt_depth, dtype=bool)
            for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, FINAL_WIDTH - 1), min(y2, FINAL_HEIGHT - 1)
                mask_bbox[y1:y2 + 1, x1:x2 + 1] = True
        else:                                          # fallback
            mask_bbox = np.ones_like(gt_depth, dtype=bool)

        mask_valid = mask_bbox & (gt_depth >= 250 / COEFFICIENT)
        if not np.any(mask_valid):
            mask_valid = gt_depth >= 250 / COEFFICIENT

        # 中值修正
        diff       = gt_depth[mask_valid] - pred_depth[mask_valid]
        c          = float(np.median(diff))
        pred_depth  = (pred_depth + c) * COEFFICIENT
        gt_depth    = gt_depth * COEFFICIENT

        # ~~~~~~~~~~~~~~~~~~ 误差计算 ~~~~~~~~~~~~~~~~~~
        if not len(pred_kps2d) or not len(gt_kps2d):
            continue        # 本帧无鼠，跳过

        # 多鼠匹配
        p_idx, g_idx = match_keypoints(pred_kps2d, gt_kps2d)

        # 当前脚本里只算第一对
        pk3d = project_to_3d(pred_kps2d[p_idx[0]], pred_depth, FX, FY)
        gk3d = project_to_3d(gt_kps2d[g_idx[0]],   gt_depth,  FX, FY)
        err  = calc_3d_error(pk3d, gk3d)

        # 只统计 z>250 mm 的点
        mean_err = float(np.mean(err[gk3d[:, 2] > 250]))
        print(f'{img_path.name}: {mean_err:.2f} mm')

        # 保存
        results[img_path.name] = {
            "pred_keypoints_3d": pk3d,
            "gt_keypoints_3d"  : gk3d,
            "per_point_errors" : err,
            "yolo_confidence"  : yolo_conf,
        }

       

    # ---------- 持久化 ----------
    with open(OUTPUT_DIR / '3d_keypoint_errors_moseq.pkl', 'wb') as f:
        pickle.dump(results, f)


# --------------------------- 入口 ------------------------------
if __name__ == '__main__':
    process_images()
