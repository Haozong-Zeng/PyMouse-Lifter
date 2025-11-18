import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from ultralytics import YOLO
import cv2
from scipy.optimize import linear_sum_assignment
import pickle
from scipy.ndimage import distance_transform_edt
from pathlib import Path          # ✨ 推荐用 pathlib 处理路径

# ------------------------------------------------------------------
# 全局设置
# ------------------------------------------------------------------
FY = 619.766  # Focal length Y
FX = 619.766  # Focal length X
FINAL_HEIGHT = 720
FINAL_WIDTH = 720
BASE_DIR             = Path('../YOLO_models/dataset/orbbec-pose-real')
IMG_DIR              = BASE_DIR / 'images' / 'val2017'   # <-- 图片
LABEL_DIR            = BASE_DIR / 'labels' / 'val2017'   # <-- YOLO keypoints
OUTPUT_DIR           = BASE_DIR                          # 结果仍写回 dataset 根目录
DATASET              = 'nyu'
COEFFICIENT          = 2.3529


# ------------------------------------------------------------------
# 读取 YOLO keypoints
# ------------------------------------------------------------------
def read_yolo_pose_labels(label_path, img_width=FINAL_WIDTH, img_height=FINAL_HEIGHT):
    keypoints = []
    if not label_path.exists():
        return keypoints

    with open(label_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if int(data[0]) != 0:          # 仅处理 class 0
                continue
            single = []
            for i in range(5, 5 + 10 * 2, 2):   # 10 个 keypoints
                rel_x, rel_y = map(float, data[i:i+2])
                single.append((rel_x * img_width, rel_y * img_height))
            keypoints.append(single)
    return keypoints


def project_to_3d(keypoints, depth, fx, fy):
    """Convert 2D keypoints and depth to 3D camera coordinates."""
    keypoints_3d = []
    for (u, v) in keypoints:
        z = depth[int(v), int(u)]
        x = (u - FINAL_WIDTH / 2) * z / fx
        y = (v - FINAL_HEIGHT / 2) * z / fy
        keypoints_3d.append((x, y, z))
    return np.array(keypoints_3d)


def calculate_3d_error(pred_keypoints_3d, gt_keypoints_3d):
    errors = []
    for pred, gt in zip(pred_keypoints_3d, gt_keypoints_3d):
        errors.append(np.linalg.norm(pred - gt))   # 去掉 axis
    return np.array(errors)


def process_images():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # ❶ 收集全部 jpg
    image_paths = sorted(IMG_DIR.glob('*.jpg'))
    results = {}

    # ❷ 初始化一次 YOLO
    yolo_model = YOLO('../YOLO_models/yolo11m-orbbec-pose-real.pt')

    for img_path in tqdm(image_paths, desc='Processing Images'):

        # ---- A. 载入 RGB
        color_img = Image.open(img_path).convert('RGB').resize((FINAL_WIDTH, FINAL_HEIGHT))

        # ---- B. YOLO 推理（bbox + keypoints + confidence）
        yolo_out       = yolo_model(color_img, verbose=False)
        boxes_xyxy     = yolo_out[0].boxes.xyxy.cpu().numpy()      # (N,4)
        conf_scores    = yolo_out[0].boxes.conf.cpu().numpy()      # (N,)
        pred_keypoints = [kp.cpu().numpy() for kp in yolo_out[0].keypoints.data]

        # 若使用第一只检测（和 pred_keypoints[0] 对应），取其置信度
        yolo_confidence = float(conf_scores[0]) if len(conf_scores) > 0 else None

        # ---- C. 读取深度图
        stem       = img_path.stem
        pred_depth = cv2.imread(str(img_path).replace('rgb', 'depth').replace('.jpg', '_pred.png'))
        gt_depth   = cv2.imread(str(img_path).replace('rgb', 'depth').replace('.jpg', '_gt.png'))

        if pred_depth.ndim > 2: pred_depth = pred_depth[:, :, 0]
        if gt_depth.ndim   > 2: gt_depth   = gt_depth[:, :, 0]

        # -------------------------------------------------------------------
        # 1. 生成 bbox 掩码
        # -------------------------------------------------------------------
        if boxes_xyxy.size > 0:
            mask_bbox = np.zeros_like(gt_depth, dtype=bool)
            for x1, y1, x2, y2 in boxes_xyxy.astype(int):
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, FINAL_WIDTH - 1), min(y2, FINAL_HEIGHT - 1)
                mask_bbox[y1:y2 + 1, x1:x2 + 1] = True
        else:
            mask_bbox = np.ones_like(gt_depth, dtype=bool)

        # -------------------------------------------------------------------
        # 2. 与 gt 深度 ≥ 250 mm 取交集
        # -------------------------------------------------------------------
        mask_valid = mask_bbox & (gt_depth >= 250 / COEFFICIENT)
        if not np.any(mask_valid):            # 回退：只用有效深度
            mask_valid = gt_depth >= 250 / COEFFICIENT

        # -------------------------------------------------------------------
        # 3. 中值修正
        # -------------------------------------------------------------------
        diff        = gt_depth[mask_valid] - pred_depth[mask_valid]
        c           = np.median(diff)
        pred_depth  = (pred_depth + c) * COEFFICIENT
        gt_depth    = gt_depth * COEFFICIENT

        # ---- D. 读取 GT keypoints
        label_path     = LABEL_DIR / f'{stem}.txt'
        gt_keypoints   = read_yolo_pose_labels(label_path)

        # ---- E. 误差计算
        pred_kp_3d = project_to_3d(pred_keypoints[0], pred_depth, FX, FY)
        gt_kp_3d   = project_to_3d(gt_keypoints[0],  gt_depth,  FX, FY)
        per_err    = calculate_3d_error(pred_kp_3d, gt_kp_3d)

        print(np.mean(per_err[gt_kp_3d[:, 2] > 250]))

        # ---- F. 保存结果
        img_key = img_path.name
        results[img_key] = {
            "pred_keypoints_3d": pred_kp_3d,
            "gt_keypoints_3d"  : gt_kp_3d,
            "per_point_errors" : per_err,
            "yolo_confidence"  : yolo_confidence,        # ← 新增
        }

    # ---- G. 持久化
    with open(OUTPUT_DIR / '3d_keypoint_errors_orbbec.pkl', 'wb') as f:
        pickle.dump(results, f)


def main():
    process_images()

if __name__ == '__main__':
    main()