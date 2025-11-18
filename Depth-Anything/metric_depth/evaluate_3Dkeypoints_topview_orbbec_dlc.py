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
import cv2
from scipy.optimize import linear_sum_assignment
import pickle
from scipy.ndimage import distance_transform_edt

# Global settings
FY = 480.66  # Focal length Y
FX = 480.66  # Focal length X
FINAL_HEIGHT = 512
FINAL_WIDTH = 512
INPUT_DIR = './my_test/orbbec-test-yolo'
OUTPUT_DIR = './my_test/orbbec-test-yolo'
DATASET = 'nyu'
COEFFICIENT = 2.3529


def read_yolo_pose_labels(file_path, img_width=FINAL_WIDTH, img_height=FINAL_HEIGHT):
    keypoints = []
    if not os.path.exists(file_path):
        return keypoints
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split()
            cls = int(data[0])
            if cls == 0:  # Only process class 0 objects
                individual_keypoints = []
                for i in range(5, 5 + 10 * 2, 2):  # Each object has 10 keypoints (x, y)
                    rel_x = float(data[i])
                    rel_y = float(data[i + 1])
                    abs_x = rel_x * img_width
                    abs_y = rel_y * img_height
                    individual_keypoints.append((abs_x, abs_y))
                keypoints.append(individual_keypoints)
    return keypoints


def read_dlc_pose_labels(file_path, index, num_individuals=1, num_keypoints=10, scale=FINAL_WIDTH / 512):
    """Read DLC pose labels, separating keypoints by individuals."""
    individuals_keypoints = []

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, 'r') as f:
        lines = f.readlines()
        if index >= len(lines) - 3:  # Check index range
            print(f"Index {index} out of range. File has {len(lines) - 3} data lines.")
            return None

        line = lines[index + 3]  # Skip header lines
        print(f"Processing line {index}: {line.strip()}")  # Debug: Print current line

        data = line.strip().split(',')[1:]  # Skip the first column (header info)

        for ind in range(num_individuals):
            start = ind * num_keypoints * 3  # Calculate start index for this individual
            end = start + num_keypoints * 3  # Calculate end index
            individual_keypoints = []
            for i in range(start, end, 3):  # Each keypoint has 3 values: x, y, confidence
                try:
                    x = float(data[i]) * scale
                    y = float(data[i + 1]) * scale
                    individual_keypoints.append((x, y))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing keypoint at index {i}: {e}")
                    continue

            if len(individual_keypoints) == num_keypoints:  # Ensure full keypoint set
                individuals_keypoints.append(individual_keypoints)

        if len(individuals_keypoints) == num_individuals:  # Ensure all individuals are parsed
            print(f"Valid keypoints found: {len(individuals_keypoints)} individuals.")
            return individuals_keypoints
        else:
            print(f"Invalid keypoints. Found {len(individuals_keypoints)} individuals.")
            return None


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
    """Calculate per-point Euclidean distance error."""
    errors = []
    for pred, gt in zip(pred_keypoints_3d, gt_keypoints_3d):
        errors.append(np.linalg.norm(pred - gt, axis=1))
    return errors


def fill_invalid_depth_nearest(gt_depth):
    """
    使用最近邻插值填充 gt_depth 中的无效值 (==0)，
    返回填充后的新数组。
    
    参数:
    -------
    gt_depth: np.ndarray, 形状 (H, W)，深度图数据
    
    返回:
    -------
    filled_gt_depth: np.ndarray, 相同形状，已填充无效值
    """
    # invalid_mask: 无效像素位置（True/False）
    invalid_mask = (gt_depth < 0)
    # valid_mask: 有效像素位置
    valid_mask = ~invalid_mask

    if not np.any(invalid_mask):
        # 如果没有无效像素，直接返回原数据
        return gt_depth.copy()

    # distance_transform_edt 支持返回 “nearest indices”
    # 这里对无效像素(==0)采用“向最近的有效像素”索引进行映射
    # 注意：distance_transform_edt 里，若要得到最近邻坐标，需要对“无效掩码的逻辑”做适当取反
    distances, nearest_indices = distance_transform_edt(
        ~valid_mask,            # ~valid_mask 表示有效像素为 "背景"，从而对无效区域求最近点
        return_distances=True,
        return_indices=True
    )

    # nearest_indices 的形状是 (2, H, W)，即 (row_index_map, col_index_map)
    # 取出最近点在 gt_depth 中的坐标
    nearest_y = nearest_indices[0]
    nearest_x = nearest_indices[1]

    # 创建填充后的输出
    filled_gt_depth = gt_depth.copy()
    # 对无效像素位置，填充为其最近有效像素的深度值
    filled_gt_depth[invalid_mask] = gt_depth[nearest_y[invalid_mask], nearest_x[invalid_mask]]

    return filled_gt_depth


def process_images(model):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
    image_paths.sort()
    results = {}
    index = 0
    for image_path in tqdm(image_paths, desc="Processing Images"):
        #try:
        # Load RGB image
        color_image = Image.open(image_path).convert('RGB').resize((FINAL_WIDTH, FINAL_HEIGHT))
        image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Predict depth
        pred = model(image_tensor, dataset=DATASET)
        pred_depth = pred.get('metric_depth', pred.get('out')).squeeze().detach().cpu().numpy()
        pred_depth = cv2.resize(pred_depth, (FINAL_WIDTH, FINAL_HEIGHT), interpolation=cv2.INTER_NEAREST) * COEFFICIENT * 1000
        

        # Load DLC predicted keypoints
        pred_keypoints = read_dlc_pose_labels(os.path.join(INPUT_DIR, 'orbbec_testDLC_Resnet50_orbbecJan11shuffle1_detector_200_snapshot_150.csv'), index)
        index = index + 1

        # Load ground truth
        gt_depth = cv2.imread(image_path.replace('jpg', 'png').replace('rgb', 'depth'), cv2.IMREAD_UNCHANGED)
        gt_depth = cv2.resize(gt_depth, (FINAL_WIDTH, FINAL_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(np.float32) * COEFFICIENT
        gt_keypoints = read_yolo_pose_labels(image_path.replace('jpg', 'txt'))
        
        if gt_depth.shape[-1] == 3:
            gt_depth = gt_depth[:,:,0]
        
        # Load contour image
        contour = cv2.imread(image_path.replace('jpg', 'png').replace('rgb', 'contour'), cv2.IMREAD_UNCHANGED)
        contour = cv2.resize(contour, (FINAL_WIDTH, FINAL_HEIGHT), interpolation=cv2.INTER_NEAREST)

        # Background is not precisely modeled
        gt_depth = fill_invalid_depth_nearest(gt_depth)

        # 利用 contour 生成 mask
        mask = (contour >= 127)

        # 计算中值并修正 pred_depth
        c = np.median((gt_depth - pred_depth)[mask])
        pred_depth = pred_depth + c

        print(np.mean(pred_depth))
        print(np.mean(gt_depth))
        
        if pred_keypoints is not None:
            
            # Convert to 3D and calculate errors
            pred_keypoints_3d = [project_to_3d(pred_keypoints[0], pred_depth, FX, FY)]
            gt_keypoints_3d = [project_to_3d(gt_keypoints[0], gt_depth, FX, FY)]
            per_point_errors = calculate_3d_error(pred_keypoints_3d, gt_keypoints_3d)
            print(np.mean(per_point_errors))

            # Save results
            results[image_path] = {
                "pred_keypoints_3d": pred_keypoints_3d,
                "gt_keypoints_3d": gt_keypoints_3d,
                "per_point_errors": per_point_errors,
            }
        else:
            pass

        #except Exception as e:
            #print(f"Error processing {image_path}: {e}")
        
    # Save all results to a pickle file
    with open(os.path.join(OUTPUT_DIR, '3d_keypoint_errors_orbbec_dlc.pkl'), 'wb') as f:
        pickle.dump(results, f)


def main(model_name, pretrained_resource):
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    process_images(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_PyMouse_HQ_orbbec_trans_synthetic_core.pt', help="Pretrained resource to use for fetching weights.")
    args = parser.parse_args()
    main(args.model, args.pretrained_resource)