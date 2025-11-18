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

# Global settings
FY = 597.6377952755906  # Focal length Y
FX = 597.6377952755906  # Focal length X
FINAL_HEIGHT = 460
FINAL_WIDTH = 460
INPUT_DIR = './my_test/moseq-white-test-yolo'
OUTPUT_DIR = './my_test/moseq-white-test-yolo'
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


def read_dlc_pose_labels(file_path, index, num_individuals=2, num_keypoints=10, scale=FINAL_WIDTH / 512):
    """Read DLC pose labels, separating keypoints by individuals."""
    individuals_keypoints = []

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, 'r') as f:
        lines = f.readlines()
        if index >= len(lines) - 4:  # Check index range
            print(f"Index {index} out of range. File has {len(lines) - 4} data lines.")
            return None

        line = lines[index + 4]  # Skip header lines
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


def match_keypoints(pred_keypoints, gt_keypoints, merge_threshold=20.0):
    """
    Match predicted keypoint groups to ground truth using Hungarian Algorithm.
    Handles cases where multiple predictions are for the same target.
    
    Args:
        pred_keypoints (list): List of predicted keypoint groups.
        gt_keypoints (list): List of ground truth keypoint groups.
        merge_threshold (float): Distance threshold to consider predictions as duplicates.

    Returns:
        matched_pred_indices (list): List of matched prediction indices.
        matched_gt_indices (list): List of matched ground truth indices.
        merged_predictions (list): List of merged prediction keypoint groups.
    """
    # Step 1: Compute cost matrix
    cost_matrix = np.zeros((len(pred_keypoints), len(gt_keypoints)))
    for i, pred_group in enumerate(pred_keypoints):
        for j, gt_group in enumerate(gt_keypoints):
            dist = np.linalg.norm(np.mean(pred_group, axis=0) - np.mean(gt_group, axis=0))
            cost_matrix[i, j] = dist

    # Step 2: Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Step 3: Identify duplicate predictions
    duplicate_pairs = []
    for i, pred_i in enumerate(pred_keypoints):
        for j, pred_j in enumerate(pred_keypoints):
            if i < j:  # Avoid self-comparison and redundant checks
                dist = np.linalg.norm(np.mean(pred_i, axis=0) - np.mean(pred_j, axis=0))
                if dist < merge_threshold:  # Check if distance is below threshold
                    duplicate_pairs.append((i, j))

    # Step 4: Merge duplicate predictions and reassign
    merged_predictions = pred_keypoints.copy()
    used_indices = set()
    for i, j in duplicate_pairs:
        if i not in used_indices and j not in used_indices:
            # Merge two groups into one by averaging their keypoints
            merged_group = [
                ((kp1[0] + kp2[0]) / 2, (kp1[1] + kp2[1]) / 2)
                for kp1, kp2 in zip(pred_keypoints[i], pred_keypoints[j])
            ]
            merged_predictions[i] = merged_group  # Replace the first with the merged group
            used_indices.add(j)  # Mark the second as used

    # Step 5: Update cost matrix and recompute assignment
    merged_cost_matrix = np.zeros((len(merged_predictions), len(gt_keypoints)))
    for i, pred_group in enumerate(merged_predictions):
        for j, gt_group in enumerate(gt_keypoints):
            dist = np.linalg.norm(np.mean(pred_group, axis=0) - np.mean(gt_group, axis=0))
            merged_cost_matrix[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(merged_cost_matrix)

    # Filter matches if there are more than 2 prediction groups
    matched_pred_indices = row_ind[:2] if len(row_ind) > 2 else row_ind
    matched_gt_indices = col_ind[:2] if len(col_ind) > 2 else col_ind

    return matched_pred_indices, matched_gt_indices


def calculate_3d_error(pred_keypoints_3d, gt_keypoints_3d):
    """Calculate per-point Euclidean distance error."""
    errors = []
    for pred, gt in zip(pred_keypoints_3d, gt_keypoints_3d):
        errors.append(np.linalg.norm(pred - gt, axis=1))
    return errors


def process_images(model):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
    results = {}

    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            # Load RGB image
            color_image = Image.open(image_path).convert('RGB').resize((FINAL_WIDTH, FINAL_HEIGHT))
            image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

            # Predict depth
            pred = model(image_tensor, dataset=DATASET)
            pred_depth = pred.get('metric_depth', pred.get('out')).squeeze().detach().cpu().numpy()
            pred_depth = cv2.resize(pred_depth, (FINAL_WIDTH, FINAL_HEIGHT), interpolation=cv2.INTER_NEAREST) * COEFFICIENT * 1000
            

            # Load DLC predicted keypoints
            index = int(image_path[-16:-4])
            pred_keypoints = read_dlc_pose_labels(os.path.join(INPUT_DIR, 'moseq-white-0-9000DLC_HrnetW32_moseq-whiteNov29shuffle1_detector_190_snapshot_060_el_filtered.csv'), index)

            # Load ground truth
            gt_depth = cv2.imread(image_path.replace('jpg', 'png').replace('rgb', 'depth'), cv2.IMREAD_UNCHANGED)
            gt_depth = cv2.resize(gt_depth, (FINAL_WIDTH, FINAL_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(np.float32) * COEFFICIENT
            gt_keypoints = read_yolo_pose_labels(image_path.replace('jpg', 'txt'))
            
            # Load contour image
            contour = cv2.imread(image_path.replace('jpg', 'png').replace('rgb', 'contour'), cv2.IMREAD_UNCHANGED)
            contour = cv2.resize(contour, (FINAL_WIDTH, FINAL_HEIGHT), interpolation=cv2.INTER_NEAREST)

            # Background is not precisely modeled
            mask = (contour >= 127)
            c = np.median((gt_depth - pred_depth)[mask])
            pred_depth = pred_depth + c

            print(np.mean(pred_depth))
            print(np.mean(gt_depth))
            
            if pred_keypoints is not None:
                # Match prediction and ground truth groups
                matched_pred_indices, matched_gt_indices = match_keypoints(pred_keypoints, gt_keypoints)

                # Convert to 3D and calculate errors
                pred_keypoints_3d = [project_to_3d(pred_keypoints[i], pred_depth, FX, FY) for i in matched_pred_indices]
                gt_keypoints_3d = [project_to_3d(gt_keypoints[i], gt_depth, FX, FY) for i in matched_gt_indices]
                per_point_errors = calculate_3d_error(pred_keypoints_3d, gt_keypoints_3d)

                # Save results
                results[image_path] = {
                    "pred_keypoints_3d": pred_keypoints_3d,
                    "gt_keypoints_3d": gt_keypoints_3d,
                    "per_point_errors": per_point_errors,
                }
            else:
                pass

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        
    # Save all results to a pickle file
    with open(os.path.join(OUTPUT_DIR, '3d_keypoint_errors.pkl'), 'wb') as f:
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
    parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_PyMouse_HQ_moseq_trans_synthetic_core.pt', help="Pretrained resource to use for fetching weights.")
    args = parser.parse_args()
    main(args.model, args.pretrained_resource)