import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
import re

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 全局设置
FINAL_HEIGHT = 720
FINAL_WIDTH = 720
DATASET = 'nyu'
COEFFICIENT = 2.3529

# 输入/输出根目录
INPUT_DIR = '/mnt/d/WSL/Depth-Anything/YOLO_models/dataset/orbbec-pose-real/images'

def rename_images_in_subfolders(root_dir):
    """
    在指定文件夹下遍历所有子文件夹，将满足
    rgb_{i:012d}_depth_pred.png 格式的图片重命名为
    depth_{i:012d}_pred.png。
    """
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            # 判断文件格式是否符合要求
            if file_name.startswith("rgb_") and file_name.endswith("_depth_pred.png"):
                # 提取中间的索引部分
                idx_str = file_name[len("rgb_"):-len("_depth_pred.png")]
                # 构建新文件名
                new_file_name = f"depth_{idx_str}_pred.png"

                old_path = os.path.join(root, file_name)
                new_path = os.path.join(root, new_file_name)

                os.rename(old_path, new_path)
                print(f"重命名: {old_path} -> {new_path}")

def process_images(model, suffix="_pred"):
    """
    遍历指定文件夹下，(可匹配特定字符的子文件夹)，
    对 .jpg 图像进行深度推理并保存为 .png。
    
    若输出文件已存在，则跳过处理。
    使用 suffix 来控制输出文件的后缀名，比如 "_depth_pred.png"、
    "_depth_pred_second.png" 等。
    """

    rename_images_in_subfolders(INPUT_DIR)

    # 需要匹配的关键字列表
    possible_keywords = ['val']
    batch_list = []

    for root, dirs, files in os.walk(INPUT_DIR):
        for d in dirs:
            # 如果文件夹 d 的名称中包含任意一个关键字，则加入 batch_list
            if any(keyword in d for keyword in possible_keywords):
                batch_list.append(d)
    
    # 遍历子文件夹
    for batch in batch_list:
        subfolder_dir = os.path.join(INPUT_DIR, batch)
        if not os.path.isdir(subfolder_dir):
            print(f"警告：未找到子文件夹 {subfolder_dir}，跳过...")
            continue
        
        # 找到所有的 .jpg 文件
        image_paths = glob.glob(os.path.join(subfolder_dir, '*.jpg'))
        if not image_paths:
            print(f"提示：子文件夹 {subfolder_dir} 下没有找到 .jpg 文件。")
            continue
        
        for image_path in tqdm(image_paths, desc=f"Processing {batch} with suffix {suffix}"):
            # 生成输出文件名，使用 suffix 替换默认 "_depth_pred"
            depth_save_path = image_path.replace('.jpg', f'{suffix}.png').replace('rgb', 'depth')
            
            # 如果已经存在则跳过
            #if os.path.exists(depth_save_path):
                #continue
            
            # 1) 读取 RGB 图片
            color_image = Image.open(image_path).convert('RGB')
            
            # 2) 转成张量并推理
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to(device)
            
            pred = model(image_tensor, dataset=DATASET)
            if isinstance(pred, dict):
                pred = pred.get('metric_depth', pred.get('out'))
            elif isinstance(pred, (list, tuple)):
                pred = pred[-1]
            
            # 转回 numpy 并缩放到 [0,255]
            pred = pred.squeeze().detach().cpu().numpy() * 1000
            # clip到 [0,255]
            pred = np.clip(pred, 0, 255)
            pred = pred.astype(np.uint8)
            
            # 3) 最近邻缩放到指定大小
            resized_pred = Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)

            # 4) 保存输出深度图
            resized_pred.save(depth_save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--model1", type=str, default='zoedepth',
                        help="Name of the first model to test")
    parser.add_argument("-p1", "--pretrained_resource1", type=str,
                        default='local::./checkpoints/depth_anything_metric_PyMouse_HQ_orbbec_trans_synthetic_core.pt',
                        help="Pretrained resource for the first model")

    parser.add_argument("-m2", "--model2", type=str, default='zoedepth',
                        help="Name of the second model to test")
    parser.add_argument("-p2", "--pretrained_resource2", type=str,
                        default='local::./checkpoints/depth_anything_metric_PyMouse_HQ_orbbec_real_core.pt',
                        help="Pretrained resource for the second model")

    args = parser.parse_args()

    # 1) 加载第一个模型
    config1 = get_config(args.model1, "eval", DATASET)
    config1.pretrained_resource = args.pretrained_resource1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model1 = build_model(config1).to(device)
    model1.eval()

    # 使用第一个模型推理并保存后缀为 "_depth_pred" 的图片
    process_images(model1, suffix="_pred")

    '''# 2) 加载第二个模型
    config2 = get_config(args.model2, "eval", DATASET)
    config2.pretrained_resource = args.pretrained_resource2
    model2 = build_model(config2).to(device)
    model2.eval()

    # 使用第二个模型推理并保存后缀为 "_depth_pred_second" 的图片
    process_images(model2, suffix="_gt")'''


if __name__ == '__main__':
    main()
