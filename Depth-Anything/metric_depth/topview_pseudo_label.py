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

# Global settings
FL = 50  # 50
FY = 1007.87 * 0.6
FX = 1007.87 * 0.6
NYU_DATA = False
FINAL_HEIGHT = 512
FINAL_WIDTH = 512
INPUT_DIR = '/home/haozong/projects/ControlNeXt/topview_mouse/target'
OUTPUT_DIR = '/home/haozong/projects/ControlNeXt/topview_mouse/source'
DATASET = 'nyu'  # Lets not pick a fight with the model's dataloader

def process_images(model):
    # 获取所有图片路径并计数
    image_paths = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(('.png', '.jpg')):
                image_paths.append(os.path.join(root, file))

    # 使用 tqdm 进度条
    for image_path in tqdm(image_paths, desc="Processing Images", unit="image"):
        # 计算输出文件路径
        relative_path = os.path.relpath(os.path.dirname(image_path), INPUT_DIR)
        output_folder_path = os.path.join(OUTPUT_DIR, relative_path)
        
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        output_image_path = os.path.join(output_folder_path, os.path.splitext(os.path.basename(image_path))[0] + "_pred_depth.png")

        try:
            # 读取和处理图片
            color_image = Image.open(image_path).convert('RGB')
            original_width, original_height = color_image.size
            image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

            pred = model(image_tensor, dataset=DATASET)
            if isinstance(pred, dict):
                pred = pred.get('metric_depth', pred.get('out'))
            elif isinstance(pred, (list, tuple)):
                pred = pred[-1]
            pred = pred.squeeze().detach().cpu().numpy()

            # Clip and scale depth values
            pred = (pred - pred.min()) / (pred.max() - pred.min()) * 255.0
            pred = pred.astype(np.uint8)
            
            # Resize depth map to final size
            resized_pred = Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)

            # Save depth map as PNG
            resized_pred = resized_pred.convert("L")
            resized_pred.save(output_image_path)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main(model_name, pretrained_resource):
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    process_images(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_PyMouse_HQ_all_core.pt', help="Pretrained resource to use for fetching weights.")

    args = parser.parse_args()
    main(args.model, args.pretrained_resource)
