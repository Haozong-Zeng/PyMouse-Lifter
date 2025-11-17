import cv2
import numpy as np
import torch
from torchvision.transforms import Compose
from ultralytics import YOLO
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import matplotlib as mpl

# ----------------- Global Configuration -----------------
FINAL_HEIGHT = 518
FINAL_WIDTH = 518
FPS = 24
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIDENCE_THRESHOLD = 0.3  # Add confidence threshold

KEYPOINT_NAMES = [
    'nose', 'head', 'left_ear', 'right_ear', 'neck',
    'spine_center', 'lumbar_spine', 'tail_base'
]
_cmap = mpl.colormaps['turbo'].resampled(len(KEYPOINT_NAMES))
KPT_COLORS = [tuple(int(255 * c) for c in _cmap(i)[:3][::-1]) for i in range(len(KEYPOINT_NAMES))]

# ----------------- Model Loading -----------------
depth_model_cfg = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}
depth_anything = DepthAnything(depth_model_cfg['vitl'])
depth_anything.load_state_dict(
    torch.load('./metric_depth/checkpoints/depth_anything_metric_PyMouse_HQ_orbbec_trans_synthetic.pt')
)
depth_anything = depth_anything.half().to(DEVICE).eval()

yolo_model = YOLO('./YOLO_models/yolo11m-orbbec-pose-real.pt')

transform = Compose([
    Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
           resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# ----------------- Visualization Functions -----------------
def draw_depth_and_kpts(depth_map, kpts_xy, radius=3):
    depth_map = np.squeeze(depth_map).astype(np.float32)
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    vis = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)

    if kpts_xy is not None:
        for idx, (x, y) in enumerate(kpts_xy.astype(int)):
            color = KPT_COLORS[idx] if idx < len(KPT_COLORS) else KPT_COLORS[-1]
            cv2.circle(vis, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return vis

# ----------------- Main Function -----------------
def run_webcam_visualization():
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FINAL_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FINAL_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        resized_frame = cv2.resize(frame, (FINAL_WIDTH, FINAL_HEIGHT))
        rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB) / 255.0
        inp = transform({'image': rgb})['image']
        inp_tensor = torch.from_numpy(inp).unsqueeze(0).half().to(DEVICE)

        with torch.no_grad():
            depth_pred = depth_anything(inp_tensor)[0].cpu().numpy()

        result = yolo_model([resized_frame], half=True, max_det=1, verbose=False)[0]

        # Filter detection results with confidence < 0.3
        filtered_keypoints = None
        if len(result.keypoints) > 0:
            # Get detection confidence
            try:
                confidence = result.boxes.conf[0].item()
                if confidence < CONFIDENCE_THRESHOLD:
                    print(f"Detection confidence {confidence:.4f} < {CONFIDENCE_THRESHOLD}，filtered")
                    filtered_keypoints = None
                else:
                    filtered_keypoints = result.keypoints.data[0].cpu().numpy()
                    print(f"Detection confidence {confidence:.4f} >= {CONFIDENCE_THRESHOLD}，kept")
            except:
                filtered_keypoints = None

        vis_depth = draw_depth_and_kpts(depth_pred, filtered_keypoints)

        stacked = np.hstack((resized_frame, vis_depth))
        cv2.imshow("Webcam | Raw (Left) + Depth+Keypoints (Right)", stacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------- Startup -----------------
if __name__ == '__main__':
    run_webcam_visualization()
