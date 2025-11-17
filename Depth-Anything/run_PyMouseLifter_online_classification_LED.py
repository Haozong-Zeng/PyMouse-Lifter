import argparse
import cv2
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from ultralytics import YOLO
import joblib
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import matplotlib as mpl
import u3
import threading
import queue
from threading import Lock, Event

# ----------------- Global Setups -----------------
FINAL_HEIGHT = 518
FINAL_WIDTH = 518
DATASET = 'nyu'
COEFFICIENT = 2.3529
FPS = 24
dt = 1.0 / FPS
CAMERA_ID = 4  # CameraID，check your computer's usb device

# KEYPOINTS definition
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
IDX_NOSE = 0
IDX_HEAD = 1
IDX_LEFT_EAR = 2
IDX_RIGHT_EAR = 3
IDX_NECK = 4
IDX_SPINE_CENTER = 5
IDX_LUMBAR_SPINE = 6
IDX_TAIL_BASE = 7

# Turbo colormap
_cmap = mpl.colormaps['turbo'].resampled(len(KEYPOINT_NAMES))
KPT_COLORS = [
    tuple(int(255 * c) for c in _cmap(i)[:3][::-1])  # BGR
    for i in range(len(KEYPOINT_NAMES))
]

FEATURE_COLUMNS = [
    'nose_3d_z', 'pixel_change', 'nose_3d_y', 'neck_3d_y', 'bend_ratio',
    'head_3d_y', 'spine_center_3d_y', 'left_ear_3d_y', 'right_ear_3d_z',
    'right_ear_3d_y', 'spine_center_3d_z', 'left_ear_3d_z', 'segment_angles_0',
    'neck_3d_z', 'lumbar_spine_3d_z', 'segment_angles_2', 'nose_3d_x',
    'head_3d_z', 'tail_speed', 'head_speed', 'lumbar_spine_3d_y', 'neck_3d_x',
    'segment_angles_1', 'right_ear_3d_x', 'left_ear_3d_x', 'segment_angles_3',
    'head_3d_x', 'spine_center_3d_x', 'head_vel_3d_z'
]

# ----------------- multiprocessing setups -----------------
classification_queue = queue.Queue(maxsize=500)
latest_classification = {"result": "None", "frame_id": -1}
classification_lock = Lock()
shutdown_event = Event()

# ----------------- video recording setups -----------------
video_queue = queue.Queue(maxsize=500)
video_writer = None
video_recording_enabled = False


def crop_to_fixed_square(image, crop_size=720):
    """
    Crop image to fixed-size square
    Args:
        image: Input image (numpy array)
        crop_size: Target square side length (default 720)
    Returns:
        cropped_image: Cropped square image
        crop_info: Crop info dictionary, containing offset and original size
    """
    height, width = image.shape[:2]
    
    # check image size
    if width < crop_size or height < crop_size:
        # resize if too small
        scale_factor = max(crop_size / width, crop_size / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]
    
    # crop a square
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    
    cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    
    # keep cropping info
    crop_info = {
        'start_x': start_x,
        'start_y': start_y,
        'crop_size': crop_size,
        'original_width': width,
        'original_height': height
    }
    
    return cropped_image, crop_info


def compute_pixel_change(prev_gray, curr_gray, mask=None):
    """compute pixel change: mean((curr - prev)^2) / mean(curr)"""
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
    """rotate the points around z axis"""
    c, s = np.cos(alpha), np.sin(alpha)
    Rz = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]], dtype=np.float32)
    return points @ Rz.T


def align_tailbase_lumbar(frame_3d):
    """align the keypoints"""
    if frame_3d is None or len(frame_3d) < 8:  # make sure there's enough keypoints
        return frame_3d
    
    tail_base = frame_3d[IDX_TAIL_BASE]
    lumbar_spine = frame_3d[IDX_LUMBAR_SPINE]
    shifted_points = frame_3d - tail_base
    vec = lumbar_spine - tail_base
    dx, dy = vec[0], vec[1]
    theta = np.arctan2(dy, dx)
    alpha = np.pi / 2 - theta
    rotated_points = rotate_around_z(shifted_points, alpha)
    return rotated_points


def compute_segment_angle_2d(p1, p2):
    """compute segment angle 2d"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)


def compute_features_for_two_Frame(prev_frame_3d, curr_frame_3d, dt):
    """compute behavior features with 2 frames"""
    if prev_frame_3d is None or curr_frame_3d is None:
        return None
    
    # make sure there's enough keypoints
    if len(prev_frame_3d) < 8 or len(curr_frame_3d) < 8:
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

    head_vel = (head_curr - head_prev) / dt
    tail_vel = (tail_curr - tail_prev) / dt
    head_speed = np.linalg.norm(head_vel)
    tail_speed = np.linalg.norm(tail_vel)

    head_acc = np.full((3,), np.nan, dtype=np.float32)
    tail_acc = np.full((3,), np.nan, dtype=np.float32)

    bend_chain = [IDX_NOSE, IDX_HEAD, IDX_NECK, IDX_SPINE_CENTER, IDX_LUMBAR_SPINE, IDX_TAIL_BASE]
    poly_len = 0.0
    for k in range(len(bend_chain) - 1):
        pA = aligned_curr[bend_chain[k]]
        pB = aligned_curr[bend_chain[k + 1]]
        poly_len += np.linalg.norm(pB - pA)
    nose_pt = aligned_curr[IDX_NOSE]
    tail_pt = aligned_curr[IDX_TAIL_BASE]
    direct_dist = np.linalg.norm(tail_pt - nose_pt) + 1e-8
    bend_ratio = poly_len / direct_dist

    nose_3d = aligned_curr[IDX_NOSE]
    head_3d = aligned_curr[IDX_HEAD]
    left_ear_3d = aligned_curr[IDX_LEFT_EAR]
    right_ear_3d = aligned_curr[IDX_RIGHT_EAR]
    neck_3d = aligned_curr[IDX_NECK]
    spine_center_3d = aligned_curr[IDX_SPINE_CENTER]
    lumbar_spine_3d = aligned_curr[IDX_LUMBAR_SPINE]
    tail_base_3d = aligned_curr[IDX_TAIL_BASE]

    feat_dict = {
        'nose_3d': nose_3d,
        'head_3d': head_3d,
        'left_ear_3d': left_ear_3d,
        'right_ear_3d': right_ear_3d,
        'neck_3d': neck_3d,
        'spine_center_3d': spine_center_3d,
        'lumbar_spine_3d': lumbar_spine_3d,
        'tail_base_3d': tail_base_3d,
        'segment_angles': angles_curr,
        'segment_ang_vel': ang_vel,
        'segment_ang_acc': ang_acc,
        'head_vel_3d': head_vel,
        'head_acc_3d': head_acc,
        'tailbase_vel_3d': tail_vel,
        'tailbase_acc_3d': tail_acc,
        'head_speed': head_speed,
        'tail_speed': tail_speed,
        'bend_ratio': bend_ratio
    }
    return feat_dict


def flatten_feat_dict(feat_dict):
    """flatten the features to 1D"""
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


def draw_depth_and_kpts(depth_map, kpts_xy, radius=3):
    """draw depth maps and keypoints"""
    depth_map = np.squeeze(depth_map).astype(np.float32, copy=False)
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    vis = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)

    if kpts_xy is not None and len(kpts_xy) > 0:
        for idx, (x, y) in enumerate(kpts_xy.astype(int)):
            color = KPT_COLORS[idx] if idx < len(KPT_COLORS) else KPT_COLORS[-1]
            cv2.circle(vis, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return vis


def get_original_image_size(cap):
    """get original camera resolusion"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height


def send_gpio_pulse(device, pin, duration=0.005):
    """
    send GPIO pulse signal
    Args:
        device: LabJack U3 
        pin: GPIO (4 for FIO4, 5 for FIO5)
        duration: pulse duration (second)
    """
    def pulse_thread():
        try:
            if device is not None:
                # GPIO high
                device.setFIOState(pin, 1)
                # wait
                time.sleep(duration)
                # GPIO low
                device.setFIOState(pin, 0)
        except Exception as e:
            print(f"GPIO pulse error (FIO{pin}): {e}")
    
    # Create and start daemon thread
    thread = threading.Thread(target=pulse_thread, daemon=True)
    thread.start()


def initialize_labjack():
    """
    initialize LabJack U3-LV device
    Returns:
        device: LabJack U3 device object, return None if failed
    """
    try:
        device = u3.U3()
        print("LabJack U3-LV device initialized")
        
        # Configure FIO4 and FIO5 as digital outputs, with an initial low level
        device.configIO(FIOAnalog=0)  # Set all FIOs to digital mode.
        device.setFIOState(4, 0)  # FIO4 is set to low level.
        device.setFIOState(5, 0)  # FIO5 is set to low level.
        print("GPIO pins FIO4 and FIO5 are configured for output mode.")
    
        return device
    except Exception as e:
        print(f"LabJack U3-LV initialization failed: {e}")
        print("The program will continue to run, but will not send GPIO signals.")
        return None


def classification_worker(rf_model, labjack_device, result_file, batch_size=1):
    """
    Background classification worker thread (supports batch processing)
    Args:
        rf_model: Random forest classification model
        labjack_device: LabJack U3 device object
        result_file: Result file object
        batch_size: Batch processing size (1=single frame, 2=dual frame batch processing)
    """
    global classification_queue, latest_classification, classification_lock, shutdown_event
    
    print(f"Classification worker thread started (batch size: {batch_size})")
    
    if batch_size == 1:
        # Single frame processing mode (original logic)
        while not shutdown_event.is_set():
            try:
                # Get feature data from queue, set timeout to avoid blocking
                frame_id, feature_data = classification_queue.get(timeout=1.0)
                
                # Check if need to skip classification to catch up
                if classification_queue.qsize() > 5:
                    print(f"Skip frame {frame_id} classification to catch up (queue size: {classification_queue.qsize()})")
                    classification_queue.task_done()
                    continue
                
                # Execute random forest classification
                try:
                    X = np.array(feature_data, dtype=np.float32).reshape(1, -1)
                    y_pred = rf_model.predict(X)[0]
                    
                    # Send inference complete signal (FIO5)
                    send_gpio_pulse(labjack_device, 5, 0.1)
                    
                    # Thread-safely update latest classification result
                    with classification_lock:
                        latest_classification["result"] = y_pred
                        latest_classification["frame_id"] = frame_id
                    
                    # Save classification result to file
                    if result_file is not None:
                        result_file.write(f"Frame {frame_id}: {y_pred}\n")
                        result_file.flush()
                    
                except Exception as e:
                    print(f"Classification prediction error (frame {frame_id}): {e}")
                    # Even if classification fails, update result to"None"
                    with classification_lock:
                        latest_classification["result"] = "None"
                        latest_classification["frame_id"] = frame_id
                
                # Mark task complete
                classification_queue.task_done()
                
            except queue.Empty:
                # Timeout, continue checkingshutdown_event
                continue
            except Exception as e:
                print(f"Classification worker thread error: {e}")
    
    elif batch_size == 2:
        # Dual frame batch processing mode
        frame_buffer = []
        
        while not shutdown_event.is_set():
            try:
                # Collect batch processing frames
                if len(frame_buffer) < batch_size:
                    try:
                        frame_data = classification_queue.get(timeout=1.0)
                        frame_buffer.append(frame_data)
                        classification_queue.task_done()
                        continue
                    except queue.Empty:
                        continue
                
                # Check if need to skip classification to catch up
                if classification_queue.qsize() > 5:
                    print(f"Skip batch classification to catch up (queue size: {classification_queue.qsize()})")
                    frame_buffer.clear()
                    continue
                
                # Process batch (2frame)
                frame1_id, frame1_features = frame_buffer[0]
                frame2_id, frame2_features = frame_buffer[1]
                
                try:
                    # Use second frame features for classification (because features are calculated based on inter-frame differences)
                    X = np.array(frame2_features, dtype=np.float32).reshape(1, -1)
                    y_pred = rf_model.predict(X)[0]
                    
                    # Send inference complete signal (FIO5) - once per batch
                    send_gpio_pulse(labjack_device, 5, 0.1)
                    
                    # Thread-safely update latest classification result - both frames use same label
                    with classification_lock:
                        latest_classification["result"] = y_pred
                        latest_classification["frame_id"] = frame2_id
                    
                    # Save classification result to file - both frames use same label
                    if result_file is not None:
                        result_file.write(f"Frame {frame1_id}: {y_pred}\n")
                        result_file.write(f"Frame {frame2_id}: {y_pred}\n")
                        result_file.flush()
                    
                    print(f"Batch classification complete: Frame {frame1_id}, {frame2_id} -> {y_pred}")
                    
                except Exception as e:
                    print(f"Batch classification prediction error (frame {frame1_id}, {frame2_id}): {e}")
                    # Even if classification fails, update result to"None"
                    with classification_lock:
                        latest_classification["result"] = "None"
                        latest_classification["frame_id"] = frame2_id
                    
                    # Save failure result
                    if result_file is not None:
                        result_file.write(f"Frame {frame1_id}: None\n")
                        result_file.write(f"Frame {frame2_id}: None\n")
                        result_file.flush()
                
                # Clear buffer for next batch
                frame_buffer.clear()
                
            except Exception as e:
                print(f"Batch processing classification worker thread error: {e}")
                frame_buffer.clear()
    
    print("Classification worker thread has exited")


def get_latest_classification():
    """
    Thread-safely get latest classification result
    Returns:
        tuple: (classification_result, frame_id)
    """
    global latest_classification, classification_lock
    
    with classification_lock:
        return latest_classification["result"], latest_classification["frame_id"]


def video_writer_worker(output_path, fps, frame_size):
    """
    Video recording worker thread
    Args:
        output_path: Output video file path
        fps: Video frame rate
        frame_size: Video frame size (width, height)
    """
    global video_queue, shutdown_event
    
    print(f"Video recording thread started: {output_path}")
    print(f"Video parameters: {frame_size[0]}x{frame_size[1]} @ {fps} FPS")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    if not video_writer.isOpened():
        print(f"Error: Cannot create video file {output_path}")
        return
    
    frames_written = 0
    
    try:
        while not shutdown_event.is_set():
            try:
                # Get frame data from queue, set timeout to avoid blocking
                left_frame, right_frame = video_queue.get(timeout=1.0)
                
                # Ensure both frame sizes match
                if left_frame.shape[:2] != right_frame.shape[:2]:
                    # Adjust left frame size to match right frame
                    right_height, right_width = right_frame.shape[:2]
                    left_frame = cv2.resize(left_frame, (right_width, right_height))
                
                # Horizontally concatenate left and right frames
                combined_frame = np.hstack([left_frame, right_frame])
                
                # Ensure concatenated frame size matches video writer
                if combined_frame.shape[:2][::-1] != frame_size:
                    combined_frame = cv2.resize(combined_frame, frame_size)
                
                # Write to video
                video_writer.write(combined_frame)
                frames_written += 1
                
                # Mark task complete
                video_queue.task_done()
                
            except queue.Empty:
                # Timeout, continue checkingshutdown_event
                continue
            except Exception as e:
                print(f"Video recording error: {e}")
                break
    
    finally:
        # Release video writer
        video_writer.release()
        print(f"Video recording complete: {frames_written} frames written to {output_path}")


def add_frame_to_video_queue(raw_frame, vis_frame):
    """
    Non-blocking add frame to video recording queue
    Args:
        raw_frame: Raw RGB frame
        vis_frame: Visualization frame (depth map + keypoints)
    """
    global video_queue, video_recording_enabled
    
    if not video_recording_enabled:
        return
    
    try:
        # Non-blocking add to queue
        video_queue.put_nowait((raw_frame.copy(), vis_frame.copy()))
    except queue.Full:
        # Queue full, skip current frame
        pass


def configure_camera_fps(cap, target_fps):
    """
    Configure camera FPS and verify settings
    Args:
        cap: OpenCV VideoCapture object
        target_fps: Target FPS value
    Returns:
        actual_fps: Actually set FPS value
    """
    print(f"Configuring camera FPS to {target_fps}...")
    
    # Set FPS
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    # Verify actually set FPS
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get other camera parameters for diagnosis
    buffer_size = cap.get(cv2.CAP_PROP_BUFFERSIZE)
    
    print(f"Camera FPS configuration result:")
    print(f"  Target FPS: {target_fps}")
    print(f"  Actual FPS: {actual_fps}")
    print(f"  Buffer size: {buffer_size}")
    
    if abs(actual_fps - target_fps) > 0.1:
        print(f"Warning: Camera cannot be set to exact {target_fps} FPS")
        print(f"       Will use camera supported {actual_fps} FPS")
    else:
        print(f"Success: Camera FPS has been set to {actual_fps}")
    
    return actual_fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_id', type=int, default=CAMERA_ID, help='Camera device ID')
    parser.add_argument('--output_dir', type=str, default='./realtime_output', help='Result save directory')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'], help='Depth model encoder')
    parser.add_argument('--crop_size', type=int, default=518, help='Fixed crop square size')
    parser.add_argument('--batch_size', type=int, default=1, choices=[1, 2], help='Classification batch size (1=single frame, 2=dual frame batch processing)')
    parser.add_argument('--record_video', action='store_true', help='Record side-by-side comparison MP4 video')
    parser.add_argument('--video_output', type=str, default='realtime_recording.mp4', help='Output video filename')
    parser.add_argument('--video_fps', type=int, default=24, help='Video FPS (default: match processing FPS)')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    print(f"Fixed crop size: {args.crop_size}x{args.crop_size}")

    # Load depth model
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    depth_anything = DepthAnything(model_configs[args.encoder])
    depth_anything.load_state_dict(
        torch.load('./metric_depth/checkpoints/depth_anything_metric_PyMouse_HQ_orbbec_trans_synthetic.pt')
    )
    depth_anything.half().to(DEVICE)
    depth_anything.eval()

    # Load YOLO model
    yolo_path = './YOLO_models/yolo11m-orbbec-pose-real.pt'
    yolo_model = YOLO(yolo_path)

    # Load random forest classification model
    rf_model = joblib.load('./YOLO_models/rf_model_realtime_demo.pkl')

    # Initialize LabJack U3-LV device
    labjack_device = initialize_labjack()

    # Image preprocessing
    transform = Compose([
        Resize(
            width=FINAL_WIDTH,
            height=FINAL_HEIGHT,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # Open camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error：Cannot open camera {args.camera_id}")
        exit(1)

    # Configure camera FPS and verify settings
    actual_fps = configure_camera_fps(cap, FPS)
    
    # If actual FPS differs from target FPS, update dt value to maintain feature calculation accuracy
    if abs(actual_fps - FPS) > 0.1:
        actual_dt = 1.0 / actual_fps
        print(f"Update time interval: dt = {actual_dt:.4f}s (based on actual FPS {actual_fps})")
    else:
        actual_dt = dt
        print(f"Use preset time interval: dt = {actual_dt:.4f}s")
    
    # Get camera native image size
    native_width, native_height = get_original_image_size(cap)
    print(f"Camera native size: {native_width}x{native_height}")
    
    # Check if camera resolution is sufficient for cropping
    min_dimension = min(native_width, native_height)
    if min_dimension < args.crop_size:
        print(f"Warning: Camera minimum dimension ({min_dimension}) less than target crop size ({args.crop_size})")
        print("Will scale then crop")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = open(os.path.join(args.output_dir, "realtime_results.txt"), 'w')

    # Start classification worker thread
    classification_thread = threading.Thread(
        target=classification_worker, 
        args=(rf_model, labjack_device, result_file, args.batch_size),
        daemon=True
    )
    classification_thread.start()
    
    print(f"Classification mode: {'single frame processing' if args.batch_size == 1 else 'dual frame batch processing'}")

    # Initialize video recording
    video_writer_thread = None
    if args.record_video:
        video_recording_enabled = True
        
        # Determine video FPS
        video_fps = args.video_fps if args.video_fps is not None else int(actual_fps)
        
        # Determine video size (side-by-side comparison: 2xdepth map width)
        video_width = FINAL_WIDTH * 2  # 1036 (518*2)
        video_height = FINAL_HEIGHT    # 518
        video_frame_size = (video_width, video_height)
        
        # Build complete video output path
        video_output_path = os.path.join(args.output_dir, args.video_output)
        
        # Start video recording thread
        video_writer_thread = threading.Thread(
            target=video_writer_worker,
            args=(video_output_path, video_fps, video_frame_size),
            daemon=True
        )
        video_writer_thread.start()
        
        print(f"Video recording enabled: {video_output_path}")
        print(f"Video format: {video_width}x{video_height} @ {video_fps} FPS")
    else:
        print("Video recording disabled")

    # Initialize previous frame data and timestamp tracking
    prev_frame_data = None
    frame_timestamps = []

    print("Real-time pose estimation and behavior classification started, press'q'key to exit...")
    frame_count = 0
    fps_list = []

    if args.batch_size == 1:
        # Single frame processing mode (using dynamic real-time FPS)
        while True:
            start_time = time.time()
            
            # Read current frame
            ret, frame = cap.read()
            if not ret:
                print("Error：Cannot get frame")
                break

            # Record frame timestamp for dynamic dt calculation
            current_timestamp = time.time()
            frame_timestamps.append(current_timestamp)
            
            # Calculate dynamic dt (actual inter-frame time)
            dynamic_dt = actual_dt  # default value
            if len(frame_timestamps) >= 2:
                dynamic_dt = frame_timestamps[-1] - frame_timestamps[-2]
                # Add sanity check to avoid extreme values
                if dynamic_dt < 0.01 or dynamic_dt > 1.0:  # limit between 10ms and 1s
                    dynamic_dt = actual_dt  # use default value
                    #print(f"Warning: Detected abnormal inter-frame time {dynamic_dt:.4f}s，use default dt")
            
            # Keep timestamp list size reasonable
            if len(frame_timestamps) > 100:
                frame_timestamps = frame_timestamps[-50:]  # keep recent 50 timestamps

            # Send camera trigger signal (FIO4)
            send_gpio_pulse(labjack_device, 4, 0.1)

            # Apply fixed size crop (crop first then detect)
            cropped_frame, crop_info = crop_to_fixed_square(frame, args.crop_size)
            
            # Keypoint detection (on cropped frame)
            yolo_result = yolo_model([cropped_frame], half=True, max_det=1, verbose=False)[0]
            keypoints = yolo_result.keypoints.data[0].cpu().numpy() if len(yolo_result.keypoints) > 0 else None

            # Preprocess cropped image
            image_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB) / 255.0
            gray_image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            
            transformed = transform({'image': image_rgb})['image']
            input_tensor = torch.from_numpy(transformed).unsqueeze(0).half().to(DEVICE)
            
            # Depth estimation
            with torch.no_grad():
                depth_map = depth_anything(input_tensor)[0].cpu().numpy()
            
            # Get depth map size
            depth_height, depth_width = depth_map.shape[:2]
            
            # Get cropped image size (now square)
            current_height, current_width = cropped_frame.shape[:2]
            
            # Calculate 2D keypoint coordinates in depth map
            curr_frame_3d = None
            if keypoints is not None and len(keypoints) > 0:
                # Map using current frame size
                x2d = keypoints[:, 0] * (depth_width / current_width)
                y2d = keypoints[:, 1] * (depth_height / current_height)
                
                # Ensure coordinates are within valid range
                x2d = np.clip(x2d, 0, depth_width - 1)
                y2d = np.clip(y2d, 0, depth_height - 1)
                
                # Get 3D coordinates from depth map
                x2d_int = x2d.astype(np.int32)
                y2d_int = y2d.astype(np.int32)
                pred_z = depth_map[y2d_int, x2d_int]
                
                # Convert to 3D coordinates (origin at image center)
                pred_x = x2d - (depth_width / 2)
                pred_y = -(y2d - (depth_height / 2))
                curr_frame_3d = np.stack((pred_x, pred_y, pred_z), axis=-1)
            else:
                print(f"Warning: Frame {frame_count} no keypoints detected")
            
            # Calculate features (using dynamic dt)
            feat_dict = None
            pixel_change = 0.0
            if prev_frame_data is not None and prev_frame_data['keypoints'] is not None and curr_frame_3d is not None:
                prev_gray = prev_frame_data['gray']
                pixel_change = compute_pixel_change(prev_gray, gray_image)
                
                prev_frame_3d = prev_frame_data['3d_points']
                feat_dict = compute_features_for_two_Frame(prev_frame_3d, curr_frame_3d, dynamic_dt)
            
            # Flatten features and submit to classification queue (non-blocking)
            if feat_dict is not None:
                feat_dict['pixel_change'] = pixel_change
                flat_feat = flatten_feat_dict(feat_dict)
                feature_data = []
                for col in FEATURE_COLUMNS:
                    feature_data.append(flat_feat.get(col, np.nan))
                
                # Non-blocking submit feature data to classification queue
                try:
                    classification_queue.put_nowait((frame_count, feature_data))
                except queue.Full:
                    print(f"Classification queue full, skip frame {frame_count} classification")
            
            # Get latest classification result for display
            y_pred, classified_frame_id = get_latest_classification()
            
            # Save current frame data for next frame calculation
            prev_frame_data = {
                'frame': frame,
                'gray': gray_image,
                'keypoints': keypoints,
                '3d_points': curr_frame_3d
            }
            
            # Visualization
            if keypoints is not None and len(keypoints) > 0:
                vis_depth = draw_depth_and_kpts(depth_map, np.column_stack((x2d, y2d)))
            else:
                vis_depth = draw_depth_and_kpts(depth_map, None)
            
            # Display classification result and FPS
            cv2.putText(vis_depth, f"Label: {y_pred}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Calculate and display FPS
            end_time = time.time()
            frame_time = end_time - start_time
            curr_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_list.append(curr_fps)
            avg_fps = sum(fps_list[-30:]) / min(len(fps_list), 30)
            cv2.putText(vis_depth, f"FPS: {avg_fps:.1f}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display dynamic dt info
            cv2.putText(vis_depth, f"dt: {dynamic_dt:.3f}s", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Add frame to video recording queue (if enabled)
            add_frame_to_video_queue(cropped_frame, vis_depth)
            
            cv2.imshow("Real-time Pose Estimation with LED - Square Mode", vis_depth)
            
            # Press'q'key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1

    elif args.batch_size == 2:
        # GPU batch processing mode (batch=2, using dynamic real-time FPS)
        frame_buffer = []
        
        while True:
            batch_start_time = time.time()
            
            # Collect 2 frames for batch processing
            for i in range(2):
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot get frame")
                    break
                
                # Record frame timestamp for dynamic dt calculation
                current_timestamp = time.time()
                frame_timestamps.append(current_timestamp)
                
                # Send camera trigger signal (FIO4) - once per frame
                send_gpio_pulse(labjack_device, 4, 0.1)
                
                frame_buffer.append({
                    'frame': frame,
                    'frame_id': frame_count + i,
                    'timestamp': current_timestamp
                })
            
            if len(frame_buffer) < 2:
                break  # Cannot get enough frames
            
            # Keep timestamp list size reasonable
            if len(frame_timestamps) > 100:
                frame_timestamps = frame_timestamps[-50:]  # keep recent 50 timestamps
            
            # Batch crop and YOLO keypoint detection
            raw_frames = [item['frame'] for item in frame_buffer]
            cropped_frames = []
            crop_infos = []
            
            # First batch crop all frames
            for frame in raw_frames:
                cropped_frame, crop_info = crop_to_fixed_square(frame, args.crop_size)
                cropped_frames.append(cropped_frame)
                crop_infos.append(crop_info)
            
            # Perform YOLO detection on cropped frames
            yolo_results = yolo_model(cropped_frames, half=True, max_det=1, verbose=False)
            
            # Batch preprocessing and depth estimation
            batch_tensors = []
            batch_data = []
            
            for i, item in enumerate(frame_buffer):
                frame = item['frame']
                frame_id = item['frame_id']
                timestamp = item['timestamp']
                
                # Get keypoints (already in cropped coordinate system)
                keypoints = yolo_results[i].keypoints.data[0].cpu().numpy() if len(yolo_results[i].keypoints) > 0 else None
                
                # Use already cropped frame
                cropped_frame = cropped_frames[i]
                crop_info = crop_infos[i]
                
                # Preprocessing
                image_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB) / 255.0
                gray_image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                
                transformed = transform({'image': image_rgb})['image']
                batch_tensors.append(torch.from_numpy(transformed))
                
                batch_data.append({
                    'frame_id': frame_id,
                    'cropped_frame': cropped_frame,
                    'gray_image': gray_image,
                    'keypoints': keypoints,
                    'crop_info': crop_info,
                    'timestamp': timestamp
                })
            
            # GPU batchDepth estimation - key optimization point！
            batch_tensor = torch.stack(batch_tensors).half().to(DEVICE)
            with torch.no_grad():
                depth_batch = depth_anything(batch_tensor)  # single GPU call processes 2 frames
            
            # Process batch results
            for i, data in enumerate(batch_data):
                frame_id = data['frame_id']
                cropped_frame = data['cropped_frame']
                gray_image = data['gray_image']
                keypoints = data['keypoints']
                timestamp = data['timestamp']
                
                # Calculate dynamic dt (actual inter-frame time)
                dynamic_dt = actual_dt  # default value
                if prev_frame_data is not None and 'timestamp' in prev_frame_data:
                    dynamic_dt = timestamp - prev_frame_data['timestamp']
                    # Add sanity check to avoid extreme values
                    if dynamic_dt < 0.01 or dynamic_dt > 1.0:  # limit between 10ms and 1s
                        dynamic_dt = actual_dt  # use default value
                        print(f"Warning: Detected abnormal inter-frame time {dynamic_dt:.4f}s, use default dt")
                
                # Get depth map
                depth_map = depth_batch[i].cpu().numpy()
                depth_height, depth_width = depth_map.shape[:2]
                current_height, current_width = cropped_frame.shape[:2]
                
                # Calculate 3D coordinates
                curr_frame_3d = None
                x2d, y2d = None, None
                if keypoints is not None and len(keypoints) > 0:
                    x2d = keypoints[:, 0] * (depth_width / current_width)
                    y2d = keypoints[:, 1] * (depth_height / current_height)
                    
                    x2d = np.clip(x2d, 0, depth_width - 1)
                    y2d = np.clip(y2d, 0, depth_height - 1)
                    
                    x2d_int = x2d.astype(np.int32)
                    y2d_int = y2d.astype(np.int32)
                    pred_z = depth_map[y2d_int, x2d_int]
                    
                    pred_x = x2d - (depth_width / 2)
                    pred_y = -(y2d - (depth_height / 2))
                    curr_frame_3d = np.stack((pred_x, pred_y, pred_z), axis=-1)
                else:
                    print(f"Warning: Frame {frame_id} no keypoints detected")
                
                # Calculate features (using dynamic dt)
                feat_dict = None
                pixel_change = 0.0
                if prev_frame_data is not None and prev_frame_data['keypoints'] is not None and curr_frame_3d is not None:
                    prev_gray = prev_frame_data['gray']
                    pixel_change = compute_pixel_change(prev_gray, gray_image)
                    
                    prev_frame_3d = prev_frame_data['3d_points']
                    feat_dict = compute_features_for_two_Frame(prev_frame_3d, curr_frame_3d, dynamic_dt)
                
                # Submit features to classification queue
                if feat_dict is not None:
                    feat_dict['pixel_change'] = pixel_change
                    flat_feat = flatten_feat_dict(feat_dict)
                    feature_data = []
                    for col in FEATURE_COLUMNS:
                        feature_data.append(flat_feat.get(col, np.nan))
                    
                    try:
                        classification_queue.put_nowait((frame_id, feature_data))
                    except queue.Full:
                        print(f"Classification queue full, skip frame {frame_id} classification")
                
                # Generate visualization for each frame (for video recording)
                y_pred, classified_frame_id = get_latest_classification()
                
                if keypoints is not None and len(keypoints) > 0:
                    vis_depth = draw_depth_and_kpts(depth_map, np.column_stack((x2d, y2d)))
                else:
                    vis_depth = draw_depth_and_kpts(depth_map, None)
                
                cv2.putText(vis_depth, f"Label: {y_pred}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add each frame to video recording queue (if enabled) - Ensure both frames are recorded in batch processing mode
                add_frame_to_video_queue(cropped_frame, vis_depth)
                
                # Update previous frame data (including timestamp)
                prev_frame_data = {
                    'frame': raw_frames[i],
                    'gray': gray_image,
                    'keypoints': keypoints,
                    '3d_points': curr_frame_3d,
                    'timestamp': timestamp
                }
                
                # Only display visualization and calculate FPS on last frame
                if i == 1:  # only display last frame of batch
                    # Calculate batch FPS
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_fps = 2.0 / batch_time if batch_time > 0 else 0  # 2 frames FPS
                    fps_list.append(batch_fps)
                    avg_fps = sum(fps_list[-15:]) / min(len(fps_list), 15)  # 15 batches average
                    
                    #cv2.putText(vis_depth, f"Batch FPS: {avg_fps:.1f}", (10, 70), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #cv2.putText(vis_depth, f"dt: {dynamic_dt:.3f}s", (10, 110), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    cv2.imshow("Real-time Pose Estimation with LED - Batch Mode", vis_depth)
            
            # Clear buffer
            frame_buffer.clear()
            frame_count += 2
            
            # Press'q'key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Stop classification worker thread
    print("Stopping classification worker thread...")
    shutdown_event.set()
    
    # Wait for tasks in queue to complete
    try:
        classification_queue.join()
    except:
        pass
    
    # Wait for classification thread to exit
    if classification_thread.is_alive():
        classification_thread.join(timeout=2.0)
    
    # Wait for video recording thread to complete and cleanup
    if video_writer_thread is not None and video_writer_thread.is_alive():
        print("Completing video recording...")
        try:
            # Wait for remaining frames in video queue to process
            video_queue.join()
        except:
            pass
        # Wait for video recording thread to exit
        video_writer_thread.join(timeout=3.0)
        if video_writer_thread.is_alive():
            print("Warning: Video recording thread did not exit normally")
    
    # Release resources
    cap.release()
    result_file.close()
    cv2.destroyAllWindows()
    
    # Close LabJack device
    if labjack_device is not None:
        try:
            # Ensure GPIO pins are set to low level
            labjack_device.setFIOState(4, 0)
            labjack_device.setFIOState(5, 0)
            labjack_device.close()
            print("LabJack U3-LV device closed")
        except Exception as e:
            print(f"Error closing LabJack device: {e}")

    # Print average FPS
    if fps_list:
        mean_fps = np.mean(fps_list)
        std_fps = np.std(fps_list)
        print(f"Processing complete. Average FPS: {mean_fps:.3f} ± {std_fps:.3f}")
    print("Program has exited")
