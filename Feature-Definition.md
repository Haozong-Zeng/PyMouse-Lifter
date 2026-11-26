# PyMouse Lifter – Complete Feature Definitions

This document provides a **complete and exhaustive specification of all behavioral features** extracted by **PyMouse Lifter**.  
It is intended to ensure **full transparency, reproducibility, and reviewer accessibility**, and directly addresses requests for detailed feature definitions.

All features described here are implemented exactly as specified in the public repository:

https://github.com/Haozong-Zeng/PyMouse-Lifter

The definitions follow the code that generated the training and test matrix of the 3D classifier, available as `PyMouse-Lifter-Classification/pymouselifter_features.py` (`extract_features_3d`, `flatten_feat_dict`, and `process_video_features_2d_single_mouse`). The real-time scripts use their own window lengths and regions of interest and are not covered here.

An extended discussion of feature motivation and usage is provided in the associated manuscript.

---

## 1. Coordinate Systems and Alignment

### 1.1 Raw 3-D Reconstruction

For each frame, 2-D anatomical keypoints are detected using a YOLO-based pose estimator and lifted into 3-D using monocular depth estimation.

Raw coordinates are defined as:

- **X**: horizontal image axis (positive to the right)
- **Y**: vertical image axis (positive upward)
- **Z**: height (positive toward the overhead camera; a larger value means higher above the floor)

For a keypoint at video pixel (u, v), with video size W × H (720 × 720) and depth-map size W_d × H_d (518 × 518):

```
x = u * W_d / W - W_d / 2
y = H_d / 2 - v * H_d / H
z = 2.3529 * D_t( floor(u * W_d / W), floor(v * H_d / H) )
```

where D_t(column, row) is the depth-network output at frame t.

**Units:** X and Y are in depth-map pixels (**px**; 1 px = 720/518 ≈ 1.39 video pixels). Z is in scaled depth-network units (**du**). The axes are not calibrated to a common physical unit. Quantities that combine horizontal and vertical components are labeled **mixed**.

All time derivatives use a fixed frame interval delta_t = 1/30 s. No smoothing, gap filling, or outlier removal is applied to the 3-D keypoints.

---

### 1.2 Rigid Body Alignment

To remove location in the arena and heading, posture features use 3-D coordinates aligned **independently for each frame** with the following steps:

1. **Translation**  
   The tail base is translated to the origin:
tail_base → (0, 0, 0)

2. **Rotation about the Z-axis**  
All keypoints are rotated so that the XY projection of the vector
tail_base → lumbar_spine
aligns with the **+Y axis**.

After alignment:
- global translation is removed,
- global yaw is removed,
- +X points to the animal's right, +Y points forward, and Z is height relative to the tail base,
- pitch, roll, and body size are retained.

Feature Group B uses **aligned coordinates**. Feature Group C uses the **original, unaligned coordinates** of Section 1.1, except for the variability of segment angular velocity, which is derived from aligned coordinates.

---

## 2. Anatomical Keypoints

The pose model returns 14 landmarks. The features use the first **8 anatomical landmarks**:

| Index | Keypoint name |
|------:|---------------|
| 0 | nose |
| 1 | head |
| 2 | left_ear |
| 3 | right_ear |
| 4 | neck |
| 5 | spine_center |
| 6 | lumbar_spine |
| 7 | tail_base |

---

## 3. Feature Group A — Appearance-Change Features (2-D)

**Purpose:**  
To capture instantaneous image-space intensity changes as a proxy for motion, posture change, or illumination transients.

### 3.1 Whole-frame Pixel Change

**Feature name:** `pixel_change`  
**Dimensionality:** 1 scalar  
**Units:** gray levels

**Definition:**

pixel_change =
mean( (I_t - I_{t-1})^2 ) / mean( I_t )

where I_t denotes the grayscale image at frame t, with floating-point intensities from 0 to 255. The value is 0 for the first frame and when mean( I_t ) is 0.

**Interpretation:**
- Sensitive to sudden body motion
- Captures motion not fully explained by keypoints
- Also responds to changes in lighting

---

### 3.2 Keypoint-Centered ROI Intensity Change

**Dimensionality:** 8 scalars  
**Units:** gray levels  

**Feature names:**
`{k}_pixel_change`, with k in the eight keypoints of Section 2 (tail_base included)

**Definition:**

The formula of Section 3.1 is evaluated over a square of at most 21 × 21 video pixels (10 pixels on each side of the center pixel), clipped at the frame border. The square is placed at the 2-D keypoint position in frame t, and the same pixels are read from frames t and t-1.

For this step only, missing 2-D positions are filled by linear interpolation over time, or with the nearest detected position at the ends of the recording. The value is 0 for the first frame, for an empty or zero-intensity region, and for a keypoint that is never detected in the recording.

The mid-tail, tail-tip, and paw landmarks are not used.

**Interpretation:**
Captures localized appearance changes associated with fine movements (e.g., grooming, head twitching).

---

## 4. Feature Group B — Posture and Kinematics (3-D)

### 4.1 Aligned 3-D Coordinates

**Dimensionality:**  
8 keypoints × 3 axes = **24 scalars**

**Feature naming convention:**
{k}_3d_x, {k}_3d_y, {k}_3d_z
where  
`k ∈ {nose, head, left_ear, right_ear, neck, spine_center, lumbar_spine, tail_base}`

**Units:** px (x, y); du (z)

**Interpretation:**
Posture expressed in the body-centered coordinate system of Section 1.2. `x` is the lateral offset from the body axis (right positive), `y` is the forward distance from the tail base along the body axis, and `z` is the height relative to the tail base.

---

### 4.2 Spine Segment Definition

The body axis is divided into **5 ordered spine segments**:

| Segment index | From → To |
|--------------:|-----------|
| 0 | nose → head |
| 1 | head → neck |
| 2 | neck → spine_center |
| 3 | spine_center → lumbar_spine |
| 4 | lumbar_spine → tail_base |

---

### 4.3 Spine Segment Angles

For each spine segment i, the angle is computed in the aligned XY plane as:

theta_i = atan2( delta_y , delta_x )

where delta_x and delta_y are the differences between the segment endpoints (rear point minus front point). The result lies in (-pi, pi]. A segment parallel to the body axis has theta_i = -pi/2; theta_i decreases when the front end swings to the animal's right and increases when it swings to the left.

**Dimensionality:** 5 scalars  
**Units:** radians  

**Feature names:**
segment_angles_0 … segment_angles_4

---

### 4.4 Spine Angular Velocity

ang_vel_i =
( theta_i(t) - theta_i(t-1) ) / delta_t

Angles are not unwrapped, so a step across the ±pi boundary appears as a jump of about 2 pi / delta_t ≈ 188.5 rad · s⁻¹.

**Dimensionality:** 5 scalars  
**Units:** radians · s⁻¹  

**Feature names:**
segment_ang_velocity_0 … segment_ang_velocity_4

---

### 4.5 Spine Angular Acceleration

ang_acc_i =
( ang_vel_i(t) - ang_vel_i(t-1) ) / delta_t

**Dimensionality:** 5 scalars  
**Units:** radians · s⁻²  

**Feature names:**
segment_ang_acc_0 … segment_ang_acc_4

---

### 4.6 Global Bend Ratio

**Feature name:** `bend_ratio`  
**Dimensionality:** 1 scalar  

bend_ratio =
( ||p_head - p_nose||
+ ||p_neck - p_head||
+ ||p_spine_center - p_neck||
+ ||p_lumbar_spine - p_spine_center||
+ ||p_tail_base - p_lumbar_spine|| )
/
( ||p_nose - p_tail_base|| + 1e-8 )

where ||a - b|| denotes the Euclidean distance between two 3-D points, over x, y, and z. The small constant avoids division by zero. The ratio is unitless.

**Interpretation:**
- ≈ 1 indicates a straight body
- larger values indicate increased curvature (e.g., turning, grooming, rearing)

---

## 5. Feature Group C — Locomotor Dynamics (3-D)

### 5.1 Velocity Vectors

Computed from the original, unaligned coordinates. For each body point (head, tail_base):

velocity(t) =
( position(t) - position(t-1) ) / delta_t

**Dimensionality:**  
2 points × 3 axes = **6 scalars**

**Feature names:**
head_vel_3d_x/y/z
tailbase_vel_3d_x/y/z

**Units:** px · s⁻¹ (x, y); du · s⁻¹ (z)

---

### 5.2 Acceleration Vectors

acceleration(t) =
( velocity(t) - velocity(t-1) ) / delta_t

**Dimensionality:** 6 scalars  

**Feature names:**
head_acc_3d_x/y/z
tailbase_acc_3d_x/y/z

**Units:** px · s⁻² (x, y); du · s⁻² (z)

---

### 5.3 Speed (Scalar Magnitude)

speed = sqrt( vx^2 + vy^2 + vz^2 )

**Dimensionality:** 2 scalars  

**Feature names:**
head_speed
tail_speed

**Units:** mixed · s⁻¹

---

### 5.4 Short-Term Motion Variability

Population standard deviation over a **rolling 11-frame window** (the current frame and the 10 preceding frames). Undefined values inside the window are ignored.

Includes:
- spine angular velocity (5 scalars; rad · s⁻¹)
- head velocity components (3 scalars; px · s⁻¹ for x, y and du · s⁻¹ for z)
- tail-base velocity components (3 scalars; same units)

**Feature names:**
segment_ang_velocity_std_0 … segment_ang_velocity_std_4
head_vel_3d_std_x/y/z
tailbase_vel_3d_std_x/y/z

**Dimensionality:** 11 scalars  

**Interpretation:**
Captures motion instability, jitter, and rapid behavioral transitions.

---

## 6. Temporal Validity and NaN Handling

Frames are numbered from 0 within each recording.

- Frames 0–8 are removed, so the first exported frame is frame 9.
- Angular velocity, angular acceleration, velocity, acceleration, and speed are `NaN` up to and including frame 9.
- Rolling variability is `NaN` at frame 9, equals 0 at frame 10 (one value in the window), and uses the full 11-frame window from frame 20.
- A keypoint that is not detected has `NaN` coordinates, and the posture and motion features computed from it are `NaN`. Rolling variability ignores these values (Section 5.4), and appearance-change features use interpolated 2-D positions (Section 3.2).

Exported feature tables can therefore contain `NaN`. These values are neither imputed nor removed; XGBoost handles them natively.

Eight features are constant by construction, apart from rounding error, because the alignment fixes the tail base and the body axis: `tail_base_3d_x`, `tail_base_3d_y`, `tail_base_3d_z`, `lumbar_spine_3d_x`, `segment_angles_4` (-pi/2), `segment_ang_velocity_4`, `segment_ang_acc_4`, and `segment_ang_velocity_std_4`.

---

## 7. Feature Count Summary

| Feature group | Number of scalars |
|---------------|------------------:|
| Appearance change (A) | 9 |
| Posture & kinematics (B) | 40 |
| Locomotor dynamics (C) | 25 |
| **Total** | **74** |

---
