# PyMouse Lifter – Complete Feature Definitions

This document provides a **complete and exhaustive specification of all behavioral features** extracted by **PyMouse Lifter**.  
It is intended to ensure **full transparency, reproducibility, and reviewer accessibility**, and directly addresses requests for detailed feature definitions.

All features described here are implemented exactly as specified in the public repository:

https://github.com/Haozong-Zeng/PyMouse-Lifter

An extended discussion of feature motivation and usage is provided in the associated manuscript.

---

## 1. Coordinate System and Alignment (Applies to All 3-D Features)

### 1.1 Raw 3-D Reconstruction

For each frame, 2-D anatomical keypoints are detected using a YOLO-based pose estimator and lifted into 3-D using monocular metric depth estimation.

Raw coordinates are defined as:

- **X**: horizontal image axis (positive to the right)
- **Y**: vertical image axis (positive upward)
- **Z**: depth (distance from camera)

All distances are expressed in **millimeters (mm)**.

---

### 1.2 Rigid Body Alignment

To remove camera-dependent motion and inter-frame yaw, all 3-D coordinates are aligned **independently for each frame** using the following steps:

1. **Translation**  
   The tail base is translated to the origin:
tail_base → (0, 0, 0)

markdown
复制代码

2. **Rotation about the Z-axis**  
The vector
tail_base → lumbar_spine

yaml
复制代码
is rotated such that it aligns with the global **+Y axis**.

After alignment:
- global translation is removed,
- global yaw is removed,
- remaining coordinates represent **body-centric posture and motion**.

All 3-D features described below use **aligned coordinates**, unless explicitly stated otherwise.

---

## 2. Anatomical Keypoints

PyMouse Lifter uses **8 anatomical landmarks**:

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
To capture instantaneous image-space intensity changes as a **hardware-agnostic proxy** for motion, posture change, or illumination transients.

### 3.1 Whole-frame Pixel Change

**Feature name:** `pixel_change`  
**Dimensionality:** 1 scalar  
**Units:** normalized intensity change (unitless)

**Definition:**

\[
\text{pixel\_change}
=
\frac{\mathbb{E}\left[(I_t - I_{t-1})^2\right]}
{\mathbb{E}[I_t]}
\]

where \( I_t \) denotes the grayscale image at frame *t*.

**Interpretation:**
- Sensitive to sudden body motion
- Captures motion not fully explained by keypoints
- Robust across camera hardware and lighting conditions

---

### 3.2 Keypoint-Centered ROI Intensity Change

**Dimensionality:** 8 scalars  
**ROI radius:** 20 pixels  

**Keypoints included:**
- nose
- head
- left_ear
- right_ear
- neck
- spine_center
- lumbar_spine

**Excluded:**
- tail keypoints

**Interpretation:**
Captures localized appearance changes associated with fine movements (e.g., grooming, head twitching).

---

## 4. Feature Group B — Posture and Kinematics (3-D)

### 4.1 Aligned 3-D Coordinates

**Dimensionality:**  
8 keypoints × 3 axes = **24 scalars**

**Feature naming convention:**
{k}_3d_x, {k}_3d_y, {k}_3d_z

yaml
复制代码

where  
`k ∈ {nose, head, left_ear, right_ear, neck, spine_center, lumbar_spine, tail_base}`

**Units:** millimeters (mm)

**Interpretation:**
Absolute posture expressed in a body-centered coordinate system.

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

For each segment *i*, the angle is computed in the aligned XY plane:

\[
\theta_i = \mathrm{atan2}(\Delta y, \Delta x)
\]

**Dimensionality:** 5 scalars  
**Units:** radians  

**Feature names:**
segment_angles_0 … segment_angles_4

yaml
复制代码

---

### 4.4 Spine Angular Velocity

\[
\dot{\theta}_i =
\frac{\theta_i(t) - \theta_i(t-1)}{\Delta t}
\]

**Dimensionality:** 5 scalars  
**Units:** radians · s⁻¹  

**Feature names:**
segment_ang_vel_0 … segment_ang_vel_4

yaml
复制代码

---

### 4.5 Spine Angular Acceleration

\[
\ddot{\theta}_i =
\frac{\dot{\theta}_i(t) - \dot{\theta}_i(t-1)}{\Delta t}
\]

**Dimensionality:** 5 scalars  
**Units:** radians · s⁻²  

**Feature names:**
segment_ang_acc_0 … segment_ang_acc_4

yaml
复制代码

---

### 4.6 Global Bend Ratio

**Feature name:** `bend_ratio`  
**Dimensionality:** 1 scalar  

\[
\text{bend\_ratio}
=
\frac{\sum_{i=1}^{5} \lVert p_{i+1} - p_i \rVert}
{\lVert p_{\text{nose}} - p_{\text{tail\_base}} \rVert}
\]

**Interpretation:**
- ≈ 1 indicates a straight body
- larger values indicate increased curvature (e.g., turning, grooming, rearing)

---

## 5. Feature Group C — Locomotor Dynamics (3-D)

### 5.1 Velocity Vectors

For each body point (head, tail_base):

\[
\vec{v}(t) =
\frac{\vec{p}(t) - \vec{p}(t-1)}{\Delta t}
\]

**Dimensionality:**  
2 points × 3 axes = **6 scalars**

**Feature names:**
head_vel_3d_x/y/z
tailbase_vel_3d_x/y/z

yaml
复制代码

**Units:** mm · s⁻¹

---

### 5.2 Acceleration Vectors

\[
\vec{a}(t) =
\frac{\vec{v}(t) - \vec{v}(t-1)}{\Delta t}
\]

**Dimensionality:** 6 scalars  
**Units:** mm · s⁻²

---

### 5.3 Speed (Scalar Magnitude)

\[
\text{speed} = \lVert \vec{v} \rVert
\]

**Dimensionality:** 2 scalars  

**Feature names:**
head_speed
tail_speed

yaml
复制代码

---

### 5.4 Short-Term Motion Variability

Computed using a **rolling 8-frame window**.

Includes:
- spine angular velocity (5 scalars)
- head velocity vector variability (3 scalars)
- tail velocity vector variability (3 scalars)

**Dimensionality:** 11 scalars  

**Interpretation:**
Captures motion instability, jitter, and rapid behavioral transitions.

---

## 6. Temporal Validity and NaN Handling

Features requiring temporal history are **undefined for the first 7 frames**.

- Undefined values are set to `NaN`
- Frames containing any `NaN` values are **removed prior to feature table export**

As a result, all exported feature tables contain **only finite values**.

---

## 7. Feature Count Summary

| Feature group | Number of scalars |
|---------------|------------------:|
| Appearance change (A) | 9 |
| Posture & kinematics (B) | 40 |
| Locomotor dynamics (C) | 25 |
| **Total** | **74** |

---

## 8. Reference Implementation

All features defined in this document are implemented in Python and publicly available at:

https://github.com/Haozong-Zeng/PyMouse-Lifter

The implementation follows **exactly** the definitions given here, without post-hoc modification.

---

*End of feature specification.*
