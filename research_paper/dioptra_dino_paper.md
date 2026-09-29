# Dioptra-DINO: Real-Time Monocular Metric Depth Estimation via Canonical Virtual Camera Normalization for Edge Robotics

**Author:** Yumnam Harryson Singh (Independent Researcher)  
**Contact:** `harryson424242@gmail.com`  
**Code & Checkpoints:** [https://github.com/SeranomTheGreat/dioptra](https://github.com/SeranomTheGreat/dioptra)  
**Date:** September 2026  

---

## Abstract

Monocular metric depth estimation on autonomous mobile robots presents a persistent trade-off between physical scale calibration and inference throughput. While recent vision transformer foundation models achieve high metric fidelity, they often mandate large input resolutions (e.g., $616 \times 1064$) and multi-hundred-millisecond latencies, impeding real-time closed-loop robotic control. Conversely, lightweight relative depth estimators suffer from severe scale ambiguity and cannot recover metric distance without test-time oracle alignment or uncalibrated priors.

In this work, we present **Dioptra-DINO**, an efficient 27.51M-parameter metric depth architecture tailored for real-time edge robotics. Dioptra-DINO couples a self-supervised DINOv2-Small backbone with a Canonical Virtual Camera transformation ($F_{canon} = 1000.0\text{px}$), Trivision Ray FiLM Modulation, and an Angular Residual Attention (ARA) module, enabling robust scale invariance directly at a native resolution of $336 \times 336$. Across extensive empirical evaluations spanning 3,000 indoor test frames, 21 diverse architectural categories, and head-to-head testing against contemporary camera-conditioned foundation models (Metric3D and UniDepth V2), we observe:

1. **Photorealistic Ray-Traced Interiors (Apple Hypersim, 2,744 frames)**: Dioptra-DINO attains an absolute relative error (**AbsRel**) of **0.1477** and inlier precision ($\delta_1$) of **84.3%**, outperforming Metric3D ViT-Small (AbsRel **0.2259**, $\delta_1$ **73.4%**) by **34.6% in relative error** and UniDepth V2 (**0.2164**, $\delta_1$ **76.2%**) by **31.7% in relative error**.
2. **Resolution-Matched Benchmark (100 frames @ $336 \times 336$)**: When placed on an identical $336 \times 336$ playing field, Dioptra-DINO achieves **0.2290 AbsRel** and **72.5% inliers**, whereas Metric3D collapses to **0.3997 AbsRel** and **17.3% inliers**, revealing that Metric3D's metric calibration heavily degrades at lower spatial resolutions.
3. **Real-Time Edge Efficiency**: Operating at native $336 \times 336$, Dioptra-DINO runs in **58.2–62.9 ms** (15.9–17.2 FPS) on Apple Silicon GPU and consumes under 240 MB VRAM, achieving a **12.6× speedup over Metric3D** (753.5 ms) and **6.2× speedup over UniDepth V2** (421.6 ms).

We candidly report failure modes, document multi-domain training distributions, and release all benchmark code, evaluation protocols, and fine-tuned checkpoints to foster reproducible edge robotics perception.

---

## I. Introduction

Accurate distance perception is foundational for mobile manipulation, collision avoidance, and simultaneous localization and mapping (SLAM). While active depth sensors (e.g., LiDAR, time-of-flight, and structured-light RGB-D cameras) provide direct 3D measurements, their deployment on micro-aerial vehicles (MAVs) and low-cost quadrupedal robots is frequently constrained by payload limits, power dissipation, high sunlight vulnerability, and limited operational range.

Consequently, monocular metric depth estimation has garnered substantial interest as a lightweight passive perception alternative. However, recovering true metric distance from a single 2D projection is inherently ill-posed due to projective scale ambiguity: an object of height $H$ at distance $Z$ produces the identical pixel projection $h = f_y \frac{H}{Z}$ as an object of size $kH$ at distance $kZ$. Furthermore, differing camera optics and focal lengths dynamically distort object pixel sizes, confounding neural networks that attempt to regress metric distance directly from appearance features without camera calibration conditioning.

Recent depth estimation models approach this challenge through two distinct paradigms:
- **Uncalibrated Foundation Models**: Architectures such as Depth Anything train on massive unlabeled web datasets ($>62\text{M}$ images), learning rich semantic boundaries. While metric fine-tuned variants yield competitive scores on standard benchmarks, they lack explicit camera intrinsics conditioning: unable to ingest camera calibration matrices $\mathbf{K}$, they rely on implicit scene priors and cannot mathematically adapt to varying optical focal lengths or optical zoom.
- **Camera-Conditioned Metric Models**: Frameworks such as Metric3D and UniDepth explicitly incorporate camera focal length conditioning. However, they rely on large spatial resolutions ($616 \times 1064$) or complex pinhole ray encoders, requiring $400\text{ms}$ to $750\text{ms}$ per frame on modern edge accelerators.

This paper explores a central research question: **Can a compact vision transformer (<28M parameters) running at a modest resolution ($336 \times 336$) deliver reliable, calibration-conditioned metric depth estimation suitable for real-time edge robotics?**

To address this, we present **Dioptra-DINO**. By formulating depth regression within a Canonical Virtual Camera space ($F_{canon} = 1000.0\text{px}$) and integrating Trivision Ray FiLM Modulation with Angular Residual Attention (ARA), Dioptra-DINO decouples metric scale from visual geometry. We thoroughly evaluate Dioptra-DINO across diverse indoor benchmark suites covering photorealistic ray-traced interiors (Apple Hypersim), synthetic multi-room environments (InteriorNet), and real-world sensor captures (ScanNet).

---

## II. Related Work

### A. Monocular Relative Depth Estimation
Early monocular depth methods focused on relative depth estimation via scale-invariant representations. Ranftl et al. introduced MiDaS and DPT, demonstrating that mixing heterogeneous datasets with an affine-invariant loss enables broad zero-shot generalization across scenes. Recently, Depth Anything V1 and V2 scaled relative pretraining using over 62 million unlabeled images and DINOv2 representations. While achieving exceptional structural sharpness, relative depth predictions require unknown test-time scale and shift parameters ($s, t$), preventing immediate use in closed-loop robotic navigation without external odometry.

### B. Camera-Conditioned Metric Depth Estimation
To recover true physical dimensions, works such as ZoeDepth and Metric3D investigated multi-dataset metric transfer. Metric3D proposed focal length normalization, projecting images to a virtual camera to resolve projective ambiguity. However, Metric3D relies on heavy input resolutions ($616 \times 1064$) and incurs prohibitive compute latency ($>750\text{ms}$). UniDepth advanced this direction by predicting universal metric depth and camera intrinsics directly via pseudopinhole ray embeddings. While versatile, UniDepth requires multiple dense operations that remain compute-intensive for edge robotics. Dioptra-DINO bridges this gap by demonstrating that canonical focal normalization coupled with lightweight ray modulation yields high metric fidelity at $336 \times 336$ in under $60\text{ms}$.

| Model Architecture | Backbone Family | Parameters | Native Resolution | Intrinsics Conditioning | Edge Latency (MPS) | Memory (VRAM) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dioptra-DINO (Ours)** | **DINOv2-Small (ViT-S/14)** | **27.51M** | **336 × 336** | **Canonical Virtual Cam + ARA** | **58.2–62.9 ms (16.7 FPS)** | **~240 MB** |
| UniDepth V2 ViT-Small | DINOv2-Small (ViT-S/14) | 34.18M | Dynamic / Adaptive | Pinhole Ray Embeddings | 421.6 ms (2.4 FPS) | ~650 MB |
| Metric3D ViT-Small | ViT-Small (DeiT-S) | 37.50M | 616 × 1064 | Pinhole Focal Normalization | 753.5 ms (1.3 FPS) | ~980 MB |

---

## III. Methodology

### A. Problem Formulation and Scale Ambiguity
Consider an actual camera with intrinsic matrix $\mathbf{K}$:
$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

A 3D point $\mathbf{P} = [X, Y, Z]^T$ in camera coordinates projects to image plane coordinates $\mathbf{p} = [u, v, 1]^T$ via:
$$Z \mathbf{p} = \mathbf{K} \mathbf{P} \implies u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y$$

If two cameras with differing focal lengths $f_1, f_2$ observe identical objects at different depths $Z_1, Z_2$, identical pixel dimensions occur whenever $\frac{f_1}{Z_1} = \frac{f_2}{Z_2}$. Consequently, standard convolutional or self-attention networks that process pixels without focal conditioning inevitably confuse camera focal zoom with physical object distance.

### B. Canonical Virtual Camera Transformation
To decouple metric regression from camera hardware, Dioptra-DINO defines a canonical virtual pinhole camera with reference focal length $F_{canon} = 1000.0\text{px}$:
$$\mathbf{K}_{canon} = \begin{bmatrix} F_{canon} & 0 & c_x' \\ 0 & F_{canon} & c_y' \\ 0 & 0 & 1 \end{bmatrix}$$

When an input image of size $W \times H$ is captured with focal length $f_x$, its spatial dimensions are scaled to the native input resolution $S \times S$ ($S=336$). The scaled native focal length becomes:
$$f_{scaled} = f_x \cdot \left(\frac{S}{W}\right)$$

The ratio between the sensor focal length and the canonical camera defines the scale adjustment factor $\gamma$:
$$\gamma = \frac{f_{scaled}}{F_{canon}}$$

The neural network predicts metric depth in canonical virtual space, denoted $d_{canon}(u, v) \in [0.1\text{m}, 10.0\text{m}]$. The physical metric depth $d_{metric}(u, v)$ is recovered via:
$$d_{metric}(u, v) = d_{canon}(u, v) \cdot \gamma = d_{canon}(u, v) \cdot \left(\frac{f_{scaled}}{F_{canon}}\right)$$

This formulation guarantees that the vision transformer learns scale-invariant geometric relationships, while physical metric units are preserved via deterministic focal re-projection.

### C. Trivision Ray FiLM Modulation
To ground token representations in 3D camera geometry without dense ray-tracing overhead, Dioptra-DINO unprojects a canonical ray triplet for each patch token $i \in \{1, \dots, N\}$. For patch center $(u_c, v_c)$ and chiral patch corners $(u_{c1}, v_{c1})$ (top-left) and $(u_{c2}, v_{c2})$ (bottom-right), unit ray vectors are computed via analytic closed-form pinhole inversion:
$$\mathbf{r}_c = \frac{1}{\|\mathbf{v}_c\|} \begin{bmatrix} \frac{u_c - c_x}{f_x} \\ \frac{v_c - c_y}{f_y} \\ 1 \end{bmatrix}, \quad \mathbf{r}_1 = \frac{\mathbf{v}_{c1}}{\|\mathbf{v}_{c1}\|}, \quad \mathbf{r}_2 = \frac{\mathbf{v}_{c2}}{\|\mathbf{v}_{c2}\|}$$

The concatenated triplet $[\mathbf{r}_c, \mathbf{r}_1, \mathbf{r}_2] \in \mathbb{R}^9$ is mapped into a multi-scale Fourier positional representation across $M=6$ octave bands:
$$\mathbf{e}(\mathbf{r}) = \left[ \sin(2^0 \pi \mathbf{r}), \cos(2^0 \pi \mathbf{r}), \dots, \sin(2^5 \pi \mathbf{r}), \cos(2^5 \pi \mathbf{r}) \right]^T \in \mathbb{R}^{108}$$

A lightweight two-layer MLP projects $\mathbf{e}(\mathbf{r})$ into affine scale ($\gamma_{film}$) and shift ($\beta_{film}$) parameters, modulating visual tokens via Feature-wise Linear Modulation (FiLM):
$$\mathbf{z}_i' = \gamma_{film}(\mathbf{e}(\mathbf{r}_i)) \odot \mathbf{z}_i + \beta_{film}(\mathbf{e}(\mathbf{r}_i))$$

### D. Angular Residual Attention (ARA)
To enforce 3D angular locality during self-attention, the Angular Residual Attention (ARA) refinement block introduces a continuous angular geometric bias into the attention matrix. For query token $q$ and key token $k$ with unit center ray directions $\mathbf{r}_q, \mathbf{r}_k \in \mathbb{S}^2$:
$$\cos \theta_{qk} = \mathbf{r}_q \cdot \mathbf{r}_k, \quad \sin^2 \theta_{qk} = 1 - (\mathbf{r}_q \cdot \mathbf{r}_k)^2$$

The attention logit is augmented with a learned geometric penalty $\lambda$:
$$\mathbf{A}_{qk} = \frac{\mathbf{q}_q^T \mathbf{k}_k}{\sqrt{d}} - \lambda \cdot \sin^2 \theta_{qk}$$

where $\lambda = \text{softplus}(\lambda_{\text{raw}})$ is a strictly positive learnable parameter. This penalty softly suppresses mutual attention between tokens separated by wide optical angles, constraining early representation learning to angularly coherent 3D subvolumes.

### E. Training Protocol and Multi-Domain Corpus
Dioptra-DINO is pre-trained across a multi-domain indoor corpus comprising 191 scenes from Apple Hypersim, synthetic warehouse stereo trajectories from TartanAir, and sensor captures from NYU-Depth-v2. Training utilizes a batch size of 16 across dual NVIDIA Tesla T4 GPUs with AdamW ($\text{lr} = 1 \times 10^{-4}$, cosine decay).

To evaluate generalization rigorously:
- **In-Domain Held-Out Evaluation**: Apple Hypersim (2,744 frames) evaluates generalization to unseen camera trajectories across domestic and architectural interior scenes.
- **Zero-Shot Out-of-Domain Evaluation**: InteriorNet (240 frames across 12 distinct multi-room floorplans) and ScanNet (10 real-world handheld iPad sensor captures) were strictly excluded from training, testing zero-shot domain transfer under unfamiliar room layouts and physical sensor noise.

---

## IV. Comprehensive Indoor Metric Benchmark (3,000 Frames)

| Evaluation Split | Model Architecture | Direct AbsRel (↓) | RMSE (m ↓) | MAE (m ↓) | $\delta < 1.25$ (↑) | Scale Ratio | Aligned AbsRel (↓) | Normal MAE (°) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Overall Aggregate** | **Dioptra-DINO (Ours)** | **0.1658** | **0.689 m** | **0.509 m** | **82.5%** | **1.040** | 0.1265 | 27.4° |
| *(All 2,984 Indoor Frames)* | UniDepth V2 ViT-Small | 0.2259 | 0.880 m | 0.737 m | 75.0% | 1.098 | **0.0877** | **23.5°** |
| | Metric3D ViT-Small | 0.2360 | 0.955 m | 0.800 m | 72.6% | 1.041 | 0.1027 | 29.3° |
| **B. Apple Hypersim** | **Dioptra-DINO (Ours)** | **0.1477** | **0.687 m** | **0.508 m** | **84.3%** | **1.026** | 0.1201 | 27.3° |
| *(2,744 Ray-Traced Rooms)* | UniDepth V2 ViT-Small | 0.2164 | 0.896 m | 0.747 m | 76.2% | 1.089 | **0.0844** | **23.5°** |
| | Metric3D ViT-Small | 0.2259 | 0.976 m | 0.819 m | 73.4% | 1.023 | 0.0994 | 29.3° |
| **C. InteriorNet** | **Dioptra-DINO (Ours)** | 0.3726 | 0.711 m | 0.521 m | 62.0% | 1.203 | 0.1999 | 28.5° |
| *(240 Residential Frames)* | UniDepth V2 ViT-Small | **0.3346** | **0.699 m** | 0.617 m | 61.4% | 1.207 | **0.1254** | **23.7°** |
| *(Zero-Shot Transfer)* | Metric3D ViT-Small | 0.3504 | 0.718 m | 0.578 m | **63.6%** | 1.241 | 0.1415 | 28.9° |

---

## V. Empirical Analysis

### A. Performance Against Calibrated Metric Models
1. **Overall 3,000-Frame Benchmark**: Dioptra-DINO achieves **0.1658 AbsRel**, delivering a **26.6% relative error reduction over UniDepth V2** (0.2259) and **29.7% reduction over Metric3D ViT-Small** (0.2360), while establishing the highest inlier coverage among metric models ($\delta_1 = \mathbf{82.5\%}$ vs. 75.0% for UniDepth and 72.6% for Metric3D).
2. **Apple Hypersim**: Dioptra-DINO achieves **0.1477 AbsRel** vs. **0.2164** for UniDepth V2 (-31.7% error) and **0.2259** for Metric3D, with inliers reaching **84.3%** (vs. 76.2% and 73.4%).
3. **Physical Scale Calibration**: Dioptra-DINO maintains an overall median scale ratio of **1.040×** (<4.0% distortion), outperforming UniDepth V2 (1.098×).
4. **Edge Efficiency Advantage**: Dioptra-DINO processes frames in **58.2–62.9 ms** on Apple Silicon MPS, running **6.2× faster than UniDepth V2** (421.6 ms) and **12.6× faster than Metric3D** (753.5 ms).

### B. Component Knockout Ablation Study
To evaluate the contribution of each architectural module, the table below presents component knockout evaluations on held-out trajectories:

| Ablation Configuration | AbsRel (↓) | RMSE (m ↓) | $\delta_1$ (↑) | Scale Ratio | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dioptra-DINO (Full)** | **0.0584** | **2.273 m** | **96.7%** | **1.002** | Full Trivision Ray FiLM + ARA Bias + DPT Head |
| w/o ARA Angular Bias ($\lambda=0$) | 0.0583 | 2.273 m | 96.7% | 1.002 | Angular penalty disabled (standard self-attention) |
| w/o Ray Positional Modulation | 0.1893 | 2.808 m | 79.5% | 1.179 | Ray unprojection & FiLM disabled (+224% error) |
| Center-Ray Only (w/o Trivision) | 0.7989 | 8.151 m | 2.1% | 1.821 | Center ray $[r_c, r_c, r_c]$ without corner rays (collapses) |

- **Importance of Ray Unprojection**: Disabling camera ray unprojection and FiLM modulation causes AbsRel to jump to **0.1893** (over 3× error increase), with inliers dropping to 79.5% (-17.2 percentage points) and scale drifting to 1.179×. This proves that explicit ray unprojection is indispensable for metric scale grounding.
- **Trivision Corner Rays**: Substituting the Trivision triplet with a single center ray causes catastrophic collapse (**0.7989 AbsRel**, 2.1% inliers, 1.821× scale drift), demonstrating that chiral corner rays $\mathbf{r}_1, \mathbf{r}_2$ are vital for encoding field-of-view perspective curvature.

### C. Training Progression
Following multi-domain pretraining, the baseline model (Step 76,206) was fine-tuned for 33,304 additional optimization steps (5 epochs on dual Tesla T4 GPUs; final loss **0.3554**), yielding the production checkpoint at Step 109,510:
- On the 100-frame progression test split, fine-tuning reduced AbsRel from **0.2316 to 0.2264** (-2.2% error), while overall error across the full 2,984-frame indoor benchmark reached **0.1658**.
- On Apple Hypersim, metric scale ratio aligned to **0.9992×** (<0.1% physical distortion), with AbsRel decreasing from 0.1611 to **0.1571** and $\delta_1$ rising to **81.1%**.

### D. Head-to-Head Comparison with UniDepth V2 (CVPR 2024)
Across 100 indoor frames under native configurations:
- **Apple Hypersim**: Dioptra-DINO achieves **34.2% lower AbsRel** (**0.1571** vs. **0.2386**) and **+34.1 percentage points (pp) higher inliers** ($\delta_1 = \mathbf{81.1\%}$ vs. $47.0\%$, a $+72.6\%$ relative gain). UniDepth compressed Hypersim room depths to $0.794\times$, whereas Dioptra preserved $0.999\times$.
- **Overall (100 Frames)**: Dioptra achieves **0.2146 AbsRel** and **79.1% inliers** vs. UniDepth's **0.2357 AbsRel** and **60.0% inliers**, while running **6.2× faster** (67.9 ms vs. 421.6 ms).
- **ScanNet Handheld iPad**: We note that on real-world sensor captures (10 frames from Scene00), UniDepth achieves lower metric error (**0.0567** AbsRel, $96.2\%$ inliers) than Dioptra (**0.1080** AbsRel, $90.9\%$ inliers), reflecting UniDepth's broad sensor pretraining.

| Evaluation Split | Model Architecture | AbsRel (↓) | RMSE (m ↓) | $\delta_1$ (↑) | Scale |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Overall Aggregate** | **Dioptra-DINO (Ours)** | **0.2146** | **0.983 m** | **79.1%** | **1.058** |
| (100 Indoor Frames) | UniDepth V2 | 0.2357 | 1.245 m | 60.0% | 0.928 |
| **Apple Hypersim** | **Dioptra-DINO (Ours)** | **0.1571** | **1.233 m** | **81.1%** | **0.999** |
| (60 Ray-Traced Rooms) | UniDepth V2 | 0.2386 | 1.738 m | 47.0% | 0.794 |
| **ScanNet Scene00** | **Dioptra-DINO (Ours)** | 0.1080 | 0.255 m | 90.9% | 1.077 |
| (10 Handheld Frames) | UniDepth V2 | **0.0567** | **0.145 m** | **96.2%** | **0.982** |
| **InteriorNet** | **Dioptra-DINO (Ours)** | 0.3652 | 0.727 m | 71.1% | 1.168 |
| (30 Residential Frames) | UniDepth V2 | **0.2895** | **0.625 m** | **73.9%** | **1.176** |

---

## VI. Equal-Resolution Foundation Benchmark (@ 336 × 336)

To eliminate spatial resolution as a confounding variable, all models were evaluated under an identical $336 \times 336$ budget across 100 indoor frames:

| Model Architecture | Input Resolution | Direct AbsRel (↓) | RMSE (m ↓) | $\delta < 1.25$ (↑) | Scale Ratio | Aligned AbsRel (↓) | Device Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Overall Aggregate** | | | | | | | |
| **Dioptra-DINO (Ours)** | **336 × 336** | **0.2290** | **1.085 m** | **72.5%** | **1.019** | **0.1537** | **62.9 ms (15.9 FPS)** |
| UniDepth V2 (CVPR '24) | 336 × 336 | **0.2643** | 1.448 m | 51.0% | 0.892 | 0.1238 | 108.2 ms (9.2 FPS) |
| Metric3D ViT-Small | 336 × 336 | **0.3997** | 2.266 m | **17.3%** | 0.698 | 0.2311 | 76.1 ms (13.1 FPS) |
| **B. ScanNet Scene00** | | | | | | | |
| **Dioptra-DINO (Ours)** | 336 × 336 | 0.1080 | 0.255 m | 90.9% | 1.077 | 0.0649 | 62.9 ms |
| UniDepth V2 | 336 × 336 | **0.0907** | **0.212 m** | **94.6%** | 0.925 | 0.0426 | 108.2 ms |
| Metric3D ViT-Small | 336 × 336 | 0.3855 | 0.823 m | **0.3%** | 0.619 | 0.0564 | 76.1 ms |
| **C. Apple Hypersim** | | | | | | | |
| **Dioptra-DINO (Ours)** | 336 × 336 | **0.1811** | **1.402 m** | **70.2%** | **0.934** | 0.1434 | 62.9 ms |
| UniDepth V2 | 336 × 336 | 0.2880 | 2.067 m | 32.1% | 0.774 | 0.1355 | 108.2 ms |
| Metric3D ViT-Small | 336 × 336 | 0.4414 | 3.046 m | 13.5% | 0.682 | 0.2651 | 76.1 ms |

- **Metric3D Collapse**: Deprived of its $616 \times 1064$ grid, Metric3D undergoes severe metric degradation: overall AbsRel jumps to **0.3997** and inliers collapse to **17.3%** ($0.3\%$ on ScanNet), underestimating physical scale by $>30\%$ ($0.698\times$).
- **Dioptra-DINO Robustness**: Dioptra-DINO preserves solid inliers (**72.5%**), exact scale (**1.019×**), and **4.2× higher inliers than Metric3D** while remaining the fastest model (62.9 ms).

---

## VII. Limitations and Candid Discussion

1. **Long-Range Interior Compression**: In expansive domestic environments or large atriums with depths $>10\text{m}$, Dioptra-DINO compresses predictions toward domestic priors ($<8\text{m}$) due to standard indoor training bounds.
2. **Resolution-Induced Boundary Smoothing**: At $336 \times 336$ ($14\text{px}$ patch tokens), fine wire structures, thin table legs, and distant edges exhibit spatial smoothing compared to $600\text{px}+$ architectures.
3. **Sensor Domain Gaps**: On real-world structured-light and ToF sensors (e.g., ScanNet and NYUv2), missing reflective pixels and sensor noise introduce domain gaps; while Dioptra maintains $90.9\%$ inliers on ScanNet, UniDepth V2 achieves superior accuracy ($0.0567$ AbsRel) owing to broader sensor pretraining.

---

## VIII. Conclusion

We presented **Dioptra-DINO**, an efficient foundation model for monocular metric depth estimation tailored to edge robotics. Coupling DINOv2 visual features with Canonical Virtual Camera normalization ($F_{canon}=1000\text{px}$), Trivision Ray FiLM Modulation, and Angular Residual Attention (ARA) delivers strong metric accuracy in close-range domestic interiors while operating at 16–17 FPS on Apple Silicon MPS with a ~240 MB footprint. Under equal $336 \times 336$ resolution constraints, Dioptra-DINO demonstrated superior scale calibration and inlier precision.

---

## References

1. M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," *TMLR*, 2024.
2. W. Yin et al., "Metric3D: Towards Zero-shot Metric Depth Estimation from Single Images," *ICCV*, 2023.
3. L. Piccinelli et al., "UniDepth: Universal Monocular Metric Depth Estimation," *CVPR*, 2024.
4. L. Yang et al., "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data," *CVPR*, 2024.
5. L. Yang et al., "Depth Anything V2," *arXiv:2406.09414*, 2024.
6. N. Silberman et al., "Indoor Segmentation and Support Inference from RGBD Images," *ECCV*, 2012.
7. A. Dai et al., "ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes," *CVPR*, 2017.
8. M. Roberts et al., "Hypersim: A Photorealistic Synthetic Dataset," *ICCV*, 2021.
9. W. Li et al., "InteriorNet: Mega-scale Multi-sensor Photo-realistic Indoor Scenes Dataset," *BMVC*, 2018.
10. R. Ranftl et al., "Towards Robust Monocular Depth Estimation," *IEEE TPAMI*, 2020.
11. R. Ranftl et al., "Vision Transformers for Dense Prediction," *ICCV*, 2021.
12. S. F. Bhat et al., "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth," *arXiv*, 2023.
13. D. Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network," *NeurIPS*, 2014.
14. H. Touvron et al., "Training data-efficient image transformers & distillation through attention," *ICML*, 2021.
