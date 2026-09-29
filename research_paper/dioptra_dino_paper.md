# Dioptra-DINO: Real-Time Monocular Metric Depth Estimation via Canonical Virtual Camera Normalization for Edge Robotics

**Author:** Yumnam Harryson Singh (Independent Researcher)  
**Contact:** `harryson424242@gmail.com`  
**Code & Checkpoints:** [https://github.com/SeranomTheGreat/dioptra](https://github.com/SeranomTheGreat/dioptra)  
**Date:** September 2026  

---

## Abstract

Monocular metric depth estimation on autonomous mobile robots presents an acute trade-off between physical scale fidelity and closed-loop inference latency. Contemporary metric vision foundation models (e.g., UniDepth V2, Metric3D) achieve impressive zero-shot transfer, but their high computational footprints ($400\text{--}750\text{ ms}$ per frame) limit throughput to $1.3\text{--}2.4\text{ FPS}$ on embedded hardware, inducing unacceptable control latency for aerial navigation and quadrupedal locomotion ($>15\text{ FPS}$ required). Conversely, uncalibrated relative depth models cannot recover physical scale without test-time oracle alignment.

In this work, we present **Dioptra-DINO**, an edge-efficient 27.51M-parameter metric depth architecture tailored for real-time mobile robotics. Dioptra-DINO couples a self-supervised DINOv2-Small backbone with Canonical Virtual Camera Normalization ($F_{canon} = 1000.0\text{px}$) and Trivision Ray FiLM Modulation, operating natively at $336 \times 336$ resolution. Evaluated under a standardized latency protocol (FP16, batch size 1 on Apple Silicon GPU), Dioptra-DINO processes frames in **58.2 ms (17.2 FPS)** with under **240 MB VRAM**—achieving a **6.2× speedup over UniDepth V2** (421.6 ms) and **12.6× speedup over Metric3D** (753.5 ms). 

On 2,744 in-domain held-out Apple Hypersim indoor frames, Dioptra-DINO attains **0.1477 AbsRel** and **84.3% inlier precision** ($\delta_1$). On zero-shot transfer benchmarks (InteriorNet and ScanNet), UniDepth V2 achieves superior accuracy ($0.0567$ vs. $0.1080$ AbsRel on ScanNet real sensor depth), demonstrating the limits of compact models. However, Dioptra-DINO establishes the Pareto frontier for edge robotics, providing real-time $17\text{ FPS}$ metric guidance where heavyweight models induce control lag. We release complete code, benchmark protocols, and weights.

---

## I. Introduction

Autonomous mobile robots operating in GPS-denied environments require accurate, low-latency 3D distance perception for reactive obstacle avoidance, visual odometry, and path planning. While active sensors such as LiDAR and structured-light RGB-D cameras provide direct metric measurements, their integration onto micro-aerial vehicles (MAVs) and low-cost ground rovers is often hindered by strict payload constraints, high power consumption, limited operational range, and susceptibility to sunlight interference.

Passive monocular metric depth estimation represents an appealing alternative, but recovering metric scale from a single 2D image is inherently ill-posed due to projective ambiguity: an object of dimension $H$ at distance $Z$ generates the identical retinal projection $h = f_y \frac{H}{Z}$ as an object of size $kH$ at distance $kZ$. Furthermore, different camera optics dynamically distort object projections, confounding networks that attempt to regress metric depth without explicit focal length conditioning.

Recent research addresses this problem along two divergent axes:
- **Uncalibrated Foundation Models**: Architectures such as Depth Anything train on massive unlabeled web datasets ($>62\text{M}$ images), learning exceptional semantic boundaries. However, their metric adaptations lack explicit camera intrinsics conditioning $\mathbf{K}$: unable to ingest focal parameters, they rely on implicit scene priors and cannot mathematically adjust to zoom lenses or varying sensor optics.
- **Heavyweight Metric Foundation Models**: Frameworks such as Metric3D and UniDepth condition on focal length or unproject dense pseudopinhole ray fields. While achieving strong zero-shot transfer, their complex encoders and large resolutions ($616 \times 1064$) require $400\text{--}750\text{ ms}$ per frame on embedded edge hardware, restricting throughput to $1.3\text{--}2.4\text{ FPS}$—insufficient for closed-loop robotic control.

This paper addresses the practical edge robotics question: **Can a lightweight vision transformer (<28M parameters) operating at modest resolution ($336 \times 336$) deliver reliable, calibration-conditioned metric depth at $>15\text{ FPS}$ on embedded hardware?**

We present **Dioptra-DINO**. By projecting depth regression into a Canonical Virtual Camera space ($F_{canon} = 1000.0\text{px}$) and modulating multi-scale patch tokens via continuous Trivision Ray FiLM vectors, Dioptra-DINO decouples scale from visual geometry. We candidly evaluate trade-offs across in-domain held-out camera trajectories (Apple Hypersim) and zero-shot cross-dataset environments (InteriorNet and ScanNet).

---

## II. Related Work

### A. Monocular Relative Depth Estimation
Early monocular depth methods focused on relative depth estimation via scale-invariant representations. Ranftl et al. introduced MiDaS and DPT, demonstrating that mixing heterogeneous datasets with an affine-invariant loss enables broad zero-shot generalization across scenes. Recently, Depth Anything V1 and V2 scaled relative pretraining using over 62 million unlabeled images and DINOv2 representations. While achieving exceptional structural sharpness, relative depth predictions require unknown test-time scale and shift parameters ($s, t$), preventing immediate use in closed-loop robotic navigation without external odometry.

### B. Camera-Conditioned Metric Depth Estimation
To recover true physical dimensions, works such as ZoeDepth and Metric3D investigated multi-dataset metric transfer. Metric3D proposed focal length normalization, projecting images to a virtual camera to resolve projective ambiguity. However, Metric3D relies on heavy input resolutions ($616 \times 1064$) and incurs prohibitive compute latency ($>750\text{ms}$). UniDepth advanced this direction by predicting universal metric depth and camera intrinsics directly via pseudopinhole ray embeddings. While versatile, UniDepth requires multiple dense operations that remain compute-intensive for edge robotics. Dioptra-DINO demonstrates that canonical focal normalization coupled with lightweight ray modulation yields high metric fidelity at $336 \times 336$ in under $60\text{ms}$.

| Model Architecture | Backbone Family | Parameters | Native Resolution | Intrinsics Conditioning | Edge Latency (MPS) | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dioptra-DINO (Ours)** | **DINOv2-Small (ViT-S/14)** | **27.51M** | **336 × 336** | **Canonical Virtual Cam + Trivision Ray** | **58.2 ms** | **17.2 FPS** |
| UniDepth V2 ViT-Small | DINOv2-Small (ViT-S/14) | 34.18M | Dynamic / Adaptive | Pinhole Ray Embeddings | 421.6 ms | 2.4 FPS |
| Metric3D ViT-Small | ViT-Small (DeiT-S) | 37.50M | 616 × 1064 | Pinhole Focal Normalization | 753.5 ms | 1.3 FPS |

---

## III. Methodology

### A. Problem Formulation and Scale Ambiguity
Consider a pinhole camera with intrinsic calibration matrix $\mathbf{K}$:
$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

A 3D coordinate $\mathbf{P} = [X, Y, Z]^T$ projects to image coordinates $\mathbf{p} = [u, v, 1]^T$ via:
$$Z \mathbf{p} = \mathbf{K} \mathbf{P} \implies u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y$$

When cameras with focal lengths $f_1, f_2$ observe identical objects at depths $Z_1, Z_2$, identical pixel extents occur whenever $\frac{f_1}{Z_1} = \frac{f_2}{Z_2}$. Without explicit focal conditioning, neural networks confuse optical zoom with physical distance.

### B. Canonical Virtual Camera Transformation
To decouple metric scale from camera hardware, Dioptra-DINO defines a canonical virtual pinhole camera with reference focal length $F_{canon} = 1000.0\text{px}$:
$$\mathbf{K}_{canon} = \begin{bmatrix} F_{canon} & 0 & c_x' \\ 0 & F_{canon} & c_y' \\ 0 & 0 & 1 \end{bmatrix}$$

When an input image of size $W \times H$ captured with focal length $f_x$ is resized to input resolution $S \times S$ ($S=336$), its scaled focal length is:
$$f_{scaled} = f_x \cdot \left(\frac{S}{W}\right)$$

The focal ratio between sensor and canonical camera defines the scale factor $\gamma$:
$$\gamma = \frac{f_{scaled}}{F_{canon}}$$

The model regresses canonical depth $d_{canon}(u, v) \in [0.1\text{m}, 10.0\text{m}]$. True metric depth $d_{metric}(u, v)$ is recovered via deterministic re-projection:
$$d_{metric}(u, v) = d_{canon}(u, v) \cdot \gamma = d_{canon}(u, v) \cdot \left(\frac{f_{scaled}}{F_{canon}}\right)$$

### C. Trivision Ray FiLM Modulation
To ground tokens in 3D camera geometry without dense ray-tracing overhead, Dioptra-DINO unprojects a canonical ray triplet for each patch token $i \in \{1, \dots, N\}$. For patch center $(u_c, v_c)$ and chiral patch corners $(u_{c1}, v_{c1})$ (top-left) and $(u_{c2}, v_{c2})$ (bottom-right), unit ray vectors are computed via closed-form inversion:
$$\mathbf{r}_c = \frac{1}{\|\mathbf{v}_c\|} \begin{bmatrix} \frac{u_c - c_x}{f_x} \\ \frac{v_c - c_y}{f_y} \\ 1 \end{bmatrix}, \quad \mathbf{r}_1 = \frac{\mathbf{v}_{c1}}{\|\mathbf{v}_{c1}\|}, \quad \mathbf{r}_2 = \frac{\mathbf{v}_{c2}}{\|\mathbf{v}_{c2}\|}$$

The concatenated triplet $[\mathbf{r}_c, \mathbf{r}_1, \mathbf{r}_2] \in \mathbb{R}^9$ is mapped to a multi-scale Fourier embedding across $M=6$ octave frequency bands:
$$\mathbf{e}(\mathbf{r}) = \left[ \sin(2^0 \pi \mathbf{r}), \cos(2^0 \pi \mathbf{r}), \dots, \sin(2^5 \pi \mathbf{r}), \cos(2^5 \pi \mathbf{r}) \right]^T \in \mathbb{R}^{108}$$

A lightweight two-layer MLP projects $\mathbf{e}(\mathbf{r})$ into affine scale ($\gamma_{film}$) and shift ($\beta_{film}$) parameters, modulating visual tokens via Feature-wise Linear Modulation (FiLM):
$$\mathbf{z}_i' = \gamma_{film}(\mathbf{e}(\mathbf{r}_i)) \odot \mathbf{z}_i + \beta_{film}(\mathbf{e}(\mathbf{r}_i))$$

### D. Angular Residual Attention (ARA)
To evaluate angular geometric locality during self-attention, we explored an Angular Residual Attention (ARA) block that augments attention logits with an angular distance penalty:
$$\mathbf{A}_{qk} = \frac{\mathbf{q}_q^T \mathbf{k}_k}{\sqrt{d}} - \lambda \cdot (1 - (\mathbf{r}_q \cdot \mathbf{r}_k)^2)$$
where $\lambda = \text{softplus}(\lambda_{\text{raw}})$. As detailed in our ablations (Section VI), while mathematically principled, empirical ablations demonstrate that ARA provides marginal benefit ($\Delta \text{AbsRel} < 0.0002$) once Trivision Ray FiLM Modulation is active; we document this frankly to avoid ungrounded novelty claims.

### E. Training Protocol and Multi-Domain Corpus
Dioptra-DINO is pre-trained across a multi-domain indoor corpus comprising 191 scenes from Apple Hypersim, synthetic warehouse stereo trajectories from TartanAir, and sensor captures from NYU-Depth-v2. Standard pinhole intrinsics are applied per domain (TartanAir: $f=320\text{px}$; Hypersim: $f_x=888.89, f_y=1000.0\text{px}$; NYUv2: $f=518.86\text{px}$). Depth targets are clamped to $[0.1\text{m}, 10.0\text{m}]$, reflecting typical indoor robotics obstacle envelopes. Optimization runs with AdamW ($\text{lr} = 1 \times 10^{-4}$, cosine decay) on dual Tesla T4 GPUs.

### F. Composite Objective Function
Training optimizes:
$$\mathcal{L}_{total} = \lambda_{SILog} \mathcal{L}_{SILog} + \lambda_{grad} \mathcal{L}_{grad} + \lambda_{norm} \mathcal{L}_{norm}$$
where $\mathcal{L}_{SILog}$ is the scale-invariant logarithmic loss ($\alpha=0.85$), $\mathcal{L}_{grad}$ penalizes multi-scale spatial edge gradients, and $\mathcal{L}_{norm}$ penalizes cosine errors on derived 3D surface normals.

---

## IV. Experimental Setup

### A. Evaluation Datasets and Protocols
- **Apple Hypersim** (*In-Domain Held-Out*): Evaluated across 2,744 unseen test camera trajectories from 460 photorealistic ray-traced interiors.
- **InteriorNet** (*Zero-Shot Transfer*): Synthetic multi-room residential environments with distinct layout distributions (240 frames across 12 unseen sequences).
- **ScanNet Scene00** (*Zero-Shot Real Sensor*): Handheld iPad Structure Sensor captures (10 frames) serving as a physical sensor sanity check.

### B. Latency Benchmarking Protocol
All latency and throughput measurements are evaluated under an identical protocol: batch size 1, FP16 precision, on an Apple Silicon GPU using Metal Performance Shaders (MPS), recording the median elapsed execution time over 100 runs following 20 warmup iterations.

---

## V. Comprehensive Indoor Metric Benchmark (3,000 Frames)

| Evaluation Split | Model Architecture | Direct AbsRel (↓) | RMSE (m ↓) | MAE (m ↓) | $\delta < 1.25$ (↑) | Scale Ratio | Aligned AbsRel (↓) | Normal MAE (°) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Overall Aggregate** | **Dioptra-DINO (Ours)** | **0.1658** | **0.689 m** | **0.509 m** | **82.5%** | **1.040** | 0.1265 | 27.4° |
| *(All 2,984 Indoor Frames)* | UniDepth V2 ViT-Small | 0.2259 | 0.880 m | 0.737 m | 75.0% | 1.098 | **0.0877** | **23.5°** |
| | Metric3D ViT-Small | 0.2360 | 0.955 m | 0.800 m | 72.6% | 1.041 | 0.1027 | 29.3° |
| **B. Apple Hypersim** | **Dioptra-DINO (Ours)** | **0.1477** | **0.687 m** | **0.508 m** | **84.3%** | **1.026** | 0.1201 | 27.3° |
| *(2,744 In-Domain Rooms)* | UniDepth V2 ViT-Small | 0.2164 | 0.896 m | 0.747 m | 76.2% | 1.089 | **0.0844** | **23.5°** |
| | Metric3D ViT-Small | 0.2259 | 0.976 m | 0.819 m | 73.4% | 1.023 | 0.0994 | 29.3° |
| **C. InteriorNet** | **Dioptra-DINO (Ours)** | 0.3726 | 0.711 m | 0.521 m | 62.0% | 1.203 | 0.1999 | 28.5° |
| *(240 Residential Frames)* | UniDepth V2 ViT-Small | **0.3346** | **0.699 m** | 0.617 m | 61.4% | 1.207 | **0.1254** | **23.7°** |
| *(Zero-Shot Transfer)* | Metric3D ViT-Small | 0.3504 | 0.718 m | 0.578 m | **63.6%** | 1.241 | 0.1415 | 28.9° |

### Benchmark Analysis
1. **In-Domain Hypersim Performance**: On 2,744 ray-traced Hypersim frames, Dioptra-DINO attains **0.1477 AbsRel** and **84.3% inliers** ($\delta_1$), outperforming Metric3D ViT-Small ($0.2259$ AbsRel, $73.4\%$ inliers) by **34.6% lower error** and UniDepth V2 ($0.2164$ AbsRel, $76.2\%$ inliers) by **31.7% lower error**. We note that Dioptra benefits here from in-domain pretraining on Hypersim training scenes.
2. **Zero-Shot Transfer Realities**: On InteriorNet (240 zero-shot frames), UniDepth V2 attains lower error (**0.3346 AbsRel**) than Dioptra ($0.3726$), and on ScanNet handheld sensor depth (Table III), UniDepth achieves **0.0567 AbsRel** vs. Dioptra's $0.1080$. This confirms that large models trained across broader sensor corpora generalize better to unseen sensor distributions.
3. **Robotic Edge Trade-Off**: Dioptra-DINO operates in **58.2 ms (17.2 FPS)**, running **6.2× faster than UniDepth V2** (421.6 ms / 2.4 FPS) and **12.6× faster than Metric3D** (753.5 ms / 1.3 FPS), establishing an efficient real-time operating point.

---

## VI. Equal-Resolution Foundation Benchmark (@ 336 × 336)

To evaluate how architectures handle low compute budgets, Table III constrains all models to $336 \times 336$:

| Model Architecture | Input Resolution | Direct AbsRel (↓) | RMSE (m ↓) | $\delta < 1.25$ (↑) | Scale Ratio | Aligned AbsRel (↓) | Device Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Overall Aggregate** | | | | | | | |
| **Dioptra-DINO (Ours)** | **336 × 336** | **0.2290** | **1.085 m** | **72.5%** | **1.019** | **0.1537** | **58.2 ms (17.2 FPS)** |
| UniDepth V2 (CVPR '24) | 336 × 336 | **0.2643** | 1.448 m | 51.0% | 0.892 | 0.1238 | 108.2 ms (9.2 FPS) |
| Metric3D ViT-Small | 336 × 336 | **0.3997** | 2.266 m | **17.3%** | 0.698 | 0.2311 | 76.1 ms (13.1 FPS) |
| **B. ScanNet Scene00 (Zero-Shot)** | | | | | | | |
| **Dioptra-DINO (Ours)** | 336 × 336 | 0.1080 | 0.255 m | 90.9% | 1.077 | 0.0649 | 58.2 ms |
| UniDepth V2 | 336 × 336 | **0.0907** | **0.212 m** | **94.6%** | 0.925 | 0.0426 | 108.2 ms |
| Metric3D ViT-Small | 336 × 336 | 0.3855 | 0.823 m | **0.3%** | 0.619 | 0.0564 | 76.1 ms |
| **C. Apple Hypersim** | | | | | | | |
| **Dioptra-DINO (Ours)** | 336 × 336 | **0.1811** | **1.402 m** | **70.2%** | **0.934** | 0.1434 | 58.2 ms |
| UniDepth V2 | 336 × 336 | 0.2880 | 2.067 m | 32.1% | 0.774 | 0.1355 | 108.2 ms |
| Metric3D ViT-Small | 336 × 336 | 0.4414 | 3.046 m | 13.5% | 0.682 | 0.2651 | 76.1 ms |

- **Metric3D Sensitivity**: Metric3D undergoes severe degradation when deprived of its $616 \times 1064$ grid: overall AbsRel jumps to **0.3997** and inliers collapse to **17.3%** ($0.3\%$ on ScanNet). This confirms that Metric3D's normalization was tailored specifically for high-resolution rectangular inputs.
- **Dioptra-DINO Resilience**: Operating natively at $336 \times 336$, Dioptra-DINO preserves solid inliers (**72.5%**), exact metric scale (**1.019×**), and achieves **4.2× higher inliers than Metric3D**.

---

## VII. Architectural Ablation Analysis

Table IV presents component knockout evaluations on held-out TartanAir trajectory sequences (where ground truth depth is dense and clean):

| Ablation Configuration | AbsRel (↓) | RMSE (m ↓) | $\delta_1$ (↑) | Scale Ratio | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dioptra-DINO (Full)** | **0.0584** | **2.273 m** | **96.7%** | **1.002** | Full Trivision Ray FiLM + ARA Bias + DPT Head |
| w/o ARA Angular Bias ($\lambda=0$) | 0.0583 | 2.273 m | 96.7% | 1.002 | Angular penalty disabled (standard self-attention) |
| w/o Ray Positional Modulation | 0.1893 | 2.808 m | 79.5% | 1.179 | Ray unprojection & FiLM disabled (+224% error) |
| Center-Ray Only (w/o Trivision) | 0.7989 | 8.151 m | 2.1% | 1.821 | Center ray $[r_c, r_c, r_c]$ without corner rays (collapses) |

- **Ray Modulation Impact**: Disabling Trivision Ray FiLM Modulation causes AbsRel to jump from **0.0584** to **0.1893** (+224% error increase), with inliers dropping to $79.5\%$ ($-17.2$ percentage points) and scale drifting to $1.179\times$. This confirms that camera ray unprojection is the primary mechanism grounding metric scale.
- **Center-Ray Only Failure**: Using a single center ray causes severe degradation ($0.7989$ AbsRel, $2.1\%$ inliers), showing that chiral corner rays $\mathbf{r}_1, \mathbf{r}_2$ are vital for encoding field-of-view perspective boundaries.
- **ARA Contribution**: Disabling the ARA angular bias ($\lambda=0$) yields $0.0583$ AbsRel vs. $0.0584$ for the full model ($\Delta < 0.0002$). We candidly conclude that ARA provides negligible empirical variance once Trivision Ray FiLM Modulation is present.

---

## VIII. Limitations and Candid Discussion

1. **10-Metre Range Limit**: The model clamps depth to $10.0\text{m}$. In expansive atriums or long hallways $>10\text{m}$, predictions compress toward indoor priors, which must be accounted for in high-speed navigation.
2. **Patch Token Boundary Smoothing**: At $336 \times 336$ ($14\text{px}$ tokens), thin chair legs and distant wires exhibit spatial smoothing compared to $1000\text{px}+$ models.
3. **Sensor Domain Gap**: On raw sensor depth (ScanNet, NYUv2), UniDepth V2 attains lower absolute error ($0.0567$ vs. $0.1080$ AbsRel), reflecting its training on extensive real sensor datasets.

---

## IX. Conclusion

We presented **Dioptra-DINO**, an efficient vision transformer architecture for monocular metric depth estimation tailored to edge robotics. By coupling Canonical Virtual Camera Normalization ($F_{canon}=1000\text{px}$) with Trivision Ray FiLM Modulation, Dioptra-DINO achieves reliable indoor metric depth while operating at **58.2 ms (17.2 FPS)** on Apple Silicon GPU under $<240\text{ MB}$ VRAM. While heavyweight foundation models attain superior zero-shot transfer on real sensor captures, Dioptra-DINO delivers the real-time throughput required for closed-loop edge robotics.

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
10. R. Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer," *IEEE TPAMI*, 2020.
11. R. Ranftl et al., "Vision Transformers for Dense Prediction," *ICCV*, 2021.
12. S. F. Bhat et al., "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth," *arXiv*, 2023.
13. D. Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network," *NeurIPS*, 2014.
14. H. Touvron et al., "Training data-efficient image transformers & distillation through attention," *ICML*, 2021.
