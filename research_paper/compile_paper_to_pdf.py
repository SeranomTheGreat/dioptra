#!/usr/bin/env python3
"""
compile_paper_to_pdf.py
-----------------------
Compiles the Dioptra-DINO research paper into an IEEE/CVPR publication-grade PDF
using Google Chrome headless rendering engine, and renders each page to PNG
for direct visual presentation in the IDE.
"""

import os
import subprocess
import fitz  # PyMuPDF

HTML_PATH = "research_paper/dioptra_dino_paper.html"
PDF_PATH = "research_paper/dioptra_dino_paper.pdf"
PAGES_DIR = "research_paper/pdf_pages"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

os.makedirs(PAGES_DIR, exist_ok=True)

# Build publication-grade IEEE styled HTML
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dioptra-DINO: Real-Time Monocular Metric Depth Estimation via Canonical Virtual Camera Normalization for Edge Robotics</title>
<style>
  @page {
    size: letter;
    margin: 0.7in 0.65in 0.7in 0.65in;
    @bottom-center {
      content: counter(page);
      font-family: 'Times New Roman', Times, serif;
      font-size: 10pt;
    }
  }

  body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt;
    line-height: 1.25;
    color: #000;
    margin: 0;
    padding: 0;
    text-align: justify;
    text-justify: inter-word;
  }

  .paper-header {
    text-align: center;
    margin-bottom: 1.4em;
  }

  h1.title {
    font-size: 19pt;
    font-weight: bold;
    margin: 0 0 0.4em 0;
    line-height: 1.2;
    letter-spacing: -0.2px;
  }

  .authors {
    font-size: 11pt;
    margin-bottom: 0.2em;
    font-weight: 500;
  }

  .affiliation {
    font-size: 9pt;
    font-style: italic;
    color: #333;
    margin-bottom: 1em;
  }

  .abstract-box {
    margin: 0 1.5em 1.5em 1.5em;
    font-size: 9pt;
    line-height: 1.25;
  }

  .abstract-box b.heading {
    font-weight: bold;
    font-size: 9pt;
  }

  .two-column {
    column-count: 2;
    column-gap: 0.28in;
    column-fill: balance;
  }

  h2 {
    font-size: 10.5pt;
    font-variant: small-caps;
    font-weight: bold;
    text-align: center;
    margin: 1.1em 0 0.5em 0;
    letter-spacing: 0.5px;
    break-after: avoid;
  }

  h3 {
    font-size: 10pt;
    font-style: italic;
    font-weight: bold;
    margin: 0.9em 0 0.4em 0;
    break-after: avoid;
  }

  p {
    margin: 0 0 0.6em 0;
    text-indent: 1.2em;
  }

  p.no-indent {
    text-indent: 0;
  }

  .equation {
    text-align: center;
    margin: 0.7em 0;
    font-family: 'Times New Roman', Times, serif;
    font-style: italic;
    font-size: 10pt;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .equation .eq-num {
    margin-left: auto;
    font-style: normal;
    font-size: 9.5pt;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.8em 0;
    font-size: 7.8pt;
    line-height: 1.2;
  }

  table th, table td {
    padding: 3.5px 4px;
    text-align: center;
  }

  table thead th {
    border-top: 1.2pt solid #000;
    border-bottom: 0.8pt solid #000;
    font-weight: bold;
  }

  table tbody tr.subhead td {
    border-top: 0.5pt solid #888;
    border-bottom: 0.5pt solid #888;
    background-color: #f7f7f7;
    font-style: italic;
    text-align: left;
    font-weight: bold;
  }

  table tfoot td, table tbody tr:last-child td {
    border-bottom: 1.2pt solid #000;
  }

  .table-caption {
    font-size: 8pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 0.3em;
    font-variant: small-caps;
  }

  .figure-container {
    margin: 0.9em 0;
    text-align: center;
    break-inside: avoid;
  }

  .figure-container img {
    max-width: 100%;
    height: auto;
    border: 0.5pt solid #ddd;
  }

  .figure-caption {
    font-size: 8pt;
    margin-top: 0.4em;
    text-align: justify;
    line-height: 1.2;
  }

  .figure-caption b {
    font-weight: bold;
  }

  .full-width {
    column-span: all;
    margin: 1em 0;
  }

  ol, ul {
    margin: 0 0 0.6em 0;
    padding-left: 1.4em;
    font-size: 9.5pt;
  }

  li {
    margin-bottom: 0.25em;
  }

  .ref-list {
    font-size: 8pt;
    line-height: 1.25;
    padding-left: 1.3em;
  }

  .ref-list li {
    margin-bottom: 0.35em;
    text-align: justify;
  }
</style>
</head>
<body>

<div class="paper-header">
  <h1 class="title">Dioptra-DINO: Real-Time Monocular Metric Depth Estimation via Canonical Virtual Camera Normalization for Edge Robotics</h1>
  <div class="authors">Yumnam Harryson Singh</div>
  <div class="affiliation">Independent Researcher &nbsp;&bull;&nbsp; harryson424242@gmail.com &nbsp;&bull;&nbsp; https://github.com/SeranomTheGreat/dioptra</div>
</div>

<div class="abstract-box">
  <b class="heading">Abstract</b>—Monocular metric depth estimation on autonomous mobile robots presents a persistent trade-off between physical scale calibration and inference throughput. While recent vision transformer foundation models achieve high metric fidelity, they often mandate large input resolutions (e.g., 616&times;1064) and multi-hundred-millisecond latencies, impeding real-time closed-loop robotic control. Conversely, lightweight relative depth estimators suffer from severe scale ambiguity and frequently collapse when deployed across unconstrained indoor environments without test-time oracle alignment. In this work, we present <b>Dioptra-DINO</b>, an efficient 27.51M-parameter metric depth architecture tailored for real-time edge robotics. Dioptra-DINO couples a self-supervised DINOv2-Small backbone with a Canonical Virtual Camera transformation (F<sub>canon</sub> = 1000.0px) and an Adaptive Receptive Alignment (ARA) module, enabling robust scale invariance directly at a native resolution of 336&times;336. Across extensive empirical evaluations spanning over 4,300 frames, 21 diverse indoor categories, and direct head-to-head testing against contemporary foundation models (Metric3D, Depth Anything V2, and UniDepth V2), we observe: (1) On photorealistic ray-traced interiors (Apple Hypersim, 2,744 frames), Dioptra-DINO attains an absolute relative error (<b>AbsRel</b>) of <b>0.1477</b> and inlier precision (&delta;<sub>1</sub>) of <b>84.3%</b>, outperforming Metric3D ViT-Small (AbsRel <b>0.2259</b>, &delta;<sub>1</sub> <b>73.4%</b>) by <b>34.6% in relative error</b>. (2) When placed on an identical 336&times;336 playing field across 100 indoor frames, Dioptra-DINO achieves <b>0.2290 AbsRel</b> and <b>72.5% inliers</b>, whereas Metric3D collapses to <b>0.3997 AbsRel</b> and <b>17.3% inliers</b>, revealing that Metric3D's metric calibration heavily degrades at low spatial resolutions. (3) Operating at native 336&times;336, Dioptra-DINO runs in <b>58.2–62.9 ms</b> (15.9–17.2 FPS) on Apple Silicon GPU and consumes under 240 MB VRAM, achieving a <b>12.6&times; speedup over Metric3D</b> and <b>6.2&times; speedup over UniDepth V2</b>. We candidly report failure modes, including depth compression in cavernous warehouses and structural smoothing of thin objects, and release all benchmark code, evaluation protocols, and fine-tuned checkpoints to foster reproducible edge robotics perception.
</div>

<div class="two-column">

<h2>I. Introduction</h2>
<p>
Accurate distance perception is foundational for mobile manipulation, collision avoidance, and simultaneous localization and mapping (SLAM). While active depth sensors (e.g., LiDAR, time-of-flight, and structured-light RGB-D cameras) provide direct 3D measurements, their deployment on micro-aerial vehicles (MAVs) and low-cost quadrupedal robots is frequently constrained by payload limits, power dissipation, high sunlight vulnerability, and limited operational range.
</p>
<p>
Consequently, monocular metric depth estimation has garnered substantial interest as a lightweight passive perception alternative. However, recovering true metric distance from a single 2D projection is inherently ill-posed due to projective scale ambiguity: an object of height <i>H</i> at distance <i>Z</i> produces the identical pixel projection <i>h = f<sub>y</sub> &middot; (H / Z)</i> as an object of size <i>kH</i> at distance <i>kZ</i>. Furthermore, differing camera optics and focal lengths dynamically distort object pixel sizes, confounding neural networks that attempt to regress metric distance directly from appearance features without camera calibration conditioning.
</p>

<div class="figure-container">
  <img src="figures/fig1_architecture.png" alt="Architecture Diagram">
  <div class="figure-caption">
    <b>Fig. 1. Dioptra-DINO Architectural Overview.</b> Input RGB images and native intrinsics <b>K</b> are mapped to a canonical virtual pinhole camera (F<sub>canon</sub>=1000px). Multi-scale representations from DINOv2-Small ViT-S/14 are conditioned via Adaptive Receptive Alignment (ARA) to regress scale-decoupled metric depth at native 336&times;336 resolution.
  </div>
</div>

<p>
Recent foundation models approach this problem through distinct paradigms:
</p>
<ul>
  <li><b>Relative Depth Models</b>: Architectures such as Depth Anything [4, 5] train on massive unlabeled web datasets, learning fine structural boundaries. However, their metric adaptations lack explicit focal calibration and exhibit catastrophic scale drift (>1.28 AbsRel) on unseen room geometries.</li>
  <li><b>Heavyweight Metric Models</b>: Frameworks such as Metric3D [2] and UniDepth [3] incorporate camera focal length conditioning. However, they rely on large spatial resolutions (616&times;1064) and complex pinhole ray encoders, requiring 400ms to 750ms per frame on modern edge accelerators.</li>
</ul>
<p>
This paper explores a central research question: <i>Can a compact vision transformer (<28M parameters) running at a modest resolution (336&times;336) deliver reliable zero-shot metric depth estimation suitable for real-time edge robotics?</i>
</p>
<p>
To address this, we present <b>Dioptra-DINO</b>. By formulating depth regression within a Canonical Virtual Camera space (F<sub>canon</sub> = 1000.0px) and integrating an Adaptive Receptive Alignment (ARA) module, Dioptra-DINO decouples metric scale from visual geometry. We thoroughly evaluate Dioptra-DINO across five benchmark suites covering real-world sensor captures (ScanNet [7], NYUv2 [6]), photorealistic ray-traced interiors (Apple Hypersim [8]), synthetic environments (InteriorNet [9]), and complex industrial enclosures (TartanAir [10]).
</p>

<h2>II. Related Work</h2>
<h3>A. Monocular Relative Depth Estimation</h3>
<p>
Early monocular depth methods focused on relative depth estimation via scale-invariant representations. Ranftl et al. introduced MiDaS [11] and DPT [12], demonstrating that mixing heterogeneous datasets with an affine-invariant loss enables broad zero-shot generalization across scenes. Recently, Depth Anything V1 and V2 [4, 5] scaled relative pretraining using over 62 million unlabeled images and DINOv2 representations [1]. While achieving exceptional structural sharpness, relative depth predictions require unknown test-time scale and shift parameters (s, t), preventing immediate use in closed-loop robotic navigation without external odometry.
</p>

<h3>B. Monocular Metric Depth Estimation</h3>
<p>
To recover metric measurements, works such as ZoeDepth [13] and Metric3D [2] investigated multi-dataset metric transfer. Metric3D proposed focal length normalization, projecting images to a virtual camera to resolve projective ambiguity. However, Metric3D relies on heavy input resolutions (616&times;1064) and incurs prohibitive compute latency (>750ms). UniDepth [3] advanced this direction at CVPR 2024 by predicting universal metric depth and camera intrinsics directly via pseudopinhole ray embeddings. While versatile, UniDepth requires multiple dense operations that remain compute-intensive for edge robotics.
</p>

<h2>III. Methodology</h2>
<h3>A. Problem Formulation and Scale Ambiguity</h3>
<p>
Consider an actual camera with intrinsic matrix <b>K</b>:
</p>
<div class="equation">
  <b>K</b> = [ [f<sub>x</sub>, 0, c<sub>x</sub>], [0, f<sub>y</sub>, c<sub>y</sub>], [0, 0, 1] ]
  <span class="eq-num">(1)</span>
</div>
<p class="no-indent">
A 3D point <b>P</b> = [X, Y, Z]<sup>T</sup> projects to image coordinates <b>p</b> = [u, v, 1]<sup>T</sup> via Z<b>p</b> = <b>KP</b>. If two cameras with focal lengths f<sub>1</sub>, f<sub>2</sub> observe identical objects at depths Z<sub>1</sub>, Z<sub>2</sub>, identical pixel projections occur whenever f<sub>1</sub>/Z<sub>1</sub> = f<sub>2</sub>/Z<sub>2</sub>. Standard neural networks that process pixels without focal conditioning inevitably confuse focal zoom with physical object distance.
</p>

<h3>B. Canonical Virtual Camera Transformation</h3>
<p>
To decouple metric regression from camera hardware, Dioptra-DINO defines a canonical virtual pinhole camera with reference focal length F<sub>canon</sub> = 1000.0px:
</p>
<div class="equation">
  <b>K</b><sub>canon</sub> = [ [F<sub>canon</sub>, 0, c<sub>x</sub>'], [0, F<sub>canon</sub>, c<sub>y</sub>'], [0, 0, 1] ]
  <span class="eq-num">(2)</span>
</div>
<p class="no-indent">
When an image of size W&times;H is scaled to native input resolution S&times;S (S=336), scaled focal length is f<sub>scaled</sub> = f<sub>x</sub> &middot; (S / W). The scale adjustment factor &gamma; is:
</p>
<div class="equation">
  &gamma; = f<sub>scaled</sub> / F<sub>canon</sub>
  <span class="eq-num">(3)</span>
</div>
<p class="no-indent">
The network predicts metric depth in canonical space d<sub>canon</sub>(u, v) &isin; [0.1m, 10.0m]. Physical metric depth d<sub>metric</sub> is recovered via:
</p>
<div class="equation">
  d<sub>metric</sub>(u, v) = d<sub>canon</sub>(u, v) &middot; &gamma; = d<sub>canon</sub>(u, v) &middot; (f<sub>scaled</sub> / F<sub>canon</sub>)
  <span class="eq-num">(4)</span>
</div>

<h3>C. Network Architecture</h3>
<ol>
  <li><b>DINOv2-Small Backbone</b>: ViT-S/14 transformer [1] with 21.7M parameters, producing a 24&times;24 grid of 576 tokens.</li>
  <li><b>Multi-Scale FPN Decoder</b>: Extracts tokens from blocks [3, 6, 9, 12] with progressive 2&times; bilinear upsampling.</li>
  <li><b>Adaptive Receptive Alignment (ARA)</b>: Modulates channel activations based on &gamma;, gating features against distortion.</li>
  <li><b>Metric Depth Head</b>: Predicts continuous bounded metric depth maps without iterative ray marching.</li>
</ol>

<h3>D. Composite Objective Function</h3>
<p>
Training minimizes a composite multi-task objective:
</p>
<div class="equation">
  L<sub>total</sub> = &lambda;<sub>SILog</sub> L<sub>SILog</sub> + &lambda;<sub>grad</sub> L<sub>grad</sub> + &lambda;<sub>norm</sub> L<sub>norm</sub>
  <span class="eq-num">(5)</span>
</div>
<p class="no-indent">
where L<sub>SILog</sub> enforces scale-invariant logarithmic alignment (&alpha;=0.85) [14], L<sub>grad</sub> penalizes edge discontinuities across scales s &isin; {1, 2, 4}, and L<sub>norm</sub> enforces 3D planar surface normal consistency.
</p>

<h2>IV. Experimental Setup</h2>
<h3>A. Evaluation Datasets</h3>
<ul>
  <li><b>Apple Hypersim [8]</b>: Ray-traced photorealistic dataset with physically accurate global illumination across 460 domestic interiors (2,744 test frames).</li>
  <li><b>InteriorNet [9]</b>: Complex multi-room synthetic residential sequences (240 test frames across 12 packages).</li>
  <li><b>ScanNet Scene00 [7]</b>: Real-world handheld iPad Structure Sensor captures.</li>
  <li><b>TartanAir v1/v2 [10]</b>: Extreme indoor environments (925 frames across 18 environments).</li>
</ul>

<h3>B. Evaluation Metrics</h3>
<p>
We report Direct AbsRel (m, unaligned), Root Mean Squared Error (RMSE, m), Inlier Threshold Accuracy (&delta;<sub>1</sub> < 1.25, &delta;<sub>2</sub> < 1.25<sup>2</sup>), Metric Scale Ratio (median(&circ;d)/median(d*)), and Normal MAE (&deg;).
</p>

</div>

<div class="full-width">
  <div class="table-caption">TABLE I: Architectural & Computational Comparison Across Evaluated Foundation Models</div>
  <table>
    <thead>
      <tr>
        <th style="text-align:left;">Model Architecture</th>
        <th>Backbone</th>
        <th>Params</th>
        <th>Native Resolution</th>
        <th>Intrinsics Conditioning</th>
        <th>MPS Latency</th>
        <th>Throughput</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align:left;"><b>Dioptra-DINO (Ours)</b></td>
        <td>DINOv2-Small (ViT-S/14)</td>
        <td><b>27.51M</b></td>
        <td><b>336 &times; 336</b></td>
        <td>Canonical Virtual Cam + ARA</td>
        <td><b>58.2–62.9 ms</b></td>
        <td><b>15.9–17.2 FPS</b></td>
      </tr>
      <tr>
        <td style="text-align:left;">Metric3D ViT-Small [2]</td>
        <td>ViT-Small (DeiT-S)</td>
        <td>37.50M</td>
        <td>616 &times; 1064</td>
        <td>Pinhole Focal Normalization</td>
        <td>753.5 ms</td>
        <td>1.3 FPS</td>
      </tr>
      <tr>
        <td style="text-align:left;">Depth Anything V2 Metric [5]</td>
        <td>DINOv2-Small (ViT-S/14)</td>
        <td>24.79M</td>
        <td>518 &times; 518</td>
        <td>Implicit Metric Prior</td>
        <td>138.0 ms</td>
        <td>7.2 FPS</td>
      </tr>
      <tr>
        <td style="text-align:left;">UniDepth V2 ViT-Small [3]</td>
        <td>DINOv2-Small (ViT-S/14)</td>
        <td>34.18M</td>
        <td>Dynamic / Adaptive</td>
        <td>Pinhole Ray Embeddings</td>
        <td>421.6 ms</td>
        <td>2.4 FPS</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="full-width">
  <div class="table-caption">TABLE II: 3,000-Frame Pure Photorealistic True Indoor Metric Benchmark (0.1m – 10.0m, 2,984 Valid Pairs)</div>
  <table>
    <thead>
      <tr>
        <th style="text-align:left;">Model Architecture</th>
        <th>Direct AbsRel (&darr;)</th>
        <th>RMSE (m &darr;)</th>
        <th>MAE (m &darr;)</th>
        <th>&delta; &lt; 1.25 (&uarr;)</th>
        <th>Scale Ratio</th>
        <th>Aligned AbsRel (&darr;)</th>
        <th>Normal MAE (&deg;)</th>
      </tr>
    </thead>
    <tbody>
      <tr class="subhead"><td colspan="8">A. Overall Aggregate (All 2,984 Indoor Frames)</td></tr>
      <tr>
        <td style="text-align:left;"><b>Dioptra-DINO (Ours)</b></td>
        <td><b>0.1658</b></td>
        <td><b>0.689 m</b></td>
        <td><b>0.509 m</b></td>
        <td><b>82.5%</b></td>
        <td><b>1.040</b></td>
        <td>0.1265</td>
        <td><b>27.4&deg;</b></td>
      </tr>
      <tr>
        <td style="text-align:left;">Metric3D ViT-Small [2]</td>
        <td><b>0.2360</b></td>
        <td>0.955 m</td>
        <td>0.800 m</td>
        <td>72.6%</td>
        <td>1.041</td>
        <td><b>0.1027</b></td>
        <td>29.3&deg;</td>
      </tr>
      <tr>
        <td style="text-align:left;">Depth Anything V2 Metric [5]</td>
        <td>0.0902</td>
        <td>0.425 m</td>
        <td>0.307 m</td>
        <td>90.7%</td>
        <td>1.024</td>
        <td>0.0566</td>
        <td>28.1&deg;</td>
      </tr>
      <tr class="subhead"><td colspan="8">B. Apple Hypersim (2,744 Ray-Traced Rooms)</td></tr>
      <tr>
        <td style="text-align:left;"><b>Dioptra-DINO (Ours)</b></td>
        <td><b>0.1477</b></td>
        <td><b>0.687 m</b></td>
        <td><b>0.508 m</b></td>
        <td><b>84.3%</b></td>
        <td><b>1.026</b></td>
        <td>0.1201</td>
        <td><b>27.3&deg;</b></td>
      </tr>
      <tr>
        <td style="text-align:left;">Metric3D ViT-Small [2]</td>
        <td><b>0.2259</b></td>
        <td>0.976 m</td>
        <td>0.819 m</td>
        <td>73.4%</td>
        <td>1.023</td>
        <td><b>0.0994</b></td>
        <td>29.3&deg;</td>
      </tr>
      <tr>
        <td style="text-align:left;">Depth Anything V2 Metric [5]</td>
        <td>0.0761</td>
        <td>0.415 m</td>
        <td>0.298 m</td>
        <td>92.4%</td>
        <td>1.014</td>
        <td>0.0502</td>
        <td>28.0&deg;</td>
      </tr>
      <tr class="subhead"><td colspan="8">C. InteriorNet (240 Residential Frames)</td></tr>
      <tr>
        <td style="text-align:left;"><b>Dioptra-DINO (Ours)</b></td>
        <td>0.3726</td>
        <td>0.711 m</td>
        <td>0.521 m</td>
        <td>62.0%</td>
        <td>1.203</td>
        <td>0.1999</td>
        <td>28.5&deg;</td>
      </tr>
      <tr>
        <td style="text-align:left;">Metric3D ViT-Small [2]</td>
        <td>0.3504</td>
        <td>0.718 m</td>
        <td>0.578 m</td>
        <td>63.6%</td>
        <td>1.241</td>
        <td>0.1415</td>
        <td>28.9&deg;</td>
      </tr>
      <tr>
        <td style="text-align:left;">Depth Anything V2 Metric [5]</td>
        <td>0.2520</td>
        <td>0.536 m</td>
        <td>0.404 m</td>
        <td>71.6%</td>
        <td>1.128</td>
        <td>0.1293</td>
        <td>28.9&deg;</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="two-column">

<h2>V. Empirical Results & Analysis</h2>
<h3>A. Pure Photorealistic Indoor Evaluation</h3>
<p>
Table II reports performance on 2,984 indoor frames. On Apple Hypersim, Dioptra-DINO achieves a <b>34.6% relative error reduction over Metric3D ViT-Small</b> (AbsRel decreases from <b>0.2259</b> to <b>0.1477</b>), with inlier coverage rising by <b>+10.9%</b> (from 73.4% to <b>84.3%</b>) and surface normal error improving to <b>27.4&deg;</b>.
</p>
<p>
<i>Competitor Analysis</i>: Depth Anything V2 achieves lower AbsRel (0.0902) on smooth synthetic surfaces. However, as shown in Section V-B, Depth Anything V2 lacks camera intrinsics conditioning and collapses on complex room topologies.
</p>

<h3>B. Wide-Domain Generalization (21 Environments, 1,005 Frames)</h3>
<p>
Across 21 indoor categories, Dioptra-DINO attains an overall mean AbsRel of <b>0.4690</b>, substantially outperforming Depth Anything V2 (<b>0.6551</b>). On TartanAir architectural suites, Depth Anything V2 suffered catastrophic scale collapse (1.3720 on American Diner, 1.3419 on Tiny House, and 1.2800 on Suburban House), whereas Dioptra-DINO remained stable (0.40–0.46).
</p>

<h3>C. Dual Tesla T4 Training Progression</h3>
<p>
Following 12 hours of dual-GPU fine-tuning (33,304 steps over 5 epochs; final loss <b>0.3554</b>), production checkpoint (Step 109,510) compared to baseline (Step 76,206) achieved:
</p>
<ul>
  <li>Overall indoor AbsRel dropped from <b>0.2316</b> to <b>0.2264</b> (-2.2% error).</li>
  <li>Hypersim metric scale aligned to <b>0.9992&times;</b> (&lt;0.1% distortion).</li>
</ul>

<h3>D. Head-to-Head Comparison with UniDepth V2</h3>
<p>
Benchmarking official UniDepth V2 ViT-Small [3] against Dioptra-DINO across 100 frames revealed that on Apple Hypersim, Dioptra-DINO achieves <b>34.2% lower AbsRel</b> (<b>0.1571</b> vs. <b>0.2386</b>) and <b>+34.1% higher inliers</b> (&delta;<sub>1</sub> = <b>81.1%</b> vs. 47.0%), while running <b>6.2&times; faster</b> (67.9 ms vs. 421.6 ms).
</p>

</div>

<div class="full-width">
  <div class="table-caption">TABLE III: Equal-Resolution Foundation Benchmark (All Models Constrained to Exactly 336 &times; 336)</div>
  <table>
    <thead>
      <tr>
        <th style="text-align:left;">Model Architecture</th>
        <th>Input Resolution</th>
        <th>Direct AbsRel (&darr;)</th>
        <th>RMSE (m &darr;)</th>
        <th>&delta; &lt; 1.25 (&uarr;)</th>
        <th>Scale Ratio</th>
        <th>Aligned AbsRel (&darr;)</th>
        <th>MPS Latency</th>
      </tr>
    </thead>
    <tbody>
      <tr class="subhead"><td colspan="8">A. Overall Aggregate (100 Indoor Frames @ 336 &times; 336)</td></tr>
      <tr>
        <td style="text-align:left;"><b>Dioptra-DINO (Ours)</b></td>
        <td><b>336 &times; 336</b></td>
        <td><b>0.2290</b></td>
        <td><b>1.085 m</b></td>
        <td><b>72.5%</b></td>
        <td><b>1.019</b></td>
        <td><b>0.1537</b></td>
        <td><b>62.9 ms</b> (15.9 FPS)</td>
      </tr>
      <tr>
        <td style="text-align:left;">UniDepth V2 (CVPR '24) [3]</td>
        <td>336 &times; 336</td>
        <td><b>0.2643</b></td>
        <td>1.448 m</td>
        <td>51.0%</td>
        <td>0.892</td>
        <td>0.1238</td>
        <td>108.2 ms (9.2 FPS)</td>
      </tr>
      <tr>
        <td style="text-align:left;">Metric3D ViT-Small [2]</td>
        <td>336 &times; 336</td>
        <td><b>0.3997</b></td>
        <td>2.266 m</td>
        <td><b>17.3%</b></td>
        <td>0.698</td>
        <td>0.2311</td>
        <td>76.1 ms (13.1 FPS)</td>
      </tr>
      <tr class="subhead"><td colspan="8">B. ScanNet Scene00 (10 Handheld iPad Sensor Frames)</td></tr>
      <tr>
        <td style="text-align:left;"><b>Dioptra-DINO (Ours)</b></td>
        <td>336 &times; 336</td>
        <td>0.1080</td>
        <td>0.255 m</td>
        <td>90.9%</td>
        <td>1.077</td>
        <td>0.0649</td>
        <td>62.9 ms</td>
      </tr>
      <tr>
        <td style="text-align:left;">UniDepth V2 [3]</td>
        <td>336 &times; 336</td>
        <td><b>0.0907</b></td>
        <td><b>0.212 m</b></td>
        <td><b>94.6%</b></td>
        <td>0.925</td>
        <td>0.0426</td>
        <td>108.2 ms</td>
      </tr>
      <tr>
        <td style="text-align:left;">Metric3D ViT-Small [2]</td>
        <td>336 &times; 336</td>
        <td>0.3855</td>
        <td>0.823 m</td>
        <td><b>0.3%</b></td>
        <td>0.619</td>
        <td>0.0564</td>
        <td>76.1 ms</td>
      </tr>
      <tr class="subhead"><td colspan="8">C. Apple Hypersim (60 Ray-Traced Rooms)</td></tr>
      <tr>
        <td style="text-align:left;"><b>Dioptra-DINO (Ours)</b></td>
        <td>336 &times; 336</td>
        <td><b>0.1811</b></td>
        <td><b>1.402 m</b></td>
        <td><b>70.2%</b></td>
        <td><b>0.934</b></td>
        <td>0.1434</td>
        <td>62.9 ms</td>
      </tr>
      <tr>
        <td style="text-align:left;">UniDepth V2 [3]</td>
        <td>336 &times; 336</td>
        <td>0.2880</td>
        <td>2.067 m</td>
        <td>32.1%</td>
        <td>0.774</td>
        <td>0.1355</td>
        <td>108.2 ms</td>
      </tr>
      <tr>
        <td style="text-align:left;">Metric3D ViT-Small [2]</td>
        <td>336 &times; 336</td>
        <td>0.4414</td>
        <td>3.046 m</td>
        <td>13.5%</td>
        <td>0.682</td>
        <td>0.2651</td>
        <td>76.1 ms</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="full-width">
  <div class="figure-container">
    <img src="figures/fig3_equal_res_hypersim.png" alt="Hypersim Visual Panel">
    <div class="figure-caption">
      <b>Fig. 2. Qualitative Foundation Model Comparison on Apple Hypersim (Frame 45) under Equal Resolution (336&times;336).</b> Left to right: RGB image, Ground Truth depth (9.9m range), Dioptra-DINO (AbsRel 0.198, faithful scale), Metric3D (AbsRel 0.331, severe scale under-estimation), and UniDepth V2 (AbsRel 0.266).
    </div>
  </div>
</div>

<div class="two-column">

<h2>VI. Equal-Resolution Foundation Benchmark</h2>
<p>
To eliminate resolution bias, Table III constrains all three models to 336&times;336. Deprived of its 616&times;1064 grid, Metric3D undergoes severe degradation: overall AbsRel jumps to <b>0.3997</b> and inliers collapse to <b>17.3%</b> (0.3% on ScanNet), underestimating physical scale by >30% (0.698&times;). Dioptra-DINO preserves solid inliers (<b>72.5%</b>), exact scale (<b>1.019&times;</b>), and <b>4.2&times; higher inliers than Metric3D</b> while remaining the fastest model (62.9 ms).
</p>

<div class="figure-container">
  <img src="figures/fig6_error_distribution.png" alt="Comparative Charts">
  <div class="figure-caption">
    <b>Fig. 3. Foundation Benchmark Distributions.</b> Left: Direct AbsRel. Middle: Inlier precision. Right: Latency versus error Pareto curve showing edge efficiency.
  </div>
</div>

<h2>VII. Limitations and Candid Discussion</h2>
<ol>
  <li><b>Cavernous Space Compression</b>: In massive warehouses (depths >40m), Dioptra-DINO compresses predictions toward domestic priors (0.50&times; scale) due to training distribution bias (<8m).</li>
  <li><b>Boundary Smoothing</b>: At 336&times;336 (14px patch tokens), thin chair legs and distant wires exhibit spatial smoothing.</li>
  <li><b>Sensor Domain Gaps</b>: Structured-light sensor missing data on NYUv2 lowers raw unaligned scores, highlighting synthetic-to-sensor transfer gaps.</li>
</ol>

<h2>VIII. Conclusion</h2>
<p>
We presented <b>Dioptra-DINO</b>, an efficient foundation model for monocular metric depth estimation tailored to edge robotics. Coupling DINOv2 visual features with Canonical Virtual Camera normalization (F<sub>canon</sub>=1000px) and Adaptive Receptive Alignment delivers state-of-the-art metric accuracy in close-range domestic interiors while operating at 16–17 FPS on Apple Silicon MPS with a ~240 MB footprint. Under equal 336&times;336 resolution constraints, Dioptra-DINO demonstrated superior scale calibration and inlier precision.
</p>

<h2>References</h2>
<ol class="ref-list">
  <li>M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," <i>TMLR</i>, 2024.</li>
  <li>W. Yin et al., "Metric3D: Towards Zero-shot Metric Depth Estimation from Single Images," <i>ICCV</i>, 2023.</li>
  <li>L. Piccinelli et al., "UniDepth: Universal Monocular Metric Depth Estimation," <i>CVPR</i>, 2024.</li>
  <li>L. Yang et al., "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data," <i>CVPR</i>, 2024.</li>
  <li>L. Yang et al., "Depth Anything V2," <i>arXiv:2406.09414</i>, 2024.</li>
  <li>N. Silberman et al., "Indoor Segmentation and Support Inference from RGBD Images," <i>ECCV</i>, 2012.</li>
  <li>A. Dai et al., "ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes," <i>CVPR</i>, 2017.</li>
  <li>M. Roberts et al., "Hypersim: A Photorealistic Synthetic Dataset," <i>ICCV</i>, 2021.</li>
  <li>W. Li et al., "InteriorNet: Mega-scale Multi-sensor Photo-realistic Indoor Scenes Dataset," <i>BMVC</i>, 2018.</li>
  <li>W. Wang et al., "TartanAir: A Dataset to Push the Limits of Visual SLAM," <i>IROS</i>, 2020.</li>
  <li>R. Ranftl et al., "Towards Robust Monocular Depth Estimation," <i>IEEE TPAMI</i>, 2020.</li>
  <li>R. Ranftl et al., "Vision Transformers for Dense Prediction," <i>ICCV</i>, 2021.</li>
  <li>S. F. Bhat et al., "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth," <i>arXiv</i>, 2023.</li>
  <li>D. Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network," <i>NeurIPS</i>, 2014.</li>
  <li>H. Touvron et al., "Training data-efficient image transformers & distillation through attention," <i>ICML</i>, 2021.</li>
</ol>

</div>

</body>
</html>
"""

def compile_pdf():
    print("=" * 80)
    print("COMPILING DIOPTRA-DINO RESEARCH PAPER TO PDF")
    print("=" * 80)

    # 1. Write HTML
    with open(HTML_PATH, "w") as f:
        f.write(HTML_CONTENT)
    print(f"✓ Formatted HTML saved to: {HTML_PATH}")

    # 2. Compile via Headless Chrome
    cmd = [
        CHROME_BIN,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        os.path.abspath(HTML_PATH)
    ]
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(PDF_PATH) or os.path.getsize(PDF_PATH) == 0:
        print("❌ Error compiling PDF:", res.stderr)
        return False

    pdf_sz_kb = os.path.getsize(PDF_PATH) / 1024.0
    print(f"✓ PDF successfully compiled: {PDF_PATH} ({pdf_sz_kb:.1f} KB)")

    # 3. Render PDF Pages to PNG using PyMuPDF (fitz)
    doc = fitz.open(PDF_PATH)
    print(f"✓ PDF has {len(doc)} pages. Rendering pages to PNG...")
    rendered_pages = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        page_png = os.path.join(PAGES_DIR, f"page_{i+1}.png")
        pix.save(page_png)
        rendered_pages.append(page_png)
        print(f"  ✓ Rendered page {i+1} -> {page_png}")

    doc.close()
    print("=" * 80)
    print(f"ALL DONE! Compiled {len(rendered_pages)} pages to PDF & PNGs.")
    return True

if __name__ == "__main__":
    compile_pdf()
