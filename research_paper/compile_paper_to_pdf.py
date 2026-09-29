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
    margin: 0.65in 0.6in 0.65in 0.6in;
    @bottom-center {
      content: counter(page);
      font-family: 'Times New Roman', Times, serif;
      font-size: 10pt;
    }
  }

  body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt;
    line-height: 1.22;
    color: #000;
    margin: 0;
    padding: 0;
    text-align: justify;
    text-justify: inter-word;
  }

  .paper-header {
    text-align: center;
    margin-bottom: 1.2em;
  }

  h1.title {
    font-size: 18pt;
    font-weight: bold;
    margin: 0 0 0.35em 0;
    line-height: 1.18;
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
    margin-bottom: 0.8em;
  }

  .abstract-box {
    margin: 0 1.2em 1.1em 1.2em;
    font-size: 8.8pt;
    line-height: 1.22;
  }

  .abstract-box b.heading {
    font-weight: bold;
    font-size: 8.8pt;
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
    margin: 0.85em 0 0.35em 0;
    letter-spacing: 0.5px;
    break-after: avoid;
  }

  h3 {
    font-size: 9.8pt;
    font-style: italic;
    font-weight: bold;
    margin: 0.65em 0 0.25em 0;
    break-after: avoid;
  }

  p {
    margin: 0 0 0.45em 0;
    text-indent: 1.2em;
  }

  p.no-indent {
    text-indent: 0;
  }

  ul, ol {
    margin: 0.2em 0 0.6em 1.4em;
    padding: 0;
  }

  li {
    margin-bottom: 0.25em;
    line-height: 1.22;
  }

  .equation {
    text-align: center;
    margin: 0.35em 0;
    font-style: italic;
    position: relative;
  }

  .equation span.eq-num {
    position: absolute;
    right: 0;
    font-style: normal;
  }

  .full-width {
    column-span: all;
    margin: 0.8em 0;
  }

  .table-caption {
    font-size: 8.5pt;
    text-align: center;
    font-weight: bold;
    margin-bottom: 0.35em;
    letter-spacing: 0.2px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8.2pt;
    margin-bottom: 0.4em;
  }

  th {
    border-top: 1.2pt solid #000;
    border-bottom: 0.8pt solid #000;
    padding: 2.5px 5px;
    font-weight: bold;
    text-align: center;
  }

  td {
    padding: 2.2px 5px;
    text-align: center;
  }

  tr.subhead td {
    font-style: italic;
    font-weight: bold;
    text-align: left;
    background-color: #f8f9fa;
    border-top: 0.5pt solid #ddd;
    border-bottom: 0.5pt solid #ddd;
    padding: 2.5px 5px;
  }

  tbody tr:last-child td {
    border-bottom: 1.2pt solid #000;
  }

  .figure-container {
    margin: 0.8em 0;
    text-align: center;
    break-inside: avoid;
  }

  .figure-container img {
    max-width: 100%;
    height: auto;
    border: 0.5pt solid #ddd;
  }

  .figure-caption {
    font-size: 8.2pt;
    text-align: justify;
    text-justify: inter-word;
    margin-top: 0.35em;
    line-height: 1.18;
  }

  .figure-caption b {
    font-weight: bold;
  }

  ol.ref-list {
    font-size: 7.5pt;
    line-height: 1.15;
    margin: 0 0 0 1.2em;
    padding: 0;
  }

  ol.ref-list li {
    margin-bottom: 0.2em;
    text-align: left;
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
  <b class="heading">Abstract</b>—Monocular metric depth estimation on autonomous mobile robots presents an acute trade-off between physical scale fidelity and closed-loop inference latency. Contemporary metric vision foundation models (e.g., UniDepth V2, Metric3D) achieve impressive zero-shot transfer, but their high computational footprints (400–750 ms per frame) limit throughput to 1.3–2.4 FPS on embedded hardware, inducing unacceptable control latency for aerial navigation and quadrupedal locomotion (where &gt;15 FPS is typically targeted for closed-loop stability). Conversely, uncalibrated relative depth models cannot recover physical scale without test-time oracle alignment. In this work, we present <b>Dioptra-DINO</b>, an edge-efficient 27.51M-parameter metric depth architecture tailored for real-time mobile robotics. Dioptra-DINO couples a self-supervised DINOv2-Small backbone with Canonical Virtual Camera Normalization (F<sub>canon</sub> = 1000.0px) and Trivision Ray FiLM Modulation, operating natively at 336&times;336 resolution. Evaluated under a standardized latency protocol (FP16, batch size 1 on Apple Silicon GPU), Dioptra-DINO processes frames in <b>58.2 ms (17.2 FPS)</b> with under <b>240 MB VRAM</b>. Compared to heavyweight baselines running at their native resolutions, Dioptra-DINO achieves a <b>7.2&times; speedup over native UniDepth V2</b> (421.6 ms) and <b>12.9&times; speedup over native Metric3D</b> (753.5 ms); when all models are constrained to an identical 336&times;336 budget, Dioptra-DINO remains <b>1.9&times; faster than UniDepth</b> (108.2 ms) and <b>1.3&times; faster than Metric3D</b> (76.1 ms). On 2,744 held-out Apple Hypersim indoor frames (in-domain, as Dioptra was pre-trained on Hypersim training scenes), Dioptra-DINO attains <b>0.1477 AbsRel</b> and <b>84.3% inlier precision</b> (&delta;<sub>1</sub>). On zero-shot transfer benchmarks (InteriorNet and ScanNet Scene00), UniDepth V2 achieves superior accuracy (0.0907 vs. 0.1080 AbsRel on ScanNet Scene00 real sensor depth at 336&times;336), demonstrating the limits of compact models. However, Dioptra-DINO establishes the Pareto frontier for edge robotics, providing real-time 17 FPS metric guidance where heavyweight models induce control lag. We release complete code, benchmark protocols, and weights.
</div>

<div class="two-column">

<h2>I. Introduction</h2>
<p>
Autonomous mobile robots operating in GPS-denied environments require accurate, low-latency 3D distance perception for reactive obstacle avoidance, visual odometry, and path planning. While active sensors such as LiDAR and structured-light RGB-D cameras provide direct metric measurements, their integration onto micro-aerial vehicles (MAVs) and low-cost ground rovers is often hindered by strict payload constraints, high power consumption, limited operational range, and susceptibility to sunlight interference.
</p>
<p>
Passive monocular metric depth estimation represents an appealing alternative, but recovering metric scale from a single 2D image is inherently ill-posed due to projective ambiguity: an object of dimension <i>H</i> at distance <i>Z</i> generates the identical retinal projection <i>h = f<sub>y</sub> &middot; (H / Z)</i> as an object of size <i>kH</i> at distance <i>kZ</i>. Furthermore, different camera optics dynamically distort object projections, confounding networks that attempt to regress metric depth without explicit focal length conditioning.
</p>

<div class="figure-container">
  <img src="figures/fig1_architecture.png" alt="Architecture Diagram">
  <div class="figure-caption">
    <b>Fig. 1. Dioptra-DINO Architectural Overview.</b> Input RGB images and camera intrinsics <b>K</b> are normalized to a canonical virtual pinhole camera (F<sub>canon</sub>=1000px). Multi-scale representations from DINOv2-Small ViT-S/14 are modulated by closed-form Trivision Ray FiLM vectors to regress metric depth at native 336&times;336 resolution.
  </div>
</div>

<p>
Recent research addresses this problem along two divergent axes:
</p>
<ul>
  <li><b>Uncalibrated Foundation Models</b>: Architectures such as Depth Anything [4, 5] train on massive unlabeled web datasets (&gt;62M images), learning exceptional semantic boundaries. However, their metric adaptations lack explicit camera intrinsics conditioning <b>K</b>: unable to ingest focal parameters, they rely on implicit scene priors and cannot mathematically adjust to zoom lenses or varying sensor optics.</li>
  <li><b>Heavyweight Metric Foundation Models</b>: Frameworks such as Metric3D [2] and UniDepth [3] condition on focal length or unproject dense pseudopinhole ray fields. While achieving strong zero-shot transfer, their complex encoders and large resolutions (616&times;1064) require 400–750 ms per frame on embedded edge hardware, restricting throughput to 1.3–2.4 FPS—insufficient for closed-loop robotic control.</li>
</ul>
<p>
This paper addresses the practical edge robotics question: <i>Can a lightweight vision transformer (&lt;28M parameters) operating at modest resolution (336&times;336) deliver reliable, calibration-conditioned metric depth at &gt;15 FPS on embedded hardware?</i>
</p>
<p>
We present <b>Dioptra-DINO</b>. By projecting depth regression into a Canonical Virtual Camera space (F<sub>canon</sub> = 1000.0px) and modulating multi-scale patch tokens via continuous Trivision Ray FiLM vectors, Dioptra-DINO decouples scale from visual geometry. We candidly evaluate trade-offs across in-domain held-out camera trajectories (Apple Hypersim [8]) and zero-shot cross-dataset environments (InteriorNet [9] and ScanNet [7]).
</p>

<h2>II. Related Work</h2>
<h3>A. Monocular Relative Depth Estimation</h3>
<p>
Early monocular depth methods focused on relative depth estimation via scale-invariant representations. Ranftl et al. introduced MiDaS [10] and DPT [11], demonstrating that mixing heterogeneous datasets with an affine-invariant loss enables broad zero-shot generalization across scenes. Recently, Depth Anything V1 and V2 [4, 5] scaled relative pretraining using over 62 million unlabeled images and DINOv2 representations [1]. While achieving exceptional structural sharpness, relative depth predictions require unknown test-time scale and shift parameters (s, t), preventing immediate use in closed-loop robotic navigation without external odometry.
</p>

<h3>B. Camera-Conditioned Metric Depth Estimation</h3>
<p>
To recover true physical dimensions, works such as ZoeDepth [12] and Metric3D [2] investigated multi-dataset metric transfer. Metric3D proposed focal length normalization, projecting images to a virtual camera to resolve projective ambiguity. However, Metric3D relies on heavy input resolutions (616&times;1064) and incurs prohibitive compute latency (&gt;750ms). UniDepth [3] advanced this direction by predicting universal metric depth and camera intrinsics directly via pseudopinhole ray embeddings. While versatile, UniDepth requires multiple dense operations that remain compute-intensive for edge robotics. Dioptra-DINO demonstrates that canonical focal normalization coupled with lightweight ray modulation yields high metric fidelity at 336&times;336 in under 60ms.
</p>

<h2>III. Methodology</h2>
<h3>A. Problem Formulation and Scale Ambiguity</h3>
<p>
Consider a pinhole camera with intrinsic calibration matrix <b>K</b>:
</p>
<div class="equation">
  <b>K</b> = [ [f<sub>x</sub>, 0, c<sub>x</sub>], [0, f<sub>y</sub>, c<sub>y</sub>], [0, 0, 1] ]
  <span class="eq-num">(1)</span>
</div>
<p class="no-indent">
A 3D coordinate <b>P</b> = [X, Y, Z]<sup>T</sup> projects to image coordinates <b>p</b> = [u, v, 1]<sup>T</sup> via Z<b>p</b> = <b>KP</b>. When cameras with focal lengths f<sub>1</sub>, f<sub>2</sub> observe identical objects at depths Z<sub>1</sub>, Z<sub>2</sub>, identical pixel extents occur whenever f<sub>1</sub>/Z<sub>1</sub> = f<sub>2</sub>/Z<sub>2</sub>. Without explicit focal conditioning, neural networks confuse optical zoom with physical distance.
</p>

<h3>B. Canonical Virtual Camera Transformation</h3>
<p>
To decouple metric scale from camera hardware, Dioptra-DINO defines a canonical virtual pinhole camera with reference focal length F<sub>canon</sub> = 1000.0px:
</p>
<div class="equation">
  <b>K</b><sub>canon</sub> = [ [F<sub>canon</sub>, 0, c<sub>x</sub>'], [0, F<sub>canon</sub>, c<sub>y</sub>'], [0, 0, 1] ]
  <span class="eq-num">(2)</span>
</div>
<p class="no-indent">
When an input image of size W&times;H captured with focal length f<sub>x</sub> is resized to input resolution S&times;S (S=336), its scaled focal length is f<sub>scaled</sub> = f<sub>x</sub> &middot; (S / W). The focal ratio between sensor and canonical camera defines scale factor &gamma;:
</p>
<div class="equation">
  &gamma; = f<sub>scaled</sub> / F<sub>canon</sub>
  <span class="eq-num">(3)</span>
</div>
<p class="no-indent">
The model regresses canonical depth d<sub>canon</sub>(u, v) &isin; [0.1m, 10.0m]. True metric depth d<sub>metric</sub> is recovered via deterministic re-projection:
</p>
<div class="equation">
  d<sub>metric</sub>(u, v) = d<sub>canon</sub>(u, v) &middot; &gamma; = d<sub>canon</sub>(u, v) &middot; (f<sub>scaled</sub> / F<sub>canon</sub>)
  <span class="eq-num">(4)</span>
</div>

<h3>C. Trivision Ray FiLM Modulation</h3>
<p class="no-indent">
To ground tokens in 3D camera geometry without dense ray-tracing overhead, Dioptra-DINO unprojects a canonical ray triplet for each patch token: center ray <b>r</b><sub>c</sub>, top-left opposing corner ray <b>r</b><sub>1</sub>, and bottom-right opposing corner ray <b>r</b><sub>2</sub>. The concatenated triplet [<b>r</b><sub>c</sub>, <b>r</b><sub>1</sub>, <b>r</b><sub>2</sub>] &isin; &reals;<sup>9</sup> is mapped to a multi-scale Fourier embedding across 6 octave bands (&reals;<sup>108</sup>), generating scale (&gamma;<sub>film</sub>) and shift (&beta;<sub>film</sub>) parameters that modulate visual tokens via FiLM:
</p>
<div class="equation">
  <b>z</b><sub>i</sub>' = &gamma;<sub>film</sub>(<b>e</b>(<b>r</b><sub>i</sub>)) &odot; <b>z</b><sub>i</sub> + &beta;<sub>film</sub>(<b>e</b>(<b>r</b><sub>i</sub>))
  <span class="eq-num">(5)</span>
</div>

<h3>D. Angular Residual Attention (ARA)</h3>
<p class="no-indent">
To evaluate angular geometric locality during self-attention, we explored an Angular Residual Attention (ARA) block that augments attention logits with an angular distance penalty:
</p>
<div class="equation">
  <b>A</b><sub>qk</sub> = (<b>q</b><sub>q</sub><sup>T</sup> <b>k</b><sub>k</sub> / &radic;d) - &lambda; &middot; (1 - (<b>r</b><sub>q</sub> &middot; <b>r</b><sub>k</sub>)<sup>2</sup>)
  <span class="eq-num">(6)</span>
</div>
<p class="no-indent">
where &lambda; = softplus(&lambda;<sub>raw</sub>). As detailed in our ablations, empirical ablations demonstrate that ARA provides marginal benefit (&Delta;AbsRel &lt; 0.0002) once Trivision Ray FiLM Modulation is active; we document this frankly to avoid ungrounded novelty claims.
</p>

<h3>E. Training Protocol & Implementation Details</h3>
<p class="no-indent">
Dioptra-DINO is trained across a multi-domain indoor corpus comprising 191 scenes from Apple Hypersim [8], synthetic warehouse trajectories from TartanAir, and sensor captures from NYU-Depth-v2 [6] (mixture: 60% Hypersim, 25% TartanAir, 15% NYUv2). Pinhole intrinsics are applied per domain (TartanAir: f=320px; Hypersim: f<sub>x</sub>=888.89, f<sub>y</sub>=1000.0px; NYUv2: f=518.86px). Depth targets are clamped to [0.1m, 10.0m]. Optimization runs using AdamW (&beta;<sub>1</sub>=0.9, &beta;<sub>2</sub>=0.999, weight decay 0.01) with initial learning rate 1&times;10<sup>-4</sup> scheduled via cosine annealing over 40 epochs with batch size 16 across dual Tesla T4 GPUs.
</p>

<h3>F. Composite Objective Function</h3>
<p class="no-indent">
Training optimizes &Lscr;<sub>total</sub> = &lambda;<sub>SILog</sub> &Lscr;<sub>SILog</sub> + &lambda;<sub>grad</sub> &Lscr;<sub>grad</sub> + &lambda;<sub>norm</sub> &Lscr;<sub>norm</sub> with &lambda;<sub>SILog</sub> = 1.0 (&alpha;=0.85), &lambda;<sub>grad</sub> = 0.5, and &lambda;<sub>norm</sub> = 0.1.
</p>

</div>

<div class="full-width">
  <div class="table-caption">TABLE I: Architectural & Computational Comparison Across Evaluated Camera-Conditioned Foundation Models</div>
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
        <td>Canonical Virtual Cam + Trivision Ray</td>
        <td><b>58.2 ms</b></td>
        <td><b>17.2 FPS</b></td>
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
      <tr>
        <td style="text-align:left;">Metric3D ViT-Small [2]</td>
        <td>ViT-Small (DeiT-S)</td>
        <td>37.50M</td>
        <td>616 &times; 1064</td>
        <td>Pinhole Focal Normalization</td>
        <td>753.5 ms</td>
        <td>1.3 FPS</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="full-width">
  <div class="table-caption">TABLE II: 3,000-Frame Comprehensive Indoor Metric Benchmark (0.1m – 10.0m, 2,984 Valid Pairs)</div>
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
      <tr class="subhead"><td colspan="8">A. Overall Aggregate (All 2,984 Pure Indoor Frames)</td></tr>
      <tr>
        <td style="text-align:left;"><b>Dioptra-DINO (Ours)</b></td>
        <td><b>0.1658</b></td>
        <td><b>0.689 m</b></td>
        <td><b>0.509 m</b></td>
        <td><b>82.5%</b></td>
        <td><b>1.040</b></td>
        <td>0.1265</td>
        <td>27.4&deg;</td>
      </tr>
      <tr>
        <td style="text-align:left;">UniDepth V2 ViT-Small [3]</td>
        <td>0.2259</td>
        <td>0.880 m</td>
        <td>0.737 m</td>
        <td>75.0%</td>
        <td>1.098</td>
        <td><b>0.0877</b></td>
        <td><b>23.5&deg;</b></td>
      </tr>
      <tr>
        <td style="text-align:left;">Metric3D ViT-Small [2]</td>
        <td>0.2360</td>
        <td>0.955 m</td>
        <td>0.800 m</td>
        <td>72.6%</td>
        <td>1.041</td>
        <td>0.1027</td>
        <td>29.3&deg;</td>
      </tr>
      <tr class="subhead"><td colspan="8">B. Apple Hypersim (2,744 Ray-Traced Rooms -- In-Domain Held-Out Split)</td></tr>
      <tr>
        <td style="text-align:left;"><b>Dioptra-DINO (Ours)</b></td>
        <td><b>0.1477</b></td>
        <td><b>0.687 m</b></td>
        <td><b>0.508 m</b></td>
        <td><b>84.3%</b></td>
        <td><b>1.026</b></td>
        <td>0.1201</td>
        <td>27.3&deg;</td>
      </tr>
      <tr>
        <td style="text-align:left;">UniDepth V2 ViT-Small [3]</td>
        <td>0.2164</td>
        <td>0.896 m</td>
        <td>0.747 m</td>
        <td>76.2%</td>
        <td>1.089</td>
        <td><b>0.0844</b></td>
        <td><b>23.5&deg;</b></td>
      </tr>
      <tr>
        <td style="text-align:left;">Metric3D ViT-Small [2]</td>
        <td>0.2259</td>
        <td>0.976 m</td>
        <td>0.819 m</td>
        <td>73.4%</td>
        <td>1.023</td>
        <td>0.0994</td>
        <td>29.3&deg;</td>
      </tr>
      <tr class="subhead"><td colspan="8">C. InteriorNet (240 Residential Frames -- Zero-Shot Transfer)</td></tr>
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
        <td style="text-align:left;">UniDepth V2 ViT-Small [3]</td>
        <td><b>0.3346</b></td>
        <td><b>0.699 m</b></td>
        <td>0.617 m</td>
        <td>61.4%</td>
        <td>1.207</td>
        <td><b>0.1254</b></td>
        <td><b>23.7&deg;</b></td>
      </tr>
      <tr>
        <td style="text-align:left;">Metric3D ViT-Small [2]</td>
        <td>0.3504</td>
        <td>0.718 m</td>
        <td>0.578 m</td>
        <td><b>63.6%</b></td>
        <td>1.241</td>
        <td>0.1415</td>
        <td>28.9&deg;</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="two-column">

<h2>IV. Comprehensive Indoor Evaluation</h2>
<h3>A. 3,000-Frame Benchmark Analysis</h3>
<p>
Table II reports evaluation across 2,984 valid test pairs. 
</p>
<ul>
  <li><b>In-Domain Hypersim Performance</b>: On 2,744 ray-traced Hypersim frames, Dioptra-DINO attains <b>0.1477 AbsRel</b> and <b>84.3% inliers</b> (&delta;<sub>1</sub>), outperforming Metric3D ViT-Small (0.2259 AbsRel, 73.4% inliers) by <b>34.6% lower error</b> and UniDepth V2 (0.2164 AbsRel, 76.2% inliers) by <b>31.7% lower error</b>. We note that Dioptra benefits here from in-domain pretraining on Hypersim training scenes.</li>
  <li><b>Zero-Shot Transfer Realities</b>: On InteriorNet (240 zero-shot frames), UniDepth V2 attains lower error (<b>0.3346 AbsRel</b>) than Dioptra (0.3726), and on ScanNet Scene00 real sensor depth under equal 336&times;336 resolution (Table III), UniDepth achieves <b>0.0907 AbsRel</b> vs. Dioptra's 0.1080. This confirms that large models trained across broader sensor corpora generalize better to unseen sensor distributions.</li>
  <li><b>Robotic Edge Trade-Off</b>: Running natively at 336&times;336 in <b>58.2 ms (17.2 FPS)</b>, Dioptra-DINO is <b>7.2&times; faster than native UniDepth V2</b> (421.6 ms / 2.4 FPS) and <b>12.9&times; faster than native Metric3D</b> (753.5 ms / 1.3 FPS), establishing an efficient real-time operating point.</li>
</ul>

<h3>B. Component Knockout Ablation Study</h3>
<p>
Table IV presents component knockout evaluations on held-out TartanAir trajectory sequences (an in-domain synthetic benchmark with dense ground truth depth, which explains the lower baseline error of 0.0584 AbsRel compared to cross-dataset tests):
</p>
<ul>
  <li><b>Ray Modulation Impact</b>: Disabling Trivision Ray FiLM Modulation causes AbsRel to jump from <b>0.0584</b> to <b>0.1893</b> (+224% error increase), with inliers dropping to 79.5% (-17.2 pp) and scale drifting to 1.179&times;. This confirms that camera ray unprojection is the primary mechanism grounding metric scale.</li>
  <li><b>Center-Ray Only Failure</b>: Using a single center ray causes severe degradation (0.7989 AbsRel, 2.1% inliers), demonstrating that opposing corner rays <b>r</b><sub>1</sub>, <b>r</b><sub>2</sub> are vital for encoding boundary field-of-view perspective cues.</li>
  <li><b>ARA Contribution</b>: Disabling the ARA angular bias (&lambda;=0) yields 0.0583 AbsRel vs. 0.0584 for the full model (&Delta;&lt;0.0002). We candidly conclude that ARA provides negligible empirical variance once Trivision Ray FiLM Modulation is present.</li>
</ul>

</div>

<div class="full-width">
  <div class="table-caption">TABLE IV: Component Knockout Ablation Study on Held-Out TartanAir Trajectory Sequences (In-Domain Synthetic Benchmark)</div>
  <table>
    <thead>
      <tr>
        <th style="text-align:left;">Ablation Configuration</th>
        <th>AbsRel (&darr;)</th>
        <th>RMSE (m &darr;)</th>
        <th>&delta;<sub>1</sub> (&uarr;)</th>
        <th>Scale Ratio</th>
        <th style="text-align:left;">Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align:left;"><b>Dioptra-DINO (Full)</b></td>
        <td><b>0.0584</b></td>
        <td><b>2.273 m</b></td>
        <td><b>96.7%</b></td>
        <td><b>1.002</b></td>
        <td style="text-align:left;">Full Trivision Ray FiLM + ARA Bias + DPT Head</td>
      </tr>
      <tr>
        <td style="text-align:left;">w/o ARA Angular Bias (&lambda;=0)</td>
        <td>0.0583</td>
        <td>2.273 m</td>
        <td>96.7%</td>
        <td>1.002</td>
        <td style="text-align:left;">Angular penalty disabled (standard self-attention)</td>
      </tr>
      <tr>
        <td style="text-align:left;">w/o Ray Positional Modulation</td>
        <td>0.1893</td>
        <td>2.808 m</td>
        <td>79.5%</td>
        <td>1.179</td>
        <td style="text-align:left;">Ray unprojection & FiLM disabled (+224% error)</td>
      </tr>
      <tr>
        <td style="text-align:left;">Center-Ray Only (w/o Trivision)</td>
        <td>0.7989</td>
        <td>8.151 m</td>
        <td>2.1%</td>
        <td>1.821</td>
        <td style="text-align:left;">Center ray [r<sub>c</sub>, r<sub>c</sub>, r<sub>c</sub>] without corner rays (collapses)</td>
      </tr>
    </tbody>
  </table>
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
        <td><b>58.2 ms</b> (17.2 FPS)</td>
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
      <tr class="subhead"><td colspan="8">B. ScanNet Scene00 (10 Handheld iPad Sensor Frames -- Zero-Shot)</td></tr>
      <tr>
        <td style="text-align:left;"><b>Dioptra-DINO (Ours)</b></td>
        <td>336 &times; 336</td>
        <td>0.1080</td>
        <td>0.255 m</td>
        <td>90.9%</td>
        <td>1.077</td>
        <td>0.0649</td>
        <td>58.2 ms</td>
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
        <td>58.2 ms</td>
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
    <img src="figures/fig2_equal_res_scannet.png" alt="ScanNet Qualitative Comparison">
    <div class="figure-caption">
      <b>Fig. 2. Qualitative Comparison on Real-World ScanNet Scene00 under Equal Resolution (336&times;336).</b> Displaying a representative frame (Dioptra-DINO AbsRel <b>0.055</b>, planar desk recovery, vs. sequence average of 0.1080; Metric3D AbsRel 0.379, scale collapse to &approx; 1m; UniDepth V2 AbsRel 0.106). Left to right: RGB sensor input, Ground Truth (2.6m max depth), Dioptra-DINO, Metric3D, and UniDepth V2.
    </div>
  </div>
</div>

<div class="two-column">

<h2>V. Equal-Resolution Foundation Benchmark</h2>
<p>
To evaluate how architectures handle low compute budgets, Table III constrains all models to an identical 336&times;336 grid.
</p>
<ul>
  <li><b>Metric3D Sensitivity</b>: Metric3D undergoes severe degradation when deprived of its 616&times;1064 grid: overall AbsRel jumps to <b>0.3997</b> and inliers collapse to <b>17.3%</b> (0.3% on ScanNet Scene00). This confirms that Metric3D's normalization was tailored specifically for high-resolution rectangular inputs.</li>
  <li><b>Dioptra-DINO Resilience</b>: Operating natively at 336&times;336, Dioptra-DINO preserves solid inliers (<b>72.5%</b>), exact metric scale (<b>1.019&times;</b>), and achieves <b>4.2&times; higher inliers than Metric3D</b> while remaining <b>1.9&times; faster than UniDepth</b> (108.2 ms) and <b>1.3&times; faster than Metric3D</b> (76.1 ms).</li>
</ul>

<div class="figure-container">
  <img src="figures/fig6_error_distribution.png" alt="Comparative Charts">
  <div class="figure-caption">
    <b>Fig. 3. Foundation Benchmark Distributions.</b> Left: Direct AbsRel. Middle: Inlier precision. Right: Latency versus error Pareto curve showing edge efficiency.
  </div>
</div>

<h2>VI. Limitations and Candid Discussion</h2>
<ol>
  <li><b>10-Metre Range Limit</b>: The model clamps depth to 10.0m. In expansive atriums or long hallways &gt;10m, predictions compress toward indoor priors, which must be accounted for in high-speed navigation.</li>
  <li><b>Patch Token Boundary Smoothing</b>: At 336&times;336 (14px tokens), thin chair legs and distant wires exhibit spatial smoothing compared to 1000px+ models.</li>
  <li><b>Sensor Domain Gap</b>: On raw sensor depth captures under equal 336&times;336 resolution (ScanNet Scene00 handheld iPad ToF), UniDepth V2 attains lower absolute error (0.0907 vs. 0.1080 AbsRel), reflecting its extensive training across broad real-world sensor datasets.</li>
</ol>

<h2>VII. Conclusion</h2>
<p>
We presented <b>Dioptra-DINO</b>, an efficient vision transformer architecture for monocular metric depth estimation tailored to edge robotics. By coupling Canonical Virtual Camera Normalization (F<sub>canon</sub>=1000px) with Trivision Ray FiLM Modulation, Dioptra-DINO achieves reliable indoor metric depth while operating at <b>58.2 ms (17.2 FPS)</b> on Apple Silicon GPU under &lt;240 MB VRAM. While heavyweight foundation models attain superior zero-shot transfer on real sensor captures, Dioptra-DINO delivers the real-time throughput required for closed-loop edge robotics.
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
  <li>R. Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer," <i>IEEE TPAMI</i>, 2020.</li>
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
