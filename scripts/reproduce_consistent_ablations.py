import os, sys, glob, torch, numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from dataclasses import replace

sys.path.insert(0, "/Users/krishnakant/Downloads/tesseract_kaggle_code_v16")
from dioptra_dino import DioptraDINO, DioptraDINOConfig
from scripts.eval_dino import load_model as load_dino_model, preprocess_sample, compute_metrics
from dioptra import Tesseract, TesseractConfig, load_pretrained_weights
from scripts.run_extensive_paper_benchmarks import compute_surface_normal_error

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using compute device:", device)

models = [
    {
        "id": "2d_vit",
        "name": "Ablation: Canonical 2D ViT",
        "type": "tesseract",
        "ckpt": "outputs/checkpoint_2d_vit_epoch24.pt",
        "pe_mode": "none",
        "enable_trivision": False,
        "enable_irer": False,
        "params": "8.1 M",
    },
    {
        "id": "center_ray",
        "name": "Ablation: Center-Ray PE",
        "type": "tesseract",
        "ckpt": "outputs/checkpoint_center_ray_epoch24.pt",
        "pe_mode": "center_ray",
        "enable_trivision": True,
        "enable_irer": True,
        "params": "8.1 M",
    },
    {
        "id": "no_ara",
        "name": "Ablation: Without ARA",
        "type": "tesseract",
        "ckpt": "outputs/checkpoint_no_irer_epoch24.pt",
        "pe_mode": "trivision",
        "enable_trivision": True,
        "enable_irer": False,
        "params": "8.1 M",
    },
    {
        "id": "stage1",
        "name": "Dioptra Stage 1 Baseline",
        "type": "tesseract",
        "ckpt": "outputs/checkpoint_epoch24.pt",
        "pe_mode": "trivision",
        "enable_trivision": True,
        "enable_irer": True,
        "params": "8.1 M",
    },
    {
        "id": "stage2",
        "name": "Dioptra Stage 2 Headline",
        "type": "tesseract",
        "ckpt": "outputs/checkpoint_stage2_epoch12.pt",
        "pe_mode": "trivision",
        "enable_trivision": True,
        "enable_irer": True,
        "params": "8.1 M",
    },
    {
        "id": "dino_ep13",
        "name": "Dioptra-DINO (Epoch 13)",
        "type": "dino",
        "ckpt": "outputs_dino/dioptra_dino_epoch_13.pt",
        "params": "27.5 M",
    },
    {
        "id": "dino_ep26",
        "name": "Dioptra-DINO (Epoch 26)",
        "type": "dino",
        "ckpt": "outputs_dino/dioptra_dino_epoch26_best.pt",
        "params": "27.5 M",
    },
]

pairs = []
for img in sorted(glob.glob("test_samples/**/*.png", recursive=True)):
    base, _ = os.path.splitext(img)
    for ext in ["_depth.npy", ".npy"]:
        cand = base + ext
        if os.path.exists(cand):
            pairs.append((img, cand))
            break
print(f"Loaded {len(pairs)} pairs.")

results = []
for m_info in models:
    m_name = m_info["name"]
    print(f"\nEvaluating {m_name}...")
    if m_info["type"] == "dino":
        model = load_dino_model(m_info["ckpt"], device=device)
    else:
        base_cfg = TesseractConfig()
        m_cfg = replace(
            base_cfg.model,
            pe_mode=m_info["pe_mode"],
            enable_trivision=m_info["enable_trivision"],
            enable_irer=m_info["enable_irer"],
        )
        model = Tesseract(m_cfg).to(device)
        load_pretrained_weights(m_info["ckpt"], model, str(device))
    model.eval()

    raw_ars, raw_rmses, raw_d1s, raw_d2s, scales, ali_ars, ali_d1s, normal_errs = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for img_p, gt_p in pairs:
            inp, gt, _, K, _ = preprocess_sample(img_p, gt_p, device=device)
            if m_info["type"] == "dino":
                pred_t = model(inp, K, ara_gate=1.0)
            else:
                out = model(inp, K)
                if isinstance(out, dict): pred_t = out.get("depth", out.get("pred"))
                elif isinstance(out, tuple): pred_t = out[0]
                else: pred_t = out
            p = pred_t.squeeze().cpu().numpy()
            m_raw = compute_metrics(p, gt)
            raw_ars.append(m_raw["abs_rel"])
            raw_rmses.append(m_raw["rmse"])
            raw_d1s.append(m_raw["a1"])
            raw_d2s.append(m_raw["a2"])
            scales.append(m_raw["scale_ratio"])

            mask = (gt > 0.1) & (gt < 80.0) & np.isfinite(gt) & (p > 0.1) & (p < 80.0) & np.isfinite(p)
            s = np.median(gt[mask]) / (np.median(p[mask]) + 1e-8)
            p_ali = p[mask] * s
            g_ali = gt[mask]
            ali_ars.append(np.mean(np.abs(g_ali - p_ali) / g_ali))
            ratio = np.maximum(g_ali / p_ali, p_ali / g_ali)
            ali_d1s.append((ratio < 1.25).mean())

            gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)
            err_deg = compute_surface_normal_error(pred_t, gt_t, K)
            if err_deg > 0.0:
                normal_errs.append(err_deg)

    res = {
        "id": m_info["id"],
        "name": m_info["name"],
        "params": m_info["params"],
        "raw_absrel": np.mean(raw_ars),
        "raw_rmse": np.mean(raw_rmses),
        "raw_d1": np.mean(raw_d1s) * 100,
        "raw_d2": np.mean(raw_d2s) * 100,
        "scale": np.mean(scales),
        "ali_absrel": np.mean(ali_ars),
        "ali_d1": np.mean(ali_d1s) * 100,
        "normal_err": np.mean(normal_errs) if normal_errs else 0.0,
    }
    results.append(res)
    print(f"-> AbsRel: {res['raw_absrel']:.4f}, RMSE: {res['raw_rmse']:.2f}m, d1: {res['raw_d1']:.1f}%, Scale: {res['scale']:.3f}, Ali AbsRel: {res['ali_absrel']:.4f}, Ali d1: {res['ali_d1']:.1f}%, Normal: {res['normal_err']:.2f} deg")

print("\n\n" + "="*80)
print("FINAL UNIFIED ABLATION RESULTS (Consistent with Table 1):")
print("="*80)
for r in results:
    print(f"{r['name']} & {r['params']} & {r['raw_absrel']:.4f} & {r['raw_rmse']:.2f}\\,m & {r['raw_d1']:.1f}\\% & {r['raw_d2']:.1f}\\% & {r['scale']:.3f} & {r['ali_absrel']:.4f} & {r['ali_d1']:.1f}\\% & {r['normal_err']:.2f}$^\\circ$ \\\\")
