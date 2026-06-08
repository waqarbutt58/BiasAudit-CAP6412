# Bias & Safety Auditor for Text-to-Image Generative Models

**Course:** CAP6412 – Advanced Computer Vision, Spring 2025  
**Instructor:** Dr. Mubarak Shah, UCF CRCV  
**Project Type:** End-to-End Research & Engineering Deliverable

---

## Overview

This project builds a complete **Bias & Safety Auditing Pipeline** for Text-to-Image (T2I) generative models (Stable Diffusion). It detects demographic biases, flags unsafe content, erases harmful concepts, and auto-generates a structured PDF audit report.

```
Prompt Set → [Module 1] Image Generation
                       ↓
           ┌─────────────────────┐
           ↓                     ↓
   [Module 2]            [Module 3]
   Bias Detection        Safety Filter
   (CLIP zero-shot)      (SAFREE-inspired)
           ↓                     ↓
   [Module 4]                    │
   Concept Erasure               │
   (Forget-Me-Not style)         │
           └──────────┬──────────┘
                      ↓
             [Module 5] Report Generation
             → PDF Audit Report
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv cap6412_env
# Windows:
cap6412_env\Scripts\activate
# Linux/Mac:
source cap6412_env/bin/activate

pip install -r requirements.txt
```

> **Requires:** Python 3.10+, CUDA GPU (≥8 GB VRAM) or Google Colab (free T4).

### 2. Run the full pipeline

```bash
# Generate images + run bias + safety analysis + report
python src/run_pipeline.py

# Skip image generation (if already done)
python src/run_pipeline.py --skip_generation

# Include concept erasure
python src/run_pipeline.py --erase "nudity"

# Fast test (fewer images, fewer steps)
python src/run_pipeline.py --n_per_prompt 3 --steps 20
```

### 3. Run modules individually

```bash
# Module 1: Generate images
python src/generate_images.py --prompts_csv prompts.csv --output_dir images --n_per_prompt 10

# Module 2: Detect bias
python src/bias_detector.py --images_dir images --output_csv results/bias_results.csv

# Module 3: Safety scan
python src/safety_filter.py --images_dir images --output_csv results/safety_results.csv

# Module 4: Erase a concept
python src/concept_erasure.py --concept "nudity" --n_steps 200

# Module 5: Generate PDF report
python src/report_generator.py --output_pdf results/audit_report.pdf
```

---

## Project Structure

```
BiasAudit-CAP6412/
├── prompts.csv                    # 60 structured prompts (occupation/activity/neutral)
├── requirements.txt
├── README.md
│
├── src/
│   ├── generate_images.py         # Module 1 — Stable Diffusion image generation
│   ├── bias_detector.py           # Module 2 — CLIP zero-shot bias classification
│   ├── safety_filter.py           # Module 3 — NSFW / unsafe content detection
│   ├── concept_erasure.py         # Module 4 — Attention fine-tuning concept erasure
│   ├── report_generator.py        # Module 5 — PDF audit report generation
│   └── run_pipeline.py            # End-to-end pipeline runner
│
├── notebooks/
│   ├── 01_image_generation.ipynb
│   ├── 02_bias_detection.ipynb
│   ├── 03_safety_filter.ipynb
│   ├── 04_concept_erasure.ipynb
│   └── 05_report_generation.ipynb
│
├── images/                        # Generated images (gitignored — created at runtime)
├── results/                       # CSVs, charts, PDF report
└── models/                        # Fine-tuned U-Net weights (gitignored)
```

---

## Modules

### Module 1 — Image Generation
Generates images from `prompts.csv` using Stable Diffusion v1.5. Uses deterministic seeds for reproducibility.

- **Model:** `runwayml/stable-diffusion-v1-5`
- **Output:** `images/{category}/{prompt}/seed_N.png`

### Module 2 — Bias Detection
Zero-shot CLIP classification for **gender**, **age**, and **apparent race** per image.

- **Bias score** = dominant proportion − (1 / n_classes). Range [0, 1].
- **Shannon entropy** for diversity measurement.
- **Output:** `results/bias_results.csv`, `results/bias_summary.json`

### Module 3 — Safety Filter (SAFREE-inspired)
Two-stage: (1) keyword taxonomy prompt check, (2) `Falconsai/nsfw_image_detection` model.

- **Output:** `results/safety_results.csv`

### Module 4 — Concept Erasure (Forget-Me-Not inspired)
Fine-tunes only cross-attention K/V weights (`~2%` of U-Net parameters) via attention re-steering MSE loss.

- **Output:** `models/erased_unet/`, `results/erasure_comparison/`

### Module 5 — Report Generation
Builds a structured PDF with bias distribution charts, safety statistics, concept erasure comparison, methodology, and references.

- **Output:** `results/audit_report.pdf`

---

## Evaluation Metrics

| Dimension | Metric | Description |
|---|---|---|
| Bias | Bias Score | max_proportion − 1/n_classes (0=fair) |
| Bias | Shannon Entropy | Higher = more demographically diverse |
| Safety | NSFW Rate | % images flagged as unsafe |
| Erasure | CLIP Similarity | Should decrease after erasure |
| Erasure | Collateral Damage | CLIP score on unrelated prompts (stays high) |

---

## Google Colab Setup

```python
!pip install diffusers transformers accelerate openai-clip reportlab -q

from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.run(["git", "clone", "https://github.com/waqarbutt58/BiasAudit-CAP6412.git"])
%cd BiasAudit-CAP6412

!python src/run_pipeline.py --n_per_prompt 5 --steps 20
```

---

## References

1. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022
2. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021
3. Chinchure et al., "OpenBias: Open-set Bias Detection in Text-to-Image Generative Models," CVPR 2024
4. SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image Generation, 2024
5. Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models, CVPR Workshop 2024
6. Poppi et al., "Safe-CLIP," ECCV 2024
7. Liu et al., "Visual Instruction Tuning (LLaVA)," NeurIPS 2023

---

*CAP6412 – Advanced Computer Vision | Spring 2025 | UCF CRCV*
