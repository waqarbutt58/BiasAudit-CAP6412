"""
Module 5: Report Generation
Auto-generates a structured PDF audit report with bias charts,
safety statistics, and concept erasure comparison.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, PageBreak,
)


# ── Chart helpers ────────────────────────────────────────────────────────────────

PALETTE = {
    "blue":  "#3949AB",
    "pink":  "#E91E63",
    "grey":  "#9E9E9E",
    "green": "#43A047",
    "orange": "#FB8C00",
}

ATTR_COLORS = {
    "gender": [PALETTE["blue"], PALETTE["pink"], PALETTE["grey"]],
    "age":    ["#5C6BC0", "#42A5F5", "#26C6DA", "#AB47BC"],
    "race":   ["#EF5350", "#26A69A", "#FFA726", "#8D6E63", "#78909C"],
}


def _bar_chart(labels, values, title, color, output_path, figsize=(6, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, values, color=color, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    ax.set_ylabel("Proportion", fontsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=15)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def _pie_chart(labels, values, title, output_path, figsize=(5, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    wedge_props = {"linewidth": 1, "edgecolor": "white"}
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140,
           wedgeprops=wedge_props)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def _bias_heatmap(summary: dict, attribute: str, output_path: str):
    categories = list(summary.keys())
    dominant_proportions = [
        summary[cat][attribute]["dominant_proportion"] for cat in categories
    ]
    bias_scores = [
        summary[cat][attribute]["bias_score"] for cat in categories
    ]

    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, dominant_proportions, width, label="Dominant group proportion",
           color=PALETTE["blue"], alpha=0.85)
    ax.bar(x + width / 2, bias_scores, width, label="Bias score",
           color=PALETTE["pink"], alpha=0.85)
    ax.axhline(y=0.33, color="grey", linestyle="--", linewidth=0.8, label="Fair baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(f"{attribute.capitalize()} Bias Across Categories", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# ── Main report class ────────────────────────────────────────────────────────────

class ReportGenerator:
    def __init__(
        self,
        bias_csv: str,
        safety_csv: str,
        bias_summary_json: str,
        output_pdf: str,
        charts_dir: str = "results/charts",
        erasure_dir: Optional[str] = None,
    ):
        self.bias_df    = pd.read_csv(bias_csv) if Path(bias_csv).exists() else pd.DataFrame()
        self.safety_df  = pd.read_csv(safety_csv) if Path(safety_csv).exists() else pd.DataFrame()
        self.summary    = json.load(open(bias_summary_json)) if Path(bias_summary_json).exists() else {}
        self.output     = output_pdf
        self.charts_dir = charts_dir
        self.erasure_dir = erasure_dir
        Path(charts_dir).mkdir(parents=True, exist_ok=True)

    # ── Chart generation ─────────────────────────────────────────────────────────

    def _make_attribute_charts(self, category: str, attribute: str) -> Optional[str]:
        if self.bias_df.empty:
            return None
        subset = self.bias_df[self.bias_df["category"] == category]
        if subset.empty:
            return None

        pred_col = f"{attribute}_pred"
        counts = subset[pred_col].value_counts(normalize=True)
        labels = list(counts.index)
        values = list(counts.values)
        pal = ATTR_COLORS.get(attribute, [PALETTE["blue"]] * len(labels))
        colors_for_bars = [pal[i % len(pal)] for i in range(len(labels))]

        out = f"{self.charts_dir}/{attribute}_{category}.png"
        return _bar_chart(
            labels, values,
            title=f"{attribute.capitalize()} Distribution — {category.capitalize()}",
            color=colors_for_bars,
            output_path=out,
        )

    def _make_heatmaps(self) -> dict[str, str]:
        paths = {}
        for attr in ["gender", "race", "age"]:
            if self.summary:
                p = f"{self.charts_dir}/heatmap_{attr}.png"
                _bias_heatmap(self.summary, attr, p)
                paths[attr] = p
        return paths

    def _make_safety_chart(self) -> Optional[str]:
        if self.safety_df.empty:
            return None
        total = len(self.safety_df)
        unsafe = int(self.safety_df["is_flagged"].sum())
        safe = total - unsafe
        out = f"{self.charts_dir}/safety_overview.png"
        return _pie_chart(
            ["Safe", "Flagged NSFW"],
            [safe, unsafe],
            "Image Safety Overview",
            out,
        )

    # ── PDF construction ─────────────────────────────────────────────────────────

    def build_pdf(self) -> str:
        doc = SimpleDocTemplate(
            self.output,
            pagesize=letter,
            leftMargin=0.8 * inch,
            rightMargin=0.8 * inch,
            topMargin=0.8 * inch,
            bottomMargin=0.8 * inch,
        )
        styles = getSampleStyleSheet()
        H1 = styles["Title"]
        H2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=4)
        H3 = ParagraphStyle("H3", parent=styles["Heading3"], spaceBefore=8, spaceAfter=2)
        NORMAL = styles["Normal"]
        story = []

        # ── Cover ──────────────────────────────────────────────────────────────
        story.append(Paragraph("Bias & Safety Audit Report", H1))
        story.append(Paragraph("Text-to-Image Model: Stable Diffusion v1.5", NORMAL))
        story.append(Paragraph("Project: CAP6412 — Advanced Computer Vision, UCF CRCV", NORMAL))
        story.append(Spacer(1, 0.2 * inch))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 0.15 * inch))

        # ── Executive summary ──────────────────────────────────────────────────
        story.append(Paragraph("1. Executive Summary", H2))
        total_images = len(self.bias_df) if not self.bias_df.empty else 0
        unsafe_count = int(self.safety_df["is_flagged"].sum()) if not self.safety_df.empty else 0
        total_safety = len(self.safety_df) if not self.safety_df.empty else 0
        nsfw_pct = 100.0 * unsafe_count / total_safety if total_safety > 0 else 0

        summary_rows = [
            ["Metric", "Value"],
            ["Total images analysed (bias)", str(total_images)],
            ["Total images scanned (safety)", str(total_safety)],
            ["Images flagged NSFW", f"{unsafe_count} ({nsfw_pct:.1f}%)"],
            ["Prompt categories", ", ".join(self.bias_df["category"].unique().tolist()) if not self.bias_df.empty else "N/A"],
        ]
        t = Table(summary_rows, colWidths=[3 * inch, 3.5 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3949AB")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("PADDING",    (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2 * inch))

        # ── Bias analysis ──────────────────────────────────────────────────────
        story.append(Paragraph("2. Bias Analysis", H2))
        story.append(Paragraph(
            "Images were classified using CLIP zero-shot classification across three "
            "demographic dimensions: gender, age, and apparent race. "
            "Bias score = dominant group proportion − expected uniform proportion (0=fair, 1=biased).",
            NORMAL,
        ))
        story.append(Spacer(1, 0.1 * inch))

        # Heatmaps
        heatmap_paths = self._make_heatmaps()
        for attr, path in heatmap_paths.items():
            if Path(path).exists():
                story.append(RLImage(path, width=5.5 * inch, height=2.4 * inch))
                story.append(Spacer(1, 0.05 * inch))

        # Per-category breakdown
        if not self.bias_df.empty:
            story.append(Paragraph("2.1 Per-Category Distributions", H3))
            for category in self.bias_df["category"].unique():
                story.append(Paragraph(f"Category: {category.upper()}", H3))
                for attr in ["gender", "race", "age"]:
                    chart_path = self._make_attribute_charts(category, attr)
                    if chart_path and Path(chart_path).exists():
                        story.append(RLImage(chart_path, width=4.5 * inch, height=2.5 * inch))
                story.append(Spacer(1, 0.1 * inch))

        # Bias score table
        if self.summary:
            story.append(Paragraph("2.2 Bias Score Summary", H3))
            header = ["Category", "Attribute", "Bias Score", "Dominant Group", "Entropy (norm.)"]
            rows = [header]
            for cat, attrs in self.summary.items():
                for attr, scores in attrs.items():
                    rows.append([
                        cat.capitalize(),
                        attr.capitalize(),
                        f"{scores['bias_score']:.3f}",
                        scores["dominant_group"].capitalize(),
                        f"{scores['normalised_entropy']:.3f}",
                    ])
            t2 = Table(rows, colWidths=[1.2*inch, 1.1*inch, 1.1*inch, 1.8*inch, 1.3*inch])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3949AB")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
                ("FONTSIZE",   (0, 0), (-1, -1), 8),
                ("PADDING",    (0, 0), (-1, -1), 4),
            ]))
            story.append(t2)

        story.append(PageBreak())

        # ── Safety analysis ────────────────────────────────────────────────────
        story.append(Paragraph("3. Safety Analysis", H2))
        safety_chart = self._make_safety_chart()
        if safety_chart and Path(safety_chart).exists():
            story.append(RLImage(safety_chart, width=3.5 * inch, height=2.8 * inch))

        if not self.safety_df.empty:
            prompt_blocked = int(self.safety_df["prompt_is_unsafe"].sum()) if "prompt_is_unsafe" in self.safety_df.columns else 0
            story.append(Paragraph(
                f"Prompts with keyword flags: {prompt_blocked} | "
                f"Images flagged NSFW: {unsafe_count} / {total_safety} ({nsfw_pct:.1f}%)",
                NORMAL,
            ))

        # ── Concept erasure ────────────────────────────────────────────────────
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("4. Concept Erasure", H2))
        story.append(Paragraph(
            "Cross-attention (K/V) weights were fine-tuned using an attention re-steering loss "
            "to suppress a target concept while preserving unrelated generation capability. "
            "Effectiveness is measured via CLIP cosine similarity between generated images and the erased concept text.",
            NORMAL,
        ))

        if self.erasure_dir:
            before_dir = Path(self.erasure_dir) / "before"
            after_dir  = Path(self.erasure_dir) / "after"
            before_imgs = sorted(before_dir.glob("*.png"))[:3] if before_dir.exists() else []
            after_imgs  = sorted(after_dir.glob("*.png"))[:3]  if after_dir.exists()  else []

            if before_imgs or after_imgs:
                story.append(Spacer(1, 0.1 * inch))
                story.append(Paragraph("Before erasure:", H3))
                for p in before_imgs:
                    story.append(RLImage(str(p), width=1.6*inch, height=1.6*inch))
                story.append(Paragraph("After erasure:", H3))
                for p in after_imgs:
                    story.append(RLImage(str(p), width=1.6*inch, height=1.6*inch))

        # ── Methodology ────────────────────────────────────────────────────────
        story.append(PageBreak())
        story.append(Paragraph("5. Methodology", H2))
        methodology_text = (
            "<b>Image Generation:</b> Stable Diffusion v1.5 was used to generate 10 images per "
            "prompt across 60 structured prompts (occupation, activity, neutral categories) "
            "using deterministic seeds for reproducibility.<br/><br/>"
            "<b>Bias Detection:</b> CLIP ViT-B/32 zero-shot classification was applied to each "
            "generated image. Demographic attributes (gender, age, apparent race) were inferred "
            "by comparing image embeddings against textual attribute descriptions using cosine "
            "similarity. Bias score = dominant proportion − (1 / n_classes).<br/><br/>"
            "<b>Safety Filtering:</b> A two-stage approach: (1) keyword-based prompt scanning "
            "using a curated taxonomy of unsafe terms; (2) image-level NSFW classification "
            "using Falconsai/nsfw_image_detection. Threshold: 0.5.<br/><br/>"
            "<b>Concept Erasure:</b> Cross-attention K/V projection weights were fine-tuned via "
            "an attention re-steering MSE loss that pushes concept representations toward a safe "
            "anchor embedding. Only ~2% of U-Net parameters are modified (Forget-Me-Not inspired)."
        )
        story.append(Paragraph(methodology_text, NORMAL))

        # ── References ─────────────────────────────────────────────────────────
        story.append(PageBreak())
        story.append(Paragraph("6. References", H2))
        refs = [
            'Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022.',
            'Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021.',
            'Chinchure et al., "OpenBias: Open-set Bias Detection in Text-to-Image Generative Models," CVPR 2024.',
            "SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image Generation, 2024.",
            "Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models, CVPR Workshop 2024.",
            "SocialCounterfactuals, CVPR 2024.",
            'Poppi et al., "Safe-CLIP: Removing NSFW Concepts from Vision-and-Language Models," ECCV 2024.',
            'Liu et al., "Visual Instruction Tuning (LLaVA)," NeurIPS 2023.',
            'Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)," ICCV 2023.',
        ]
        for i, ref in enumerate(refs, 1):
            story.append(Paragraph(f"[{i}] {ref}", NORMAL))
            story.append(Spacer(1, 0.05 * inch))

        doc.build(story)
        print(f"\nAudit report saved: {self.output}")
        return self.output


def main():
    parser = argparse.ArgumentParser(description="Module 5: Report Generation")
    parser.add_argument("--bias_csv",     default="results/bias_results.csv")
    parser.add_argument("--safety_csv",   default="results/safety_results.csv")
    parser.add_argument("--summary_json", default="results/bias_summary.json")
    parser.add_argument("--output_pdf",   default="results/audit_report.pdf")
    parser.add_argument("--charts_dir",   default="results/charts")
    parser.add_argument("--erasure_dir",  default=None)
    args = parser.parse_args()

    rg = ReportGenerator(
        bias_csv=args.bias_csv,
        safety_csv=args.safety_csv,
        bias_summary_json=args.summary_json,
        output_pdf=args.output_pdf,
        charts_dir=args.charts_dir,
        erasure_dir=args.erasure_dir,
    )
    rg.build_pdf()


if __name__ == "__main__":
    main()
