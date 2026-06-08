"""
Module 3: Safety Filter (SAFREE-inspired)
Detects unsafe/NSFW content at prompt and image levels.
Supports both keyword-based prompt filtering and model-based image classification.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline


# ── Unsafe keyword taxonomy ────────────────────────────────────────────────────
UNSAFE_KEYWORDS: dict[str, list[str]] = {
    "nudity": [
        "nude", "naked", "nsfw", "explicit", "topless", "undressed",
        "pornographic", "erotic", "sexual",
    ],
    "violence": [
        "gore", "blood", "weapon", "murder", "kill", "attack",
        "torture", "corpse", "decapitate", "massacre",
    ],
    "hate": [
        "hate speech", "racial slur", "discriminatory", "racist",
        "sexist", "homophobic", "terrorist", "extremist",
    ],
    "self_harm": [
        "suicide", "self-harm", "cutting", "overdose",
    ],
}


class SafetyFilter:
    """
    Combined prompt-level keyword filter and image-level NSFW classifier.
    SAFREE-inspired: no fine-tuning required; works at inference time.
    """

    def __init__(
        self,
        nsfw_model: str = "Falconsai/nsfw_image_detection",
        nsfw_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.nsfw_threshold = nsfw_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_index = 0 if self.device == "cuda" else -1

        print(f"Loading NSFW detector: {nsfw_model}")
        self.nsfw_detector = hf_pipeline(
            "image-classification",
            model=nsfw_model,
            device=device_index,
        )

    # ── Prompt-level ────────────────────────────────────────────────────────────

    def check_prompt(self, prompt: str) -> dict:
        prompt_lower = prompt.lower()
        flags: dict[str, list[str]] = {}

        for category, keywords in UNSAFE_KEYWORDS.items():
            matched = [kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", prompt_lower)]
            if matched:
                flags[category] = matched

        return {
            "prompt": prompt,
            "is_unsafe": len(flags) > 0,
            "flags": flags,
            "safe_prompt": self._sanitize_prompt(prompt) if flags else prompt,
        }

    def _sanitize_prompt(self, prompt: str) -> str:
        safe = prompt
        for keywords in UNSAFE_KEYWORDS.values():
            for kw in keywords:
                safe = re.sub(r"\b" + re.escape(kw) + r"\b", "[removed]", safe, flags=re.IGNORECASE)
        return safe

    # ── Image-level ─────────────────────────────────────────────────────────────

    def check_image(self, image: Image.Image) -> dict:
        results = self.nsfw_detector(image)
        nsfw_score = next(
            (r["score"] for r in results if r["label"].lower() in ("nsfw", "unsafe")), 0.0
        )
        return {
            "nsfw_score": round(nsfw_score, 4),
            "is_flagged": nsfw_score > self.nsfw_threshold,
            "label": "UNSAFE" if nsfw_score > self.nsfw_threshold else "SAFE",
        }

    def check_image_path(self, path: str) -> dict:
        img = Image.open(path).convert("RGB")
        result = self.check_image(img)
        result["image_path"] = path
        return result

    # ── Batch processing ────────────────────────────────────────────────────────

    def scan_directory(self, images_dir: str, prompts_csv: str) -> pd.DataFrame:
        df_prompts = pd.read_csv(prompts_csv)
        records = []

        for _, row in df_prompts.iterrows():
            prompt: str = row["prompt"]
            category: str = row["category"]
            safe_name = prompt.replace(" ", "_").replace("/", "-")[:50]
            prompt_dir = Path(images_dir) / category / safe_name

            prompt_check = self.check_prompt(prompt)

            if not prompt_dir.exists():
                continue

            for img_path in sorted(prompt_dir.glob("*.png")):
                img_result = self.check_image_path(str(img_path))
                records.append({
                    "category": category,
                    "prompt": prompt,
                    "image": img_path.name,
                    "image_path": str(img_path),
                    "prompt_is_unsafe": prompt_check["is_unsafe"],
                    "prompt_flags": str(prompt_check["flags"]),
                    **img_result,
                })
            print(f"Scanned: {prompt[:60]}")

        return pd.DataFrame(records)

    # ── Safe generation wrapper ─────────────────────────────────────────────────

    def generate_safe(self, pipe, prompt: str, **kwargs) -> dict:
        """Generate an image only if the prompt passes the safety check."""
        prompt_check = self.check_prompt(prompt)
        if prompt_check["is_unsafe"]:
            return {
                "blocked": True,
                "reason": "unsafe_prompt",
                "flags": prompt_check["flags"],
                "image": None,
                "nsfw_score": None,
            }

        image = pipe(prompt, **kwargs).images[0]
        image_check = self.check_image(image)

        return {
            "blocked": image_check["is_flagged"],
            "reason": "unsafe_image" if image_check["is_flagged"] else None,
            "flags": {},
            "nsfw_score": image_check["nsfw_score"],
            "image": None if image_check["is_flagged"] else image,
        }


def main():
    parser = argparse.ArgumentParser(description="Module 3: Safety Filter")
    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--prompts_csv", default="prompts.csv")
    parser.add_argument("--output_csv", default="results/safety_results.csv")
    parser.add_argument("--nsfw_threshold", type=float, default=0.5)
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    sf = SafetyFilter(nsfw_threshold=args.nsfw_threshold)
    df = sf.scan_directory(args.images_dir, args.prompts_csv)

    if df.empty:
        print("No images found. Run generate_images.py first.")
        return

    df.to_csv(args.output_csv, index=False)
    n_unsafe = df["is_flagged"].sum()
    print(f"\nSafety scan complete.")
    print(f"  Total images : {len(df)}")
    print(f"  Flagged NSFW : {n_unsafe} ({100*n_unsafe/len(df):.1f}%)")
    print(f"  Results saved: {args.output_csv}")


if __name__ == "__main__":
    main()
