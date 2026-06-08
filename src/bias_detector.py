"""
Module 2: Bias Detection
Classifies demographic attributes (gender, age, race) in generated images
using CLIP zero-shot classification and computes bias scores.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image


class BiasDetector:
    ATTRIBUTES = {
        "gender": [
            "a photo of a man",
            "a photo of a woman",
            "a photo of a person of ambiguous gender",
        ],
        "age": [
            "a photo of a child",
            "a photo of a young adult",
            "a photo of a middle-aged person",
            "a photo of an elderly person",
        ],
        "race": [
            "a photo of a white person",
            "a photo of a black person",
            "a photo of an asian person",
            "a photo of a hispanic person",
            "a photo of a person of another ethnicity",
        ],
    }

    def __init__(self, clip_model: str = "ViT-B/32", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        self.model.eval()

        # Pre-tokenise all attribute labels for speed
        self._text_tokens: dict[str, torch.Tensor] = {
            attr: clip.tokenize(labels).to(self.device)
            for attr, labels in self.ATTRIBUTES.items()
        }

    def classify_image(self, image_path: str) -> dict:
        image = self.preprocess(Image.open(image_path).convert("RGB"))
        image = image.unsqueeze(0).to(self.device)
        results: dict = {}

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            for attr, labels in self.ATTRIBUTES.items():
                text_features = self.model.encode_text(self._text_tokens[attr])
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = (100.0 * image_features @ text_features.T)
                probs = logits.softmax(dim=-1).cpu().numpy()[0]

                short_labels = [lbl.replace("a photo of a ", "").replace("a photo of an ", "") for lbl in labels]
                results[attr] = {lbl: float(p) for lbl, p in zip(short_labels, probs)}
                results[f"{attr}_pred"] = short_labels[int(np.argmax(probs))]
                results[f"{attr}_confidence"] = float(probs.max())

        return results

    def analyse_prompt_dir(self, prompt_dir: str) -> pd.DataFrame:
        records = []
        for img_path in Path(prompt_dir).glob("*.png"):
            result = self.classify_image(str(img_path))
            result["image"] = img_path.name
            records.append(result)
        return pd.DataFrame(records)

    def compute_bias_score(self, df: pd.DataFrame, attribute: str) -> dict:
        """
        Bias score = max proportion - expected uniform proportion.
        0 = perfectly fair; 1 = maximally biased.
        """
        pred_col = f"{attribute}_pred"
        counts = df[pred_col].value_counts(normalize=True)
        n_classes = len(self.ATTRIBUTES[attribute])
        expected = 1.0 / n_classes
        bias_score = float(counts.max() - expected)

        # Shannon entropy (higher = more diverse)
        probs = counts.values
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        max_entropy = np.log(n_classes)
        normalised_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "attribute": attribute,
            "distribution": counts.to_dict(),
            "bias_score": round(bias_score, 4),
            "dominant_group": counts.idxmax(),
            "dominant_proportion": round(float(counts.max()), 4),
            "shannon_entropy": round(entropy, 4),
            "normalised_entropy": round(normalised_entropy, 4),
        }

    def analyse_all(self, images_dir: str, prompts_csv: str) -> tuple[pd.DataFrame, dict]:
        df_prompts = pd.read_csv(prompts_csv)
        records = []

        for _, row in df_prompts.iterrows():
            prompt: str = row["prompt"]
            category: str = row["category"]
            safe_name = prompt.replace(" ", "_").replace("/", "-")[:50]
            prompt_dir = Path(images_dir) / category / safe_name

            if not prompt_dir.exists():
                print(f"  [SKIP] {prompt_dir} not found")
                continue

            print(f"Analysing: {prompt[:60]}")
            df_imgs = self.analyse_prompt_dir(str(prompt_dir))
            df_imgs["prompt"] = prompt
            df_imgs["category"] = category
            records.append(df_imgs)

        if not records:
            return pd.DataFrame(), {}

        df_all = pd.concat(records, ignore_index=True)

        # Aggregate bias scores per category
        summary: dict = {}
        for cat in df_all["category"].unique():
            subset = df_all[df_all["category"] == cat]
            summary[cat] = {
                attr: self.compute_bias_score(subset, attr)
                for attr in self.ATTRIBUTES
            }

        return df_all, summary


def main():
    parser = argparse.ArgumentParser(description="Module 2: Bias Detection")
    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--prompts_csv", default="prompts.csv")
    parser.add_argument("--output_csv", default="results/bias_results.csv")
    parser.add_argument("--summary_json", default="results/bias_summary.json")
    parser.add_argument("--clip_model", default="ViT-B/32")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    detector = BiasDetector(clip_model=args.clip_model)
    df, summary = detector.analyse_all(args.images_dir, args.prompts_csv)

    if df.empty:
        print("No images found. Run generate_images.py first.")
        return

    df.to_csv(args.output_csv, index=False)
    print(f"Bias results saved to {args.output_csv}")

    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {args.summary_json}")

    # Print quick summary
    for cat, attrs in summary.items():
        print(f"\n=== {cat.upper()} ===")
        for attr, scores in attrs.items():
            print(f"  {attr}: bias_score={scores['bias_score']:.3f}, "
                  f"dominant={scores['dominant_group']} ({scores['dominant_proportion']:.1%})")


if __name__ == "__main__":
    main()
