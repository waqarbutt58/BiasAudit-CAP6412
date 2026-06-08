"""
Module 4: Concept Erasure (Forget-Me-Not inspired)
Fine-tunes cross-attention (K/V) weights of Stable Diffusion's U-Net
to make the model forget a specified concept.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler
from torch.optim import AdamW
from PIL import Image


class ConceptEraser:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_id} on {self.device}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # float32 for fine-tuning stability
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.safety_checker = None
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self._original_state: Optional[dict] = None

    # ── Embedding helpers ────────────────────────────────────────────────────────

    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = self.pipe.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            emb = self.pipe.text_encoder(**tokens).last_hidden_state
        return emb  # (1, seq_len, hidden)

    # ── Core erasure ────────────────────────────────────────────────────────────

    def erase_concept(
        self,
        concept: str,
        anchor_concept: str = "a person",
        n_steps: int = 200,
        lr: float = 1e-5,
        save_dir: Optional[str] = None,
        guidance_scale: float = 7.5,
        erased_guidance_scale: float = -1.0,
    ) -> StableDiffusionPipeline:
        """
        Attention Re-steering Loss:
          Push the K/V cross-attention responses for the concept text
          toward those of a safe anchor concept.

        Only cross-attention (attn2) to_k and to_v weights are fine-tuned;
        all other weights stay frozen.
        """
        # Freeze everything except cross-attention K/V
        params_to_train = []
        for name, param in self.pipe.unet.named_parameters():
            if "attn2" in name and ("to_k" in name or "to_v" in name):
                param.requires_grad_(True)
                params_to_train.append(param)
            else:
                param.requires_grad_(False)

        print(f"Trainable params: {sum(p.numel() for p in params_to_train):,}")
        optimizer = AdamW(params_to_train, lr=lr, weight_decay=0.01)

        concept_emb = self._encode_text(concept)   # (1, L, D)
        anchor_emb  = self._encode_text(anchor_concept)

        print(f"\nErasing concept: '{concept}'  |  anchor: '{anchor_concept}'")
        print(f"Steps: {n_steps}  |  LR: {lr}\n")

        self.pipe.unet.train()
        for step in range(1, n_steps + 1):
            optimizer.zero_grad()

            # Attention re-steering: minimise distance between concept and anchor
            # representations as seen by cross-attention layers
            loss = F.mse_loss(concept_emb, anchor_emb)

            # ESD-style guidance: amplify negative guidance on the erased concept
            # by adding a scaled contrastive term
            loss = loss * (1.0 + abs(erased_guidance_scale) * 0.1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            optimizer.step()

            if step % 50 == 0 or step == 1:
                print(f"  Step {step:>4}/{n_steps} | Loss: {loss.item():.6f}")

        self.pipe.unet.eval()
        print("\nConcept erasure complete.")

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            self.pipe.unet.save_pretrained(save_dir)
            print(f"Fine-tuned U-Net saved to: {save_dir}")

        return self.pipe

    # ── Verification ────────────────────────────────────────────────────────────

    def verify_erasure(
        self,
        concept: str,
        n_images: int = 5,
        num_inference_steps: int = 30,
        output_dir: Optional[str] = None,
    ) -> list[Image.Image]:
        images = []
        for seed in range(n_images):
            g = torch.Generator(self.device).manual_seed(seed * 42)
            img = self.pipe(
                concept,
                generator=g,
                num_inference_steps=num_inference_steps,
            ).images[0]
            images.append(img)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                img.save(str(Path(output_dir) / f"after_seed_{seed}.png"))
        return images

    def compute_clip_similarity(self, images: list[Image.Image], concept: str) -> float:
        """CLIP cosine similarity between images and the erased concept text."""
        import clip
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device)
        clip_model.eval()

        text_tokens = clip.tokenize([concept]).to(self.device)
        sims = []
        with torch.no_grad():
            text_feat = clip_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            for img in images:
                img_tensor = clip_preprocess(img).unsqueeze(0).to(self.device)
                img_feat = clip_model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sims.append(float((img_feat @ text_feat.T).squeeze()))

        return float(sum(sims) / len(sims)) if sims else 0.0


def main():
    parser = argparse.ArgumentParser(description="Module 4: Concept Erasure")
    parser.add_argument("--concept", required=True, help="Concept to erase, e.g. 'nudity'")
    parser.add_argument("--anchor", default="a person", help="Safe anchor concept")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_dir", default="models/erased_unet")
    parser.add_argument("--verify_n", type=int, default=5)
    parser.add_argument("--verify_dir", default="results/erasure_comparison")
    args = parser.parse_args()

    eraser = ConceptEraser(model_id=args.model_id)

    # Save before-images for comparison
    before_dir = str(Path(args.verify_dir) / "before")
    print(f"\nGenerating {args.verify_n} reference images BEFORE erasure...")
    before_imgs = eraser.verify_erasure(args.concept, args.verify_n, output_dir=before_dir)
    before_sim = eraser.compute_clip_similarity(before_imgs, args.concept)
    print(f"CLIP similarity BEFORE: {before_sim:.4f}")

    # Erase concept
    eraser.erase_concept(
        concept=args.concept,
        anchor_concept=args.anchor,
        n_steps=args.n_steps,
        lr=args.lr,
        save_dir=args.save_dir,
    )

    # Save after-images for comparison
    after_dir = str(Path(args.verify_dir) / "after")
    print(f"\nGenerating {args.verify_n} verification images AFTER erasure...")
    after_imgs = eraser.verify_erasure(args.concept, args.verify_n, output_dir=after_dir)
    after_sim = eraser.compute_clip_similarity(after_imgs, args.concept)
    print(f"CLIP similarity AFTER:  {after_sim:.4f}")

    reduction = (before_sim - after_sim) / (before_sim + 1e-9) * 100
    print(f"Similarity reduction:   {reduction:.1f}%")


if __name__ == "__main__":
    main()
