"""
Module 1: Image Generation
Generates images from prompts using Stable Diffusion.
"""

import os
import argparse
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path


def load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5",
                  device: str = "cuda") -> StableDiffusionPipeline:
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.safety_checker = None  # disable built-in filter; we use our own Module 3
    pipe.enable_attention_slicing()
    # Warmup: compile CUDA kernels so all subsequent images run at full speed
    print("Warming up CUDA kernels...")
    _g = torch.Generator(device).manual_seed(0)
    pipe("warmup", generator=_g, num_inference_steps=3)
    print("Warmup done.")
    return pipe


def generate_images(
    prompts_csv: str,
    output_dir: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    n_per_prompt: int = 10,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    device: str = "cuda",
) -> None:
    pipe = load_pipeline(model_id, device)
    df = pd.read_csv(prompts_csv)
    total = len(df) * n_per_prompt
    generated = 0

    for _, row in df.iterrows():
        prompt: str = row["prompt"]
        category: str = row["category"]
        safe_name = prompt.replace(" ", "_").replace("/", "-")[:50]
        prompt_dir = Path(output_dir) / category / safe_name
        prompt_dir.mkdir(parents=True, exist_ok=True)

        for seed in range(n_per_prompt):
            out_path = prompt_dir / f"seed_{seed}.png"
            if out_path.exists():
                generated += 1
                continue

            generator = torch.Generator(device).manual_seed(seed * 42)
            image = pipe(
                prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            ).images[0]
            image.save(str(out_path))
            generated += 1

        print(f"[{generated}/{total}] {category} | {prompt[:60]}")

    print(f"\nDone. {generated} images saved to '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Module 1: Image Generation")
    parser.add_argument("--prompts_csv", default="prompts.csv")
    parser.add_argument("--output_dir", default="images")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--n_per_prompt", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    generate_images(
        prompts_csv=args.prompts_csv,
        output_dir=args.output_dir,
        model_id=args.model_id,
        n_per_prompt=args.n_per_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
    )


if __name__ == "__main__":
    main()
