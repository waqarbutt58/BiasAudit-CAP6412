"""
End-to-end pipeline runner.
Runs all 5 modules in sequence and produces the final audit report.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --skip_generation   # if images already exist
    python src/run_pipeline.py --erase "nudity"    # include concept erasure step
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="BiasAudit end-to-end pipeline")
    parser.add_argument("--prompts_csv",  default="prompts.csv")
    parser.add_argument("--images_dir",   default="images")
    parser.add_argument("--results_dir",  default="results")
    parser.add_argument("--model_id",     default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--n_per_prompt", type=int, default=10)
    parser.add_argument("--steps",        type=int, default=50)
    parser.add_argument("--device",       default=None)
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip Module 1 (images already generated)")
    parser.add_argument("--skip_safety",     action="store_true")
    parser.add_argument("--skip_bias",       action="store_true")
    parser.add_argument("--erase",           default=None,
                        help="Concept to erase (Module 4), e.g. 'nudity'")
    parser.add_argument("--erase_steps",     type=int, default=200)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.results_dir).mkdir(exist_ok=True)
    Path("results/charts").mkdir(parents=True, exist_ok=True)

    # ── Module 1: Image Generation ────────────────────────────────────────────
    if not args.skip_generation:
        print("\n" + "="*60)
        print("MODULE 1: Image Generation")
        print("="*60)
        from src.generate_images import generate_images
        generate_images(
            prompts_csv=args.prompts_csv,
            output_dir=args.images_dir,
            model_id=args.model_id,
            n_per_prompt=args.n_per_prompt,
            num_inference_steps=args.steps,
            device=device,
        )
    else:
        print("\n[SKIP] Module 1 — using existing images.")

    # ── Module 2: Bias Detection ──────────────────────────────────────────────
    if not args.skip_bias:
        print("\n" + "="*60)
        print("MODULE 2: Bias Detection")
        print("="*60)
        from src.bias_detector import BiasDetector
        import json
        detector = BiasDetector(device=device)
        df, summary = detector.analyse_all(args.images_dir, args.prompts_csv)
        if not df.empty:
            df.to_csv(f"{args.results_dir}/bias_results.csv", index=False)
            with open(f"{args.results_dir}/bias_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Bias results: {args.results_dir}/bias_results.csv")
    else:
        print("\n[SKIP] Module 2.")

    # ── Module 3: Safety Filter ───────────────────────────────────────────────
    if not args.skip_safety:
        print("\n" + "="*60)
        print("MODULE 3: Safety Filter")
        print("="*60)
        from src.safety_filter import SafetyFilter
        sf = SafetyFilter(device=device)
        df_safety = sf.scan_directory(args.images_dir, args.prompts_csv)
        if not df_safety.empty:
            df_safety.to_csv(f"{args.results_dir}/safety_results.csv", index=False)
            print(f"Safety results: {args.results_dir}/safety_results.csv")
    else:
        print("\n[SKIP] Module 3.")

    # ── Module 4: Concept Erasure (optional) ──────────────────────────────────
    erasure_dir = None
    if args.erase:
        print("\n" + "="*60)
        print(f"MODULE 4: Concept Erasure — '{args.erase}'")
        print("="*60)
        from src.concept_erasure import ConceptEraser
        erasure_dir = f"{args.results_dir}/erasure_comparison"
        eraser = ConceptEraser(model_id=args.model_id, device=device)
        eraser.erase_concept(
            concept=args.erase,
            n_steps=args.erase_steps,
            save_dir="models/erased_unet",
        )
        eraser.verify_erasure(args.erase, n_images=5, output_dir=f"{erasure_dir}/after")
    else:
        print("\n[SKIP] Module 4 — no --erase concept specified.")

    # ── Module 5: Report Generation ───────────────────────────────────────────
    print("\n" + "="*60)
    print("MODULE 5: Report Generation")
    print("="*60)
    from src.report_generator import ReportGenerator
    rg = ReportGenerator(
        bias_csv=f"{args.results_dir}/bias_results.csv",
        safety_csv=f"{args.results_dir}/safety_results.csv",
        bias_summary_json=f"{args.results_dir}/bias_summary.json",
        output_pdf=f"{args.results_dir}/audit_report.pdf",
        charts_dir=f"{args.results_dir}/charts",
        erasure_dir=erasure_dir,
    )
    rg.build_pdf()

    print("\n" + "="*60)
    print("Pipeline complete.")
    print(f"  Audit report : {args.results_dir}/audit_report.pdf")
    print("="*60)


if __name__ == "__main__":
    main()
