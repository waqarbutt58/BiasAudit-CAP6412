"""Full pipeline runner — called directly to avoid shell quoting issues."""
import sys, json
sys.path.insert(0, 'C:/Users/bsef0/Documents/Phd/SelfStudy/BiasAudit')

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('GPU:', torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU')

# ── Module 1: Image Generation ────────────────────────────────────────
print('\n' + '='*60)
print('MODULE 1: Image Generation (20 steps x 600 images)')
print('='*60)
from src.generate_images import generate_images
generate_images(prompts_csv='prompts.csv', output_dir='images',
                n_per_prompt=10, num_inference_steps=20, device=device)

# ── Module 2: Bias Detection ──────────────────────────────────────────
print('\n' + '='*60)
print('MODULE 2: Bias Detection')
print('='*60)
from src.bias_detector import BiasDetector
detector = BiasDetector(device=device)
df, summary = detector.analyse_all('images', 'prompts.csv')
df.to_csv('results/bias_results.csv', index=False)
with open('results/bias_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Saved {len(df)} bias records.')

# ── Module 3: Safety Filter ───────────────────────────────────────────
print('\n' + '='*60)
print('MODULE 3: Safety Filter')
print('='*60)
from src.safety_filter import SafetyFilter
sf = SafetyFilter(device=device)
df_s = sf.scan_directory('images', 'prompts.csv')
df_s.to_csv('results/safety_results.csv', index=False)
flagged = int(df_s['is_flagged'].sum())
print(f'Safety scan: {len(df_s)} images, {flagged} flagged NSFW.')

# ── Module 5: Report Generation ───────────────────────────────────────
print('\n' + '='*60)
print('MODULE 5: Report Generation')
print('='*60)
from src.report_generator import ReportGenerator
rg = ReportGenerator(
    bias_csv='results/bias_results.csv',
    safety_csv='results/safety_results.csv',
    bias_summary_json='results/bias_summary.json',
    output_pdf='results/audit_report.pdf',
    charts_dir='results/charts',
)
rg.build_pdf()

print('\n' + '='*60)
print('PIPELINE COMPLETE.')
print('='*60)
