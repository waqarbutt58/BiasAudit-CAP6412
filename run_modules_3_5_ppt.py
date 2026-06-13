"""Run Module 3 (Safety), Module 5 (Report), and Presentation builder."""
import sys, os, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

def run(label, args):
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    sys.stdout.flush()
    result = subprocess.run(args, cwd=BASE)
    if result.returncode != 0:
        print(f"ERROR: {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

run("MODULE 3: Safety Filter",
    [PY, "-u", os.path.join(BASE, "src", "safety_filter.py")])

run("MODULE 5: Report Generation",
    [PY, "-u", os.path.join(BASE, "src", "report_generator.py")])

run("PRESENTATION: Building slides",
    [PY, "-u", os.path.join(BASE, "slides", "build_presentation.py")])

print("\nAll done!")
