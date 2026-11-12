"""Runs the whole pipeline in order. Each step reads the previous steps' outputs from outputs/ and writes its own.
Usage: python src/run_all.py [--from p04_pressure] [--skip-external] [--skip-deck]"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    "p00_audit", "p00x_external", "p01_state", "p02_features", "p03_context_models", "p03b_model_zoo",
    "p04_pressure", "p05_value", "p06_ratings", "p07_unsupervised", "p08_selection", "p09_experiments",
    "p10_rank", "p11_robustness", "p14_external", "p14b_family_check", "p12_figures", "p12_report",
    "p13_notebook", "p16_slides_pdf",
]
# steps that need the Cricsheet downloads, or read tables the outside-match test writes
EXTERNAL = {"p00x_external", "p14_external", "p14b_family_check", "p12_figures", "p12_report", "p13_notebook", "p16_slides_pdf"}

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", default=STEPS[0], choices=STEPS)
ap.add_argument("--skip-external", action="store_true", help="rankings only: skip the outside-match test and the reports built on it")
ap.add_argument("--skip-deck", action="store_true", help="skip the slide deck (needs local image assets)")
a = ap.parse_args()

for step in STEPS[STEPS.index(a.start):]:
    if (a.skip_external and step in EXTERNAL) or (a.skip_deck and step == "p16_slides_pdf"):
        print(f"-- skip {step}")
        continue
    t = time.time()
    print(f"== {step}", flush=True)
    # p00x reads downloaded zips: run it isolated (-I) so nothing is imported from the working directory
    cmd = [sys.executable] + (["-I"] if step == "p00x_external" else []) + [os.path.join(HERE, step + ".py")]
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode:
        sys.exit(f"{step} failed (exit {r.returncode})")
    print(f"   done in {time.time() - t:.0f}s", flush=True)
