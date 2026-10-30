"""Run the whole pipeline in order, from the project root.

  python3 run_all.py            forecast -> explain -> reorder -> back-tests -> results.json -> app data (uses saved models)
  python3 run_all.py --retrain  first retrain v2 + range models (src/train.py), then the steps above

Needs raw/ and clean/ locally (they are never committed). Stops at the first failing step.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["src/forecast.py", "src/explain.py", "src/reorder.py", "src/backtest.py", "src/results.py", "src/build_app_data.py"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="retrain v2 and the prediction-range models first")
    args = ap.parse_args()
    steps = (["src/train.py"] if args.retrain else []) + STEPS
    if not args.retrain and not (ROOT / "models" / "lgbm_final_v2.txt").exists():
        sys.exit("models/lgbm_final_v2.txt is missing: run python3 run_all.py --retrain")
    for step in steps:
        print(f"\n=== {step} ===", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, step], cwd=ROOT, check=True)
        print(f"--- done in {time.time() - t0:.0f}s", flush=True)
    print("\nPipeline finished. Open the dashboard with: streamlit run app/app.py")
