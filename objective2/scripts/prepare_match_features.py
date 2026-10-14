"""Rebuild match-level features from the raw tournament and historical files."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    ("Clean tournament matches", "clean_world_cup_matches.py"),
    ("Standardise historical teams", "standardise_historical_teams.py"),
    ("Calculate historical form", "calculate_historical_form.py"),
    ("Calculate World Cup experience", "calculate_world_cup_experience.py"),
    ("Calculate rest intervals", "calculate_rest_days.py"),
    ("Build match-level dataset", "build_match_dataset.py"),
]
for label, script in TASKS:
    print(f"  {label} ...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "preprocessing" / script)], cwd=ROOT, check=True)
print("Match-level preprocessing completed.")
