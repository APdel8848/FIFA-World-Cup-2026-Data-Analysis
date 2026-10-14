"""Run all data preparation, modelling, sensitivity and validation stages."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("Prepare match-level features", "prepare_match_features.py"),
    ("Prepare team-level features", "prepare_team_features.py"),
    ("Fit and evaluate regression models", "fit_and_evaluate_models.py"),
    ("Assess rest-day sensitivity", "analyse_rest_day_sensitivity.py"),
    ("Compare rest-day caps", "compare_rest_day_caps.py"),
    ("Validate datasets and outputs", "validate_datasets.py"),
    ("Audit external source comparisons", "audit_external_sources.py"),
    ("Assess club-mapping uncertainty", "analyse_club_uncertainty.py"),
]
for label, script in STAGES:
    print(f"\n{'=' * 64}\n{label.upper()}\n{'=' * 64}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
print("\nPipeline completed successfully. All analyses and checks finished.")
