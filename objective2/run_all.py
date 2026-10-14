"""Convenience entry point for the analysis pipeline."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py")], cwd=ROOT, check=True)
