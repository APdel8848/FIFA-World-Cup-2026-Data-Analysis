# FIFA World Cup 2026: Match and Team Goal Prediction

## Scope
Two regression tasks: match goal difference (104 observations) and team goals scored (208 observations). Each modelling dataset contains exactly eight explanatory variables. The tasks share four predictor concepts.

## Run
Python 3.11+ (tested with Python 3.14). From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
```

## Workflow
`run_all.py` → `scripts/run_pipeline.py` → match preparation → team preparation → model evaluation → sensitivity analyses → structural validation → source audit → club-mapping uncertainty. Preprocessing tasks are in `scripts/preprocessing/`.

## Folders
- `data/raw/`: source snapshots and league reference tables.
- `data/processed/`: intermediate and final datasets.
- `output/results/`: performance metrics, inferential statistics, sensitivity results and source checks.
- `output/figures/`: generated charts.
- `Group9_Objective2_Report.docx`: analysis report.

## Interpretation and provenance
The match-level rest-day difference is capped at ±7 days for the primary analysis; uncapped and alternative treatments are retained. The opponent Elo measure is computed locally from historical results and preceding tournament matches, not sourced from an official rating provider. The top-five-league player feature includes provisional non-top-five assignments for unresolved clubs. The June 11 FIFA rankings have been compared with an independent 48-team listing; 104 match scores and all player-to-club classifications have not been independently verified record by record. Source URLs and verification scope are recorded in `output/results/source_provenance_register.csv`.
