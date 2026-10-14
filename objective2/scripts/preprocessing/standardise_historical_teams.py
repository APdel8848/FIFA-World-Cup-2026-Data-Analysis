
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Load datasets
historical = pd.read_csv(
    RAW / "historical_matches.csv"
)

worldcup = pd.read_csv(
    PROCESSED / "worldcup_matches_clean.csv"
)

# Standardise historical team names
TEAM_NAMES = {
    "Bosnia and Herzegovina": "Bosnia–Herz",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "Turkey": "Türkiye",
    "United States": "USA",
}

historical["home_team"] = (
    historical["home_team"].replace(TEAM_NAMES)
)

historical["away_team"] = (
    historical["away_team"].replace(TEAM_NAMES)
)

# Parse dates
historical["date"] = pd.to_datetime(
    historical["date"],
    errors="raise"
)

# Identify World Cup teams
worldcup_teams = (
    set(worldcup["Home"])
    | set(worldcup["Away"])
)

historical_teams = (
    set(historical["home_team"])
    | set(historical["away_team"])
)

unmatched = worldcup_teams - historical_teams

if unmatched:
    print("Unmatched World Cup teams:")
    print(sorted(unmatched))
    raise ValueError(
        "Some World Cup teams have no historical match records."
    )

# Check historical match coverage before the tournament
cutoff = pd.Timestamp("2026-06-11")

previous = historical[
    historical["date"] < cutoff
]

coverage = []

for team in sorted(worldcup_teams):

    team_matches = previous[
        (previous["home_team"] == team)
        | (previous["away_team"] == team)
    ]

    coverage.append({
        "team": team,
        "historical_matches": len(team_matches)
    })

coverage = pd.DataFrame(coverage)

insufficient = coverage[
    coverage["historical_matches"] < 5
]

if not insufficient.empty:
    print("Teams with fewer than five matches:")
    print(insufficient.to_string(index=False))
    raise ValueError(
        "Insufficient historical match coverage."
    )

# Save processed historical dataset
PROCESSED.mkdir(parents=True, exist_ok=True)

output_path = (
    PROCESSED / "historical_matches_standardised.csv"
)

historical.to_csv(output_path, index=False)

coverage_path = (
    PROCESSED / "historical_team_coverage.csv"
)

coverage.to_csv(coverage_path, index=False)

print("=" * 65)
print("GROUP 9 - HISTORICAL TEAM STANDARDISATION")
print("=" * 65)

print("Total historical matches:", len(historical))
print("World Cup teams:", len(worldcup_teams))
print("Unmatched teams:", len(unmatched))

print("\nHistorical match coverage:")
print(
    coverage.sort_values("historical_matches")
    .to_string(index=False)
)

print("\nTeams with fewer than five matches:",
      len(insufficient))

print("\nSaved:", output_path)
print("Saved:", coverage_path)

print("Preprocessing completed successfully.")
