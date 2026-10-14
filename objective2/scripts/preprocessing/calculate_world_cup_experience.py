
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

historical = pd.read_csv(
    PROCESSED / "historical_matches_standardised.csv"
)

worldcup = pd.read_csv(
    PROCESSED / "worldcup_matches_clean.csv"
)

historical["date"] = pd.to_datetime(
    historical["date"], errors="raise"
)

worldcup["match_date"] = pd.to_datetime(
    worldcup["match_date"], errors="raise"
)

# Include World Cup finals matches before 2026 only.
cutoff = worldcup["match_date"].min()

past_world_cups = historical[
    (historical["tournament"] == "FIFA World Cup")
    & (historical["date"] < cutoff)
].copy()

teams = sorted(
    set(worldcup["Home"]) | set(worldcup["Away"])
)

records = []

for team in teams:

    previous_matches = past_world_cups[
        (past_world_cups["home_team"] == team)
        | (past_world_cups["away_team"] == team)
    ]

    records.append({
        "team": team,
        "worldcup_experience": len(previous_matches)
    })

experience = pd.DataFrame(records)

# Validation
assert len(experience) == 48
assert experience["team"].nunique() == 48
assert experience["worldcup_experience"].notna().all()
assert (experience["worldcup_experience"] >= 0).all()

# Save dataset
output_path = (
    PROCESSED / "worldcup_experience_2026.csv"
)

experience.to_csv(output_path, index=False)

print("=" * 65)
print("GROUP 9 - WORLD CUP EXPERIENCE FEATURES")
print("=" * 65)

print("Historical World Cup matches:", len(past_world_cups))
print("National teams:", len(experience))

print("\nWorld Cup experience by team:")
print(
    experience.sort_values(
        "worldcup_experience",
        ascending=False
    ).to_string(index=False)
)

print("\nMissing values:")
print(experience.isna().sum())

print("\nSaved:", output_path)
print("Preprocessing completed successfully.")
