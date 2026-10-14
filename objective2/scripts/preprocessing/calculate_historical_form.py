
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

teams = sorted(
    set(worldcup["Home"]) | set(worldcup["Away"])
)

# Exclude all World Cup 2026 matches and later results.
cutoff = worldcup["match_date"].min()

historical = historical[
    historical["date"] < cutoff
].copy()

records = []

for team in teams:

    team_matches = historical[
        (historical["home_team"] == team)
        | (historical["away_team"] == team)
    ].copy()

    # Use the five most recent historical matches.
    team_matches = (
        team_matches
        .sort_values("date", kind="stable")
        .tail(5)
    )

    if len(team_matches) < 5:
        raise ValueError(
            f"Insufficient matches for {team}"
        )

    goals_scored = []
    goals_conceded = []
    points = []
    clean_sheets = []

    for _, match in team_matches.iterrows():

        if match["home_team"] == team:
            scored = int(match["home_score"])
            conceded = int(match["away_score"])
        else:
            scored = int(match["away_score"])
            conceded = int(match["home_score"])

        goals_scored.append(scored)
        goals_conceded.append(conceded)

        if scored > conceded:
            points.append(3)
        elif scored == conceded:
            points.append(1)
        else:
            points.append(0)

        clean_sheets.append(
            1 if conceded == 0 else 0
        )

    records.append({
        "team": team,
        "recent_form": sum(points),
        "avg_goals_scored": sum(goals_scored) / 5,
        "avg_goals_conceded": sum(goals_conceded) / 5,
        "clean_sheet_rate": sum(clean_sheets) / 5,
        "last_historical_match": (
            team_matches["date"].max().strftime("%Y-%m-%d")
        ),
        "historical_matches_used": 5
    })

features = pd.DataFrame(records)

assert len(features) == 48
assert features["team"].nunique() == 48
assert features["historical_matches_used"].eq(5).all()

assert features["recent_form"].between(0, 15).all()
assert features["clean_sheet_rate"].between(0, 1).all()

assert (
    pd.to_datetime(features["last_historical_match"])
    < cutoff
).all(), "Historical data leakage detected"

output_path = (
    PROCESSED / "historical_form_features.csv"
)

features.to_csv(output_path, index=False)

print("=" * 65)
print("GROUP 9 - HISTORICAL FORM FEATURES")
print("=" * 65)

print("Historical cutoff:", cutoff.date())
print("National teams:", len(features))
print("Matches used per team: 5")

print("\nFirst 15 teams:")
print(features.head(15).to_string(index=False))

print("\nMissing values:")
print(features.isna().sum())

print("\nSaved:", output_path)
print("Preprocessing completed successfully.")
