
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

worldcup = pd.read_csv(
    PROCESSED / "worldcup_matches_clean.csv"
)

historical = pd.read_csv(
    PROCESSED / "historical_matches_standardised.csv"
)

worldcup["match_date"] = pd.to_datetime(
    worldcup["match_date"]
)

historical["date"] = pd.to_datetime(
    historical["date"]
)

worldcup = worldcup.sort_values(
    ["match_date", "match_id"]
).copy()

teams = sorted(
    set(worldcup["Home"]) | set(worldcup["Away"])
)

# Initialise previous match dates using historical data.
previous_dates = {}

tournament_start = worldcup["match_date"].min()

for team in teams:

    previous = historical[
        (
            (historical["home_team"] == team)
            | (historical["away_team"] == team)
        )
        & (historical["date"] < tournament_start)
    ]

    if previous.empty:
        raise ValueError(
            f"No previous international match for {team}"
        )

    previous_dates[team] = previous["date"].max()

# Calculate rest days for each match.
records = []

for _, match in worldcup.iterrows():

    match_date = match["match_date"]
    home_team = match["Home"]
    away_team = match["Away"]

    home_rest = (
        match_date - previous_dates[home_team]
    ).days

    away_rest = (
        match_date - previous_dates[away_team]
    ).days

    if home_rest < 0 or away_rest < 0:
        raise ValueError(
            f"Negative rest days at match {match['match_id']}"
        )

    records.append({
        "match_id": match["match_id"],
        "match_date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_days_difference": home_rest - away_rest
    })

    # Update each team's last match date.
    previous_dates[home_team] = match_date
    previous_dates[away_team] = match_date

rest = pd.DataFrame(records)

# Validation
assert len(rest) == 104
assert rest["match_id"].nunique() == 104
assert rest[
    ["home_rest_days", "away_rest_days"]
].ge(0).all().all()

# Save results.
output_path = (
    PROCESSED / "worldcup_rest_days_2026.csv"
)

rest.to_csv(output_path, index=False)

print("=" * 65)
print("GROUP 9 - WORLD CUP REST DAYS")
print("=" * 65)

print("Matches:", len(rest))
print("Missing values:", rest.isna().sum().sum())

print("\nFirst 15 matches:")
print(rest.head(15).to_string(index=False))

print("\nRest days summary:")
print(
    rest[
        ["home_rest_days", "away_rest_days", "rest_days_difference"]
    ].describe().to_string()
)

print("\nSaved:", output_path)
print("Preprocessing completed successfully.")
