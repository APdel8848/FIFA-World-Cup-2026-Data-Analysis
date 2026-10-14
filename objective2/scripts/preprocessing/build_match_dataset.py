
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

matches = pd.read_csv(
    PROCESSED / "worldcup_matches_clean.csv"
)

form = pd.read_csv(
    PROCESSED / "historical_form_features.csv"
)

experience = pd.read_csv(
    PROCESSED / "worldcup_experience_2026.csv"
)

rest = pd.read_csv(
    PROCESSED / "worldcup_rest_days_2026.csv"
)

rankings = pd.read_csv(
    RAW / "fifa_rankings_2026_06_11_user_transcribed.csv"
)

# Use only FIFA ranking, not the unverified
# squad features in the rankings file.
rankings = rankings[["team", "fifa_rank"]]

def add_team_features(
    dataset, features, team_column, prefix
):
    renamed = features.rename(
        columns={
            col: f"{prefix}_{col}"
            for col in features.columns
            if col != "team"
        }
    ).rename(columns={"team": team_column})

    return dataset.merge(
        renamed,
        on=team_column,
        how="left",
        validate="many_to_one"
    )

# Add home and away rankings.
data = add_team_features(
    matches, rankings, "Home", "home"
)

data = add_team_features(
    data, rankings, "Away", "away"
)

# Add historical form statistics.
form_columns = [
    "team",
    "recent_form",
    "avg_goals_scored",
    "avg_goals_conceded",
    "clean_sheet_rate"
]

data = add_team_features(
    data, form[form_columns], "Home", "home"
)

data = add_team_features(
    data, form[form_columns], "Away", "away"
)

# Add previous World Cup experience.
data = add_team_features(
    data, experience, "Home", "home"
)

data = add_team_features(
    data, experience, "Away", "away"
)

# Add rest days.
data = data.merge(
    rest[
        ["match_id", "rest_days_difference"]
    ],
    on="match_id",
    how="left",
    validate="one_to_one"
)

# Construct exactly eight explanatory variables.
data["fifa_ranking_difference"] = (
    data["away_fifa_rank"] - data["home_fifa_rank"]
)

data["recent_form_difference"] = (
    data["home_recent_form"] - data["away_recent_form"]
)

data["avg_goals_scored_difference"] = (
    data["home_avg_goals_scored"]
    - data["away_avg_goals_scored"]
)

data["avg_goals_conceded_difference"] = (
    data["home_avg_goals_conceded"]
    - data["away_avg_goals_conceded"]
)

data["clean_sheet_rate_difference"] = (
    data["home_clean_sheet_rate"]
    - data["away_clean_sheet_rate"]
)

data["worldcup_experience_difference"] = (
    data["home_worldcup_experience"]
    - data["away_worldcup_experience"]
)

data["knockout_stage_indicator"] = (
    data["Round"] != "Group stage"
).astype(int)

data["goal_difference"] = (
    data["home_goals"] - data["away_goals"]
)

predictors = [
    "fifa_ranking_difference",
    "recent_form_difference",
    "avg_goals_scored_difference",
    "avg_goals_conceded_difference",
    "clean_sheet_rate_difference",
    "worldcup_experience_difference",
    "rest_days_difference",
    "knockout_stage_indicator"
]

final_columns = [
    "match_id",
    "match_date",
    "Home",
    "Away",
    "Round",
    "goal_difference"
] + predictors

final = data[final_columns].copy()

# Validation.
assert len(final) == 104
assert final["match_id"].nunique() == 104
assert len(predictors) == 8

missing = final.isna().sum()

if missing.any():
    print("Missing values:")
    print(missing[missing > 0])
    raise ValueError("Missing values detected")

assert (
    final["knockout_stage_indicator"]
    .isin([0, 1])
    .all()
)

output_path = (
    PROCESSED / "objective2_1_final.csv"
)

final.to_csv(PROCESSED / "objective2_1_uncapped.csv", index=False)
final["rest_days_difference"] = final["rest_days_difference"].clip(-7, 7)
final.to_csv(output_path, index=False)

print("=" * 65)
print("GROUP 9 - LINEAR REGRESSION 2.1 DATASET")
print("=" * 65)

print("Matches:", len(final))
print("Explanatory variables:", len(predictors))
print("Missing values:", final.isna().sum().sum())

print("\nExplanatory variables:")
for number, variable in enumerate(predictors, 1):
    print(f"{number}. {variable}")

print("\nFirst five matches:")
print(final.head().to_string(index=False))

print("\nRest days difference summary:")
print(final["rest_days_difference"].describe())

print("\nSaved:", output_path)
print("Preprocessing completed successfully.")
