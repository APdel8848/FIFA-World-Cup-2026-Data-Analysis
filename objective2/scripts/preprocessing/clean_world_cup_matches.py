
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

matches = pd.read_csv(
    RAW / "worldcup_matches.csv"
)

rankings = pd.read_csv(
    RAW / "fifa_rankings_2026_06_11_user_transcribed.csv"
)

valid_teams = set(rankings["team"].astype(str).str.strip())

# Country codes appearing in the original match data.

COUNTRY_CODES = {
    "ar", "at", "au", "ba", "be", "br", "ca", "cd",
    "ch", "ci", "co", "cv", "cw", "cz", "de", "dz",
    "ec", "eg", "eng", "es", "fr", "gh", "hr", "ht",
    "iq", "ir", "jo", "jp", "kr", "ma", "mx", "nl",
    "no", "nz", "pa", "pt", "py", "qa", "sa", "sct",
    "se", "sn", "tn", "tr", "us", "uy", "uz", "za"
}


def clean_team_name(value):
    name = str(value).strip()

    # Already a valid national team name.
    if name in valid_teams:
        return name

    # Remove country code from beginning or end.
    for code in sorted(COUNTRY_CODES, key=len, reverse=True):

        if name.startswith(code + " "):
            candidate = name[len(code):].strip()
            if candidate in valid_teams:
                return candidate

        if name.endswith(" " + code):
            candidate = name[:-len(code)].strip()
            if candidate in valid_teams:
                return candidate

        # Handle codes attached directly to names.
        if name.startswith(code):
            candidate = name[len(code):].strip()
            if candidate in valid_teams:
                return candidate

        if name.endswith(code):
            candidate = name[:-len(code)].strip()
            if candidate in valid_teams:
                return candidate

    # Special case from original dataset.
    SPECIAL = {
        "IRIran": "IR Iran",
        "NewZealand": "New Zealand",
    }

    for original, corrected in SPECIAL.items():
        if original in name:
            return corrected

    return name

# Remove blank rows.
matches = matches.dropna(
    subset=["Date", "Home", "Away", "Score"]
).copy()

# Clean team names.
matches["Home"] = matches["Home"].apply(clean_team_name)
matches["Away"] = matches["Away"].apply(clean_team_name)

# Validate all team names.
match_teams = set(matches["Home"]) | set(matches["Away"])

unmatched = match_teams - valid_teams

if unmatched:
    print("Unmatched teams:")
    print(sorted(unmatched))
    raise ValueError("Some team names could not be standardised.")

# Extract regulation/final score shown in the Score column.
def parse_score(score):
    text = str(score).strip()

    match = re.search(r"(\d+)\s*[–—-]\s*(\d+)", text)

    if not match:
        raise ValueError(f"Cannot parse score: {score}")

    return int(match.group(1)), int(match.group(2))

scores = matches["Score"].apply(parse_score)

matches["home_goals"] = scores.apply(lambda x: x[0])
matches["away_goals"] = scores.apply(lambda x: x[1])

matches["goal_difference"] = (
    matches["home_goals"] - matches["away_goals"]
)

# Convert dates to datetime.
matches["match_date"] = pd.to_datetime(
    matches["Date"],
    format="%d/%m/%Y",
    errors="raise"
)

# Create a unique match ID.
matches = matches.reset_index(drop=True)
matches.insert(0, "match_id", range(1, len(matches) + 1))

# Final validation.
assert len(matches) == 104, "Expected 104 matches"
assert len(match_teams) == 48, "Expected 48 teams"
assert matches["home_goals"].notna().all()
assert matches["away_goals"].notna().all()

# Save cleaned match data.
PROCESSED.mkdir(parents=True, exist_ok=True)

output_path = PROCESSED / "worldcup_matches_clean.csv"

matches.to_csv(output_path, index=False)

print("=" * 65)
print("GROUP 9 - WORLD CUP MATCH CLEANING")
print("=" * 65)

print("Total matches:", len(matches))
print("Unique teams:", len(match_teams))
print("Missing team names:", len(unmatched))

print("\nFirst 10 cleaned matches:")
print(
    matches[
        [
            "match_id",
            "match_date",
            "Home",
            "home_goals",
            "away_goals",
            "Away",
            "goal_difference"
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved:", output_path)
print("Preprocessing completed successfully.")
