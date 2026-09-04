# Task-4 Do defenders have a significantly higher mean number of tackles won per 90 minutes than midfielders at the FIFA World Cup 2026?

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# 1. Load Data 
# FBref csv contains an extra header row (skiprows=1) 
df = pd.read_csv("misc_stats.csv", skiprows=1)

# remover repeated header rows from data
df = df[df["Player"] != "Player"].copy()

print("Original dataset shape:", df.shape)


# 2.Select required variables
df = df[
    [
        "Player",
        "Squad",
        "Pos",
        "90s",
        "TklW"
    ]
].copy()

print("\nSelected columns:")
print(df.head())

# 3. Data Wrangling / Cleaning
#convert numerical columns to numbers
df["90s"] = pd.to_numeric(
    df["90s"],
    errors="coerce" #change invalid values to NaN
)

df["TklW"] = pd.to_numeric(
    df["TklW"],
    errors="coerce"
)

# remove rows with missing values
df = df.dropna(
    subset=[
        "Player",
        "Pos",
        "90s",
        "TklW"
    ]
)

# Use only players whose position is clearly (Defender (DF) or Midfielder (MF))
# Mixed positions such as FWMF, MFFW, DFMF are excluded so that the two groups are clearly defined.
df = df[
    df["Pos"].isin(["DF", "MF"])
].copy()

# to keep players who played at least 1 full 90 minutes of tournament playing time.
df = df[
    df["90s"] >= 1.0
].copy()

print("\nEligible players after cleaning:", len(df))

print("\nPosition counts:")
print(df["Pos"].value_counts())

# 4. Create tackles won per 90
df["TklW_per90"] = (
    df["TklW"] / df["90s"]
)

print("\nCleaned data:")
print(
    df[
        [
            "Player",
            "Squad",
            "Pos",
            "90s",
            "TklW",
            "TklW_per90"
        ]
    ].head()
)


# 5. Define population
# Population: Eligible defenders and midfielders at the FIFA World Cup 2026 who played at least 90 tournament minutes.

population = df.copy()
print("\nPopulation size:", len(population))

print("\nPopulation groups:")
print(population["Pos"].value_counts())

# 6. Stratified Random Sampling

# We sample separately from defenders and midfielders.
# random_state=42, to get the same random sample every time

SAMPLE_FRACTION = 0.70 #selected 70% of the eligibel players

defenders_population = population[
    population["Pos"] == "DF"
]

midfielders_population = population[
    population["Pos"] == "MF"
]

defenders_sample = defenders_population.sample(
    frac=SAMPLE_FRACTION,
    random_state=42
)

midfielders_sample = midfielders_population.sample(
    frac=SAMPLE_FRACTION,
    random_state=42
)

sample = pd.concat(
    [
        defenders_sample,
        midfielders_sample
    ],
    ignore_index=True
)

print("\nSample size:", len(sample))

print("\nSample group counts:")
print(sample["Pos"].value_counts())


# 7. Create two groups
defenders = sample[
    sample["Pos"] == "DF"
]["TklW_per90"]

midfielders = sample[
    sample["Pos"] == "MF"
]["TklW_per90"]


# 8. Descriptive Statistics
print("\nDESCRIPTIVE STATISTICS")
print("\n-- Defenders --")
print("Count:  ", len(defenders))
print("Mean:   ", round(defenders.mean(), 3))
print("Median: ", round(defenders.median(), 3))
print("Std Dev:", round(defenders.std(), 3))
print("Min:    ", round(defenders.min(), 3))
print("Max:    ", round(defenders.max(), 3))

print("\n-- Midfielders --")
print("Count:  ", len(midfielders))
print("Mean:   ", round(midfielders.mean(), 3))
print("Median: ", round(midfielders.median(), 3))
print("Std Dev:", round(midfielders.std(), 3))
print("Min:    ", round(midfielders.min(), 3))
print("Max:    ", round(midfielders.max(), 3))


# 9. 95% Confidence Intervals

def confidence_interval(series):
    n = len(series)
    mean = series.mean()
    sem = stats.sem(series)

    ci = stats.t.interval(
        confidence=0.95,
        df=n - 1,
        loc=mean,
        scale=sem
    )

    return ci


defenders_ci = confidence_interval(defenders)
midfielders_ci = confidence_interval(midfielders)

print("\n95% CONFIDENCE INTERVALS")

print(
    "\nDefenders mean tackles won/90:",
    round(defenders.mean(), 3)
)

print(
    "Defenders 95% CI:",
    [round(float(x), 3) for x in defenders_ci]
)

print(
    "\nMidfielders mean tackles won/90:",
    round(midfielders.mean(), 3)
)

print(
    "Midfielders 95% CI:",
    [round(float(x), 3) for x in midfielders_ci]
)


# 10. Visualisation

sample.boxplot(
    column="TklW_per90",
    by="Pos",
    figsize=(7, 5),
    grid=True
)

plt.title("Tackles Won per 90: Defenders vs Midfielders")
plt.suptitle("")
plt.xlabel("Player Position")
plt.ylabel("Tackles Won per 90 Minutes")
plt.tight_layout()

plt.savefig(
    "task4_tackles_boxplot.png",
    dpi=150,
    bbox_inches ="tight"
)

plt.show()


# 11. Two-sample t-test
# H0: Defenders do NOT have a higher mean tackles-won-per-90
# than midfielders.
# H1: Defenders have a higher mean tackles-won-per-90
# than midfielders.

t_stat, p_value = stats.ttest_ind(
    defenders,
    midfielders,
    equal_var=False,
    alternative="greater"
)

print("\nTWO-SAMPLE T-TEST")

print("t-statistic:",round(t_stat, 4))
print("p-value:",round(p_value, 4))

# 12. Statistical Decision
alpha = 0.05

if p_value < alpha:

    print(
        "\nDecision: Reject H0."
    )

else:

    print(
        "\nDecision: Do not reject H0."
    )

# Save the graph
print(
    "\nSaved boxplot as:"
    " task4_tackles_boxplot.png"
)