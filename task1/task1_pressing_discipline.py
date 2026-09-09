import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

os.makedirs("task1/output", exist_ok=True)

# Task 1
# Question:
# Do outfield players from teams eliminated in the group stage
# have a different mean fouls-per-90 rate compared with players
# from teams that reached the knockout stage?


# 1. Load data

standard = pd.read_csv("task1/standard_stats.csv", skiprows=1)
misc = pd.read_csv("task1/misc_stats.csv", skiprows=1)

print("Standard shape:", standard.shape)
print("Misc shape:", misc.shape)


# 2. Clean and prepare data

standard = standard[standard["Player"] != "Player"].copy()
misc = misc[misc["Player"] != "Player"].copy()

standard = standard[
    ["Player", "Squad", "Pos", "MP", "90s"]
].copy()

misc = misc[
    ["Player", "Squad", "Fls"]
].copy()

print("\nMissing values in standard:")
print(standard.isnull().sum())

print("\nMissing values in misc:")
print(misc.isnull().sum())

print("\nDuplicate keys:")
print(
    "Standard:",
    standard.duplicated(["Player", "Squad"]).sum()
)
print(
    "Misc:",
    misc.duplicated(["Player", "Squad"]).sum()
)

data = pd.merge(
    standard,
    misc,
    on=["Player", "Squad"],
    how="inner"
)

print("\nMerged shape:", data.shape)

data["Squad"] = data["Squad"].str.replace(
    r"^[a-z]{2,3}\s+",
    "",
    regex=True
)

data["90s"] = pd.to_numeric(
    data["90s"],
    errors="coerce"
)

data["Fls"] = pd.to_numeric(
    data["Fls"],
    errors="coerce"
)

data = data.dropna(
    subset=["90s", "Fls"]
)

# remove goalkeepers
data = data[
    ~data["Pos"].str.contains("GK", na=False)
].copy()

# only players with at least 90 minutes played
data = data[
    data["90s"] >= 1.0
].copy()

data["Fls_per90"] = (
    data["Fls"] / data["90s"]
)

print("\nPlayers after cleaning:", len(data))


# Create tournament stage groups

eliminated_teams = [
    "Korea Republic",
    "Czechia",
    "Qatar",
    "Scotland",
    "Haiti",
    "Türkiye",
    "Curaçao",
    "Tunisia",
    "IR Iran",
    "New Zealand",
    "Uruguay",
    "Saudi Arabia",
    "Iraq",
    "Jordan",
    "Uzbekistan",
    "Panama"
]

data["Stage"] = "Knockout Stage"

for team in eliminated_teams:
    data.loc[
        data["Squad"].str.contains(
            team,
            case=False,
            na=False
        ),
        "Stage"
    ] = "Group Stage Exit"

print("\nPlayers by tournament stage:")
print(data["Stage"].value_counts())


# 3. Outlier check

q1 = data["Fls_per90"].quantile(0.25)
q3 = data["Fls_per90"].quantile(0.75)

iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = data[
    (data["Fls_per90"] < lower) |
    (data["Fls_per90"] > upper)
]

print("\nOutlier check:")
print("Q1 =", round(q1, 2))
print("Q3 =", round(q3, 2))
print("IQR =", round(iqr, 2))
print("Lower limit =", round(lower, 2))
print("Upper limit =", round(upper, 2))
print("Number of outliers =", len(outliers))

print(
    "Outliers are kept because high foul rates "
    "can represent real player performances."
)


# 4. Levels of measurement

print("\nLevels of measurement:")
print("Player -> Nominal")
print("Squad -> Nominal")
print("Pos -> Nominal")
print("Stage -> Nominal")
print("MP -> Ratio, Discrete")
print("90s -> Ratio, Continuous")
print("Fls -> Ratio, Discrete")
print("Fls_per90 -> Ratio, Continuous")


# 5. Sampling

exit_population = data[
    data["Stage"] == "Group Stage Exit"
]

ko_population = data[
    data["Stage"] == "Knockout Stage"
]

print("\nPopulation sizes:")
print("Group Stage Exit =", len(exit_population))
print("Knockout Stage =", len(ko_population))

exit_sample = exit_population.sample(
    n=30,
    random_state=42
)

ko_sample = ko_population.sample(
    n=30,
    random_state=42
)

sample = pd.concat(
    [exit_sample, ko_sample],
    ignore_index=True
)

print("\nSample sizes:")
print(sample["Stage"].value_counts())
print("Total sample =", len(sample))

print(
    "\nBoth groups have n = 30, which is sufficiently "
    "large for the CLT approximation used in this analysis."
)

exit_group = exit_sample["Fls_per90"]
ko_group = ko_sample["Fls_per90"]


# 6. Descriptive statistics

exit_mean = exit_group.mean()
exit_median = exit_group.median()
exit_mode = exit_group.mode()[0]
exit_range = exit_group.max() - exit_group.min()
exit_iqr = (
    exit_group.quantile(0.75) -
    exit_group.quantile(0.25)
)
exit_variance = exit_group.var()
exit_sd = exit_group.std()

ko_mean = ko_group.mean()
ko_median = ko_group.median()
ko_mode = ko_group.mode()[0]
ko_range = ko_group.max() - ko_group.min()
ko_iqr = (
    ko_group.quantile(0.75) -
    ko_group.quantile(0.25)
)
ko_variance = ko_group.var()
ko_sd = ko_group.std()

print("\nDescriptive statistics")

print("\nGroup Stage Exit:")
print("Mean =", round(exit_mean, 2))
print("Median =", round(exit_median, 2))
print("Mode =", round(exit_mode, 2))
print("Range =", round(exit_range, 2))
print("IQR =", round(exit_iqr, 2))
print("Variance =", round(exit_variance, 2))
print("Standard deviation =", round(exit_sd, 2))

print("\nKnockout Stage:")
print("Mean =", round(ko_mean, 2))
print("Median =", round(ko_median, 2))
print("Mode =", round(ko_mode, 2))
print("Range =", round(ko_range, 2))
print("IQR =", round(ko_iqr, 2))
print("Variance =", round(ko_variance, 2))
print("Standard deviation =", round(ko_sd, 2))


# 7. Visualisations

plt.hist(
    exit_group,
    bins=8,
    alpha=0.6,
    label="Group Stage Exit"
)

plt.hist(
    ko_group,
    bins=8,
    alpha=0.6,
    label="Knockout Stage"
)

plt.xlabel("Fouls per 90 Minutes")
plt.ylabel("Frequency")
plt.title(
    "Distribution of Fouls per 90 Minutes by Tournament Stage"
)
plt.legend()
plt.tight_layout()

plt.savefig(
    "task1/output/fouls_histogram.png"
)

plt.close()


plt.boxplot(
    [exit_group, ko_group],
    tick_labels=[
        "Group Stage Exit",
        "Knockout Stage"
    ]
)

plt.xlabel("Tournament Stage")
plt.ylabel("Fouls per 90 Minutes")
plt.title(
    "Fouls per 90 Minutes by Tournament Stage"
)
plt.tight_layout()

plt.savefig(
    "task1/output/fouls_boxplot.png"
)

plt.close()


# 8. 95% confidence intervals

z = 1.96

n1 = len(exit_group)
n2 = len(ko_group)

exit_se = exit_sd / np.sqrt(n1)
exit_margin = z * exit_se
exit_lower = exit_mean - exit_margin
exit_upper = exit_mean + exit_margin

ko_se = ko_sd / np.sqrt(n2)
ko_margin = z * ko_se
ko_lower = ko_mean - ko_margin
ko_upper = ko_mean + ko_margin

print("\n95% Confidence Intervals")

print("\nGroup Stage Exit:")
print("Mean =", round(exit_mean, 2))
print("Standard Error =", round(exit_se, 2))
print("Margin of Error =", round(exit_margin, 2))
print(
    "95% CI =",
    round(exit_lower, 2),
    "to",
    round(exit_upper, 2)
)

print("\nKnockout Stage:")
print("Mean =", round(ko_mean, 2))
print("Standard Error =", round(ko_se, 2))
print("Margin of Error =", round(ko_margin, 2))
print(
    "95% CI =",
    round(ko_lower, 2),
    "to",
    round(ko_upper, 2)
)


plt.bar(
    ["Group Stage Exit", "Knockout Stage"],
    [exit_mean, ko_mean],
    yerr=[exit_margin, ko_margin],
    capsize=8
)

plt.ylabel("Mean Fouls per 90 Minutes")
plt.title(
    "Mean Fouls per 90 with 95% Confidence Intervals"
)
plt.tight_layout()

plt.savefig(
    "task1/output/fouls_ci_comparison.png"
)

plt.close()


# 9. Two-sample t-test

print("\nTwo-sample t-test")

print("\nState:")
print(
    "Do players from group-stage exit teams have "
    "a different mean fouls-per-90 rate compared "
    "with knockout-stage players?"
)

print("\nPlan:")
print(
    "H0: Mean fouls per 90 are equal for the two groups."
)
print(
    "Ha: Mean fouls per 90 are different for the two groups."
)
print("Significance level = 0.05")

t_value = (
    exit_mean - ko_mean
) / np.sqrt(
    (exit_sd ** 2 / n1) +
    (ko_sd ** 2 / n2)
)

df = min(
    n1 - 1,
    n2 - 1
)

p_value = 2 * stats.t.sf(
    abs(t_value),
    df
)

print("\nSolve:")
print(
    "Group Stage Exit mean =",
    round(exit_mean, 2)
)
print(
    "Knockout Stage mean =",
    round(ko_mean, 2)
)
print(
    "Difference in means =",
    round(exit_mean - ko_mean, 2)
)
print(
    "t-statistic =",
    round(t_value, 3)
)
print(
    "Degrees of freedom =",
    df
)
print(
    "p-value =",
    round(p_value, 4)
)

print("\nConclusion:")

if p_value <= 0.05:
    print("Reject H0.")
    print(
        "There is sufficient evidence that the mean "
        "fouls-per-90 rates are different between "
        "the two groups."
    )
else:
    print("Do not reject H0.")
    print(
        "There is not sufficient evidence that the mean "
        "fouls-per-90 rates are different between "
        "the two groups."
    )
