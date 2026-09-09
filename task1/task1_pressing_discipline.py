import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


# create output folder
os.makedirs("task1/output", exist_ok=True)


# ==========================================
# TASK 1 - FOULS PER 90 MINUTES
# ==========================================

# Research question:
# Do outfield players from teams eliminated in the group stage
# have a different average fouls per 90 compared with players
# from teams that reached the knockout stage?


# ==========================================
# STEP 1 - LOAD DATA
# ==========================================

standard = pd.read_csv(
    "task1/standard_stats.csv",
    skiprows=1
)

misc = pd.read_csv(
    "task1/misc_stats.csv",
    skiprows=1
)

print("Standard shape:", standard.shape)
print("Misc shape:", misc.shape)


# ==========================================
# STEP 2 - DATA WRANGLING
# ==========================================

# remove repeated header rows from FBref
standard = standard[
    standard["Player"] != "Player"
]

misc = misc[
    misc["Player"] != "Player"
]


# keep only the columns needed
standard = standard[
    [
        "Player",
        "Squad",
        "Pos",
        "MP",
        "90s"
    ]
]

misc = misc[
    [
        "Player",
        "Squad",
        "Fls"
    ]
]


# check missing values
print("\nMissing values in standard:")
print(standard.isnull().sum())

print("\nMissing values in misc:")
print(misc.isnull().sum())


# check duplicate player and team keys
print("\nDuplicate keys:")

print(
    "Standard:",
    standard.duplicated(
        ["Player", "Squad"]
    ).sum()
)

print(
    "Misc:",
    misc.duplicated(
        ["Player", "Squad"]
    ).sum()
)


# merge the two datasets
data = pd.merge(
    standard,
    misc,
    on=["Player", "Squad"],
    how="inner"
)

print(
    "\nMerged shape:",
    data.shape
)


# clean team names after merging
data["Squad"] = data["Squad"].str.replace(
    r"^[a-z]{2,3}\s+",
    "",
    regex=True
)


# convert columns to numeric
data["90s"] = pd.to_numeric(
    data["90s"],
    errors="coerce"
)

data["Fls"] = pd.to_numeric(
    data["Fls"],
    errors="coerce"
)


# remove missing values
data = data.dropna(
    subset=[
        "90s",
        "Fls"
    ]
)


# remove goalkeepers
data = data[
    ~data["Pos"].str.contains(
        "GK",
        na=False
    )
]


# keep players who played at least 90 minutes
data = data[
    data["90s"] >= 1.0
]


print(
    "Players after cleaning:",
    len(data)
)


# calculate fouls per 90 minutes
data["Fls_per90"] = (
    data["Fls"] / data["90s"]
)


# ==========================================
# CREATE TOURNAMENT STAGE GROUPS
# ==========================================

# teams eliminated in the group stage
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


# start by assigning all players to knockout stage
data["Stage"] = "Knockout Stage"


# change players from eliminated teams
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
print(
    data["Stage"].value_counts()
)


# ==========================================
# OUTLIER CHECK
# ==========================================

Q1 = data["Fls_per90"].quantile(0.25)
Q3 = data["Fls_per90"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = data[
    (data["Fls_per90"] < lower)
    |
    (data["Fls_per90"] > upper)
]


print("\nOutlier check:")
print("Q1 =", round(Q1, 2))
print("Q3 =", round(Q3, 2))
print("IQR =", round(IQR, 2))
print("Lower limit =", round(lower, 2))
print("Upper limit =", round(upper, 2))
print("Number of outliers =", len(outliers))

print(
    "Outliers are kept because high foul rates "
    "can be real player performances."
)


# ==========================================
# STEP 3 - DATA PREPARATION: LEVELS OF MEASUREMENT
# ==========================================

print(
    "\n--- STEP 3: LEVELS OF MEASUREMENT ---"
)

print(
    "Player  -> Nominal (identifier, no order/magnitude)"
)

print(
    "Squad   -> Nominal (country/team name)"
)

print(
    "Pos     -> Nominal (position label)"
)

print(
    "Stage   -> Nominal (Group Stage Exit/Knockout Stage, "
    "derived label)"
)

print(
    "MP      -> Ratio, Discrete (count of matches played, "
    "true zero)"
)

print(
    "90s     -> Ratio, Continuous (minutes converted to "
    "90-minute equivalents)"
)

print(
    "Fls     -> Ratio, Discrete (count of fouls committed, "
    "true zero)"
)

print(
    "Fls_per90 -> Ratio, Continuous (derived rate variable, "
    "true zero, meaningful ratios)"
)

print(
    "\nNote: no Ordinal or Interval variables are used in "
    "this task. Squad is Nominal and used only to derive "
    "the Stage grouping, not analysed directly."
)


# ==========================================
# STEP 3 - SAMPLING
# ==========================================

exit_population = data[
    data["Stage"] == "Group Stage Exit"
]

ko_population = data[
    data["Stage"] == "Knockout Stage"
]


print("\nPopulation sizes:")

print(
    "Group Stage Exit =",
    len(exit_population)
)

print(
    "Knockout Stage =",
    len(ko_population)
)


# take 30 random players from each group
exit_sample = exit_population.sample(
    n=30,
    random_state=42
)

ko_sample = ko_population.sample(
    n=30,
    random_state=42
)


# combine both samples
sample = pd.concat(
    [
        exit_sample,
        ko_sample
    ]
)


print("\nSample sizes:")
print(
    sample["Stage"].value_counts()
)

print(
    "Total sample =",
    len(sample)
)

print(
    "\nBoth groups have n = 30, "
    "so the CLT requirement is satisfied."
)


# variables for analysis
exit_group = exit_sample["Fls_per90"]
ko_group = ko_sample["Fls_per90"]


# ==========================================
# STEP 4 - DESCRIPTIVE STATISTICS
# ==========================================

print(
    "\n--- STEP 4: DESCRIPTIVE STATISTICS ---"
)


# Group Stage Exit statistics
exit_mean = exit_group.mean()
exit_median = exit_group.median()
exit_mode = exit_group.mode()[0]
exit_range = exit_group.max() - exit_group.min()

exit_q1 = exit_group.quantile(0.25)
exit_q3 = exit_group.quantile(0.75)
exit_iqr = exit_q3 - exit_q1

exit_variance = exit_group.var()
exit_sd = exit_group.std()


# Knockout Stage statistics
ko_mean = ko_group.mean()
ko_median = ko_group.median()
ko_mode = ko_group.mode()[0]
ko_range = ko_group.max() - ko_group.min()

ko_q1 = ko_group.quantile(0.25)
ko_q3 = ko_group.quantile(0.75)
ko_iqr = ko_q3 - ko_q1

ko_variance = ko_group.var()
ko_sd = ko_group.std()


print("\nGroup Stage Exit:")

print(
    "Mean =",
    round(exit_mean, 2)
)

print(
    "Median =",
    round(exit_median, 2)
)

print(
    "Mode =",
    round(exit_mode, 2)
)

print(
    "Range =",
    round(exit_range, 2)
)

print(
    "IQR =",
    round(exit_iqr, 2)
)

print(
    "Variance =",
    round(exit_variance, 2)
)

print(
    "Standard deviation =",
    round(exit_sd, 2)
)


print("\nKnockout Stage:")

print(
    "Mean =",
    round(ko_mean, 2)
)

print(
    "Median =",
    round(ko_median, 2)
)

print(
    "Mode =",
    round(ko_mode, 2)
)

print(
    "Range =",
    round(ko_range, 2)
)

print(
    "IQR =",
    round(ko_iqr, 2)
)

print(
    "Variance =",
    round(ko_variance, 2)
)

print(
    "Standard deviation =",
    round(ko_sd, 2)
)


# ==========================================
# HISTOGRAM
# ==========================================

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

plt.xlabel(
    "Fouls per 90 Minutes"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Fouls per 90 Minutes by Tournament Stage"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "task1/output/fouls_histogram.png"
)

plt.close()


# ==========================================
# BOXPLOT
# ==========================================

plt.boxplot(
    [
        exit_group,
        ko_group
    ],
    tick_labels=[
        "Group Stage Exit",
        "Knockout Stage"
    ]
)

plt.xlabel(
    "Tournament Stage"
)

plt.ylabel(
    "Fouls per 90 Minutes"
)

plt.title(
    "Fouls per 90 Minutes by Tournament Stage"
)

plt.tight_layout()

plt.savefig(
    "task1/output/fouls_boxplot.png"
)

plt.close()


# ==========================================
# STEP 5 - 95% CONFIDENCE INTERVAL
# ==========================================

print(
    "\n--- STEP 5: 95% CONFIDENCE INTERVAL ---"
)


z = 1.96

n1 = len(exit_group)
n2 = len(ko_group)


# Group Stage Exit confidence interval
exit_se = (
    exit_sd / np.sqrt(n1)
)

exit_margin = (
    z * exit_se
)

exit_lower = (
    exit_mean - exit_margin
)

exit_upper = (
    exit_mean + exit_margin
)


print("\nGroup Stage Exit 95% CI:")

print(
    "Mean =",
    round(exit_mean, 2)
)

print(
    "Standard Error =",
    round(exit_se, 2)
)

print(
    "Margin of Error =",
    round(exit_margin, 2)
)

print(
    "95% CI =",
    round(exit_lower, 2),
    "to",
    round(exit_upper, 2)
)


# Knockout Stage confidence interval
ko_se = (
    ko_sd / np.sqrt(n2)
)

ko_margin = (
    z * ko_se
)

ko_lower = (
    ko_mean - ko_margin
)

ko_upper = (
    ko_mean + ko_margin
)


print("\nKnockout Stage 95% CI:")

print(
    "Mean =",
    round(ko_mean, 2)
)

print(
    "Standard Error =",
    round(ko_se, 2)
)

print(
    "Margin of Error =",
    round(ko_margin, 2)
)

print(
    "95% CI =",
    round(ko_lower, 2),
    "to",
    round(ko_upper, 2)
)


# ==========================================
# CI COMPARISON CHART
# ==========================================

group_labels = [
    "Group Stage Exit",
    "Knockout Stage"
]

group_means = [
    exit_mean,
    ko_mean
]

group_margins = [
    exit_margin,
    ko_margin
]

plt.bar(
    group_labels,
    group_means,
    yerr=group_margins,
    capsize=8,
    color=["#4C72B0", "#DD8452"]
)

plt.ylabel(
    "Mean Fouls per 90 Minutes"
)

plt.title(
    "Mean Fouls per 90 with 95% Confidence Intervals"
)

plt.tight_layout()

plt.savefig(
    "task1/output/fouls_ci_comparison.png"
)

plt.close()


# ==========================================
# STEP 6 - TWO-SAMPLE T-TEST
# ==========================================

print(
    "\n--- STEP 6: TWO-SAMPLE T-TEST ---"
)


# STATE
print("\nSTATE:")

print(
    "Do players from group-stage exit teams "
    "have a different average fouls per 90 "
    "compared with knockout-stage players?"
)


# PLAN
print("\nPLAN:")

print(
    "H0: Mean fouls per 90 are equal "
    "for the two groups."
)

print(
    "Ha: Mean fouls per 90 are different "
    "for the two groups."
)

print(
    "Significance level = 0.05"
)

print(
    "Test = Independent two-sample t-test"
)


# SOLVE
t_value = (
    exit_mean - ko_mean
) / np.sqrt(
    (exit_sd ** 2 / n1)
    +
    (ko_sd ** 2 / n2)
)


# conservative degrees of freedom
df = min(
    n1 - 1,
    n2 - 1
)


# two-sided p-value
p_value = (
    2 *
    stats.t.sf(
        abs(t_value),
        df
    )
)


print("\nSOLVE:")

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
    round(
        exit_mean - ko_mean,
        2
    )
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


# CONCLUDE
print("\nCONCLUDE:")

if p_value <= 0.05:

    print(
        "Reject H0."
    )

    print(
        "There is sufficient evidence that "
        "the average fouls per 90 are different "
        "between the two groups."
    )

else:

    print(
        "Do not reject H0."
    )

    print(
        "There is not sufficient evidence that "
        "the average fouls per 90 are different "
        "between the two groups."
    )