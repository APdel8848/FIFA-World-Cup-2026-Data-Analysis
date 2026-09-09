import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


# create output folder
os.makedirs("task3/output", exist_ok=True)


# ==========================================
# TASK 3 - PLAYING TIME
# ==========================================

# Research question:
# Do defenders have a different average minutes per appearance
# compared with forwards at the FIFA World Cup 2026?


# ==========================================
# STEP 1 - LOAD DATA
# ==========================================

playing_time = pd.read_csv(
    "task3/playing_time.csv"
)

print(
    "Playing Time shape:",
    playing_time.shape
)


# ==========================================
# STEP 2 - DATA WRANGLING
# ==========================================

# remove repeated header rows if they exist
playing_time = playing_time[
    playing_time["Player"] != "Player"
]


# keep only columns needed
data = playing_time[
    [
        "Player",
        "Pos",
        "Squad",
        "MP",
        "Min",
        "90s",
        "Starts",
        "Subs"
    ]
].copy()


# check missing values
print("\nMissing values:")
print(
    data.isnull().sum()
)


# check duplicate player and squad keys
print("\nDuplicate keys:")

print(
    data.duplicated(
        ["Player", "Squad"]
    ).sum()
)


# clean squad names
data["Squad"] = data["Squad"].str.replace(
    r"^[a-z]{2,3}\s+",
    "",
    regex=True
)


# convert columns to numeric
data["MP"] = pd.to_numeric(
    data["MP"],
    errors="coerce"
)

data["Min"] = pd.to_numeric(
    data["Min"],
    errors="coerce"
)

data["90s"] = pd.to_numeric(
    data["90s"],
    errors="coerce"
)

data["Starts"] = pd.to_numeric(
    data["Starts"],
    errors="coerce"
)

data["Subs"] = pd.to_numeric(
    data["Subs"],
    errors="coerce"
)


# remove missing MP and Min
data = data.dropna(
    subset=[
        "MP",
        "Min"
    ]
)


# remove players with zero appearances
data = data[
    data["MP"] > 0
]


# ==========================================
# CREATE DEFENDER AND FORWARD GROUPS
# ==========================================

# keep players who have DF or FW in their position
data = data[
    data["Pos"].str.contains(
        "DF|FW",
        na=False
    )
]


# remove players listed as both defender and forward
data = data[
    ~(
        data["Pos"].str.contains(
            "DF",
            na=False
        )
        &
        data["Pos"].str.contains(
            "FW",
            na=False
        )
    )
]


# create position groups
data["Group"] = np.where(
    data["Pos"].str.contains(
        "DF",
        na=False
    ),
    "Defender",
    "Forward"
)


# calculate minutes per appearance
data["Min_per_MP"] = (
    data["Min"] / data["MP"]
)


print(
    "\nPlayers after cleaning:",
    len(data)
)


print(
    "\nPlayers by position group:"
)

print(
    data["Group"].value_counts()
)


# ==========================================
# OUTLIER CHECK
# ==========================================

Q1 = data["Min_per_MP"].quantile(0.25)

Q3 = data["Min_per_MP"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR


outliers = data[
    (data["Min_per_MP"] < lower)
    |
    (data["Min_per_MP"] > upper)
]


print("\nOutlier check:")

print(
    "Q1 =",
    round(Q1, 2)
)

print(
    "Q3 =",
    round(Q3, 2)
)

print(
    "IQR =",
    round(IQR, 2)
)

print(
    "Lower limit =",
    round(lower, 2)
)

print(
    "Upper limit =",
    round(upper, 2)
)

print(
    "Number of outliers =",
    len(outliers)
)

print(
    "Outliers are kept because short or long playing "
    "times can represent real player usage."
)


# ==========================================
# POPULATION INFORMATION
# ==========================================

defender_population = data[
    data["Group"] == "Defender"
]

forward_population = data[
    data["Group"] == "Forward"
]


print("\nPopulation sizes:")

print(
    "Defender =",
    len(defender_population)
)

print(
    "Forward =",
    len(forward_population)
)


print("\nPopulation means:")

print(
    "Defender mean =",
    round(
        defender_population["Min_per_MP"].mean(),
        2
    )
)

print(
    "Forward mean =",
    round(
        forward_population["Min_per_MP"].mean(),
        2
    )
)


print("\nPopulation standard deviations:")

print(
    "Defender SD =",
    round(
        defender_population["Min_per_MP"].std(
            ddof=0
        ),
        2
    )
)

print(
    "Forward SD =",
    round(
        forward_population["Min_per_MP"].std(
            ddof=0
        ),
        2
    )
)


# ==========================================
# STEP 3 - DATA PREPARATION:
# LEVELS OF MEASUREMENT
# ==========================================

print(
    "\n--- STEP 3: LEVELS OF MEASUREMENT ---"
)

print(
    "Player -> Nominal (player identifier)"
)

print(
    "Pos -> Nominal (player position)"
)

print(
    "Squad -> Nominal (team/country name)"
)

print(
    "Group -> Nominal (Defender/Forward)"
)

print(
    "MP -> Ratio, Discrete "
    "(number of matches played)"
)

print(
    "Starts -> Ratio, Discrete "
    "(number of starts)"
)

print(
    "Subs -> Ratio, Discrete "
    "(number of substitute appearances)"
)

print(
    "Min -> Ratio, Continuous "
    "(total playing time)"
)

print(
    "90s -> Ratio, Continuous "
    "(playing time in 90-minute units)"
)

print(
    "Min_per_MP -> Ratio, Continuous "
    "(derived minutes-per-appearance variable)"
)

print(
    "\nThe main response variable for this task "
    "is Min_per_MP."
)


# ==========================================
# STEP 3 - SAMPLING
# ==========================================

# take 30 random defenders
defender_sample = defender_population.sample(
    n=30,
    random_state=42
)


# take 30 random forwards
forward_sample = forward_population.sample(
    n=30,
    random_state=42
)


# combine samples
sample = pd.concat(
    [
        defender_sample,
        forward_sample
    ]
)


print("\nSample sizes:")

print(
    sample["Group"].value_counts()
)

print(
    "Total sample =",
    len(sample)
)


print(
    "\nBoth groups have n = 30, "
    "so the sample sizes are sufficiently large "
    "for the CLT approximation."
)


# variables for analysis
defender_group = defender_sample[
    "Min_per_MP"
]

forward_group = forward_sample[
    "Min_per_MP"
]


# ==========================================
# STEP 4 - DESCRIPTIVE STATISTICS
# ==========================================

print(
    "\n--- STEP 4: DESCRIPTIVE STATISTICS ---"
)


# Defender statistics
d_mean = defender_group.mean()

d_median = defender_group.median()

d_mode = defender_group.mode()[0]

d_range = (
    defender_group.max()
    -
    defender_group.min()
)

d_q1 = defender_group.quantile(0.25)

d_q3 = defender_group.quantile(0.75)

d_iqr = d_q3 - d_q1

d_variance = defender_group.var()

d_sd = defender_group.std()


# Forward statistics
f_mean = forward_group.mean()

f_median = forward_group.median()

f_mode = forward_group.mode()[0]

f_range = (
    forward_group.max()
    -
    forward_group.min()
)

f_q1 = forward_group.quantile(0.25)

f_q3 = forward_group.quantile(0.75)

f_iqr = f_q3 - f_q1

f_variance = forward_group.var()

f_sd = forward_group.std()


print("\nDefender:")

print(
    "Mean =",
    round(d_mean, 2)
)

print(
    "Median =",
    round(d_median, 2)
)

print(
    "Mode =",
    round(d_mode, 2)
)

print(
    "Range =",
    round(d_range, 2)
)

print(
    "IQR =",
    round(d_iqr, 2)
)

print(
    "Variance =",
    round(d_variance, 2)
)

print(
    "Standard deviation =",
    round(d_sd, 2)
)


print("\nForward:")

print(
    "Mean =",
    round(f_mean, 2)
)

print(
    "Median =",
    round(f_median, 2)
)

print(
    "Mode =",
    round(f_mode, 2)
)

print(
    "Range =",
    round(f_range, 2)
)

print(
    "IQR =",
    round(f_iqr, 2)
)

print(
    "Variance =",
    round(f_variance, 2)
)

print(
    "Standard deviation =",
    round(f_sd, 2)
)


# ==========================================
# HISTOGRAM
# ==========================================

plt.hist(
    defender_group,
    bins=8,
    alpha=0.6,
    label="Defender"
)

plt.hist(
    forward_group,
    bins=8,
    alpha=0.6,
    label="Forward"
)

plt.xlabel(
    "Minutes per Appearance"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Minutes per Appearance"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "task3/output/histogram.png"
)

plt.close()


# ==========================================
# BOXPLOT
# ==========================================

plt.boxplot(
    [
        defender_group,
        forward_group
    ],
    tick_labels=[
        "Defender",
        "Forward"
    ]
)

plt.xlabel(
    "Player Group"
)

plt.ylabel(
    "Minutes per Appearance"
)

plt.title(
    "Minutes per Appearance by Position"
)

plt.tight_layout()

plt.savefig(
    "task3/output/boxplot.png"
)

plt.close()


# ==========================================
# STEP 5 - 95% CONFIDENCE INTERVAL
# ==========================================

print(
    "\n--- STEP 5: 95% CONFIDENCE INTERVAL ---"
)


z = 1.96

n1 = len(defender_group)

n2 = len(forward_group)


# Defender confidence interval
d_se = (
    d_sd / np.sqrt(n1)
)

d_margin = (
    z * d_se
)

d_lower = (
    d_mean - d_margin
)

d_upper = (
    d_mean + d_margin
)


print("\nDefender 95% CI:")

print(
    "Mean =",
    round(d_mean, 2)
)

print(
    "Standard Error =",
    round(d_se, 2)
)

print(
    "Margin of Error =",
    round(d_margin, 2)
)

print(
    "95% CI =",
    round(d_lower, 2),
    "to",
    round(d_upper, 2)
)


# Forward confidence interval
f_se = (
    f_sd / np.sqrt(n2)
)

f_margin = (
    z * f_se
)

f_lower = (
    f_mean - f_margin
)

f_upper = (
    f_mean + f_margin
)


print("\nForward 95% CI:")

print(
    "Mean =",
    round(f_mean, 2)
)

print(
    "Standard Error =",
    round(f_se, 2)
)

print(
    "Margin of Error =",
    round(f_margin, 2)
)

print(
    "95% CI =",
    round(f_lower, 2),
    "to",
    round(f_upper, 2)
)


# ==========================================
# CI COMPARISON CHART
# ==========================================

group_labels = [
    "Defender",
    "Forward"
]

group_means = [
    d_mean,
    f_mean
]

group_margins = [
    d_margin,
    f_margin
]

plt.bar(
    group_labels,
    group_means,
    yerr=group_margins,
    capsize=8
)

plt.ylabel(
    "Mean Minutes per Appearance"
)

plt.title(
    "Mean Minutes per Appearance "
    "with 95% Confidence Intervals"
)

plt.tight_layout()

plt.savefig(
    "task3/output/ci_comparison.png"
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
    "Do defenders have a different average "
    "minutes per appearance compared with forwards?"
)


# PLAN
print("\nPLAN:")

print(
    "H0: Mean minutes per appearance are equal "
    "for defenders and forwards."
)

print(
    "Ha: Mean minutes per appearance are different "
    "for defenders and forwards."
)

print(
    "Significance level = 0.05"
)

print(
    "Test = Independent two-sample t-test"
)


# SOLVE
t_value = (
    d_mean - f_mean
) / np.sqrt(
    (d_sd ** 2 / n1)
    +
    (f_sd ** 2 / n2)
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
    "Defender mean =",
    round(d_mean, 2)
)

print(
    "Forward mean =",
    round(f_mean, 2)
)

print(
    "Difference in means =",
    round(
        d_mean - f_mean,
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


# ==========================================
# CONCLUSION
# ==========================================

print("\nCONCLUDE:")

if p_value <= 0.05:

    print(
        "Reject H0."
    )

    print(
        "There is sufficient evidence that "
        "the average minutes per appearance "
        "are different between defenders "
        "and forwards."
    )

else:

    print(
        "Do not reject H0."
    )

    print(
        "There is not sufficient evidence that "
        "the average minutes per appearance "
        "are different between defenders "
        "and forwards."
    )