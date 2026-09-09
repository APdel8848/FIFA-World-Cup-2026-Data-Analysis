import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

os.makedirs("task3/output", exist_ok=True)

# Task 3
# Question:
# Do defenders have a significantly different mean
# minutes played per appearance compared with forwards
# at the FIFA World Cup 2026?


# 1. Load data

playing_time = pd.read_csv(
    "task3/playing_time.csv"
)

print(
    "Playing Time shape:",
    playing_time.shape
)


# 2. Clean and prepare data

playing_time = playing_time[
    playing_time["Player"] != "Player"
]

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

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate keys:")
print(
    data.duplicated(
        ["Player", "Squad"]
    ).sum()
)

data["Squad"] = data["Squad"].str.replace(
    r"^[a-z]{2,3}\s+",
    "",
    regex=True
)

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

data = data.dropna(
    subset=[
        "Player",
        "Pos",
        "MP",
        "Min"
    ]
).copy()

data = data[
    data["MP"] > 0
].copy()


# Keep defenders and forwards

data = data[
    data["Pos"].str.contains(
        "DF|FW",
        na=False
    )
].copy()

# remove players who are listed as both DF and FW
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
].copy()

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
    data["Min"] /
    data["MP"]
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


# 3. Outlier check

q1 = data["Min_per_MP"].quantile(0.25)
q3 = data["Min_per_MP"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = data[
    (data["Min_per_MP"] < lower) |
    (data["Min_per_MP"] > upper)
]

print("\nOutlier check:")
print("Q1 =", round(q1, 2))
print("Q3 =", round(q3, 2))
print("IQR =", round(iqr, 2))
print("Lower limit =", round(lower, 2))
print("Upper limit =", round(upper, 2))
print("Number of outliers =", len(outliers))

print(
    "Outliers are kept because short or long "
    "playing times can represent real player usage."
)


# 4. Population information

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
        defender_population[
            "Min_per_MP"
        ].mean(),
        2
    )
)

print(
    "Forward mean =",
    round(
        forward_population[
            "Min_per_MP"
        ].mean(),
        2
    )
)

print(
    "\nPopulation standard deviations:"
)

print(
    "Defender SD =",
    round(
        defender_population[
            "Min_per_MP"
        ].std(
            ddof=0
        ),
        2
    )
)

print(
    "Forward SD =",
    round(
        forward_population[
            "Min_per_MP"
        ].std(
            ddof=0
        ),
        2
    )
)


# 5. Levels of measurement

print("\nLevels of measurement:")
print("Player -> Nominal")
print("Squad -> Nominal")
print("Pos -> Nominal")
print("Group -> Nominal")
print("MP -> Ratio, Discrete")
print("Min -> Ratio, Continuous")
print("90s -> Ratio, Continuous")
print("Starts -> Ratio, Discrete")
print("Subs -> Ratio, Discrete")
print("Min_per_MP -> Ratio, Continuous")


# 6. Sampling

defender_sample = defender_population.sample(
    n=30,
    random_state=42
)

forward_sample = forward_population.sample(
    n=30,
    random_state=42
)

sample = pd.concat(
    [
        defender_sample,
        forward_sample
    ],
    ignore_index=True
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
    "\nBoth groups have n = 30, which is sufficiently "
    "large for the CLT approximation used in this analysis."
)

defender_group = defender_sample[
    "Min_per_MP"
]

forward_group = forward_sample[
    "Min_per_MP"
]


# 7. Descriptive statistics

d_mean = defender_group.mean()
d_median = defender_group.median()
d_mode = defender_group.mode()[0]

d_range = (
    defender_group.max() -
    defender_group.min()
)

d_iqr = (
    defender_group.quantile(0.75) -
    defender_group.quantile(0.25)
)

d_variance = defender_group.var()
d_sd = defender_group.std()


f_mean = forward_group.mean()
f_median = forward_group.median()
f_mode = forward_group.mode()[0]

f_range = (
    forward_group.max() -
    forward_group.min()
)

f_iqr = (
    forward_group.quantile(0.75) -
    forward_group.quantile(0.25)
)

f_variance = forward_group.var()
f_sd = forward_group.std()


print("\nDescriptive statistics")

print("\nDefender:")
print("Mean =", round(d_mean, 2))
print("Median =", round(d_median, 2))
print("Mode =", round(d_mode, 2))
print("Range =", round(d_range, 2))
print("IQR =", round(d_iqr, 2))
print("Variance =", round(d_variance, 2))
print(
    "Standard deviation =",
    round(d_sd, 2)
)

print("\nForward:")
print("Mean =", round(f_mean, 2))
print("Median =", round(f_median, 2))
print("Mode =", round(f_mode, 2))
print("Range =", round(f_range, 2))
print("IQR =", round(f_iqr, 2))
print("Variance =", round(f_variance, 2))
print(
    "Standard deviation =",
    round(f_sd, 2)
)


# 8. Visualisations

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


# 9. 95% confidence intervals

z = 1.96

n1 = len(defender_group)
n2 = len(forward_group)

d_se = (
    d_sd /
    np.sqrt(n1)
)

d_margin = (
    z *
    d_se
)

d_lower = (
    d_mean -
    d_margin
)

d_upper = (
    d_mean +
    d_margin
)

f_se = (
    f_sd /
    np.sqrt(n2)
)

f_margin = (
    z *
    f_se
)

f_lower = (
    f_mean -
    f_margin
)

f_upper = (
    f_mean +
    f_margin
)


print("\n95% Confidence Intervals")

print("\nDefender:")
print("Mean =", round(d_mean, 2))
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


print("\nForward:")
print("Mean =", round(f_mean, 2))
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


plt.bar(
    [
        "Defender",
        "Forward"
    ],
    [
        d_mean,
        f_mean
    ],
    yerr=[
        d_margin,
        f_margin
    ],
    capsize=8
)

plt.ylabel(
    "Mean Minutes per Appearance"
)

plt.title(
    "Mean Minutes per Appearance with 95% Confidence Intervals"
)

plt.tight_layout()

plt.savefig(
    "task3/output/ci_comparison.png"
)

plt.close()


# 10. Two-sample t-test

print("\nTwo-sample t-test")

print("\nState:")

print(
    "Do defenders have a different mean "
    "minutes per appearance compared with forwards?"
)

print("\nPlan:")

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

t_value = (
    d_mean -
    f_mean
) / np.sqrt(
    (d_sd ** 2 / n1) +
    (f_sd ** 2 / n2)
)

df = min(
    n1 - 1,
    n2 - 1
)

p_value = (
    2 *
    stats.t.sf(
        abs(t_value),
        df
    )
)

print("\nSolve:")

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


print("\nConclusion:")

if p_value <= 0.05:

    print("Reject H0.")

    print(
        "There is sufficient evidence that "
        "the mean minutes per appearance are "
        "different between defenders and forwards."
    )

else:

    print("Do not reject H0.")

    print(
        "There is not sufficient evidence that "
        "the mean minutes per appearance are "
        "different between defenders and forwards."
    )