import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

os.makedirs("task4/output", exist_ok=True)

# Task 4
# Question:
# Do defenders have a significantly higher mean number
# of tackles won per 90 minutes than midfielders
# at the FIFA World Cup 2026?


# 1. Load data

misc = pd.read_csv(
    "task4/misc_stats.csv",
    skiprows=1
)

print(
    "Misc dataset shape:",
    misc.shape
)


# 2. Clean and prepare data

misc = misc[
    misc["Player"] != "Player"
]

data = misc[
    [
        "Player",
        "Squad",
        "Pos",
        "90s",
        "TklW"
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

data["90s"] = pd.to_numeric(
    data["90s"],
    errors="coerce"
)

data["TklW"] = pd.to_numeric(
    data["TklW"],
    errors="coerce"
)

data = data.dropna(
    subset=[
        "Player",
        "Pos",
        "90s",
        "TklW"
    ]
).copy()


# Keep only clear defenders and midfielders

data = data[
    data["Pos"].isin(
        [
            "DF",
            "MF"
        ]
    )
].copy()

# only include players who played at least 90 minutes
data = data[
    data["90s"] >= 1.0
].copy()

# calculate tackles won per 90 minutes
data["TklW_per90"] = (
    data["TklW"] /
    data["90s"]
)

data["Group"] = np.where(
    data["Pos"] == "DF",
    "Defender",
    "Midfielder"
)

print(
    "\nEligible players after cleaning:",
    len(data)
)

print(
    "\nPlayers by position group:"
)

print(
    data["Group"].value_counts()
)


# 3. Outlier check

q1 = data["TklW_per90"].quantile(0.25)
q3 = data["TklW_per90"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = data[
    (data["TklW_per90"] < lower) |
    (data["TklW_per90"] > upper)
]

print("\nOutlier check:")
print("Q1 =", round(q1, 2))
print("Q3 =", round(q3, 2))
print("IQR =", round(iqr, 2))
print("Lower limit =", round(lower, 2))
print("Upper limit =", round(upper, 2))
print("Number of outliers =", len(outliers))

print(
    "Outliers are kept because unusually high or low "
    "tackle rates can represent real player performances."
)


# 4. Population information

defender_population = data[
    data["Group"] == "Defender"
]

midfielder_population = data[
    data["Group"] == "Midfielder"
]

print("\nPopulation sizes:")

print(
    "Defender =",
    len(defender_population)
)

print(
    "Midfielder =",
    len(midfielder_population)
)

print("\nPopulation means:")

print(
    "Defender mean =",
    round(
        defender_population[
            "TklW_per90"
        ].mean(),
        2
    )
)

print(
    "Midfielder mean =",
    round(
        midfielder_population[
            "TklW_per90"
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
            "TklW_per90"
        ].std(
            ddof=0
        ),
        2
    )
)

print(
    "Midfielder SD =",
    round(
        midfielder_population[
            "TklW_per90"
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
print("90s -> Ratio, Continuous")
print("TklW -> Ratio, Discrete")
print("TklW_per90 -> Ratio, Continuous")


# 6. Sampling

defender_sample = defender_population.sample(
    n=30,
    random_state=42
)

midfielder_sample = midfielder_population.sample(
    n=30,
    random_state=42
)

sample = pd.concat(
    [
        defender_sample,
        midfielder_sample
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
    "TklW_per90"
]

midfielder_group = midfielder_sample[
    "TklW_per90"
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


m_mean = midfielder_group.mean()
m_median = midfielder_group.median()
m_mode = midfielder_group.mode()[0]

m_range = (
    midfielder_group.max() -
    midfielder_group.min()
)

m_iqr = (
    midfielder_group.quantile(0.75) -
    midfielder_group.quantile(0.25)
)

m_variance = midfielder_group.var()
m_sd = midfielder_group.std()


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

print("\nMidfielder:")
print("Mean =", round(m_mean, 2))
print("Median =", round(m_median, 2))
print("Mode =", round(m_mode, 2))
print("Range =", round(m_range, 2))
print("IQR =", round(m_iqr, 2))
print("Variance =", round(m_variance, 2))
print(
    "Standard deviation =",
    round(m_sd, 2)
)


# 8. Visualisations

plt.hist(
    defender_group,
    bins=8,
    alpha=0.6,
    label="Defender"
)

plt.hist(
    midfielder_group,
    bins=8,
    alpha=0.6,
    label="Midfielder"
)

plt.xlabel(
    "Tackles Won per 90 Minutes"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Tackles Won per 90 Minutes"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "task4/output/histogram.png"
)

plt.close()


plt.boxplot(
    [
        defender_group,
        midfielder_group
    ],
    tick_labels=[
        "Defender",
        "Midfielder"
    ]
)

plt.xlabel(
    "Player Group"
)

plt.ylabel(
    "Tackles Won per 90 Minutes"
)

plt.title(
    "Tackles Won per 90 by Position"
)

plt.tight_layout()

plt.savefig(
    "task4/output/boxplot.png"
)

plt.close()


# 9. 95% confidence intervals

z = 1.96

n1 = len(defender_group)
n2 = len(midfielder_group)

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

m_se = (
    m_sd /
    np.sqrt(n2)
)

m_margin = (
    z *
    m_se
)

m_lower = (
    m_mean -
    m_margin
)

m_upper = (
    m_mean +
    m_margin
)


print("\n95% Confidence Intervals")

print("\nDefender:")
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


print("\nMidfielder:")
print(
    "Mean =",
    round(m_mean, 2)
)
print(
    "Standard Error =",
    round(m_se, 2)
)
print(
    "Margin of Error =",
    round(m_margin, 2)
)
print(
    "95% CI =",
    round(m_lower, 2),
    "to",
    round(m_upper, 2)
)


plt.bar(
    [
        "Defender",
        "Midfielder"
    ],
    [
        d_mean,
        m_mean
    ],
    yerr=[
        d_margin,
        m_margin
    ],
    capsize=8
)

plt.ylabel(
    "Mean Tackles Won per 90 Minutes"
)

plt.title(
    "Mean Tackles Won per 90 with 95% Confidence Intervals"
)

plt.tight_layout()

plt.savefig(
    "task4/output/ci_comparison.png"
)

plt.close()


# 10. One-sided two-sample t-test

print("\nTwo-sample t-test")

print("\nState:")

print(
    "Do defenders have a significantly higher "
    "mean tackles-won-per-90 rate than midfielders?"
)

print("\nPlan:")

print(
    "H0: Mean tackles won per 90 for defenders "
    "is less than or equal to midfielders."
)

print(
    "Ha: Mean tackles won per 90 for defenders "
    "is greater than midfielders."
)

print(
    "Significance level = 0.05"
)

t_value = (
    d_mean -
    m_mean
) / np.sqrt(
    (d_sd ** 2 / n1) +
    (m_sd ** 2 / n2)
)

df = min(
    n1 - 1,
    n2 - 1
)

p_value = stats.t.sf(
    t_value,
    df
)

print("\nSolve:")

print(
    "Defender mean =",
    round(d_mean, 2)
)

print(
    "Midfielder mean =",
    round(m_mean, 2)
)

print(
    "Difference in means =",
    round(
        d_mean - m_mean,
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
    "One-sided p-value =",
    round(p_value, 4)
)


print("\nConclusion:")

if p_value <= 0.05:

    print("Reject H0.")

    print(
        "There is sufficient evidence that "
        "defenders have a higher mean tackles-won-"
        "per-90 rate than midfielders."
    )

else:

    print("Do not reject H0.")

    print(
        "There is not sufficient evidence that "
        "defenders have a higher mean tackles-won-"
        "per-90 rate than midfielders."
    )