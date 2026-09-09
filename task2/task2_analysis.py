import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

os.makedirs("task2/output", exist_ok=True)

# Task 2
# Question:
# Do forwards have a significantly different mean
# shots-on-target percentage compared with
# non-forward outfield players?


# 1. Load data

standard = pd.read_csv("task2/standard.csv")
shooting = pd.read_csv("task2/shooting.csv")

print("Standard shape:", standard.shape)
print("Shooting shape:", shooting.shape)


# 2. Clean and prepare data

standard = standard[
    [
        "Player",
        "Pos",
        "Squad",
        "Age",
        "MP",
        "Starts",
        "Min",
        "90s"
    ]
].copy()

shooting = shooting[
    [
        "Player",
        "Squad",
        "Sh",
        "SoT",
        "SoT%"
    ]
].copy()

print("\nMissing values in standard:")
print(standard.isnull().sum())

print("\nMissing values in shooting:")
print(shooting.isnull().sum())

print("\nDuplicate keys:")
print(
    "Standard:",
    standard.duplicated(
        ["Player", "Squad"]
    ).sum()
)

print(
    "Shooting:",
    shooting.duplicated(
        ["Player", "Squad"]
    ).sum()
)

data = pd.merge(
    standard,
    shooting,
    on=["Player", "Squad"],
    how="inner"
)

print("\nMerged shape:", data.shape)

data["Squad"] = data["Squad"].str.replace(
    r"^[a-z]{2,3}\s+",
    "",
    regex=True
)

data["Sh"] = pd.to_numeric(
    data["Sh"],
    errors="coerce"
)

data["SoT"] = pd.to_numeric(
    data["SoT"],
    errors="coerce"
)

data["SoT%"] = pd.to_numeric(
    data["SoT%"],
    errors="coerce"
)

# remove goalkeepers
data = data[
    ~data["Pos"].str.contains(
        "GK",
        na=False
    )
].copy()

print(
    "\nPlayers after removing goalkeepers:",
    len(data)
)

print(
    "Missing SoT% after removing goalkeepers:",
    data["SoT%"].isnull().sum()
)

# remove missing shooting values
data = data.dropna(
    subset=[
        "Sh",
        "SoT",
        "SoT%"
    ]
).copy()

# only players who attempted at least one shot
data = data[
    data["Sh"] > 0
].copy()

print(
    "Players after removing missing SoT% "
    "and zero-shot players:",
    len(data)
)

# calculate shots-on-target percentage
data["SoT_pct"] = (
    data["SoT"] /
    data["Sh"]
) * 100


# Create position groups

data["Group"] = np.where(
    data["Pos"].str.contains(
        "FW",
        na=False
    ),
    "Forward",
    "Non-Forward"
)

print("\nPlayers by position group:")
print(data["Group"].value_counts())


# 3. Outlier check

q1 = data["SoT_pct"].quantile(0.25)
q3 = data["SoT_pct"].quantile(0.75)

iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = data[
    (data["SoT_pct"] < lower) |
    (data["SoT_pct"] > upper)
]

print("\nOutlier check:")
print("Q1 =", round(q1, 2))
print("Q3 =", round(q3, 2))
print("IQR =", round(iqr, 2))
print("Lower limit =", round(lower, 2))
print("Upper limit =", round(upper, 2))
print("Number of outliers =", len(outliers))

print(
    "Outliers are kept because unusual shooting "
    "percentages can represent real performances."
)


# 4. Population information

forward_population = data[
    data["Group"] == "Forward"
]

nonforward_population = data[
    data["Group"] == "Non-Forward"
]

print("\nPopulation sizes:")
print("Forward =", len(forward_population))
print("Non-Forward =", len(nonforward_population))

print("\nPopulation means:")
print(
    "Forward mean =",
    round(
        forward_population["SoT_pct"].mean(),
        2
    )
)

print(
    "Non-Forward mean =",
    round(
        nonforward_population["SoT_pct"].mean(),
        2
    )
)

print("\nPopulation standard deviations:")
print(
    "Forward SD =",
    round(
        forward_population["SoT_pct"].std(
            ddof=0
        ),
        2
    )
)

print(
    "Non-Forward SD =",
    round(
        nonforward_population["SoT_pct"].std(
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
print("Sh -> Ratio, Discrete")
print("SoT -> Ratio, Discrete")
print("SoT_pct -> Ratio, Continuous")


# 6. Sampling

forward_sample = forward_population.sample(
    n=30,
    random_state=42
)

nonforward_sample = nonforward_population.sample(
    n=30,
    random_state=42
)

sample = pd.concat(
    [
        forward_sample,
        nonforward_sample
    ],
    ignore_index=True
)

print("\nSample sizes:")
print(sample["Group"].value_counts())
print("Total sample =", len(sample))

print(
    "\nBoth groups have n = 30, which is sufficiently "
    "large for the CLT approximation used in this analysis."
)

forward_group = forward_sample["SoT_pct"]
nonforward_group = nonforward_sample["SoT_pct"]


# 7. Descriptive statistics

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

nf_mean = nonforward_group.mean()
nf_median = nonforward_group.median()
nf_mode = nonforward_group.mode()[0]
nf_range = (
    nonforward_group.max() -
    nonforward_group.min()
)
nf_iqr = (
    nonforward_group.quantile(0.75) -
    nonforward_group.quantile(0.25)
)
nf_variance = nonforward_group.var()
nf_sd = nonforward_group.std()

print("\nDescriptive statistics")

print("\nForward:")
print("Mean =", round(f_mean, 2))
print("Median =", round(f_median, 2))
print("Mode =", round(f_mode, 2))
print("Range =", round(f_range, 2))
print("IQR =", round(f_iqr, 2))
print("Variance =", round(f_variance, 2))
print("Standard deviation =", round(f_sd, 2))

print("\nNon-Forward:")
print("Mean =", round(nf_mean, 2))
print("Median =", round(nf_median, 2))
print("Mode =", round(nf_mode, 2))
print("Range =", round(nf_range, 2))
print("IQR =", round(nf_iqr, 2))
print("Variance =", round(nf_variance, 2))
print("Standard deviation =", round(nf_sd, 2))


# 8. Visualisations

plt.hist(
    forward_group,
    bins=8,
    alpha=0.6,
    label="Forward"
)

plt.hist(
    nonforward_group,
    bins=8,
    alpha=0.6,
    label="Non-Forward"
)

plt.xlabel("Shots on Target Percentage")
plt.ylabel("Frequency")
plt.title(
    "Distribution of Shots on Target Percentage"
)
plt.legend()
plt.tight_layout()

plt.savefig(
    "task2/output/histogram.png"
)

plt.close()


plt.boxplot(
    [
        forward_group,
        nonforward_group
    ],
    tick_labels=[
        "Forward",
        "Non-Forward"
    ]
)

plt.xlabel("Player Group")
plt.ylabel("Shots on Target Percentage")
plt.title(
    "Shots on Target Percentage by Player Group"
)
plt.tight_layout()

plt.savefig(
    "task2/output/boxplot.png"
)

plt.close()


# 9. 95% confidence intervals

z = 1.96

n1 = len(forward_group)
n2 = len(nonforward_group)

f_se = f_sd / np.sqrt(n1)
f_margin = z * f_se
f_lower = f_mean - f_margin
f_upper = f_mean + f_margin

nf_se = nf_sd / np.sqrt(n2)
nf_margin = z * nf_se
nf_lower = nf_mean - nf_margin
nf_upper = nf_mean + nf_margin

print("\n95% Confidence Intervals")

print("\nForward:")
print("Mean =", round(f_mean, 2))
print("Standard Error =", round(f_se, 2))
print("Margin of Error =", round(f_margin, 2))
print(
    "95% CI =",
    round(f_lower, 2),
    "to",
    round(f_upper, 2)
)

print("\nNon-Forward:")
print("Mean =", round(nf_mean, 2))
print("Standard Error =", round(nf_se, 2))
print("Margin of Error =", round(nf_margin, 2))
print(
    "95% CI =",
    round(nf_lower, 2),
    "to",
    round(nf_upper, 2)
)


plt.bar(
    [
        "Forward",
        "Non-Forward"
    ],
    [
        f_mean,
        nf_mean
    ],
    yerr=[
        f_margin,
        nf_margin
    ],
    capsize=8
)

plt.ylabel(
    "Mean Shots on Target Percentage"
)

plt.title(
    "Mean Shots on Target Percentage with 95% Confidence Intervals"
)

plt.tight_layout()

plt.savefig(
    "task2/output/ci_comparison.png"
)

plt.close()


# 10. Two-sample t-test

print("\nTwo-sample t-test")

print("\nState:")
print(
    "Do forwards have a different mean "
    "shots-on-target percentage compared with "
    "non-forward outfield players?"
)

print("\nPlan:")
print(
    "H0: Mean shots-on-target percentage is equal "
    "for forwards and non-forwards."
)

print(
    "Ha: Mean shots-on-target percentage is different "
    "for forwards and non-forwards."
)

print("Significance level = 0.05")

t_value = (
    f_mean - nf_mean
) / np.sqrt(
    (f_sd ** 2 / n1) +
    (nf_sd ** 2 / n2)
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
print("Forward mean =", round(f_mean, 2))
print("Non-Forward mean =", round(nf_mean, 2))

print(
    "Difference in means =",
    round(
        f_mean - nf_mean,
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
        "There is sufficient evidence that the mean "
        "shots-on-target percentage is different "
        "between forwards and non-forwards."
    )

else:
    print("Do not reject H0.")

    print(
        "There is not sufficient evidence that the mean "
        "shots-on-target percentage is different "
        "between forwards and non-forwards."
    )