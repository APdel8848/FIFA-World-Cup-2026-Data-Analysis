import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


# create output folder
os.makedirs("task2/output", exist_ok=True)


# ==========================================
# TASK 2 - SHOTS ON TARGET PERCENTAGE
# ==========================================

# Research question:
# Do forwards have a different average shots-on-target
# percentage compared with non-forward outfield players?


# ==========================================
# STEP 1 - LOAD DATA
# ==========================================

standard = pd.read_csv(
    "task2/standard.csv"
)

shooting = pd.read_csv(
    "task2/shooting.csv"
)

print("Standard shape:", standard.shape)
print("Shooting shape:", shooting.shape)


# ==========================================
# STEP 2 - DATA WRANGLING
# ==========================================

# keep only the columns needed
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
]

shooting = shooting[
    [
        "Player",
        "Squad",
        "Sh",
        "SoT",
        "SoT%"
    ]
]


# check missing values
print("\nMissing values in standard:")
print(standard.isnull().sum())

print("\nMissing values in shooting:")
print(shooting.isnull().sum())


# check duplicate player and team keys
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


# merge datasets
data = pd.merge(
    standard,
    shooting,
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


# convert shooting columns to numeric
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
]

print(
    "Players after removing goalkeepers:",
    len(data)
)


# check missing SoT percentage
print(
    "Missing SoT% after removing goalkeepers:",
    data["SoT%"].isnull().sum()
)


# remove players with missing SoT%
data = data.dropna(
    subset=[
        "SoT%"
    ]
)

print(
    "Players after removing missing SoT%:",
    len(data)
)


# construct shots-on-target percentage ourselves
data["SoT_pct"] = (
    data["SoT"] / data["Sh"]
) * 100


# ==========================================
# CREATE POSITION GROUPS
# ==========================================

# any position containing FW is treated as Forward
data["Group"] = np.where(
    data["Pos"].str.contains(
        "FW",
        na=False
    ),
    "Forward",
    "Non-Forward"
)


print("\nPlayers by position group:")
print(
    data["Group"].value_counts()
)


# ==========================================
# OUTLIER CHECK
# ==========================================

Q1 = data["SoT_pct"].quantile(0.25)
Q3 = data["SoT_pct"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = data[
    (data["SoT_pct"] < lower)
    |
    (data["SoT_pct"] > upper)
]


print("\nOutlier check:")
print("Q1 =", round(Q1, 2))
print("Q3 =", round(Q3, 2))
print("IQR =", round(IQR, 2))
print("Lower limit =", round(lower, 2))
print("Upper limit =", round(upper, 2))
print("Number of outliers =", len(outliers))

print(
    "Outliers are kept because unusual shooting "
    "percentages can represent real performances."
)


# ==========================================
# POPULATION INFORMATION
# ==========================================

forward_population = data[
    data["Group"] == "Forward"
]

nonforward_population = data[
    data["Group"] == "Non-Forward"
]


print("\nPopulation sizes:")

print(
    "Forward =",
    len(forward_population)
)

print(
    "Non-Forward =",
    len(nonforward_population)
)


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


# ==========================================
# STEP 3 - SAMPLING
# ==========================================

# take 30 random players from each group
forward_sample = forward_population.sample(
    n=30,
    random_state=42
)

nonforward_sample = nonforward_population.sample(
    n=30,
    random_state=42
)


# combine samples
sample = pd.concat(
    [
        forward_sample,
        nonforward_sample
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
    "so the CLT requirement is satisfied."
)


# variables for analysis
forward_group = forward_sample["SoT_pct"]
nonforward_group = nonforward_sample["SoT_pct"]


# ==========================================
# STEP 4 - DESCRIPTIVE STATISTICS
# ==========================================

print(
    "\n--- STEP 4: DESCRIPTIVE STATISTICS ---"
)


# Forward statistics
f_mean = forward_group.mean()
f_median = forward_group.median()
f_mode = forward_group.mode()[0]
f_range = forward_group.max() - forward_group.min()

f_q1 = forward_group.quantile(0.25)
f_q3 = forward_group.quantile(0.75)
f_iqr = f_q3 - f_q1

f_variance = forward_group.var()
f_sd = forward_group.std()


# Non-Forward statistics
nf_mean = nonforward_group.mean()
nf_median = nonforward_group.median()
nf_mode = nonforward_group.mode()[0]
nf_range = (
    nonforward_group.max()
    -
    nonforward_group.min()
)

nf_q1 = nonforward_group.quantile(0.25)
nf_q3 = nonforward_group.quantile(0.75)
nf_iqr = nf_q3 - nf_q1

nf_variance = nonforward_group.var()
nf_sd = nonforward_group.std()


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


print("\nNon-Forward:")

print(
    "Mean =",
    round(nf_mean, 2)
)

print(
    "Median =",
    round(nf_median, 2)
)

print(
    "Mode =",
    round(nf_mode, 2)
)

print(
    "Range =",
    round(nf_range, 2)
)

print(
    "IQR =",
    round(nf_iqr, 2)
)

print(
    "Variance =",
    round(nf_variance, 2)
)

print(
    "Standard deviation =",
    round(nf_sd, 2)
)


# ==========================================
# HISTOGRAM
# ==========================================

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

plt.xlabel(
    "Shots on Target Percentage"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Shots on Target Percentage"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "task2/output/histogram.png"
)

plt.close()


# ==========================================
# BOXPLOT
# ==========================================

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

plt.xlabel(
    "Player Group"
)

plt.ylabel(
    "Shots on Target Percentage"
)

plt.title(
    "Shots on Target Percentage by Player Group"
)

plt.tight_layout()

plt.savefig(
    "task2/output/boxplot.png"
)

plt.show()


# ==========================================
# STEP 5 - 95% CONFIDENCE INTERVAL
# ==========================================

print(
    "\n--- STEP 5: 95% CONFIDENCE INTERVAL ---"
)


z = 1.96

n1 = len(forward_group)
n2 = len(nonforward_group)


# Forward confidence interval
f_se = (
    f_sd / np.sqrt(n1)
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


# Non-Forward confidence interval
nf_se = (
    nf_sd / np.sqrt(n2)
)

nf_margin = (
    z * nf_se
)

nf_lower = (
    nf_mean - nf_margin
)

nf_upper = (
    nf_mean + nf_margin
)


print("\nNon-Forward 95% CI:")

print(
    "Mean =",
    round(nf_mean, 2)
)

print(
    "Standard Error =",
    round(nf_se, 2)
)

print(
    "Margin of Error =",
    round(nf_margin, 2)
)

print(
    "95% CI =",
    round(nf_lower, 2),
    "to",
    round(nf_upper, 2)
)


# ==========================================
# STEP 6 - TWO-SAMPLE T-TEST
# ==========================================

print(
    "\n--- STEP 6: TWO-SAMPLE T-TEST ---"
)


# STATE
print("\nSTATE:")

print(
    "Do forwards have a different average "
    "shots-on-target percentage compared with "
    "non-forward outfield players?"
)


# PLAN
print("\nPLAN:")

print(
    "H0: Mean shots-on-target percentage is equal "
    "for forwards and non-forwards."
)

print(
    "Ha: Mean shots-on-target percentage is different "
    "for forwards and non-forwards."
)

print(
    "Significance level = 0.05"
)

print(
    "Test = Independent two-sample t-test"
)


# SOLVE
t_value = (
    f_mean - nf_mean
) / np.sqrt(
    (f_sd ** 2 / n1)
    +
    (nf_sd ** 2 / n2)
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
    "Forward mean =",
    round(f_mean, 2)
)

print(
    "Non-Forward mean =",
    round(nf_mean, 2)
)

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


# CONCLUDE
print("\nCONCLUDE:")

if p_value <= 0.05:

    print(
        "Reject H0."
    )

    print(
        "There is sufficient evidence that "
        "the average shots-on-target percentage "
        "is different between forwards and "
        "non-forward outfield players."
    )

else:

    print(
        "Do not reject H0."
    )

    print(
        "There is not sufficient evidence that "
        "the average shots-on-target percentage "
        "is different between forwards and "
        "non-forward outfield players."
    )