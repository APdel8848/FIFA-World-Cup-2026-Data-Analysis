import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


# create output folder
os.makedirs("task4/output", exist_ok=True)


# ==========================================
# TASK 4 - AGE AND STARTING STATUS
# ==========================================

# Research question:
# Do regular starters have a different average age
# compared with non-regular starters at the FIFA World Cup 2026?


# ==========================================
# STEP 1 - LOAD DATA
# ==========================================

playing_time = pd.read_csv(
    "task4/playing_time.csv"
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
        "Age",
        "MP",
        "Min",
        "Starts"
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
data["Age"] = pd.to_numeric(
    data["Age"],
    errors="coerce"
)

data["MP"] = pd.to_numeric(
    data["MP"],
    errors="coerce"
)

data["Min"] = pd.to_numeric(
    data["Min"],
    errors="coerce"
)

data["Starts"] = pd.to_numeric(
    data["Starts"],
    errors="coerce"
)


# remove missing values needed for analysis
data = data.dropna(
    subset=[
        "Age",
        "MP",
        "Starts"
    ]
)


# only include players who actually appeared
data = data[
    data["MP"] > 0
]


print(
    "\nPlayers after cleaning:",
    len(data)
)


# ==========================================
# CREATE STARTING STATUS GROUPS
# ==========================================

# Regular Starter:
# player started at least 2 matches
#
# Non-Regular Starter:
# player started fewer than 2 matches

data["Group"] = np.where(
    data["Starts"] >= 2,
    "Regular Starter",
    "Non-Regular Starter"
)


print(
    "\nPlayers by starting group:"
)

print(
    data["Group"].value_counts()
)


# ==========================================
# OUTLIER CHECK
# ==========================================

Q1 = data["Age"].quantile(0.25)
Q3 = data["Age"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR


outliers = data[
    (data["Age"] < lower)
    |
    (data["Age"] > upper)
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
    "Outliers are kept because younger or older players "
    "can represent real tournament selections."
)


# ==========================================
# POPULATION INFORMATION
# ==========================================

regular_population = data[
    data["Group"] == "Regular Starter"
]

nonregular_population = data[
    data["Group"] == "Non-Regular Starter"
]


print("\nPopulation sizes:")

print(
    "Regular Starter =",
    len(regular_population)
)

print(
    "Non-Regular Starter =",
    len(nonregular_population)
)


print("\nPopulation means:")

print(
    "Regular Starter mean age =",
    round(
        regular_population["Age"].mean(),
        2
    )
)

print(
    "Non-Regular Starter mean age =",
    round(
        nonregular_population["Age"].mean(),
        2
    )
)


print("\nPopulation standard deviations:")

print(
    "Regular Starter SD =",
    round(
        regular_population["Age"].std(
            ddof=0
        ),
        2
    )
)

print(
    "Non-Regular Starter SD =",
    round(
        nonregular_population["Age"].std(
            ddof=0
        ),
        2
    )
)


# ==========================================
# STEP 3 - SAMPLING
# ==========================================

# take 30 random regular starters
regular_sample = regular_population.sample(
    n=30,
    random_state=42
)


# take 30 random non-regular starters
nonregular_sample = nonregular_population.sample(
    n=30,
    random_state=42
)


# combine samples
sample = pd.concat(
    [
        regular_sample,
        nonregular_sample
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
regular_group = regular_sample["Age"]

nonregular_group = nonregular_sample["Age"]


# ==========================================
# STEP 4 - DESCRIPTIVE STATISTICS
# ==========================================

print(
    "\n--- STEP 4: DESCRIPTIVE STATISTICS ---"
)


# Regular Starter statistics
r_mean = regular_group.mean()

r_median = regular_group.median()

r_mode = regular_group.mode()[0]

r_range = (
    regular_group.max()
    -
    regular_group.min()
)

r_q1 = regular_group.quantile(0.25)

r_q3 = regular_group.quantile(0.75)

r_iqr = r_q3 - r_q1

r_variance = regular_group.var()

r_sd = regular_group.std()


# Non-Regular Starter statistics
nr_mean = nonregular_group.mean()

nr_median = nonregular_group.median()

nr_mode = nonregular_group.mode()[0]

nr_range = (
    nonregular_group.max()
    -
    nonregular_group.min()
)

nr_q1 = nonregular_group.quantile(0.25)

nr_q3 = nonregular_group.quantile(0.75)

nr_iqr = nr_q3 - nr_q1

nr_variance = nonregular_group.var()

nr_sd = nonregular_group.std()


print("\nRegular Starter:")

print(
    "Mean =",
    round(r_mean, 2)
)

print(
    "Median =",
    round(r_median, 2)
)

print(
    "Mode =",
    round(r_mode, 2)
)

print(
    "Range =",
    round(r_range, 2)
)

print(
    "IQR =",
    round(r_iqr, 2)
)

print(
    "Variance =",
    round(r_variance, 2)
)

print(
    "Standard deviation =",
    round(r_sd, 2)
)


print("\nNon-Regular Starter:")

print(
    "Mean =",
    round(nr_mean, 2)
)

print(
    "Median =",
    round(nr_median, 2)
)

print(
    "Mode =",
    round(nr_mode, 2)
)

print(
    "Range =",
    round(nr_range, 2)
)

print(
    "IQR =",
    round(nr_iqr, 2)
)

print(
    "Variance =",
    round(nr_variance, 2)
)

print(
    "Standard deviation =",
    round(nr_sd, 2)
)


# ==========================================
# HISTOGRAM
# ==========================================

plt.hist(
    regular_group,
    bins=8,
    alpha=0.6,
    label="Regular Starter"
)

plt.hist(
    nonregular_group,
    bins=8,
    alpha=0.6,
    label="Non-Regular Starter"
)

plt.xlabel(
    "Age"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Age by Starting Status"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "task4/output/histogram.png"
)

plt.close()


# ==========================================
# BOXPLOT
# ==========================================

plt.boxplot(
    [
        regular_group,
        nonregular_group
    ],
    tick_labels=[
        "Regular Starter",
        "Non-Regular Starter"
    ]
)

plt.xlabel(
    "Starting Status"
)

plt.ylabel(
    "Age"
)

plt.title(
    "Age by Starting Status"
)

plt.tight_layout()

plt.savefig(
    "task4/output/boxplot.png"
)

plt.close()


# ==========================================
# STEP 5 - 95% CONFIDENCE INTERVAL
# ==========================================

print(
    "\n--- STEP 5: 95% CONFIDENCE INTERVAL ---"
)


z = 1.96

n1 = len(regular_group)

n2 = len(nonregular_group)


# Regular Starter confidence interval
r_se = (
    r_sd / np.sqrt(n1)
)

r_margin = (
    z * r_se
)

r_lower = (
    r_mean - r_margin
)

r_upper = (
    r_mean + r_margin
)


print("\nRegular Starter 95% CI:")

print(
    "Mean =",
    round(r_mean, 2)
)

print(
    "Standard Error =",
    round(r_se, 2)
)

print(
    "Margin of Error =",
    round(r_margin, 2)
)

print(
    "95% CI =",
    round(r_lower, 2),
    "to",
    round(r_upper, 2)
)


# Non-Regular Starter confidence interval
nr_se = (
    nr_sd / np.sqrt(n2)
)

nr_margin = (
    z * nr_se
)

nr_lower = (
    nr_mean - nr_margin
)

nr_upper = (
    nr_mean + nr_margin
)


print("\nNon-Regular Starter 95% CI:")

print(
    "Mean =",
    round(nr_mean, 2)
)

print(
    "Standard Error =",
    round(nr_se, 2)
)

print(
    "Margin of Error =",
    round(nr_margin, 2)
)

print(
    "95% CI =",
    round(nr_lower, 2),
    "to",
    round(nr_upper, 2)
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
    "Do regular starters have a different "
    "average age compared with non-regular starters?"
)


# PLAN
print("\nPLAN:")

print(
    "H0: Mean age is equal for regular starters "
    "and non-regular starters."
)

print(
    "Ha: Mean age is different for regular starters "
    "and non-regular starters."
)

print(
    "Significance level = 0.05"
)

print(
    "Test = Independent two-sample t-test"
)


# SOLVE
t_value = (
    r_mean - nr_mean
) / np.sqrt(
    (r_sd ** 2 / n1)
    +
    (nr_sd ** 2 / n2)
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
    "Regular Starter mean age =",
    round(r_mean, 2)
)

print(
    "Non-Regular Starter mean age =",
    round(nr_mean, 2)
)

print(
    "Difference in means =",
    round(
        r_mean - nr_mean,
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
        "the average age is different between "
        "regular starters and non-regular starters."
    )

else:

    print(
        "Do not reject H0."
    )

    print(
        "There is not sufficient evidence that "
        "the average age is different between "
        "regular starters and non-regular starters."
    )