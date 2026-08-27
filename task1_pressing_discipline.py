import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# load datasets (skip top header from fbref export)
standard = pd.read_csv('standard_stats.csv', skiprows=1)
misc = pd.read_csv('misc_stats.csv', skiprows=1)

# clean out duplicate header rows inside data
standard = standard[standard['Player'] != 'Player']
misc = misc[misc['Player'] != 'Player']

# combine the columns we need
df = pd.merge(
    standard[['Player', 'Squad', 'Pos', 'MP', '90s']],
    misc[['Player', 'Squad', 'Fls']],
    on=['Player', 'Squad']
)

# convert to numeric
df['90s'] = pd.to_numeric(df['90s'], errors='coerce')
df['Fls'] = pd.to_numeric(df['Fls'], errors='coerce')

# drop missing rows, zero minutes, and goalies
df = df.dropna(subset=['90s', 'Fls'])
df = df[(df['90s'] > 0) & (df['Pos'] != 'GK')]

# foul rate per 90
df['Fls_per90'] = df['Fls'] / df['90s']

# label teams that got knocked out in group stage
eliminated_teams = [
    'South Korea', 'Czechia', 'Qatar', 'Scotland', 'Haiti', 'Turkey',
    'Curacao', 'Tunisia', 'Iran', 'New Zealand', 'Uruguay',
    'Saudi Arabia', 'Iraq', 'Jordan', 'Uzbekistan', 'Panama'
]

df['Stage'] = 'Knockout Stage'
for team in eliminated_teams:
    df.loc[df['Squad'].str.contains(team, case=False, na=False), 'Stage'] = 'Group Stage Exit'

print("Total players:", len(df))

# grab random 60% sample from each group
sample_exit = df[df['Stage'] == 'Group Stage Exit'].sample(frac=0.6, random_state=42)
sample_ko = df[df['Stage'] == 'Knockout Stage'].sample(frac=0.6, random_state=42)
sample = pd.concat([sample_exit, sample_ko])

group_exit = sample[sample['Stage'] == 'Group Stage Exit']['Fls_per90']
group_ko = sample[sample['Stage'] == 'Knockout Stage']['Fls_per90']

# summary stats
print("\nGroup Stage Exit:")
print("Mean:", group_exit.mean())
print("Median:", group_exit.median())
print("Std Dev:", group_exit.std())

print("\nKnockout Stage:")
print("Mean:", group_ko.mean())
print("Median:", group_ko.median())
print("Std Dev:", group_ko.std())

# plot comparison
sample.boxplot(column='Fls_per90', by='Stage')
plt.title('Fouls per 90 Mins: Group Exit vs Knockout')
plt.suptitle('')
plt.ylabel('Fouls per 90')
plt.savefig('fouls_plot.png')
plt.show()

# 95% confidence intervals
ci_exit = stats.t.interval(0.95, df=len(group_exit)-1, loc=group_exit.mean(), scale=stats.sem(group_exit))
print("\n95% CI (Exit):", ci_exit)

ci_ko = stats.t.interval(0.95, df=len(group_ko)-1, loc=group_ko.mean(), scale=stats.sem(group_ko))
print("95% CI (Knockout):", ci_ko)

# t-test
t_stat, p_val = stats.ttest_ind(group_exit, group_ko, equal_var=False)

print("\nt-test results:")
print("t-stat:", t_stat)
print("p-value:", p_val)

if p_val < 0.05:
    print("Result: Significant difference (reject H0)")
else:
    print("Result: No significant difference (fail to reject H0)")