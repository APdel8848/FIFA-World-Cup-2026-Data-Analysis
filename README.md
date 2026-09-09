# FIFA World Cup 2026 Data Analysis

## HIT140 – Foundations of Data Science

This repository contains our group project for HIT140 Foundations of Data Science at Charles Darwin University.

Our project looks at FIFA World Cup 2026 player statistics. We used Python to clean the datasets, create new variables, take random samples, calculate descriptive statistics and confidence intervals, create graphs, and perform hypothesis tests.

## Group Members

- Arun Paudel
- Bishal Siris
- Kanchan Lamichhane
- Rahul K C

Lecturer: Yakub Sabastian

---

## Project Tasks

We divided the project into four analysis tasks.

### Task 1 – Fouls and Tournament Progression

**Question:**  
Among outfield players who played at least 90 minutes, is the mean fouls-per-90 rate significantly different between players from teams eliminated in the group stage and players from teams that reached the knockout stage?

We calculated fouls per 90 minutes and compared players from group-stage exit teams with players from knockout-stage teams.

A random sample of 30 players was taken from each group.

**Results:**

- Group Exit mean = 0.94 fouls per 90
- Knockout mean = 1.03 fouls per 90
- p-value = 0.6409

Since the p-value was greater than 0.05, we did not reject the null hypothesis. We did not find enough evidence of a significant difference between the two groups.

Main code:

`task1/task1_pressing_discipline.py`

---

### Task 2 – Shooting Performance

**Question:**  
Do forwards have a significantly different mean shots-on-target percentage compared with non-forward outfield players?

Shots-on-target percentage was calculated using:

`SoT_pct = (SoT / Sh) * 100`

Goalkeepers and players with no shots were excluded. We then randomly sampled 30 forwards and 30 non-forwards.

**Results:**

- Forward mean = 34.94%
- Non-forward mean = 24.80%
- p-value = 0.2045

The forwards had a higher mean in our sample, but the difference was not statistically significant at the 0.05 level.

Files used:

- `task2/task2_analysis.py`
- `task2/standard.csv`
- `task2/shooting.csv`

Graphs are saved in `task2/output/`.

---

### Task 3 – Player Utilisation

**Question:**  
Do defenders have a significantly different mean minutes played per appearance compared with forwards?

For this task we calculated:

`Min_per_MP = Min / MP`

We compared a random sample of 30 defenders with 30 forwards.

**Results:**

- Defender mean = 74.67 minutes per appearance
- Forward mean = 42.27 minutes per appearance
- Difference = 32.40 minutes
- p-value = 0.0001

This was the only task where we found a statistically significant difference. Since the p-value was below 0.05, we rejected the null hypothesis.

The result shows a difference in playing time between the two position groups in our sample, but it does not tell us the cause of the difference.

Files:

- `task3/task3_analysis.py`
- `task3/playing_time.csv`

Graphs are saved in `task3/output/`.

---

### Task 4 – Tackles Won

**Question:**  
Do defenders have a significantly higher mean number of tackles won per 90 minutes than midfielders?

For this task we used:

`TklW_per90 = TklW / 90s`

We only included players listed clearly as defenders (DF) or midfielders (MF) who played at least 90 minutes.

After cleaning the data there were 253 defenders and 254 midfielders. We randomly selected 30 players from each group using `random_state=42`.

We also checked for outliers using the IQR method. There were 17 outliers, but we kept them because they could represent genuine player performances.

**Results:**

- Defender mean = 1.06 tackles won per 90
- Midfielder mean = 1.05 tackles won per 90
- Defender 95% CI = 0.74 to 1.38
- Midfielder 95% CI = 0.71 to 1.40
- t-statistic = 0.03
- p-value = 0.4880

We used a one-sided two-sample t-test because our question asked whether defenders had a higher mean.

The p-value was greater than 0.05, so we did not reject the null hypothesis. There was not enough evidence to say that defenders had a higher mean tackles-won-per-90 rate than midfielders.

Files:

- `task4/task4_tackles_position.py`
- `task4/misc_stats.csv`

Graphs are saved in `task4/output/`.

---

## Overall Results

| Task | Comparison | p-value | Result |
|---|---|---:|---|
| 1 | Group Exit vs Knockout | 0.6409 | Not significant |
| 2 | Forward vs Non-Forward | 0.2045 | Not significant |
| 3 | Defender vs Forward | 0.0001 | Significant |
| 4 | Defender vs Midfielder | 0.4880 | Not significant |

The most noticeable result from our project was Task 3. Defenders had a much higher average number of minutes per appearance than forwards in the sample.

For the other three tasks, we did not find enough statistical evidence to reject the null hypotheses.

---

## Methods Used

Some of the Python and statistical methods used in the project were:

- pandas for loading and cleaning data
- NumPy for calculations
- Matplotlib for graphs
- SciPy for statistical calculations
- data cleaning and filtering
- creating new variables
- random sampling
- descriptive statistics
- outlier checking
- 95% confidence intervals
- two-sample t-tests

For the samples we used `random_state=42` so that the same random samples can be reproduced.

---

## Project Structure

```text
FIFA-World-Cup-2026-Data-Analysis/
│
├── README.md
│
├── task1/
│   ├── task1_pressing_discipline.py
│   └── output/
│
├── task2/
│   ├── task2_analysis.py
│   ├── standard.csv
│   ├── shooting.csv
│   └── output/
│
├── task3/
│   ├── task3_analysis.py
│   ├── playing_time.csv
│   └── output/
│
└── task4/
    ├── task4_tackles_position.py
    ├── misc_stats.csv
    └── output/
```

---

## How to Run

Python 3 is required.

Install the packages if needed:

```bash
python3 -m pip install pandas numpy matplotlib scipy
```

Clone the repository:

```bash
git clone https://github.com/APdel8848/FIFA-World-Cup-2026-Data-Analysis.git
```

Open the project folder:

```bash
cd FIFA-World-Cup-2026-Data-Analysis
```

Run the tasks from the main project folder:

```bash
python3 task1/task1_pressing_discipline.py
python3 task2/task2_analysis.py
python3 task3/task3_analysis.py
python3 task4/task4_tackles_position.py
```

---

## Limitations

There are some limitations in our analysis. The data only represents the tournament, so it does not show a player's overall performance across a full season. Players also had different amounts of playing time, and some mixed-position players had to be excluded from certain comparisons.

The statistical tests show whether there is evidence of a difference between groups, but they do not explain why the difference happened.

---

## Conclusion

This project helped us apply the data science methods covered in HIT140 to a football dataset.

We used Python to prepare and analyse the data and then used confidence intervals and hypothesis tests to answer our four questions.

Three tasks did not show statistically significant differences. Task 3 showed a significant difference in minutes played per appearance between defenders and forwards.