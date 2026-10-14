# FIFA World Cup 2026 Data Analysis

## HIT140 – Foundations of Data Science

**Charles Darwin University | Group 9**

This repository contains our group project for HIT140 – Foundations of Data Science at Charles Darwin University.

The project investigates FIFA World Cup 2026 player and match statistics using Python, statistical hypothesis testing, data visualisation and predictive modelling.

The project is divided into two objectives:

- **Objective 1:** Statistical Analysis and Hypothesis Testing
- **Objective 2:** Linear Regression and Predictive Modelling

Both objectives are organised into separate folders to improve reproducibility, maintainability and collaboration.

## Group Members

| Name | Student ID |
|---|---|
| Arun Paudel | S397796 |
| Bishal Siris | S397750 |
| Kanchan Lamichhane | S398347 |
| Rahul K C | S397990 |

**Lecturer:** Yakub Sabastian

---

# Objective 1 – Statistical Analysis and Hypothesis Testing

## Overview

Objective 1 investigates four research questions using FIFA World Cup 2026 player performance statistics.

The analyses include:

- Data cleaning and preprocessing
- Feature engineering
- Random sampling
- Descriptive statistics
- Outlier detection
- 95% confidence intervals
- Two-sample hypothesis testing
- Statistical visualisation and interpretation

All four tasks are located in the `objective1/` directory.

## Task 1 – Fouls and Tournament Progression

### Research Question

Among outfield players who played at least 90 minutes, is the mean fouls-per-90 rate significantly different between players from teams eliminated in the group stage and players from teams that reached the knockout stage?

### Methodology

We calculated fouls per 90 minutes and compared players from group-stage exit teams with players from knockout-stage teams.

A random sample of 30 players was selected from each group.

### Results

| Measure | Value |
|---|---:|
| Group Exit mean | 0.94 fouls per 90 |
| Knockout mean | 1.03 fouls per 90 |
| p-value | 0.6409 |
| Significance level | 0.05 |

**Conclusion:** Since the p-value exceeds 0.05, we failed to reject the null hypothesis. There was insufficient statistical evidence of a difference between the two groups.

**Files:**

- `objective1/task1/task1_pressing_discipline.py`
- `objective1/task1/misc_stats.csv`
- `objective1/task1/standard_stats.csv`

Visualisations are stored in `objective1/task1/output/`.

---

## Task 2 – Shooting Performance

### Research Question

Do forwards have a significantly different mean shots-on-target percentage compared with non-forward outfield players?

### Methodology

Shots-on-target percentage was calculated as:

`SoT_pct = (SoT / Sh) * 100`

Goalkeepers and players with zero shots were excluded.

We randomly selected 30 forwards and 30 non-forward outfield players for comparison.

### Results

| Measure | Value |
|---|---:|
| Forward mean | 34.94% |
| Non-forward mean | 24.80% |
| p-value | 0.2045 |
| Significance level | 0.05 |

**Conclusion:** Although forwards had a higher sample mean, the difference was not statistically significant at the 5% significance level.

**Files:**

- `objective1/task2/task2_analysis.py`
- `objective1/task2/standard.csv`
- `objective1/task2/shooting.csv`

Visualisations are stored in `objective1/task2/output/`.

---

## Task 3 – Player Utilisation

### Research Question

Do defenders have a significantly different mean minutes played per appearance compared with forwards?

### Methodology

Minutes played per appearance were calculated using:

`Min_per_MP = Min / MP`

A random sample of 30 defenders was compared with 30 forwards.

### Results

| Measure | Value |
|---|---:|
| Defender mean | 74.67 minutes |
| Forward mean | 42.27 minutes |
| Mean difference | 32.40 minutes |
| p-value | 0.0001 |
| Significance level | 0.05 |

**Conclusion:** The p-value was below 0.05, so we rejected the null hypothesis.

Defenders had significantly higher mean minutes played per appearance than forwards in the analysed sample.

However, this result does not establish the cause of the difference.

**Files:**

- `objective1/task3/task3_analysis.py`
- `objective1/task3/playing_time.csv`

Visualisations are stored in `objective1/task3/output/`.

---

## Task 4 – Tackles Won

### Research Question

Do defenders have a significantly higher mean number of tackles won per 90 minutes than midfielders?

### Methodology

Tackles won per 90 minutes were calculated using:

`TklW_per90 = TklW / 90s`

The analysis included players classified as defenders (DF) or midfielders (MF) who played at least 90 minutes.

After preprocessing, the dataset contained 253 defenders and 254 midfielders.

We randomly sampled 30 players from each group using `random_state=42`.

Outliers were examined using the interquartile range (IQR) method. Seventeen outliers were identified and retained because they could represent genuine performances.

A one-sided two-sample t-test was performed.

### Results

| Measure | Value |
|---|---:|
| Defender mean | 1.06 tackles won per 90 |
| Midfielder mean | 1.05 tackles won per 90 |
| Defender 95% CI | 0.74–1.38 |
| Midfielder 95% CI | 0.71–1.40 |
| t-statistic | 0.03 |
| p-value | 0.4880 |

**Conclusion:** Since the p-value was greater than 0.05, we failed to reject the null hypothesis.

There was insufficient evidence that defenders had a higher mean tackles-won-per-90 rate than midfielders.

**Files:**

- `objective1/task4/task4_tackles_position.py`
- `objective1/task4/misc_stats.csv`

Visualisations are stored in `objective1/task4/output/`.

---

## Objective 1 – Overall Results

| Task | Comparison | p-value | Statistical Result |
|---|---|---:|---|
| Task 1 | Group Exit vs Knockout | 0.6409 | Not significant |
| Task 2 | Forward vs Non-forward | 0.2045 | Not significant |
| Task 3 | Defender vs Forward | 0.0001 | Significant |
| Task 4 | Defender vs Midfielder | 0.4880 | Not significant |

### Key Findings

Task 3 demonstrated a statistically significant difference in minutes played per appearance between defenders and forwards.

The other three tasks did not provide sufficient evidence to reject their respective null hypotheses at the 5% significance level.

These findings demonstrate the importance of statistical testing rather than relying only on observed differences in sample means.

---

# Objective 2 – Linear Regression and Predictive Modelling

## Overview

Objective 2 develops and evaluates regression models to investigate the relationship between pre-match explanatory variables and FIFA World Cup 2026 match outcomes.

Unlike Objective 1, which focuses on hypothesis testing, Objective 2 focuses on predictive modelling and evaluating model performance.

Two related but distinct regression tasks were undertaken.

## Linear Regression 2.1 – Match Goal Difference Prediction

### Research Objective

Develop a regression model to predict the goal difference between two opposing teams in a FIFA World Cup 2026 match.

### Dataset Requirements

- 104 observations representing actual matches
- Exactly eight explanatory variables
- All explanatory variables must be available before the match
- Goal difference is the target variable

### Methodology

The analysis includes:

- Data cleaning and preprocessing
- Construction of pre-match explanatory variables
- Exploratory data analysis
- Correlation and multicollinearity analysis
- Linear regression modelling
- Comparison of alternative regression algorithms
- Cross-validation and holdout evaluation
- Residual diagnostics and visualisations
- Sensitivity analysis

### Model Evaluation

Model performance is evaluated using regression metrics such as:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

Alternative regression approaches are compared to assess predictive performance and model stability.

The detailed implementation and generated results are available in the `objective2/` directory.

---

## Linear Regression 2.2 – Team Goals Scored Prediction

### Research Objective

Develop a regression model to predict the number of goals scored by an individual team in a FIFA World Cup 2026 match.

### Dataset Requirements

- 208 team-match observations
- Exactly eight explanatory variables
- All explanatory variables must be available before the match
- Goals scored by a team is the target variable

### Methodology

The analysis includes:

- Team-level data preparation
- Pre-match feature engineering
- Descriptive and exploratory analysis
- Regression model development
- Cross-validation
- Comparison of competing models
- Evaluation of predictive accuracy
- Residual analysis
- Model interpretation and visualisation

### Model Evaluation

Regression models are evaluated and compared using RMSE, MAE and R².

The analysis examines model performance, explanatory relationships and limitations associated with predicting football outcomes.

---

## Objective 2 – Python Implementation

Objective 2 includes an automated Python analysis pipeline.

The pipeline supports:

- Data preprocessing and validation
- Feature engineering
- Regression model training
- Cross-validation and model comparison
- Generation of statistical results
- Generation of figures and diagnostic plots
- Reproducible execution of the analysis

**Main execution file:**

`objective2/run_all.py`

**Additional documentation:**

`objective2/README.md`

---

# Project Structure

```text
FIFA-World-Cup-2026-Data-Analysis/
│
├── README.md
├── .gitignore
│
├── objective1/
│   ├── task1/
│   │   ├── task1_pressing_discipline.py
│   │   ├── misc_stats.csv
│   │   ├── standard_stats.csv
│   │   └── output/
│   │
│   ├── task2/
│   │   ├── task2_analysis.py
│   │   ├── standard.csv
│   │   ├── shooting.csv
│   │   └── output/
│   │
│   ├── task3/
│   │   ├── task3_analysis.py
│   │   ├── playing_time.csv
│   │   └── output/
│   │
│   └── task4/
│       ├── task4_tackles_position.py
│       ├── misc_stats.csv
│       └── output/
│
└── objective2/
    ├── README.md
    ├── requirements.txt
    ├── run_all.py
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── scripts/
    │   ├── preprocessing/
    │   └── Python analysis scripts
    │
    └── output/
        ├── figures/
        └── results/
```

---

# Installation and Execution

## Prerequisites

- Python 3.11 or later recommended
- Git
- Python packages specified in the project dependencies

## Clone the Repository

```bash
git clone https://github.com/APdel8848/FIFA-World-Cup-2026-Data-Analysis.git
cd FIFA-World-Cup-2026-Data-Analysis
```

## Run Objective 1

Install the required packages:

```bash
python3 -m pip install pandas numpy matplotlib scipy
```

Execute the four tasks from the repository root:

```bash
python3 objective1/task1/task1_pressing_discipline.py
python3 objective1/task2/task2_analysis.py
python3 objective1/task3/task3_analysis.py
python3 objective1/task4/task4_tackles_position.py
```

**Note:** If an Objective 1 script expects a CSV file relative to its own folder, run that script from its task directory or adjust its data-file path accordingly.

## Run Objective 2

Navigate to the Objective 2 folder:

```bash
cd objective2
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the full pipeline:

```bash
python3 run_all.py
```

Generated results and visualisations are stored in:

- `objective2/output/results/`
- `objective2/output/figures/`

For detailed instructions, refer to `objective2/README.md`.

---

# Tools and Technologies

| Technology | Purpose |
|---|---|
| Python | Data analysis and predictive modelling |
| pandas | Data cleaning and manipulation |
| NumPy | Numerical calculations |
| SciPy | Statistical hypothesis testing |
| Matplotlib | Data visualisation |
| scikit-learn | Regression modelling and evaluation |
| Git | Version control |
| GitHub | Team collaboration and code review |
| VS Code | Development environment |

---

# Team Collaboration and Version Control

The project was developed as a group using Git and GitHub to manage code, datasets and documentation.

Our collaboration workflow included:

- Organising analyses into separate task and objective folders
- Using branches for development and integration
- Submitting pull requests to integrate project work
- Reviewing changes before merging
- Maintaining commit history for version tracking
- Using shared project documentation to support reproducibility

GitHub provides a record of contributions, pull requests, code reviews and project integration.

---

# Limitations

## Objective 1

The analyses focus on FIFA World Cup 2026 tournament statistics rather than full-season player performance.

Differences in player participation, position classification and sample size may affect the generalisability of the findings.

Statistical significance does not establish causation.

## Objective 2

Football match outcomes are influenced by numerous factors, including team selection, tactical decisions, injuries, match conditions and unpredictable events.

Predictive models based on pre-match information may not capture all sources of variation.

The analysis therefore focuses on model evaluation, predictive accuracy and the interpretation of limitations rather than assuming that match outcomes can be predicted perfectly.

---

# Conclusion

This project demonstrates the application of foundational data science techniques to FIFA World Cup 2026 player and match statistics.

**Objective 1** used data preprocessing, random sampling, confidence intervals and hypothesis testing to investigate four research questions. Three tasks produced non-significant results, while Task 3 identified a statistically significant difference in playing time between defenders and forwards.

**Objective 2** extended the analysis through predictive modelling, developing regression models for match goal difference and team goals scored using pre-match explanatory variables.

Together, both objectives demonstrate practical applications of Python programming, statistical reasoning, data visualisation, predictive modelling and collaborative software development.