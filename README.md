
# FIFA World Cup 2026 Data Analysis (PRT304)

This repository contains the complete analytical workflow, datasets, scripts, and output visualisations for PRT304 Checkpoint 2 (Objective 1).

## Tasks Overview
* **Task 1: Defensive Discipline & Fouls (`task1/`)**
  * Evaluates fouls committed per 90 minutes between group-stage eliminated teams and advancing knockout teams.
* **Task 2: Shooting Accuracy & Forwards (`task2/`)**
  * Compares shots-on-target percentages between forwards and non-forward outfield players.
* **Task 3: Playing Time & Midfield Performance (`task3/`)**
  * Analyzes match duration and playing time metrics across domestic league cohorts.
* **Task 4: Age & Playing Role Analysis (`task4/`)**
  * Investigates tournament playing time between Under-23 and senior (24+) players.

## How to Run
Each task contains its respective dataset, Python analysis script, and `output/` folder with generated histograms, boxplots, and confidence interval figures. Run any script directly:
```bash
python task1/task1_pressing_discipline.py
python task2/task2_analysis.py
python task3/task3_analysis.py
python task4/task4_analysis.py
