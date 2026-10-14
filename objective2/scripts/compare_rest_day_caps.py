"""Exploratory rest-day robustness grid: uncapped, ±3, ±7, ±14, ±30, and excluded.
These are sensitivity checks, not independently pre-registered choices.
Use the same 5 CV folds and fixed model parameters in every comparison.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/processed'; R=ROOT/'output/results';R.mkdir(exist_ok=True,parents=True)
d=pd.read_csv(D/'objective2_1_uncapped.csv')
y=d['goal_difference']
cols=['fifa_ranking_difference','recent_form_difference','avg_goals_scored_difference','avg_goals_conceded_difference','clean_sheet_rate_difference','worldcup_experience_difference','rest_days_difference','knockout_stage_indicator']
rows=[]
for scenario,cap in [('uncapped',None),('cap_3',3),('cap_7',7),('cap_14',14),('cap_30',30),('exclude_rest_days','exclude')]:
 X=d[cols].copy()
 if cap=='exclude':X=X.drop(columns=['rest_days_difference'])
 elif cap is not None:X['rest_days_difference']=X['rest_days_difference'].clip(-cap,cap)
 for name,model in {'Baseline':DummyRegressor(), 'OLS':Pipeline([('scale',StandardScaler()),('model',LinearRegression())]),'Ridge':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=10))])}.items():
  vals=[]
  for tr,te in KFold(n_splits=5,shuffle=True,random_state=42).split(X):
   model.fit(X.iloc[tr],y.iloc[tr]);pred=model.predict(X.iloc[te]);vals.append((np.sqrt(mean_squared_error(y.iloc[te],pred)),r2_score(y.iloc[te],pred)))
  rows.append({'scenario':scenario,'model':name,'cv_RMSE_mean':np.mean([x[0] for x in vals]),'cv_RMSE_sd':np.std([x[0] for x in vals],ddof=1),'cv_R2_mean':np.mean([x[1] for x in vals])})
pd.DataFrame(rows).to_csv(R/'lr21_rest_day_cap_grid.csv',index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
