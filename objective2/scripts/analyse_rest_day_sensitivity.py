"""Post-hoc sensitivity across uncapped, ±3, ±7, ±14 and no-rest specifications."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
root=Path(__file__).resolve().parents[1]
proc=root/'data/processed'; out=root/'output/results'
cols=['fifa_ranking_difference','recent_form_difference','avg_goals_scored_difference','avg_goals_conceded_difference','clean_sheet_rate_difference','worldcup_experience_difference','rest_days_difference','knockout_stage_indicator']
rows=[]
for version in ['uncapped','capped_pm3','capped_pm7','capped_pm14','without_rest_variable']:
 d=pd.read_csv(proc/'objective2_1_uncapped.csv')
 current_cols=cols.copy()
 if version.startswith('capped_pm'):
  cap=int(version.replace('capped_pm',''))
  d['rest_days_difference']=d['rest_days_difference'].clip(-cap,cap)
 if version=='without_rest_variable':
  current_cols.remove('rest_days_difference')
 X=d[current_cols];y=d.goal_difference
 folds=KFold(n_splits=5,shuffle=True,random_state=42)
 for name,model in [('OLS',make_pipeline(StandardScaler(),LinearRegression())),('Ridge',make_pipeline(StandardScaler(),Ridge(alpha=10))),('Random Forest',RandomForestRegressor(n_estimators=150,min_samples_leaf=3,max_features=0.8,random_state=42,n_jobs=-1))]:
  for fold,(tr,te) in enumerate(folds.split(X),1):
   model.fit(X.iloc[tr],y.iloc[tr]);p=model.predict(X.iloc[te]);rows.append({'version':version,'model':name,'fold':fold,'RMSE':np.sqrt(mean_squared_error(y.iloc[te],p)),'R2':r2_score(y.iloc[te],p)})
result=pd.DataFrame(rows);result.to_csv(out/'lr21_rest_cap_sensitivity_folds.csv',index=False)
summary=result.groupby(['version','model'])[['RMSE','R2']].mean().round(4)
summary.to_csv(out/'lr21_rest_cap_sensitivity_summary.csv')
print(summary.to_string())
