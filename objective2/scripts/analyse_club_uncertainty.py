"""Sensitivity of LR2.2 results to unresolved club classifications.
This is NOT validation of provisional classifications; extremes are hypothetical bounds.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge,Lasso,LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/'data/processed'; OUT=ROOT/'output/results'
d=pd.read_csv(PROC/'objective2_2_final.csv')
s=pd.read_csv(PROC/'squad_features_submission.csv')
tm=pd.read_csv(PROC/'objective2_2_team_matches.csv')
assert len(d)==len(tm)==208
# Order-preserving ID join; no use of same-match target-derived variables.
tm=tm[['match_id','team']]
assert (d.match_id.to_numpy()==tm.match_id.to_numpy()).all()
prov=tm.merge(s[['team','provisional_club_players']],on='team',validate='many_to_one')
assert (prov.match_id.to_numpy()==d.match_id.to_numpy()).all()
features=[c for c in d if c not in ('match_id','goals_scored')]
assert len(features)==8
folds=list(GroupKFold(n_splits=5).split(d[features],d.goals_scored,groups=d.match_id))
models={'OLS':LinearRegression(),'Ridge':Ridge(alpha=10),'Lasso':Lasso(alpha=.05,max_iter=20000)}
rows=[]
for scenario in ('mapped_only_lower_bound','all_unresolved_upper_bound','exclude_top5_predictor'):
 X=d[features].copy()
 if scenario=='all_unresolved_upper_bound':
  X['top5_league_player_count']+=prov.provisional_club_players.to_numpy()
 if scenario=='exclude_top5_predictor':X=X.drop(columns=['top5_league_player_count'])
 for modelname,reg in models.items():
  for fold,(train,test) in enumerate(folds,1):
   model=Pipeline([('scale',StandardScaler()),('regression',reg)])
   model.fit(X.iloc[train],d.goals_scored.iloc[train])
   pred=model.predict(X.iloc[test])
   rows.append({'scenario':scenario,'model':modelname,'fold':fold,
    'RMSE':np.sqrt(mean_squared_error(d.goals_scored.iloc[test],pred)),
    'R2':r2_score(d.goals_scored.iloc[test],pred)})
res=pd.DataFrame(rows)
res.to_csv(OUT/'lr22_top5_uncertainty_folds.csv',index=False)
summary=res.groupby(['scenario','model'],as_index=False).agg(cv_RMSE=('RMSE','mean'),cv_R2=('R2','mean'))
summary.to_csv(OUT/'lr22_top5_uncertainty_summary.csv',index=False)
print('Top-five count sensitivity (hypothetical bounds, not source verification):')
print(summary.round(3).to_string(index=False))
