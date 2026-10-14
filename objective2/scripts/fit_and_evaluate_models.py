"""Reproducible, leakage-aware LR2.1/LR2.2 model comparison.

LR2.2 groups both team rows by match_id in every split and CV fold.
Scaling is fitted only within training folds. Hyperparameters are fixed, not tuned.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/'data/processed'; RES=ROOT/'output/results'; FIG=ROOT/'output/figures'
RES.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
CONFIG={
 'lr21':{'path':'objective2_1_final.csv','target':'goal_difference','features':['fifa_ranking_difference','recent_form_difference','avg_goals_scored_difference','avg_goals_conceded_difference','clean_sheet_rate_difference','worldcup_experience_difference','rest_days_difference','knockout_stage_indicator']},
 'lr22':{'path':'objective2_2_final.csv','target':'goals_scored','features':['fifa_rank','recent_form','avg_goals_scored','avg_goals_conceded','opponent_elo_rating','squad_value_m','avg_age','top5_league_player_count']}
}

def models():
 return {
 'Baseline':DummyRegressor(strategy='mean'),
 'OLS':Pipeline([('scale',StandardScaler()),('model',LinearRegression())]),
 'Ridge':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=10.0))]),
 'Lasso':Pipeline([('scale',StandardScaler()),('model',Lasso(alpha=0.05,max_iter=20000))]),
 'Random Forest':RandomForestRegressor(n_estimators=150,min_samples_leaf=3,max_features=0.8,random_state=42,n_jobs=-1),
 'Gradient Boosting':GradientBoostingRegressor(n_estimators=100,learning_rate=0.05,max_depth=2,random_state=42)
 }

def metrics(y,p):
 return {'MAE':float(mean_absolute_error(y,p)),'MSE':float(mean_squared_error(y,p)),
         'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}

def holdout(d,X,y,frac,seed):
 if d['match_id'].duplicated().any():
  splitter=GroupShuffleSplit(n_splits=1,test_size=frac,random_state=seed)
  tr,te=next(splitter.split(X,y,groups=d.match_id))
 else:
  tr,te=train_test_split(np.arange(len(d)),test_size=frac,random_state=seed)
 return tr,te

def run(name,cfg):
 d=pd.read_csv(PROC/cfg['path']);cols=cfg['features'];target=cfg['target']
 assert len(cols)==8 and d[cols+[target]].notna().all().all()
 if name=='lr22': assert list(d.columns)==['match_id','goals_scored']+cols
 assert len(d)==(104 if name=='lr21' else 208)
 X=d[cols];y=d[target]; grouped=(name=='lr22')
 print(f'\n{name}: {len(d)} observations, eight explanatory variables, grouped={grouped}')
 # Descriptives, correlations, collinearity
 d[cols+[target]].describe().T.to_csv(RES/f'{name}_descriptives.csv')
 d[cols+[target]].corr().to_csv(RES/f'{name}_correlations.csv')
 Xv=sm.add_constant(X)
 vif=pd.DataFrame({'variable':cols,'VIF':[variance_inflation_factor(Xv.values,i+1) for i in range(len(cols))]})
 vif.to_csv(RES/f'{name}_vif.csv',index=False)
 # Inferential OLS on a fixed holdout TRAIN subset, not on test observations.
 tr,te=holdout(d,X,y,0.2,42)
 # LR2.1 uses HC3 heteroskedasticity-robust t inference.
 # LR2.2 clusters paired team observations by match ID.
 if grouped:
  ols=sm.OLS(y.iloc[tr],sm.add_constant(X.iloc[tr])).fit(
   cov_type='cluster',cov_kwds={'groups':d.match_id.iloc[tr]},use_t=True)
  inference='cluster_t'
 else:
  ols=sm.OLS(y.iloc[tr],sm.add_constant(X.iloc[tr])).fit(cov_type='HC3',use_t=True)
  inference='hc3_t'
 (RES/f'{name}_ols_{inference}_summary.txt').write_text(ols.summary().as_text())
 pd.DataFrame({'feature':ols.params.index,'coefficient':ols.params.values,
  'robust_standard_error':ols.bse.values,'t_statistic':ols.tvalues.values,'robust_p_value':ols.pvalues.values,'CI_lower':ols.conf_int().iloc[:,0].values,
  'CI_upper':ols.conf_int().iloc[:,1].values}).to_csv(RES/f'{name}_ols_{inference}_coefficients.csv',index=False)
 # 80/20 and 70/30; same holdout for all models
 rows=[]; predictions=[]
 for test_fraction in [0.2,0.3]:
  tr,te=holdout(d,X,y,test_fraction,42)
  if grouped: assert not(set(d.match_id.iloc[tr])&set(d.match_id.iloc[te]))
  for label,model in models().items():
   model.fit(X.iloc[tr],y.iloc[tr]);p=model.predict(X.iloc[te]);m=metrics(y.iloc[te],p)
   rows.append({'split':f'{int((1-test_fraction)*100)}/{int(test_fraction*100)}','model':label,'train_rows':len(tr),'test_rows':len(te),**m})
   if test_fraction==0.2:
    for j,idx in enumerate(te):predictions.append({'model':label,'row_index':int(idx),'match_id':int(d.match_id.iloc[idx]),'actual':float(y.iloc[idx]),'predicted':float(p[j])})
 hold=pd.DataFrame(rows);hold.to_csv(RES/f'{name}_holdout_comparison.csv',index=False)
 pd.DataFrame(predictions).to_csv(RES/f'{name}_test_predictions.csv',index=False)
 # 5-fold CV: groups remain intact for LR2.2
 if grouped: folds=GroupKFold(n_splits=5).split(X,y,groups=d.match_id)
 else:
  from sklearn.model_selection import KFold
  folds=KFold(n_splits=5,shuffle=True,random_state=42).split(X,y)
 folds=list(folds);cv=[]
 for fold,(tr,te) in enumerate(folds,1):
  if grouped:assert not(set(d.match_id.iloc[tr])&set(d.match_id.iloc[te]))
  for label,model in models().items():
   model.fit(X.iloc[tr],y.iloc[tr]);cv.append({'fold':fold,'model':label,**metrics(y.iloc[te],model.predict(X.iloc[te]))})
 cv=pd.DataFrame(cv);cv.to_csv(RES/f'{name}_fivefold_details.csv',index=False)
 cv_summary=cv.groupby('model')[['MAE','MSE','RMSE','R2']].agg(['mean','std'])
 cv_summary.columns=['_'.join(x) for x in cv_summary.columns]
 cv_summary.sort_values('RMSE_mean').to_csv(RES/f'{name}_fivefold_summary.csv')
 # Repeated holdouts, same seeds across algorithms
 rep=[]
 for seed in range(10):
  tr,te=holdout(d,X,y,0.2,seed)
  for label,model in models().items():
   model.fit(X.iloc[tr],y.iloc[tr]);rep.append({'seed':seed,'model':label,**metrics(y.iloc[te],model.predict(X.iloc[te]))})
 pd.DataFrame(rep).to_csv(RES/f'{name}_repeated_splits.csv',index=False)
 rep_summary=pd.DataFrame(rep).groupby('model')[['RMSE','MAE','R2']].agg(['mean','std'])
 rep_summary.columns=['_'.join(c) for c in rep_summary.columns]
 rep_summary.reset_index().to_csv(RES/f'{name}_repeated_summary.csv',index=False)
 # Eight predictor-versus-target scatterplots, Pearson r and linear trend.
 fig,axes=plt.subplots(2,4,figsize=(17,8.5),constrained_layout=True)
 for ax,col in zip(axes.flat,cols):
  xv=d[col].to_numpy(dtype=float);yv=y.to_numpy(dtype=float)
  ax.scatter(xv,yv,s=14,alpha=.42)
  r=float(np.corrcoef(xv,yv)[0,1]) if np.std(xv)>0 else float('nan')
  if np.unique(xv).size>1:
   slope,intercept=np.polyfit(xv,yv,1)
   xx=np.linspace(xv.min(),xv.max(),100)
   ax.plot(xx,slope*xx+intercept,linewidth=1.4,color='darkorange')
  ax.set_title(f'{col}\nPearson r = {r:.3f}',fontsize=9)
  ax.set_xlabel(col.replace('_',' '),fontsize=8)
  ax.set_ylabel(target.replace('_',' '),fontsize=8)
  ax.tick_params(labelsize=8)
 fig.suptitle(f'{name.upper()}: pre-match predictors versus target',fontsize=14)
 fig.savefig(FIG/f'{name}_predictor_target_scatterplots.png',dpi=175)
 plt.close(fig)
 # Data storytelling figures
 plt.figure(figsize=(8,5));plt.hist(y,bins=range(int(y.min())-1,int(y.max())+3) if name=='lr22' else 18,edgecolor='white')
 plt.xlabel(target.replace('_',' ').title());plt.ylabel('Observations');plt.title(f'{name.upper()}: target distribution');plt.tight_layout();plt.savefig(FIG/f'{name}_target_distribution.png',dpi=170);plt.close()
 plt.figure(figsize=(9,6));cor=d[cols+[target]].corr();plt.imshow(cor,cmap='coolwarm',vmin=-1,vmax=1);plt.colorbar(label='Pearson r');plt.xticks(range(len(cor)),cor.columns,rotation=75,ha='right',fontsize=8);plt.yticks(range(len(cor)),cor.columns,fontsize=8);plt.title(f'{name.upper()}: correlations');plt.tight_layout();plt.savefig(FIG/f'{name}_correlations.png',dpi=170);plt.close()
 summary=cv.groupby('model')['RMSE'].agg(['mean','std']).sort_values('mean');plt.figure(figsize=(9,5));plt.barh(summary.index[::-1],summary['mean'][::-1],xerr=summary['std'][::-1],capsize=3);plt.xlabel('5-fold CV RMSE (lower is better)');plt.title(f'{name.upper()}: model comparison');plt.tight_layout();plt.savefig(FIG/f'{name}_cv_models.png',dpi=170);plt.close()
 best=summary.index[0];p=pd.DataFrame(predictions);p=p[p.model==best]
 plt.figure(figsize=(6,5));plt.scatter(p.actual,p.predicted,alpha=0.7);lo=min(p.actual.min(),p.predicted.min());hi=max(p.actual.max(),p.predicted.max());plt.plot([lo,hi],[lo,hi],'k--',lw=1);plt.xlabel('Actual');plt.ylabel('Predicted');plt.title(f'{name.upper()}: {best} holdout predictions');plt.tight_layout();plt.savefig(FIG/f'{name}_predicted_vs_actual.png',dpi=170);plt.close()
 plt.figure(figsize=(7,4));plt.scatter(p.predicted,p.actual-p.predicted,alpha=0.7);plt.axhline(0,color='black',ls='--');plt.xlabel('Predicted');plt.ylabel('Residual (actual - predicted)');plt.title(f'{name.upper()}: {best} residuals');plt.tight_layout();plt.savefig(FIG/f'{name}_residuals.png',dpi=170);plt.close()
 print('5-fold CV best:',best,'RMSE=',round(summary.iloc[0]['mean'],3))
 print('80/20 results:',hold[hold.split=='80/20'][['model','RMSE','R2']].round(3).to_string(index=False))

for name,cfg in CONFIG.items():run(name,cfg)
print('\nSUCCESS: all datasets evaluated; results and figures written.')
