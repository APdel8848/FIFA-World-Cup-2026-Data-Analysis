"""Independent integrity tests for the submitted datasets and results."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/processed';R=ROOT/'output/results'
a=pd.read_csv(D/'objective2_1_final.csv');b=pd.read_csv(D/'objective2_2_final.csv');w=pd.read_csv(D/'worldcup_matches_clean.csv');c=pd.read_csv(D/'club_classification_submission.csv');s=pd.read_csv(D/'squad_features_submission.csv')
f1=['fifa_ranking_difference','recent_form_difference','avg_goals_scored_difference','avg_goals_conceded_difference','clean_sheet_rate_difference','worldcup_experience_difference','rest_days_difference','knockout_stage_indicator']
f2=['fifa_rank','recent_form','avg_goals_scored','avg_goals_conceded','opponent_elo_rating','squad_value_m','avg_age','top5_league_player_count']
assert len(a)==104 and len(b)==208 and len(w)==104

assert 'goals_conceded' not in b.columns and 'goal_difference' not in b.columns
assert a.rest_days_difference.abs().max()<=7
uncapped=pd.read_csv(D/'objective2_1_uncapped.csv')
assert uncapped.rest_days_difference.min()==-70
assert a.drop(columns=['rest_days_difference']).equals(uncapped.drop(columns=['rest_days_difference']))
assert len(f1)==len(f2)==8
assert list(b.columns)==['match_id','goals_scored']+f2, list(b.columns)
assert set(pd.read_csv(ROOT/'data/raw/fifa_rankings_2026_06_11_user_transcribed.csv').columns)=={'team','fifa_rank'}
assert (ROOT/'output/figures/lr21_predictor_target_scatterplots.png').exists()
assert (ROOT/'output/figures/lr22_predictor_target_scatterplots.png').exists()
assert (R/'lr21_ols_hc3_t_coefficients.csv').exists()
assert (R/'lr22_ols_cluster_t_coefficients.csv').exists()
assert a[f1+['goal_difference']].notna().all().all()
assert b[f2+['goals_scored']].notna().all().all()
assert b.groupby('match_id').size().eq(2).all()
assert not pd.read_csv(D/'objective2_2_team_matches.csv')[['match_id','team']].duplicated().any()
assert b.groupby('match_id')['goals_scored'].sum().equals(w.set_index('match_id').loc[b.match_id.drop_duplicates().sort_values(),'home_goals'].add(w.set_index('match_id').loc[b.match_id.drop_duplicates().sort_values(),'away_goals']).rename('goals_scored'))
assert len(c)==453 and len(s)==48
assert c.player_count.sum()==1246
assert s.top5_league_player_count.sum()==487
for n in ['lr21','lr22']:
 for f in [f'{n}_holdout_comparison.csv',f'{n}_fivefold_summary.csv',f'{n}_repeated_summary.csv',f'{n}_vif.csv']:
  assert (R/f).exists(),f
print('PASS: row counts, eight predictors, goals, match pairing, player counts, output tables')
