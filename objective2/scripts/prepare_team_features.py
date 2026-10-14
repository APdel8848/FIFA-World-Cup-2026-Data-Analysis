"""Build LR2.2 from the supplied source snapshot. No invented external ratings.

Elo is an explicitly MODEL-DERIVED proxy, NOT an independently sourced official Elo.
League counts rely on a reviewed 88-club positive list; 8 missing league
participants have candidate IDs only, so a conservative uncertainty flag is kept.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'; PROC=ROOT/'data/processed'; RESULTS=ROOT/'output/results'
RESULTS.mkdir(parents=True,exist_ok=True)

team_matches=pd.read_csv(PROC/'objective2_2_team_matches.csv',parse_dates=['match_date'])
match_data=pd.read_csv(PROC/'worldcup_matches_clean.csv',parse_dates=['match_date']).sort_values(['match_date','match_id'])
history=pd.read_csv(PROC/'historical_matches_standardised.csv',parse_dates=['date'])
rank=pd.read_csv(RAW/'fifa_rankings_2026_06_11_user_transcribed.csv')
form=pd.read_csv(PROC/'historical_form_features.csv')
squad=pd.read_csv(PROC/'squad_features_2026.csv')
players=pd.read_csv(RAW/'players_market_values_2026.csv')
clubs=pd.read_csv(PROC/'club_league_classification_2026.csv')
leagues=pd.read_csv(PROC/'top5_league_clubs_2025_26.csv')
assert len(team_matches)==208 and team_matches.match_id.nunique()==104
assert len(players)==1246 and clubs.player_count.sum()==len(players)
assert len(leagues)==96 and (clubs.is_top5_league==1).sum()==88
assert players.player_id.notna().all() and players.club_id.notna().all()
assert not clubs.club_id.duplicated().any()
assert set(players.club_id)==set(clubs.club_id)

# User-reviewed Scottish Premiership screenshot; Rangers FC was not applied previously.
scottish_id=124
scot=clubs.club_id==scottish_id
assert scot.sum()==1 and clubs.loc[scot,'is_top5_league'].isna().all()
clubs.loc[scot,'is_top5_league']=0
clubs.loc[scot,'match_status']='VERIFIED_NON_TOP5'
clubs.loc[scot,'verification_source']='2025-26 Scottish Premiership standings screenshot supplied by user'

# These are a provisional exhaustive classification based on 88 mapped members
# and the absence of plausible unresolved senior-club names. NOT independently verified.
clubs['is_top5_league_final']=clubs.is_top5_league.fillna(0).astype(int)
clubs['classification_confidence']=np.where(clubs.is_top5_league.eq(1),'mapped_top5',
 np.where(clubs.match_status.eq('VERIFIED_NON_TOP5'),'user_evidence_non_top5',
 np.where(clubs.is_top5_league.eq(0),'previously_reviewed_non_top5','provisional_non_top5')))
clubs.to_csv(PROC/'club_classification_submission.csv',index=False)
players['team']=players['team'].replace({'Turkiye':'Türkiye','United States':'USA','South Korea':'Korea Republic','Bosnia-Herzegovina':'Bosnia–Herz',"Ivory Coast":"Côte d\'Ivoire",'Iran':'IR Iran','Democratic Republic of the Congo':'Congo DR','Cape Verde':'Cabo Verde'})
players=players.merge(clubs[['club_id','is_top5_league_final','classification_confidence']],on='club_id',validate='many_to_one')
counts=players.groupby('team',as_index=False).agg(top5_league_player_count=('is_top5_league_final','sum'),
 provisional_club_players=('classification_confidence',lambda x:(x=='provisional_non_top5').sum()))
# use authoritative team-snapshot values already compiled from source (June 3)
squad=squad.drop(columns=['top5_league_players'],errors='ignore').merge(counts,on='team',validate='one_to_one')
assert len(squad)==48 and squad[['squad_value_m','avg_age','top5_league_player_count']].notna().all().all()
squad.to_csv(PROC/'squad_features_submission.csv',index=False)

# Standard Elo from match results (transparent model-derived opponent strength).
# Train on all historical internationals before WC kickoff; then update after each
# WC match. Ratings before the match are used, preventing same-match leakage.
# Neutral venue => no home bonus. K=20, base=1500, goal-independent result update.
elo={}
def get(t):return elo.get(t,1500.0)
def update(home,away,hg,ag):
 a,b=get(home),get(away)
 e=1/(1+10**((b-a)/400))
 outcome=1.0 if hg>ag else (0.5 if hg==ag else 0.0)
 elo[home]=a+20*(outcome-e)
 elo[away]=b+20*((1-outcome)-(1-e))
cutoff=pd.Timestamp('2026-06-11')
hist=history[(history.date<cutoff)&history.home_score.notna()&history.away_score.notna()].sort_values('date')
for r in hist.itertuples(index=False):update(r.home_team,r.away_team,r.home_score,r.away_score)
pre_elo=[]
# Without independently verified kickoff times, avoid within-day result leakage.
# Snapshot all teams' Elo before updating ANY match played on the same date.
for day, day_matches in match_data.groupby('match_date', sort=True):
 for r in day_matches.itertuples(index=False):
  pre_elo.extend([{'match_id':r.match_id,'team':r.Home,'pre_match_elo':get(r.Home)},
                  {'match_id':r.match_id,'team':r.Away,'pre_match_elo':get(r.Away)}])
 for r in day_matches.itertuples(index=False):
  update(r.Home,r.Away,r.home_goals,r.away_goals)
elo_frame=pd.DataFrame(pre_elo)
opp=elo_frame.rename(columns={'team':'opponent','pre_match_elo':'opponent_elo_rating'})[['match_id','opponent','opponent_elo_rating']]

# Form features deliberately frozen before kickoff; no WC result included.
features=(team_matches.merge(rank[['team','fifa_rank']],on='team',validate='many_to_one')
 .merge(form[['team','recent_form','avg_goals_scored','avg_goals_conceded']],on='team',validate='many_to_one')
 .merge(squad[['team','squad_value_m','avg_age','top5_league_player_count','provisional_club_players']],on='team',validate='many_to_one')
 .merge(opp,on=['match_id','opponent'],validate='one_to_one'))
columns=['fifa_rank','recent_form','avg_goals_scored','avg_goals_conceded','opponent_elo_rating',
         'squad_value_m','avg_age','top5_league_player_count']
assert len(features)==208 and features.match_id.nunique()==104
assert features[columns+['goals_scored']].notna().all().all()
assert not features[['match_id','team']].duplicated().any()
assert features.groupby('match_id').size().eq(2).all()
# Export exactly one ID field, one response and eight explanatory variables.
# Team and opponent metadata remain in objective2_2_team_matches.csv.
features=features[['match_id','goals_scored']+columns].copy()
features.to_csv(PROC/'objective2_2_final.csv',index=False,date_format='%Y-%m-%d')
summary={'world_cup_matches':104,'team_match_rows':208,'explanatory_variables':columns,
 'players':len(players),'clubs':len(clubs),'mapped_top5_clubs':int((clubs.is_top5_league==1).sum()),
 'reviewed_non_top5_clubs':int(clubs.is_top5_league.eq(0).sum()),
 'provisional_non_top5_clubs':int(clubs.is_top5_league.isna().sum()),
 'provisional_players':int(squad.provisional_club_players.sum()),
 'total_top5_players_under_provisional_mapping':int(squad.top5_league_player_count.sum()),
 'elo_definition':'Calculated Elo (initial 1500, K=20, neutral, no official Elo claims)',
 'snapshot_date':'2026-06-03',
 'caveat':'Eight 2025-26 top-five participant names remain unmatched to the supplied player club-ID list; candidate IDs not independently validated.'}
(RESULTS/'data_audit.json').write_text(json.dumps(summary,indent=2))
print('PASS: 104 match records, 208 team-match records, exactly 8 complete predictors')
print('Top-five players (provisional club mapping):',summary['total_top5_players_under_provisional_mapping'])
print('Provisional non-top-five players:',summary['provisional_players'])
print('Derived pre-match Elo ratings computed from historical results and preceding WC matches')
