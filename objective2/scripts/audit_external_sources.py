from pathlib import Path
import pandas as pd, json, zipfile, subprocess, sys
p=Path(__file__).resolve().parents[1]
raw=p/'data/raw'; proc=p/'data/processed'; results=p/'output/results'; scripts=p/'scripts'
# Independent June 11 2026 participant list from MLSsoccer.com, corroborated by FIFA news and Sporting News.
ranking_pairs='''Argentina:1,Spain:2,France:3,England:4,Portugal:5,Brazil:6,Morocco:7,Netherlands:8,Belgium:9,Germany:10,Croatia:11,Colombia:13,Mexico:14,Senegal:15,Uruguay:16,USA:17,Japan:18,Switzerland:19,IR Iran:20,Türkiye:22,Ecuador:23,Austria:24,Korea Republic:25,Australia:27,Algeria:28,Egypt:29,Canada:30,Norway:31,Côte d'Ivoire:33,Panama:34,Sweden:38,Czechia:40,Paraguay:41,Scotland:42,Tunisia:45,Congo DR:46,Uzbekistan:50,Qatar:56,Iraq:57,South Africa:60,Saudi Arabia:61,Jordan:63,Bosnia–Herz:64,Cabo Verde:67,Ghana:73,Curaçao:82,Haiti:83,New Zealand:85'''
reference=pd.DataFrame([{'team':k,'published_rank':int(v)} for k,v in [pair.split(':') for pair in ranking_pairs.split(',')]])
assert len(reference)==48 and reference.team.nunique()==48
reference['source']='MLSsoccer.com: FIFA World Rankings Every Team at 2026 World Cup (as of 11 June 2026)'
reference['source_url']='https://www.mlssoccer.com/competitions/fifa-world-cup/news/fifa-world-rankings-every-team-at-the-world-cup'
reference.to_csv(raw/'ranking_independent_crosscheck_2026_06_11.csv',index=False)
ranking=pd.read_csv(raw/'fifa_rankings_2026_06_11_user_transcribed.csv')
a=ranking.merge(reference,on='team',how='outer',indicator=True)
a['exact_match']=a['fifa_rank'].eq(a['published_rank']) & a['_merge'].eq('both')
a.to_csv(results/'ranking_48_team_external_comparison.csv',index=False)
print('RANKING matches',a.exact_match.sum(),'of',len(a));print(a.loc[~a.exact_match,['team','fifa_rank','published_rank']].to_string(index=False))
# Only explicitly supported independent match crosschecks. Do not imply all 104 scores verified.
wc=pd.read_csv(proc/'worldcup_matches_clean.csv')
match_reference=[('2026-06-11','Mexico','South Africa',2,0),('2026-06-11','Korea Republic','Czechia',2,1)]
rows=[]
for date,home,away,hg,ag in match_reference:
 r=wc[(wc.match_date==date)&(wc.Home==home)&(wc.Away==away)]
 rows.append({'date':date,'home':home,'away':away,'source_home_goals':hg,'source_away_goals':ag,'matched':len(r)==1 and int(r.iloc[0].home_goals)==hg and int(r.iloc[0].away_goals)==ag,'source_url':'https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums'})
pd.DataFrame(rows).to_csv(results/'independent_match_spot_checks.csv',index=False)
# Transparent club classification status; do not convert unresolved clubs to 'verified'.
clubs=pd.read_csv(proc/'club_classification_submission.csv')
players=pd.read_csv(raw/'players_market_values_2026.csv')
assert len(clubs)==453 and len(players)==1246
counts=clubs.groupby('classification_confidence',dropna=False).agg(clubs=('club_id','size'),players=('player_count','sum')).reset_index()
counts.to_csv(results/'club_classification_coverage.csv',index=False)
assert counts.clubs.sum()==453 and counts.players.sum()==1246
# Source register distinguishes independently checked from metadata attribution.
sources=[
 ('FIFA ranking snapshot date','FIFA', 'https://inside.fifa.com/fifa-rankings/world-ranking/men/news/argentina-reclaim-top-spot-mens-world-ranking-fifa-coca-cola','confirmed publication June 11 2026; full 48 values compared with independent MLSsoccer list'),
 ('FIFA ranking participant values','MLSsoccer','https://www.mlssoccer.com/competitions/fifa-world-cup/news/fifa-world-rankings-every-team-at-the-world-cup','independent 48-team cross-check; not direct FIFA table extraction'),
 ('World Cup match results','FBref','https://fbref.com/en/comps/1/schedule/schedule-Stats','column schema and opening results match; full 104-row external comparison not performed'),
 ('World Cup opening results','FIFA','https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums','two June 11 results checked'),
 ('Historical international results','Mart Jurisoo','https://github.com/martj42/international_results','column schema and row count consistent; exact source file checksum not checked'),
 ('2026 squad player snapshot','littlebiscuit666/world-cup-web; Transfermarkt attributed','https://github.com/littlebiscuit666/world-cup-web','public repository confirms June 3 2026 snapshot, 48 teams, 1246 players; direct source HTML not independently compared'),
 ('Top-five league club memberships','Five 2025-26 football-data.co.uk league CSVs','https://www.football-data.co.uk/data.php','96 league participants from local league files; 88 positive mapped; unresolved identities remain'),
 ('Opponent Elo','Locally calculated','', 'K=20 base=1500; no official external Elo source'),
]
pd.DataFrame(sources,columns=['item','publisher_or_method','url','verification_scope']).to_csv(results/'source_provenance_register.csv',index=False)
assert a.exact_match.all(),'Ranking mismatch must be resolved before claiming verified'
assert all(r['matched'] for r in rows)
print('PASS: independent 48/48 FIFA ranking cross-check; 2/2 FIFA match checks; source coverage logged')
