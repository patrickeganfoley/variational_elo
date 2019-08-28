import pandas as pd
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

df_games = pd.read_csv('games.csv')

#  Make IDs for schools
unique_teams = list(set(df_games['Schl']) | set(df_games['Opp']))
n_teams = len(unique_teams)
n_games = df_games.shape[0]

df_team_ids = pd.DataFrame({
    'team_id': range(n_teams),
    'team_name': unique_teams
})
logger.info(f'Team IDs are {df_team_ids.head()}')
logger.info(f'There are {n_teams} teams and {n_games} games.')

#  Merge in to get School ids
df_team_ids.columns = ['home_team_id', 'Schl']
df_games = pd.merge(
    df_games, df_team_ids,
    on='Schl', how='left'
)
df_team_ids.columns = ['away_team_id', 'Opp']
df_games = pd.merge(
    df_games, df_team_ids,
    on='Opp', how='left'
)
logger.info('Merged in Ids...')
logger.info(df_games.head())
logger.info(f'df_games has cols {df_games.columns}')

df_games['win'] = 1.0 * (df['MOV'] >= 0.0)

df_small = df_games[[
    'home_team_id', 'away_team_id', 'win'
]]
df_small = torch.tensor(df_games.values)





