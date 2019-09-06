import torch
import pyro
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

#  import pyro.contrib.autoguide as ag
from elo_model import EloModel
from auto_normal import AutoNormal

pyro.enable_validation(True)
torch.set_default_dtype(torch.double)

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
df_team_ids.columns = ['team_id', 'team_name']

df_games['win'] = 1.0 * (df_games['MOV'] >= 0.0)


df_small = df_games[[
    'home_team_id', 'away_team_id', 'win'
]]
df_small = torch.Tensor(df_small.values.astype('double'))

elo_model = EloModel(
    n_games=n_games, n_teams=n_teams
)

logger.info(f'Running model once.')
elo_model.model(df_small)

adam_params = {
    "lr": 0.01,
    "betas": (0.9, 0.999)
}
optimizer = pyro.optim.Adam(adam_params)

svi = pyro.infer.SVI(
    elo_model.model,
    AutoNormal(elo_model.model),
    optimizer,
    loss=pyro.infer.JitTrace_ELBO()
)

logger.info(f'All params are ')
logger.info(pyro.get_param_store().get_all_param_names())


train_elbo = []
test_elbo = []

num_epochs = 2500
test_gap = 50
torch.manual_seed(0)

fig = plt.gcf()
fig.show()
fig.canvas.draw()

for epoch in range(num_epochs):
    total_epoch_loss_train = svi.step(df_small) / n_games
    train_elbo.append(total_epoch_loss_train)
    if epoch % 2 == 0:
        logger.info(
            "[epoch %03d]  average training loss: %.4f" % (
                epoch, total_epoch_loss_train
            )
        )
    if epoch % 50 == 0:
        lb = -5.0
        rb = 5.0

        logger.info('All params are ')
        logger.info(pyro.get_param_store().get_all_param_names())

        bins = np.linspace(lb, rb, 500)
        plt.clf()

        plt.subplot(211)
        plt.hist(
            pyro.param('auto_team_etas_loc').detach().numpy(),
            bins, alpha=0.5, label='teams'
        )
        plt.xlabel("Team Quality")
        # plt.xlim([lb, rb])
        # plt.ylim([0, 1000])

        plt.subplot(212)
        plt.scatter(
            pyro.param('auto_team_etas_loc').detach().numpy(),
            pyro.param('auto_team_etas_scale').detach().numpy(),
            s=0.25, alpha=0.5
        )
        plt.xlabel("Team Quality")
        plt.ylabel("Team Quality Sigma")

        plt.pause(0.05)
        fig.canvas.draw()
        plt.pause(0.05)

df_team_params = pd.DataFrame({
    'team_id': range(0, n_teams),
    'team_loc': pyro.param('auto_team_etas_loc').detach().numpy(),
    'team_scale': pyro.param('auto_team_etas_scale').detach().numpy()
})

df_team_params = pd.merge(
    df_team_params, df_team_ids,
    on='team_id', how='left'
)
df_team_params.sort_values(
    by='team_loc', ascending=False, inplace=True
)

logger.info(f'Top 20 teams are')
logger.info(df_team_params.head(20))

df_team_params.sort_values(
    by='team_scale', ascending=False, inplace=True
)
logger.info(f'Top 10 most uncertain teams are')
logger.info(df_team_params.head(10))

df_team_params.sort_values(
    by='team_scale', ascending=False, inplace=True
)
logger.info(f'Top 10 least uncertain teams are')
logger.info(df_team_params.tail(10))

