import torch

import pyro
from pyro.distributions import Normal, Bernoulli


class EloModel:
    def __init__(self, n_games, n_teams, annual=True):
        self.n_games = n_games
        self.n_teams = n_teams
        self.annual = annual

    def model(self, df_games):
        """Takes a DF with columns
        home_team, away_team, outcome"""
        home_team_ids = df_games[:, 0].long()
        away_team_ids = df_games[:, 1].long()
        game_outcomes = df_games[:, 2]

        mu_home_team = pyro.sample(
            'mu_home_team',
            Normal(0., 1.)
        )

        log_sigma = pyro.sample(
            'log_sigma',
            Normal(0., 1.)
        )
        sigma = torch.exp(log_sigma)

        with pyro.plate('teams', self.n_teams):
            team_etas = pyro.sample(
                'team_etas',
                Normal(torch.zeros(self.n_teams), 1. * sigma)
            )
        with pyro.plate('team_years', self.n_teams * 10):
            #  I need to construct year-to-year similarity.
            #  Not just a sample for the whole team.
            #  The cumsum thing could work.

        game_outcome_logodds = mu_home_team + \
            team_etas[home_team_ids] - \
            team_etas[away_team_ids]

        with pyro.plate('games', self.n_games):
            pyro.sample(
                'game_outcomes',
                Bernoulli(
                    logits=game_outcome_logodds
                ),
                obs=game_outcomes
            )
