import torch

import os
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import pyro
from pyro.distributions import Normal, Bernoulli
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import pyro.optim as optim

# for CI testing
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('0.4.0')
pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)

class EloModel:
    def __init__(self, n_games, n_teams):
        self.n_games = n_games
        self.n_teams = n_teams

    def model(self, df_games):
        """Takes a DF with columns
        home_team, away_team, outcome"""
        home_team_ids= df_games[:, 0].long()
        away_team_ids= df_games[:, 1].long()
        game_outcomes= df_games[:, 2]

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
                Normal(torch.zeros(n_teams), 1. * sigma)

        game_outcome_logodds = mu_home_team + \
                team_etas[home_team_id] - \
                team_etas[away_team_id]

        with pyro.plate('games', self.n_games):
                pyro.sample(
                    'game_outcomes',
                    Bernoulli(
                        logits=game_outcome_logodds
                    ),
                    obs=game_outcomes
                )
