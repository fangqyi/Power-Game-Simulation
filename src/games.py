import numpy as np


class Game():
    def __init__(self, n_players, is_mixed=True, is_partial=True):
        self.n_players = n_players
        self.is_mixed = is_mixed
        self.is_partial = is_partial
        self.rewards = []

    def init_game(self):
        """
        Initialize the game
        :return: initial state and a list of obs (optional)
        """
        self.rewards = []
        state = None
        obs = []
        return state, obs

    def step(self, actions):
        """

        :param actions: a list of n actions from players
        :return: a list of rewards, next state, a list of obs for the next state
        """
        rewards = []
        state = None
        obs = []
        return rewards, state, obs

    def is_mixed(self):
        return self.is_mixed

    def is_partial(self):
        return self.is_partial

class PseudoGame(Game):
    def __init__(self, n_players, orders, is_mixed=True, is_partial=True):
        super().__init__(n_players, is_mixed, is_partial)
        self.orders = self._generate_order_matrix(orders)

    def compute_game_rewards(self, actions_profile):
        def calc_distance(x, y):
            return np.abs(x-y)
        return np.sum(1 - calc_distance(actions_profile, self.orders), axis=1)

    def _generate_order_matrix(self, orders):
        o = np.zeros(self.n_players, self.n_players)
        for (x, y) in orders:
            o[x][y] = 1
        for i in range(self.n_players):
            n = np.sum(o[i], axis=0)
            o[i] = o[i]/n if n != 0 else o[i]
        return o

