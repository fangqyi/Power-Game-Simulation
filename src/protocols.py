import numpy as np


class Protocol:
    def __init__(self, n_players):
        self.n_players = n_players
        self.num_edge_weights = 0

    def get_game_action_profiles(self, raw_action_profiles):
        """
        :param raw_action_profiles:  [num_action_profiles, num_players, num_edge_weights, num_players]
        :return: [num_action_profiles, num_players, num_players]
        """


class OnlyP(Protocol):
    def __init__(self, n_players):
        super().__init__(n_players)
        self.num_edge_weights = 1

    def get_game_action_profiles(self, raw_action_profiles):
        ret = np.transpose(np.squeeze(raw_action_profiles, axis=2), axes=(0, 2, 1))
        return ret


class OnlyQ(Protocol):
    def __init__(self, n_players):
        super().__init__(n_players)
        self.num_edge_weights = 1

    def get_game_action_profiles(self, raw_action_profiles):
        return np.squeeze(raw_action_profiles, axis=2)


class DiffuseQConcentratedQ(Protocol):
    def __init__(self, n_players):
        super().__init__(n_players)
        self.q_beta = 0.5
        self.p_beta = 10
        self.num_edge_weights = 2

    def get_game_action_profiles(self, raw_action_profiles):

        def softmax(x, beta):
            return np.exp(beta*x) / sum(np.exp(beta*x))

        p = softmax(raw_action_profiles[:, :, 0], self.p_beta)
        q = softmax(raw_action_profiles[:, :, 1], self.q_beta)
        sq_pq = np.square(np.multiply(p, q))

        return sq_pq

class MinPQ(Protocol):
    def __init__(self, n_players):
        super().__init__(n_players)
        self.num_edge_weights = 2

    def get_game_action_profiles(self, raw_action_profiles):
        p_tp = np.transpose(raw_action_profiles[:, :, 0], axes=(0, 2, 1))
        q = raw_action_profiles[:, :, 1]
        return np.minimum(p_tp, q)
