import copy
from itertools import product
import numpy as np


class Simulation():
    def __init__(self, n_players, game, protocol):
        self.n_players = n_players
        self.game = game
        self.protocol = protocol

    def compute_equilibrium(self):
        """

        :return: a list that contains equilibirum(s)
        """
        return []


class BruteForceSimulation(Simulation):
    def __init__(self, n_players, game, protocol):
        super().__init__(n_players, game, protocol)

    def compute_equilibrium(self):

        equilibriums = []
        num_edge_weights = self.protocol.num_edge_weights
        raw_action_profiles = self.get_all_reduced_action_profiles()
        game_action_profiles = self.protocol.get_game_action_profiles(raw_action_profiles)

        # get rewards from game env
        game_rewards = []
        for action_profile in game_action_profiles:
            game_rewards.append(self.game.compute_game_rewards(action_profile))
        game_rewards = np.stack(game_rewards, axis=0)

        # redistribute rewards
        rewards = [np.einsum("ij,i->j", game_action_profiles[idx], game_rewards[idx])
                   for idx in range(game_action_profiles.shape[0])]

        #
        total_action_profiles = game_rewards.shape[0]
        if num_edge_weights == 2:
            for idx in range(total_action_profiles):
                flag = True
                for i in self.n_players:
                    for j in self.n_players:
                        # find p* and q*
                        p_star_idx = 0
                        q_star_idx = 0
                        interval = (2 ** self.n_players) ** (num_edge_weights * self.n_players - 1)
                        mod = idx % interval
                        for x in range(0, 2 ** self.n_players):
                            if rewards[p_star_idx][j] <= rewards[interval * x + mod][j]:
                                p_star_idx = interval * x + mod
                            if rewards[q_star_idx][i] <= rewards[interval * x + mod][i]:
                                q_star_idx = interval * x + mod
                        if raw_action_profiles[p_star_idx][j][0][i] != raw_action_profiles[idx][j][0][i]:
                            flag = False
                            break
                        if raw_action_profiles[q_star_idx][i][1][j] != raw_action_profiles[idx][i][1][j]:
                            flag = False
                            break
                    if not flag:
                        break
                if flag:
                    equilibriums.append((raw_action_profiles[idx],
                                         np.sum(rewards[idx], axis=0),
                                         np.sum(game_rewards[idx], axis=0)))
        elif num_edge_weights == 1:
            for idx in range(total_action_profiles):
                flag = True
                for i in self.n_players:
                    # find p* or q*
                    star_idx = 0
                    interval = (2 ** self.n_players) ** (self.n_players - 1)
                    mod = idx % interval
                    for x in range(0, 2 ** self.n_players):
                        if rewards[star_idx][i] <= rewards[interval * x + mod][i]:
                            star_idx = interval * x + mod
                    if star_idx != idx:
                        flag = False
                        break
                if flag:
                    equilibriums.append((raw_action_profiles[idx],
                                         np.sum(rewards[idx], axis=0),
                                         np.sum(game_rewards[idx], axis=0)))
        return equilibriums

    def get_all_reduced_action_profiles(self):
        """

        :return: a list of all possible of actions profiles, discrete
        """

        def _dec_to_action_list(dec_num):
            bin_list = [int(i) for i in list('{0:0b}'.format(dec_num))]
            bin_list = [0] * (self.n_players - len(bin_list)) + bin_list
            n = sum(bin_list)
            bin_list = [i / n for i in bin_list] if n is not 0 else bin_list
            return bin_list

        num_edge_weights = self.protocol.num_edge_weights
        actions = [_dec_to_action_list(dec) for dec in list(range(0, 2 ** self.n_players))]
        all_action_profiles = list(product(actions, actions)) if num_edge_weights is 2 else actions
        ret = copy.deepcopy(all_action_profiles)
        for _ in range(self.n_players - 1):
            ret = list(product(ret, all_action_profiles))
        return np.array(ret)
        # [num_action_profiles, num_players, num_edge_weights, num_players]

