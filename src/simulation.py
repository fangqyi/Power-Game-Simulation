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
                for i in range(self.n_players):
                    for j in range(self.n_players):
                        # find p* and q*
                        p_star_idx = idx
                        q_star_idx = idx
                        comp_p_ap = copy.deepcopy(raw_action_profiles[idx][:][0])
                        comp_p_ap[j] = np.zeros(comp_p_ap[j].shape)
                        comp_q_ap = copy.deepcopy(raw_action_profiles[idx][:][1])
                        comp_q_ap[i] = np.zeros(comp_q_ap[i].shape)

                        for idx_t in range(total_action_profiles):
                            temp_p_ap = copy.deepcopy(raw_action_profiles[idx_t][:][0])
                            temp_p_ap[j] = np.zeros(temp_p_ap[j].shape)
                            temp_q_ap = copy.deepcopy(raw_action_profiles[idx_t][:][1])
                            temp_q_ap[i] = np.zeros(temp_q_ap[i].shape)
                            if np.array_equal(temp_p_ap, comp_p_ap) and rewards[p_star_idx][j] < rewards[idx_t][j]:
                                p_star_idx = idx_t
                            if np.array_equal(temp_q_ap, comp_q_ap) and rewards[q_star_idx][i] < rewards[idx_t][i]:
                                q_star_idx = idx_t
                        if raw_action_profiles[p_star_idx][j][0][i] != raw_action_profiles[idx][j][0][i]:
                            flag = False
                            break
                        if raw_action_profiles[q_star_idx][i][1][j] != raw_action_profiles[idx][i][1][j]:
                            flag = False
                            break
                    if not flag:
                        break
                if flag:
                    #print(raw_action_profiles[idx])
                    equilibriums.append((raw_action_profiles[idx],
                                         np.sum(rewards[idx], axis=0),
                                         np.sum(game_rewards[idx], axis=0)))
        elif num_edge_weights == 1:
            for idx in range(total_action_profiles):
                flag = True

                for i in range(self.n_players):
                    # find p* or q*
                    star_idx = idx
                    comp_ap = copy.deepcopy(raw_action_profiles[idx])
                    comp_ap[i] = np.zeros(comp_ap[i].shape)
                    for idx_t in range(total_action_profiles):
                        temp_ap = copy.deepcopy(raw_action_profiles[idx_t])
                        temp_ap[i] = np.zeros(temp_ap[i].shape)
                        if np.array_equal(temp_ap, comp_ap) and rewards[star_idx][i] < rewards[idx_t][i]:
                            star_idx = idx_t
                    for j in range(self.n_players):
                        if raw_action_profiles[star_idx][i][0][j] != raw_action_profiles[idx][i][0][j]:
                            flag = False
                            break
                    if not flag:
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
        actions = [_dec_to_action_list(dec) for dec in list(range(1, 2 ** self.n_players))]
        all_action_profiles = list(product(actions, repeat=num_edge_weights))
        ret = list(product(all_action_profiles, repeat=self.n_players))
        return np.array(ret)
        # [num_action_profiles, num_players, num_edge_weights, num_players]
