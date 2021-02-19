from protocols import OnlyP,OnlyQ, DiffuseQConcentratedQ, MinPQ
from games import PseudoGame
from simulation import BruteForceSimulation


def visualize_power_structures(l, protocol):
    if len(power_structures) == 0:
        print("Power structures not found")
    else:
        for idx in range(len(l)):
            print("=========================================================")
            print("Power structure {}:".format(idx))
            if protocol.num_edge_weights == 1:
                print(l[idx][0])
            else:
                print("* Action profile for p:")
                print(l[idx][0][:, 0])
                print("* Action profile for q:")
                print(l[idx][0][:, 1])
            print("-----------------------------------------------------------")
            print("social welfare: {}, task rewards: {}".format(l[idx][1], l[idx][2]))
            print("=========================================================")



n_players = 3
orders = [(1,1), (1,2), (1,0), (0,1), (0,2),(0,0), (2,0), (2,1), (2,2)]
game = PseudoGame(n_players, orders, is_mixed=True, is_partial=False)

# protocol = OnlyP(n_players)
# simulation = BruteForceSimulation(n_players, game, protocol)
# power_structures = simulation.compute_equilibrium()
# print("OnlyP:")
# visualize_power_structures(power_structures, protocol)

# protocol = OnlyQ(n_players)
# simulation = BruteForceSimulation(n_players, game, protocol)
# power_structures = simulation.compute_equilibrium()
# print("OnlyQ")
# visualize_power_structures(power_structures, protocol)
#
protocol = DiffuseQConcentratedQ(n_players)
simulation = BruteForceSimulation(n_players, game, protocol)
power_structures = simulation.compute_equilibrium()
print("DiffuseQConcentratedQ")
# print(len(power_structures))
visualize_power_structures(power_structures, protocol)
#
# protocol = MinPQ(n_players)
# simulation = BruteForceSimulation(n_players, game, protocol)
# power_structures = simulation.compute_equilibrium()
# print("MinPQ")
# print(len(power_structures))
# #visualize_power_structures(power_structures, protocol)




