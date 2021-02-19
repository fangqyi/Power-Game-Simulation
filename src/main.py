from protocols import OnlyP, OnlyQ, DiffuseQConcentratedQ, MinPQ
from games import PseudoGame
from simulation import BruteForceSimulation


def visualize_power_structures(l, protocol, game, o):
    if len(power_structures) == 0:
        print("Power structures not found")
    else:
        print("Desired:")
        for (x, y) in o:
            print("{} --> {}".format(x, y))
        # print(game.orders)
        for idx in range(len(l)):
            print("=========================================================")
            print("Power structure {}:".format(idx))
            if protocol.num_edge_weights == 1:
                for i in range(protocol.n_players):
                    print("player {}:".format(i))
                    print("{}".format(l[idx][0][i, 0]))
            else:
                for i in range(protocol.n_players):
                    print("player {}:".format(i))
                    print("* p: {}".format(l[idx][0][i, 0]))
                    print("* q: {}".format(l[idx][0][i, 1]))
            print("-----------------------------------------------------------")
            print("social welfare: {}, task rewards: {}".format(l[idx][1], l[idx][2]))
            print("=========================================================")
        print()
        print()
        print()


n_players = 2
orders = [(1, 1), (1, 0)]
game = PseudoGame(n_players, orders, is_mixed=True, is_partial=False)

protocol = OnlyP(n_players)
simulation = BruteForceSimulation(n_players, game, protocol)
power_structures = simulation.compute_equilibrium()
print("OnlyP:")
visualize_power_structures(power_structures, protocol, game, orders)

protocol = OnlyQ(n_players)
simulation = BruteForceSimulation(n_players, game, protocol)
power_structures = simulation.compute_equilibrium()
print("OnlyQ")
visualize_power_structures(power_structures, protocol, game, orders)
#
protocol = DiffuseQConcentratedQ(n_players)
simulation = BruteForceSimulation(n_players, game, protocol)
power_structures = simulation.compute_equilibrium()
print("DiffuseQConcentratedQ")
# print(len(power_structures))
visualize_power_structures(power_structures, protocol, game, orders)
#
protocol = MinPQ(n_players)
simulation = BruteForceSimulation(n_players, game, protocol)
power_structures = simulation.compute_equilibrium()
print("MinPQ")
# print(len(power_structures))
visualize_power_structures(power_structures, protocol, game, orders)
