import matplotlib.pyplot as plt
import random

from aco import AntColony


plt.style.use("dark_background")


COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),

    (44, 55),
    (25, 50),
    (99, 88),
    (64, 34)
)

# zadanie 2
# mniej więcej tyle samo

def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

colony = AntColony(COORDS, ant_count=400, alpha=0.8, beta=0.2, 
                    pheromone_evaporation_rate=0.20, pheromone_constant=200.0,
                    iterations=150)

# gdy tylko jeden się zmnienia
# większa ilość mrówek spowolniła, mniejsza przyspiesza
# alpha nic nie zmienia
# beta nic
# pheromone_evaporation_rate wieksze przyspiesza? jednak nic
# pheromone_constant przyspiesza? jednak nic
# iterations im więc tym dłużej idzie, bo więc musi przebyć

# colony = AntColony(COORDS, ant_count=200, alpha=0.8, beta=2.2, 
#                    pheromone_evaporation_rate=0.50, pheromone_constant=1500.0,
#                    iterations=150)

# idzie 1
# 295.63582179351096

# colony = AntColony(COORDS, ant_count=200, alpha=0.2, beta=0.2, 
#                    pheromone_evaporation_rate=0.20, pheromone_constant=500.0,
#                    iterations=150)

# idzie jeszcze szybciej
# 287.174132328464

# colony = AntColony(COORDS, ant_count=400, alpha=0.8, beta=0.2, 
#                    pheromone_evaporation_rate=0.20, pheromone_constant=200.0,
#                    iterations=150)

# idzie wolniej
# 295.63582179351096

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )


plt.show()