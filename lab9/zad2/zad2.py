import gym
import numpy as np 

import pygad
import numpy as np

# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP
# labirynt: 1 - ściana, 1 - jest 2 - start, 3 - koniec, 4 - dziura

board = [
    "SFFHFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

def descToList(board):
    li = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    for row in board:
        ins = [1]
        for l in row:
            if l == 'S':
                ins.append(2)
            elif l == "F":
                ins.append(0)
            elif l == "G":
                ins.append(3)
            else:
                ins.append(4)

        ins.append(1)
        li.append(ins)

    li.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    return li


labirynt = descToList(board)

print(labirynt)

def find_start():
    i = 0
    for row in labirynt:
        j = 0
        for column in row:
            if column == 2:
                return [i, j]
            j = j + 1
        i = i + 1

def find_end():
    i = 0
    for row in labirynt:
        j = 0
        for column in row:
            if column == 3:
                return [i, j]
            j = j + 1
        i = i + 1


#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = range(4)

#definiujemy funkcję fitness
def fitness_func(solution, solution_idx):
    i, j = find_start()
    si, sj = find_start()
    endi, endj = find_end()

    fitness = 0
    for move in solution:
        if move == 0:
            if labirynt[i][j - 1] == 1:
                fitness -= 10
            elif labirynt[i][j - 1] == 4:
                fitness -= 50
                j = sj
            else:
                j -= 1
                fitness += 10
        if move == 1:
            if labirynt[i + 1][j] == 1:
                fitness -= 10
            elif labirynt[i + 1][j] == 4:
                fitness -= 50
                i = si
            else:
                i += 1
                fitness += 10
        if move == 2:
            if labirynt[i][j + 1] == 1:
                fitness -= 10
            elif labirynt[i][j + 1] == 4:
                fitness -= 50
                j = sj
            else:
                j += 1
                fitness += 10
        if move == 3:
            if labirynt[i - 1][j] == 1:
                fitness -= 10
            elif labirynt[i- 1][j] == 4:
                fitness -= 50
                i = si
            else:
                i -= 1
                fitness += 10
        
        if i == endi and j == endj:
            fitness += 10000
            break

    distance = abs(i - endi) + abs(j - endj) + 1
    fitness /= distance

    return fitness

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 200
num_genes = 15

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 300
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 7

#inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       stop_criteria="reach_10000"
                       )

#uruchomienie algorytmu
ga_instance.run()

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Generations passed: {generations_completed}".format(generations_completed=ga_instance.generations_completed)) # zad2 b

#tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
#prediction = numpy.sum(S*solution)
#print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
# ga_instance.plot_fitness()


def floatToInt(l):
    ret = []
    for i in range(len(l)):
        ret.append(int(l[i]))
    return ret

s = floatToInt(solution)
    
env = gym.make('FrozenLake-v1', desc=board, map_name="8x8", render_mode="human", is_slippery=False)

observation, info = env.reset(seed=42)

actions = s

for _ in range(60):
   if _ == len(actions):
       break
   action = actions[_]
   observation, reward, terminated, truncated, info = env.step(action)
   
   if terminated or truncated:
      observation, info = env.reset()
env.close()

# chromosomy to akcje
# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP
# labirynt: 1 - ściana, 1 - jest 2 - start, 3 - koniec, 4 - dziura

# fitness działa tak jak labirynt sprawdza czy jest krawędź, dziura, czy wolna droga na tej podstawie
# odejmuje punty albo dodaje, za ściane to -10, za dziure -50, dobre +10
# gdy dotrze do celu dodaje 10000 punktów i zakończa
# im większy wynik tym lepiej