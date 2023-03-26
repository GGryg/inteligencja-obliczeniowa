import pygad
import time
import math

# S = [1, 2, 3, 6, 10, 17, 25, 29, 30, 41, 51, 60, 70, 79, 80]

# prawo = 0, dół = 1, lewo = 2, góra = 3
# labirynt: 0 - nie ma ściany, 1 - jest 2 - start, 3 - koniec

labirynt = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y-math.sin(x))**2) + math.sin(z * u) + math.cos(v * w)

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
    endi, endj = find_end()

    fitness = 0
    for move in solution:
        if move == 0:
            if labirynt[i][j + 1] == 1:
                fitness -= 10
            else:
                j += 1
                fitness += 10
        if move == 1:
            if labirynt[i + 1][j] == 1:
                fitness -= 10
            else:
                i += 1
                fitness += 10
        if move == 2:
            if labirynt[i][j - 1] == 1:
                fitness -= 10
            else:
                j -= 1
                fitness += 10
        if move == 3:
            if labirynt[i - 1][j] == 1:
                fitness -= 10
            else:
                i -= 1
                fitness += 10
        
        if i == endi and j == endj:
            fitness += 500
            break

    distance = abs(i - endi) + abs(j - endj) + 1
    fitness /= distance

    return fitness

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 200
num_genes = 30

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 3000
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

start = time.time()

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
                       stop_criteria="reach_500"
                       )

#uruchomienie algorytmu
ga_instance.run()
end = time.time()

t = end - start
print(t)

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Generations passed: {generations_completed}".format(generations_completed=ga_instance.generations_completed)) # zad2 b

#tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
#prediction = numpy.sum(S*solution)
#print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()

times = [4.299352169036865, 1.4842016696929932, 0.3863046169281006, 2.442382574081421, 0.10264945030212402,
         0.038338661193847656, 0.9255707263946533, 3.24267315864563, 2.9554920196533203, 10.585959672927856]

avg = sum(times) / 10
print("AVG: ", avg)