import pygad
import numpy as np

input4x4_1 = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
])

row_sums_1 = [20, 14, 10]
col_sums_1 = [23, 14, 7]
input4x4_2 = [
    [-1, 18, 15, 7],
    [20, 0, 0, 0],
    [13, 0, 0, 0],
    [7, 0, 0, 0],
]

input4x4_3 = [
    [-1, 23, 10, 7],
    [18, 0, 0, 0],
    [12, 0, 0, 0],
    [10, 0, 0, 0]
]

puzzle = np.array([
    [0, 0, 0, 17, 0, 0, 0],
    [0, 0, 29, 0, 0, 15, 0],
    [0, 24, 0, 0, 21, 0, 0],
    [16, 0, 0, 0, 0, 0, 23],
    [0, 0, 25, 0, 0, 0, 0],
    [0, 20, 0, 0, 13, 0, 0],
    [0, 0, 0, 11, 0, 0, 0],
])

# Define the target sums for each row and column
row_sums = [29, 24, 12, 39, 25, 33, 11]
col_sums = [16, 21, 14, 17, 28, 13, 23]

def fitness_func(solution, solution_idx):
    grid = np.array(solution.reshape((3, 3)))

    row_diff = [abs(np.sum(grid[i, :]) - row_sums_1[i]) for i in range(3)]
    col_diff = [abs(np.sum(grid[:, j]) - col_sums_1[j]) for j in range(3)]

    fitness = sum(row_diff) + sum(col_diff)

    return -fitness

fitness_function = fitness_func

num_solutions = 20

gene_space = range(10)

sol_per_pop = 200
num_genes = 3 * 3

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 200
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 12

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
                       stop_criteria="reach_0"
                       )

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
ga_instance.plot_fitness()