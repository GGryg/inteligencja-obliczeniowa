import pygad
import numpy as np


# input
puzzle_15x15 = np.array([
    [-1, -1, -1, -1,  1,  2, -1, -1, -1,  3,  4, -1, -1, -1, -1],
    [-1, -1, -1,  5,  0,  0,  6, -1,  7,  0,  0,  8, -1, -1, -1],
    [-1, -1,  9,  0,  0,  0,  0, 10,  0,  0,  0,  0, 11, -1, -1],
    [-1, 12,  0,  0, 13, 14,  0,  0,  0, -1, 15,  0,  0, 16, 17],
    [-1, 18,  0,  0,  0,  0,  0,  0, -1, -1, -1, 19,  0,  0,  0],
    [20,  0,  0, 21,  0,  0,  0,  0, 22, 23, -1, 24, 25,  0,  0],
    [26,  0,  0,  0,  0, -1, 27,  0,  0,  0, 28,  0,  0, -1, -1],
    [-1, 29,  0,  0, 30, -1, 31,  0,  0,  0, 32,  0,  0, 33, -1],
    [-1, -1, 34,  0,  0, 35,  0,  0,  0, -1, -1, 36,  0,  0, 37],
    [-1, 38, 39,  0,  0, 40,  0,  0,  0, 41, 42,  0,  0,  0,  0],
    [43,  0,  0, 44, -1, -1, -1, 45,  0,  0,  0,  0, 46,  0,  0],
    [47,  0,  0,  0, 48, -1, -1, 49,  0,  0,  0,  0,  0,  0, -1],
    [-1, -1, 50,  0,  0, 51, 52,  0,  0,  0, 53, 54,  0,  0, -1],
    [-1, -1, -1, 55,  0,  0,  0,  0, 56,  0,  0,  0,  0, -1, -1],
    [-1, -1, -1, -1, 57,  0,  0, -1, -1, 58,  0,  0, -1, -1, -1]
])

puzzle_row_sums = np.array(
    [0,
      0,  0,  0,  0,
     10,  0, 12,  0,
     27, 21,  0,
     10,  0, 15,  9,  0,  0,
     34, 11,
     12, 23,  0,  0,  0, 12,
     19, 16, 10,
     13,  0, 24, 12,  0,
     11, 23,  8,  0,
      0,  7, 18,  0, 24,
     10,  0, 17,  9,
     13,  0, 29,
      4,  0, 13,  0,  8,
     25, 15,
     17, 10 
     ]
)

puzzle_col_sums = np.array(
    [0,
     17,  4,  4, 17,
     23, 30,  4, 17,
     15, 42,  4,
      0,  6, 16,  0, 16,  4,
     16,  0,
      0, 30, 42, 17, 17, 11,
      0,  0,  0,
      0,  4, 16,  0, 35,
      0,  0, 24,  3,
     16,  4,  0, 10,  3,
      0,  4,  0,  6,
      0,  4, 17,
      0, 17, 16, 17,  3,
      0,  0,
      0,  0,
     ]
)

# count how many places are avaiable for the chromosoms length
def how_many_blanks(puzzle):
    count = 0
    for i in range(len(puzzle)):
        for j in range(len(puzzle[i])):
            if puzzle[i, j] == 0:
                count += 1
    return count

def make_2D_array(s):
    array_2D = []
    count = 0
    for i in range(len(puzzle_15x15)):
        inside_array = []
        for j in range(len(puzzle_15x15[i])):
            if puzzle_15x15[i, j] != 0 or count > len(s) - 1:
                inside_array.append(0)
            else:
                inside_array.append(s[count])
                count += 1
        array_2D.append(inside_array)

    return array_2D

def print_array(arr):
    for i in arr:
        print(i)
        print("")

def fitness_func(solution, solution_idx):
    grid = make_2D_array(solution)
    col_diff = 0
    row_diff = 0

    for i in range(len(puzzle_15x15)):
        for j in range(len(puzzle_15x15[i])):
            if puzzle_15x15[i, j] > 0:
                sum_idx = puzzle_15x15[i, j]
                if puzzle_col_sums[sum_idx] != 0:
                    target_sum = puzzle_col_sums[sum_idx]
                    col_sum = 0
                    for i2 in range(i + 1, len(puzzle_15x15)):
                        if grid[i2][j] != 0:
                            col_sum += grid[i2][j]
                        else:
                            break
                    col_diff += abs(target_sum - col_sum)

                if puzzle_row_sums[sum_idx] != 0:
                    target_sum = puzzle_row_sums[sum_idx]
                    row_sum = 0
                    for j2 in range(j + 1, len(puzzle_15x15[j])):
                        if grid[i][j2] != 0:
                            row_sum += grid[i][j2]
                        else:
                            break
                    row_diff += abs(target_sum - row_sum)

    fitness = col_diff + row_diff
    return -fitness


fitness_function = fitness_func

num_solutions = 20

gene_space = [1, 2, 3, 4, 5, 6, 7, 8, 9]

sol_per_pop = 200
num_genes = how_many_blanks(puzzle_15x15)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 60% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 1000
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 1 # bo ma ponad sto.

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
print("Generations passed: {generations_completed}".format(generations_completed=ga_instance.generations_completed))


#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()