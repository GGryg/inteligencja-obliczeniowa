import gym
import pygad
import numpy as np

def fitness_func(solution, solution_idx):
    env = gym.make("LunarLander-v2")
    env.reset(seed=8)

    total_reward = 0
    
    for action in solution:
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    return total_reward


fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 25
num_genes = 200

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 100
keep_parents = 4

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 1

ga_instance = pygad.GA(gene_space=(0, 1, 2, 3),
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
                       gene_type=int,
                       stop_criteria="reach_200"
                       )

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

if solution_fitness > 0:
    env = gym.make("LunarLander-v2", render_mode="human")
    env.reset(seed=8)

    for action in solution:
        env.render()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

# chromosomy to akcje
#nie rób nic, lewy silnik, główny silnik, prawy silnik

# fitness
# na podstawie punktów z gry czyli
# 100-140 punktów za wylądowanie, im dalej od miejsca lądowania to traci punkty, jeżeli się rozbije to -100 punktów
# Jeżeli wylądouje to +100 punktów, z nogami na ziemi +10 punktów
# za odpalenie silników jest -0.3 punktów za każdy frame
# za zrobienie poprawnie jest 200 punktów
# im większy wynik tym lepiej