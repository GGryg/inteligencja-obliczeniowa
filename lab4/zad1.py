# Import modules
import numpy as np
from matplotlib import pyplot as plt
import math
# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history

def endurance(arr):
    return -(math.exp(-2 * (arr[1]-math.sin(arr[0]))**2) + math.sin(arr[2] * arr[3]) + math.cos(arr[4] * arr[5]))

def f(x):
    j = [endurance(x[i]) for i in range(len(x))]
    return np.array(j)


# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.7}

# Create bounds
x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)
bounds = my_bounds

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=1000)

# Obtain cost history from optimizer instance
cost_history = optimizer.cost_history

# Plot!
plot_cost_history(cost_history)
plt.show()