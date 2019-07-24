from scipy.optimize import differential_evolution
from scipy.optimize import fmin_bfgs
import numpy as np

count = 0
def ackley(x):
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    global count
    count += 1
    print(count)
    return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e



bounds = [(-5, 5), (-5, 5)]
result = differential_evolution(ackley, bounds, popsize=2, maxiter=3)
