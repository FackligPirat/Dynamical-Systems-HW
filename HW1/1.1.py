#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root_scalar

#%% Plot burification
# Define the dimensionless equation
def dxdt(x, r):
    return 1/5 + 7/(10 * (1 + np.exp(80 * (1 - x)))) - r * x**4

def stability(x, r):
    h = 1e-5  
    return (dxdt(x + h, r) - dxdt(x, r)) / h

r_values = np.linspace(0.01, 1.5, 600)

fixed_points = []
stability_info = []

for r in r_values:
    guesses = np.linspace(0, 1.5, 10)
    solutions = []
    for guess in guesses:
        sol, _, ier, _ = fsolve(dxdt, guess, args=(r,), full_output=True)
        if ier == 1: 
            sol = np.round(sol, 6)  
            if sol not in solutions:
                solutions.append(sol)
    
    stability_status = [stability(x, r) for x in solutions]
    fixed_points.append(solutions)

    stability_info.append(stability_status)

plt.figure(figsize=(10, 6))
for i, r in enumerate(r_values):
    for x, stab in zip(fixed_points[i], stability_info[i]):
        if stab < 0:  # Stable
            plt.plot(r, x, 'bo', markersize=2)
        else:  # Unstable
            plt.plot(r, x,'ro', markersize=2)
            print(f'rvalue: {r}, xvalue: {x}')

plt.title("Bifurcation Diagram")
plt.xlabel("r")
plt.ylabel("x*")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.grid()
plt.show()

