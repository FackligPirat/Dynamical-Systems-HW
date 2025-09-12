#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
#%% Functions
# Define the analytical solution for x(t) and y(t)
def analytical_solution(t, sigma, u, v):
    omega = np.sqrt(5)  # Frequency
    exp_term = np.exp(sigma * t)

    x = exp_term * (u * np.cos(omega * t) + (u + 3 * v) / omega * np.sin(omega * t))
    y = exp_term * (v * np.cos(omega * t) - (2 * u + v) / omega * np.sin(omega * t))

    return x, y
#%% Main b)
# Time range for the plots
t = np.linspace(0, 20, 1000)

# Parameters for initial conditions
u, v = 1, 1  

sigma_values = [-1/10, 0, 1/10]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, sigma in enumerate(sigma_values):
    x, y = analytical_solution(t, sigma, u, v)
    
    axes[i].plot(x, y, label=f"$\\sigma = {sigma}$")
    axes[i].set_title(f"$\\sigma = {sigma}$")
    axes[i].set_xlabel("x(t)")
    axes[i].set_ylabel("y(t)")
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()
