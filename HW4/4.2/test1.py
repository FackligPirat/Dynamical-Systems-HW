#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sp

# Define symbols for symbolic differentiation
t = sp.symbols('t')
x1, x2 = sp.symbols('x1 x2', cls=sp.symbols)
mu = 1/10
omega = 1
nu = 1

# Define the functions f1 and f2
f1 = mu * x1 - nu * x1**2 * x2 - x1 * x2**2 - x1**3 - nu * x2**3 - omega * x2
f2 = nu * x1**3 + nu * x1 * x2**2 - x1**2 * x2 + omega * x1 + mu * x2 - x2**3

# Compute the Jacobian elements
J11 = sp.diff(f1, x1)
J12 = sp.diff(f1, x2)
J21 = sp.diff(f2, x1)
J22 = sp.diff(f2, x2)

# Convert Jacobian to functions for numerical evaluation
J11_func = sp.lambdify((x1, x2), J11)
J12_func = sp.lambdify((x1, x2), J12)
J21_func = sp.lambdify((x1, x2), J21)
J22_func = sp.lambdify((x1, x2), J22)

# Define the system of differential equations
def system(t, Y):
    x1, x2, M11, M12, M21, M22 = Y
    
    # Evaluate the Jacobian at current x1, x2
    J11_val = J11_func(x1, x2)
    J12_val = J12_func(x1, x2)
    J21_val = J21_func(x1, x2)
    J22_val = J22_func(x1, x2)
    
    # Define the differential equations
    dx1_dt = mu * x1 - nu * x1**2 * x2 - x1 * x2**2 - x1**3 - nu * x2**3 - omega * x2
    dx2_dt = nu * x1**3 + nu * x1 * x2**2 - x1**2 * x2 + omega * x1 + mu * x2 - x2**3
    
    dM11_dt = J11_val * M11 + J12_val * M21
    dM12_dt = J11_val * M12 + J12_val * M22
    dM21_dt = J21_val * M11 + J22_val * M21
    dM22_dt = J21_val * M12 + J22_val * M22
    
    return [dx1_dt, dx2_dt, dM11_dt, dM12_dt, dM21_dt, dM22_dt]

# Initial conditions
x1_0 = np.sqrt(mu)
x2_0 = 0
M11_0, M12_0 = 1, 0
M21_0, M22_0 = 0, 1

Y0 = [x1_0, x2_0, M11_0, M12_0, M21_0, M22_0]

# Time parameters
t0 = 0
t_max = 2 * np.pi / (omega + nu * mu)
t_span = (t0, t_max)
t_eval = np.linspace(t0, t_max, 1000)

# Solve the system
sol = solve_ivp(system, t_span, Y0, t_eval=t_eval)

# Extract solution components
t_vals = sol.t
x1_vals, x2_vals, M11_vals, M12_vals, M21_vals, M22_vals = sol.y

# Plot the trajectories
plt.figure(figsize=(10, 6))
plt.plot(t_vals, x1_vals, label='x1', color='red', linewidth=2)
plt.plot(t_vals, x2_vals, label='x2', color='green', linewidth=2)
plt.plot(t_vals, M11_vals, label='M11', color='blue', linewidth=2)
plt.plot(t_vals, M12_vals, label='M12', color='orange', linewidth=2)
plt.plot(t_vals, M21_vals, label='M21', color='purple', linewidth=2)
plt.plot(t_vals, M22_vals, label='M22', color='brown', linewidth=2)

# Add labels and legends
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Trajectories of x1, x2, M11, M12, M21, M22')
plt.legend()
plt.grid(True)
plt.show()

# %%
M11_at_t_max = M11_vals[-1]
M12_at_t_max = M12_vals[-1]
M21_at_t_max = M21_vals[-1]
M22_at_t_max = M22_vals[-1]

# Round the results to 4 decimal places
M_matrix = np.round([[M11_at_t_max, M12_at_t_max], [M21_at_t_max, M22_at_t_max]], 4)

# Display the result as a matrix
print("Matrix at t = t_max:")
print(M_matrix)
# %% f
solMat = np.array([[M11_at_t_max, M12_at_t_max],
                   [M21_at_t_max, M22_at_t_max]])

# Compute the eigenvalues
eigenvalues = np.linalg.eigvals(solMat)

# Compute 1/t_max * log of each eigenvalue
sol_tilde = np.round([1 / t_max * np.log(eigenvalues[1]), 1 / t_max * np.log(eigenvalues[0])], 4)

# Display the result
print("solTilde:")
print(sol_tilde)
# %%
