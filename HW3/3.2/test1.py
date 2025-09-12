#%%
import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import eigvals
import matplotlib.pyplot as plt

# Constants
a = 4/9
b = 5/9
epsilon = 1
I_c = 68/405  # Hopf bifurcation point

# Function to define the system's equations
def system_equations(vars, I):
    x, y = vars
    eq1 = (1/epsilon) * (x - (1/3) * x**3 - y + I)
    eq2 = x + a - b * y
    return [eq1, eq2]

# Jacobian of the system
def jacobian(x, y):
    J = np.array([
        [1/epsilon * (1 - x**2), -1/epsilon],
        [1, -b]
    ])
    return J

# Range of I values
I_vals = np.linspace(0, 1, 200)

# Arrays to store the real and imaginary parts of the eigenvalues
Re_eigenvals = []
Im_eigenvals = []

# Initial guess for the fixed point
initial_guess = [0, 0]

# Loop over the range of I values
for I in I_vals:
    # Find the fixed point for the current I
    fixed_point = fsolve(system_equations, initial_guess, args=(I,))
    x_fp, y_fp = fixed_point

    # Compute the Jacobian at the fixed point
    J = jacobian(x_fp, y_fp)

    # Compute eigenvalues of the Jacobian
    eigenvalues = eigvals(J)

    # Store the real and imaginary parts of the eigenvalues
    Re_eigenvals.append(np.real(eigenvalues))
    Im_eigenvals.append(np.imag(eigenvalues))

# Convert lists to numpy arrays for easier plotting
Re_eigenvals = np.array(Re_eigenvals)
Im_eigenvals = np.array(Im_eigenvals)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(I_vals, Re_eigenvals[:, 0], label='Re[$\lambda_1$]', color='blue')
plt.plot(I_vals, Re_eigenvals[:, 1], label='Re[$\lambda_2$]', color='blue', linestyle='--')
plt.plot(I_vals, np.abs(Im_eigenvals[:, 0]), label='|Im[$\lambda_1$]|', color='red')
plt.plot(I_vals, np.abs(Im_eigenvals[:, 1]), label='|Im[$\lambda_2$]|', color='red', linestyle='--')

# Indicate the Hopf bifurcation point
plt.axvline(I_c, color='purple', linestyle=':', label=f'Hopf Bifurcation ($I_c = {I_c:.4f}$)')

# Labels and legend
plt.xlabel('I')
plt.ylabel('Eigenvalues')
plt.title('Real and Imaginary Parts of Eigenvalues vs I')
plt.legend()
plt.grid(True)
plt.show()
