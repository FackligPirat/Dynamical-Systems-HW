#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
#%% a)
def jacobian(x, alpha):
    J = np.array([[0, 1],
                  [-np.cos(x), -alpha]])
    return J

def classify_fixed_points(alpha):
    fixed_points = [0, np.pi]
    for x_fp in fixed_points:
        J = jacobian(x_fp, alpha)
        eigenvalues, _ = np.linalg.eig(J)
        
        if np.all(np.real(eigenvalues) < 0) and eigenvalues[0] != eigenvalues[1]:
            stability = "Stable (Sink)"
        elif np.all(np.real(eigenvalues)< 0):
            stability = "Degenerate"
        elif np.all(np.real(eigenvalues) > 0):
            stability = "Unstable (Source)"
        else:
            stability = "Saddle Point"
        
        print(f"Fixed point x = {x_fp:.2f}, Eigenvalues: {eigenvalues}, Stability: {stability}")

alphas = [0, 0.5, 1, 2]

for alpha in alphas:
    print(f"\nAlpha = {alpha}")
    classify_fixed_points(alpha)

#%% b)

import numpy as np
import matplotlib.pyplot as plt

def pendulum_system_vector_field(x, y, alpha):
    dxdt = y
    dydt = -np.sin(x) - alpha * y
    return dxdt, dydt

def plot_phase_portrait_stream(ax, alpha, fixed_point, title):
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    x_values = np.linspace(fixed_point - np.pi / 2, fixed_point + np.pi / 2, 100)
    y_values = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_values, y_values)
    
    U, V = pendulum_system_vector_field(X, Y, alpha)
    
    ax.streamplot(X, Y, U, V, color='b', density=1.5, linewidth=1)
    
    ax.plot(fixed_point, 0, 'ro', markersize=10)
    ax.grid(True)
    ax.set_xlim([fixed_point - np.pi / 2, fixed_point + np.pi / 2])
    ax.set_ylim([-2, 2])

fixed_points = [0, np.pi]

alphas = [0, 0.5, 1, 2]

for alpha in alphas:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 
    fig.suptitle(f"Phase Portraits for α = {alpha}", fontsize=16)
    
    for j, fixed_point in enumerate(fixed_points):
        title = f"x = {fixed_point:.2f}"
        plot_phase_portrait_stream(axs[j], alpha, fixed_point, title)
    
    plt.tight_layout()
    plt.show()