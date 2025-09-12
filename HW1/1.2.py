#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D

x, h, r = sp.symbols('x h r')

f = h + x * (r - x)

fixed_points = sp.solve(f, x)
fixed_points

df_dx = sp.diff(f, x)

stabilities = [df_dx.subs(x, fp) for fp in fixed_points]

r_vals = np.linspace(-2, 2, 400)
bifurication_curve = -r_vals**2 / 4

plt.figure(figsize=(8, 6))
plt.plot(bifurication_curve, r_vals, label="Bifurcation Curve: $h = -r^2 / 4$", color="blue", linewidth=2)

plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("$h$", fontsize=14)
plt.ylabel("$r$", fontsize=14)
plt.title("Bifurcation Diagram in $(h, r)$-Plane", fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

#%% 3D plot
h_vals = np.linspace(-1, 1, 200)
r_vals = np.linspace(-2, 2, 200)
H, R = np.meshgrid(h_vals, r_vals)

sqrtTerm = np.sqrt(4 * H + R**2)
x_fixed_1 = R / 2 - sqrtTerm / 2  
x_fixed_2 = R / 2 + sqrtTerm / 2 

mask = (4 * H + R**2 < 0)
x_fixed_1[mask] = np.nan
x_fixed_2[mask] = np.nan

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(R, H, x_fixed_1, color='red', alpha=0.6, label="Unstable Fixed Point")
ax.plot_surface(R, H, x_fixed_2, color='green', alpha=0.6, label="Stable Fixed Point")

ax.set_title("3D Plot of Fixed Points $x^*(h, r)$", fontsize=16)
ax.set_xlabel("$r$", fontsize=14)
ax.set_ylabel("$h$", fontsize=14)
ax.set_zlabel("$x^*$", fontsize=14)
plt.show()

# %%
