#%%
import numpy as np
import matplotlib.pyplot as plt

#%% Functions a)
def dx_dt(x, y):
    return y

def dy_dt(x, y):
    return x**2

#%% Functions b)
def h(r, a=1):
    return a * r

def dx_dt2(x, y, a=1):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return h(r, a) * np.cos(theta)

def dy_dt2(x, y, a=1):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return h(r, a) * np.sin(theta)

#%% Functions c)

def dx_dt3(x, y):
    return y**3

def dy_dt3(x, y):
    return x

#%% Functions d)

def dx_dt4(x, y, n=3):
    r = x**2+y**2
    theta = np.arctan2(y, x)
    return r ** (n/2) * np.cos(n*theta)

def dy_dt4(x, y, n=3):
    r = x**2+y**2
    theta = np.arctan2(y, x)
    return r ** (n/2) * np.sin(n*theta)

# Generate a grid of points
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)

#%% Plot a)
U = dx_dt(X, Y)
V = dy_dt(X, Y)

# Plot the stream plot
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, U, V, color='b', linewidth=1)
plt.title('Phase Portrait of a)')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid()
plt.show()

# %%
#%% Plot b)
U = dx_dt2(X, Y)
V = dy_dt2(X, Y)

# Plot the stream plot
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, U, V, color='b', linewidth=1)
plt.title('Phase Portrait of b)')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid()
plt.show()

#%% Plot c)
U = dx_dt3(X, Y)
V = dy_dt3(X, Y)

# Plot the stream plot
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, U, V, color='b', linewidth=1)
plt.title('Phase Portrait of c)')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid()
plt.show()

#%% Plot d)
U = dx_dt4(X, Y, 2)
V = dy_dt4(X, Y, 2)

# Plot the stream plot
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, U, V, color='b', linewidth=1)
plt.title('Phase Portrait of d) with n = 3')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid()
plt.show()

# %%
