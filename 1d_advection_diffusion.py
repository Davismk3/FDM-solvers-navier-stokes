import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Finite Difference Simulation of 1D Advection and Diffusion:

∂φ/∂t + α⋅∂φ/∂x = 0

∂θ/∂t = β⋅∂²θ/∂x²

This script provided me insight into numerically solving the transport and diffusion 
terms of the Navier-Stokes equations (excluding pressure). Upwind and central difference 
schemes are used for spatial derivatives, and forward Euler for time integration.
"""

# Parameters & Grid Setup -----------------------------------------------------
Lx = 1.0
Nx = 50
dx = Lx / Nx
x = np.linspace(0, Lx, Nx)

alpha = 1.0  # advection speed
beta = 0.1  # diffusion coefficient
buffer = 0.9  # CFL stability buffer

dt_advection = dx / alpha * buffer
dt_diffusion = dx**2 / (2 * beta) * buffer
Nt = 100 

print(f"Advection Time Step: {dt_advection}")
print(f"Diffusion Time Step: {dt_diffusion}")

# Initial Conditions ----------------------------------------------------------
phi = np.ones_like(x)
phi[int(0.25 * Nx): int(0.75 * Nx)] = 5.0  # rectangular pulse for advection

theta = np.ones_like(x)
theta[int(0.25 * Nx): int(0.75 * Nx)] = 5.0  # same for diffusion

# Advection & Diffusion Solvers -----------------------------------------------
def advection():
    """
    Solves the 1D linear advection equation using an upwind scheme.
    """

    global phi
    phi_new = phi.copy()

    # Forward differencing in space (flows to left)
    phi_new[1:-1] = phi[1:-1] + dt_advection * alpha * (phi[2:] - phi[1:-1]) / (2 * dx)

    # Boundary Conditions (these are not required for the transport equation)
    phi_new[0] = phi[1]  # Neumann
    phi_new[-1] = 1.0  # Dirichlet        

    phi = phi_new.copy()
    return phi

def diffusion():
    """
    Solves the 1D linear diffusion equation using a central difference scheme.
    """

    global theta
    theta_new = theta.copy()

    # Central difference for diffusion
    theta_new[1:-1] = theta[1:-1] + dt_diffusion * beta * (
        theta[2:] - 2 * theta[1:-1] + theta[:-2]) / dx**2

    # Boundary Conditions
    theta_new[0] = theta[1]  # Neumann
    theta_new[-1] = theta[-2]  # Neumann 

    theta = theta_new.copy()
    return theta

# Visualization ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
ax1, ax2 = axes

# Advection Plot Setup
phi_line, = ax1.plot(x, phi)
ax1.set_xlim(0, Lx)
ax1.set_ylim(0, 6)
ax1.set_xlabel('x')
ax1.set_ylabel(r'$\phi$')
ax1.set_title('1D Advection')

# Diffusion Plot Setup
theta_line, = ax2.plot(x, theta)
ax2.set_xlim(0, Lx)
ax2.set_ylim(0, 6)
ax2.set_xlabel('x')
ax2.set_ylabel(r'$\theta$')
ax2.set_title('1D Diffusion')

def animate(frame):
    global phi, theta
    phi = advection()
    theta = diffusion()

    phi_line.set_ydata(phi)
    theta_line.set_ydata(theta)

    return phi_line, theta_line

anim = FuncAnimation(fig, animate, frames=Nt, interval=50, blit=False)
plt.show()