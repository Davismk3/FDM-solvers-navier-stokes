import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------
# Simulation Parameters
# -----------------------------------------

# Mesh
Lx = 1.0
Nx = 70
dx = Lx / Nx
x = np.linspace(0, Lx, Nx)  # from 0 to Lx, with Nx points

# Constants
alpha = 1.0
beta = 0.1

# Time Step CFL Condition
buffer = 0.9  # for CFL condition to further stability

dt_advection = dx / alpha * buffer
print(f"Advection Time Step: {dt_advection}")

dt_diffusion = (dx ** 2) / (beta * 2) * buffer
print(f"Diffusion Time Step: {dt_diffusion}")

Nt = 100

# -----------------------------------------
# Initialize Scalar Fields
# -----------------------------------------

# IC for advection
phi = np.ones_like(x)  # phi field matches the x mesh with phi = 1 at each point
phi[int(0.25 * Nx): int(0.75 * Nx)] = 5.0  # initial pulse

# IC for diffusion
theta = np.ones_like(x)
theta[int(0.25 * Nx): int(0.75 * Nx)] = 5.0  # initial pulse

# -----------------------------------------
# Advection & Diffusion Algorithms
# -----------------------------------------


def advection():
    # It is interesting to note that this advection equation is the 1D equivalent of the material derivative if set to
    # equal zero. So it is a convection equation for phi, except rather than the velocity "u", there is the constant
    # "a", so the speed of the advection equation is generally constant, whereas it usually varies and is in more than
    # one dimension for convection, such as in the Navier-Stokes equations.

    global phi

    phi_new = phi.copy()

    # Solving for phi_new with forward difference for time (arrays are same size despite shifts)

    # If advection is to the left, forward difference for space is used
    phi_new[1:-1] = phi[1:-1] + dt_advection * alpha * (phi[2:] - phi[1:-1]) / (2 * dx)

    # If advection is to the right, backward difference for space is used
    # phi_new[1:-1] = phi[1:-1] - dt_advection * alpha * (phi[1:-1] - phi[:-2]) / (2 * dx)

    # This is called the 'Upwind Scheme', and is a fundamental aspect of solving advection/convection numerically

    # BCs
    phi_new[0] = phi[1]  # Neumann, no flux
    phi_new[-1] = 1.0  # Dirichlet

    # Cyclic BCs
    # phi_new[0] = phi[1]
    # phi_new[-1] = phi_new[0]

    phi = phi_new.copy()

    return phi


def diffusion():

    global theta

    theta_new = theta.copy()

    # Diffusion
    theta_new[1:-1] = theta[1:-1] + dt_diffusion * beta * (theta[2:] + theta[:-2] - 2 * theta[1:-1]) / (dx ** 2)

    # BCs
    theta[0] = theta[1]  # Neumann, no flux
    theta[-1] = 1.0  # Dirichlet

    theta = theta_new.copy()

    return theta


# -----------------------------------------
# Visuals
# -----------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
ax1, ax2 = axes.flatten()

font_size = 10

# Advection
phi_line, = ax1.plot(x, phi)
ax1.set_xlim(0, Lx)
ax1.set_ylim(0, 6)
ax1.set_xlabel('Position (m)', fontsize=14)
ax1.set_ylabel('Phi', fontsize=14)
ax1.set_title('1D Advection Simulation', fontsize=16)

# Diffusion
theta_line, = ax2.plot(x, theta)
ax2.set_xlim(0, Lx)
ax2.set_ylim(0, 6)
ax2.set_xlabel('Position (m)', fontsize=14)
ax2.set_ylabel('Phi', fontsize=14)
ax2.set_title('1D Diffusion Simulation', fontsize=16)


def animate(frame):
    global phi, theta

    phi = advection()
    phi_line.set_ydata(phi)
    ax1.set_title(f'1D Advection Simulation', fontsize=16)

    theta = diffusion()
    theta_line.set_ydata(theta)
    ax2.set_title(f'1D Diffusion Simulation', fontsize=16)

    return phi_line, theta_line


# Create the animation object
anim = FuncAnimation(fig, animate, frames=Nt, interval=50, blit=False)

# Display the animation
plt.show()
