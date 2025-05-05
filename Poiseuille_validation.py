import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

# Navier-Stokes Fluid Model | Finite Volume Method | Staggered Grid | Origin Upper Left Corner

# Poiseuille Validation:
# for all x,y in the mesh, the following must be true for the steady state:
# u(x,y) = poiseuille_force * y * (ly - y) / (2 * alpha)
# v(x,y) = 0

# Array Notation:
# array[0] - first item
# array[-1] - last item
# array[1] - second item
# array[-2] - second-to-last item
# array[1: -1] - second item to second-to-last item (does not include -1)

# -----------------------------------------
# Simulation Parameters
# -----------------------------------------

# Mesh
Nx = 100
Ny = 100
Lx = 30
Ly = 30
dx = Lx / Nx
dy = Ly / Ny
x = np.linspace(0, Lx, Nx + 2)
y = np.linspace(0, Ly, Ny + 2)
cell_area = dx * dy
X, Y = np.meshgrid(x, y)

# print(dx, dy)

# Constants
alpha = 0.01  # velocity diffusion speed (viscosity)
beta = 0.01  # temperature diffusion speed
kappa = 0.0  # temperature buoyancy constant
g = - 10.0
rho = 1.0  # fluid density

# Pressure
tolerance = 0.0000001
max_iteration = 100000
mu = 0.1

# Time Step Control
buffer = 0.1

# Poiseuille force
poiseuille_force = 0.0001

# -----------------------------------------
# ICs
# -----------------------------------------

# Scalars
P = np.ones([Ny + 2, Nx + 2])  # numpy is row-based, so Ny before Nx
u = np.zeros([Ny + 2, Nx + 2])
v = np.zeros([Ny + 2, Nx + 2])
T = np.zeros([Ny + 2, Nx + 2])

P_star = np.ones_like(P)
u_star = np.zeros_like(u)  # for the intermediate step with pressure to conserve mass
v_star = np.zeros_like(v)  # for the intermediate step with pressure to conserve mass

T_new = np.zeros_like(T)

# IC is the steady state, which should stay steady if my model is valid
for j in range(1, Ny + 1):  # y-direction, excluding ghost cells
    y_ = j * dy  # Center of the cell in the y-direction
    u[Ny - j, :] = poiseuille_force * (y_ * (Ly - y_)) / (2 * alpha)  # Linear velocity profile
    print(j)

# -----------------------------------------
# The Algorithm
# -----------------------------------------


def animate(frame):
    global u, v, P, T, t

    # Step 0: Time Step # -----------------------------------------

    # Compute maximum velocities for CFL condition
    u_max = np.max(np.abs(u)) + 1e-5
    v_max = np.max(np.abs(v)) + 1e-5

    # compute term time steps
    dt_velocity = min(dx / u_max, dy / v_max) * buffer
    dt_advection = min(dx / alpha, dy / alpha) * buffer
    dt_diffusion = min((dx ** 2) / (beta * 2), (dy ** 2) / (beta * 2)) * buffer

    # Choose the smallest dt to ensure stability
    dt = min(dt_advection, dt_diffusion, dt_velocity)
    # print(f"Dynamic Time Step: {dt}")

    # Step 1: BCs # -----------------------------------------

    # BC Types:
    # U or V = #.# if BC is #.#
    # U or V = 0.0 if BC is no slip
    # U or V = neighbor if BC is slip
    # U or V = opposite wall if BC is periodic

    # wall velocities are (where not explicitly defined) averaged of nearest two values

    # u velocity at tangential boundaries:
    U_t = 0.0  # no slip
    U_b = 0.0  # no slip

    # u velocities:
    u[:, -1] = u[:, 1]  # Right: Periodic
    u[:, 0] = u[:, -2]  # Left: Periodic
    u[-1, :] = 2 * U_b - u[-2, :]  # Bottom: u_b
    u[0, :] = 2 * U_t - u[1, :]  # Top: u_t

    # v velocities:
    v[-1, :] = 0.0  # Bottom: must be 0.0 to conserve mass
    v[0, :] = 0.0  # Top: must be 0.0 to conserve mass
    v[:, -1] = v[:, 1]  # Right: Periodic
    v[:, 0] = v[:, -2]  # Left: Periodic

    # temperature
    T[0, :] = T[1, :]  # Top: No Flux
    T[-1, :] = T[-2, :]  # Bottom: No Flux
    T[:, 0] = T[:, -2]  # Left: Periodic
    T[:, -1] = T[:, 1]  # Right: Periodic

    # pressure:
    P[:, 0] = P[:, -2]  # Left: periodic BC
    P[:, -1] = P[:, 1]  # Right: periodic BC
    P[0, :] = P[1, :]  # Top: Neumann BC (gradient = 0)
    P[-1, :] = P[-2, :]  # Bottom: Neumann BC (gradient = 0)

    # Step 2.1: Intermediate X-Velocities # -----------------------------------------

    # Pre-computations
    u_l = 0.5 * (u[1:-1, 1:-1] + u[1:-1, :-2])  # self + left average
    u_r = 0.5 * (u[1:-1, 1:-1] + u[1:-1, 2:])  # self + right average
    u_t = 0.5 * (u[1:-1, 1:-1] + u[:-2, 1:-1])  # self + top average
    u_b = 0.5 * (u[1:-1, 1:-1] + u[2:, 1:-1])  # self + right average

    v_b = 0.5 * (v[2:, 1:-1] + v[2:, :-2])  # see staggered grid, average of corner vs
    v_t = 0.5 * (v[1:-1, 1:-1] + v[1:-1, :-2])  # see staggered grid, average of corner vs

    u_p = u[1:-1, 1:-1]

    # Advection/Convection terms for u
    u_conv = - ((u_r ** 2 - u_l ** 2) * dy + (u_t * v_t - u_b * v_b) * dx)

    # Diffusion terms for u
    u_diff = 2 * alpha * (
            (u_t + u_b - 2 * u_p) * (dx / dy) + (u_l + u_r - 2 * u_p) * (dy / dx)
    )

    # FVM changes the force term a bit
    poiseuille_source = rho * poiseuille_force * cell_area

    # Update intermediate u_star
    u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + (dt / cell_area) * (u_conv + u_diff + poiseuille_source)

    # BCs
    u_star[:, -1] = u_star[:, 1]  # Right: Periodic
    u_star[:, 0] = u_star[:, -2]  # Left: Periodic

    # Step 2.2: Intermediate Y-Velocities # -----------------------------------------

    # Pre-computations
    v_l = 0.5 * (v[1:-1, 1:-1] + v[1:-1, :-2])  # self + left average
    v_r = 0.5 * (v[1:-1, 1:-1] + v[1:-1, 2:])  # self + right average
    v_b = 0.5 * (v[1:-1, 1:-1] + v[2:, 1:-1])  # self + top average
    v_t = 0.5 * (v[1:-1, 1:-1] + v[:-2, 1:-1])  # self + right average

    u_l = 0.5 * (u[:-2, 1:-1] + u[1:-1, 1:-1])  # see staggered grid, average of corner vs
    u_r = 0.5 * (u[:-2, 2:] + u[1:-1, 2:])  # see staggered grid, average of corner vs

    v_p = v[1:-1, 1:-1]

    # Advection/Convection terms for v
    v_conv = - ((u_r * v_r - u_l * v_l) * dy + (v_t ** 2 - v_b ** 2) * dx)

    # Diffusion terms for v
    v_diff = 2 * alpha * (
            (v_t + v_b - 2 * v_p) * (dx / dy) + (v_l + v_r - 2 * v_p) * (dy / dx)
    )

    # Buoyancy term for v
    buoyancy = g * kappa * (T[1:-1, 1:-1] - 0.0)

    # Update intermediate v_star
    v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + (dt / cell_area) * (v_conv + v_diff + buoyancy)

    v_star[:, -1] = v_star[:, 1]  # Right: Periodic
    v_star[:, 0] = v_star[:, -2]  # Left: Periodic

    # Step 3: Pressure Correction # -----------------------------------------

    # We want to solve for P, so that we can then use it for calculating its gradient

    # Pre-computations
    u_star_r = u_star[1:-1, 2:]
    u_star_p = u_star[1:-1, 1:-1]

    # Pre-computations
    v_star_b = v_star[2:, 1:-1]
    v_star_p = v_star[1:-1, 1:-1]

    # Compute divergence of velocity vector
    # because of the staggered grid, the velocity components are on either side of the cell center, allowing for
    # the divergence of velocity to be calculated at exactly the cell center, rather than forward or central
    # differences, for example
    div_velocity_star = np.zeros([Ny + 2, Nx + 2])
    div_velocity_star[1:-1, 1:-1] = (u_star_r - u_star_p) / dx + (v_star_b - v_star_p) / dy

    # Recall the right hand side (rhs) of the pressure poisson equation:
    poisson_rhs = rho * div_velocity_star / dt

    # Pre-computations
    P_t = P[:-2, 1:-1]
    P_b = P[2:, 1:-1]
    P_l = P[1:-1, :-2]
    P_r = P[1:-1, 2:]

    # The left hand side of the pressure equation can be expanded using central difference for the second derivative
    # poisson_lhs = (P_t + P_b - 2 * P_p) / (dy ** 2) + (P_l + P_r - 2 * P_p) / (dx ** 2)
    # note that the poisson_lhs is not used as a variable, rather, it is needed to solve for P_p
    # we solve the poisson equation for P, which is denoted as P_p

    # Iteration
    iteration = 0
    error = 1e10
    while error > tolerance and iteration < max_iteration:
        P_copy = P.copy()

        # BCs
        P[1: -1, 0] = P[1: -1, -2]  # Left: Periodic
        P[1: -1, -1] = P[1: -1, 1]  # Right: Periodic
        P[0, 1: -1] = P[1, 1: -1]
        P[-1, 1: -1] = P[-2, 1: -1]

        # Solving for P
        P_star[1:-1, 1:-1] = (
                (dx ** 2 * (P_t + P_b) + dy ** 2 * (P_l + P_r) - poisson_rhs[1:-1, 1:-1] * dx ** 2 * dy ** 2) /
                (2 * (dx ** 2 + dy ** 2))
        )

        # This is just saying P = P_star, but mu is there to control the 'speed of pressure flow'
        P[1:-1, 1:-1] = P_copy[1:-1, 1:-1] + mu * (P_star[1:-1, 1:-1] - P_copy[1:-1, 1:-1])

        error = np.linalg.norm(P.ravel() - P_copy.ravel(), 2)
        iteration += 1

    # print out correction note
    # print(f'The pressure was corrected with an error of: {error}')

    # save previous x velocity for error convergence check
    u_old = u.copy()

    # correct the velocities with the known pressure gradient
    # include only the velocities that are between interior pressure cells, so not those along the 'red' boundary on
    # the staggered grid, hence the funny array shapes
    u[1:-1, 1:] = u_star[1:-1, 1:] - (dt / rho) * (P[1:-1, 1:] - P[1:-1, :-1]) / dx  # not boundary us
    v[2:-1, :] = v_star[2:-1, :] - (dt / rho) * (P[2:-1, :] - P[1:-2, :]) / dy  # not boundary vs
    # cell area (the integration over cell volume) cancels out for both transient and pressure gradient terms, so a
    # standard FDM is appropriate here due to the structured nature of the grid

    # Step 4: Temperature # -----------------------------------------

    # bring velocities at faces to the cell centers for calculations
    u_c = 0.5 * (u[1:-1, 1:-1] + u[1:-1, 2:])  # self + left average
    v_c = 0.5 * (v[1:-1, 1:-1] + v[2:, 1:-1])  # self + bottom average

    # Upwind scheme is required for advection/convection terms
    # For u (x-direction)
    T_adv_x = np.where(
        u_c > 0,
        u_c * (T[1:-1, 1:-1] - T[1:-1, :-2]) / dx,  # Positive velocity
        u_c * (T[1:-1, 2:] - T[1:-1, 1:-1]) / dx  # Negative velocity
    )

    # For v (y-direction)
    T_conv_y = np.where(
        v_c > 0,
        v_c * (T[1:-1, 1:-1] - T[:-2, 1:-1]) / dy,  # Positive velocity
        v_c * (T[2:, 1:-1] - T[1:-1, 1:-1]) / dy  # Negative velocity
    )

    T_conv = - (T_adv_x + T_conv_y)

    # Diffusion
    T_diff = beta * (
            (T[1:-1, 2:] + T[1:-1, :-2] - 2 * T[1:-1, 1:-1]) / dx ** 2 +
            (T[2:, 1:-1] + T[:-2, 1:-1] - 2 * T[1:-1, 1:-1]) / dy ** 2
    )

    # Update temperature
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * (T_conv + T_diff)

    T[1:-1, 1:-1] = T_new[1:-1, 1:-1]

    # Visuals # -----------------------------------------

    t += dt

    # Update the image data without clearing the axis
    im.set_data(T[1:-1, 1:-1])
    ax.set_title('Temperature & Velocity Field')

    # Update quiver plot
    # Interpolate velocities to cell centers for plotting
    u_center = 0.5 * (u[1:-1, 1:-1] + u[1:-1, 2:])
    v_center = 0.5 * (v[1:-1, 1:-1] + v[2:, 1:-1])

    Uq = u_center[::skip, ::skip]
    Vq = - v_center[::skip, ::skip]

    quiver.set_UVC(Uq, Vq)

    # Poiseuille Validator # -----------------------------------------

    print(f"Max y velocity: {np.max(np.abs(v))}")

    # Coordinates for validating | In terms of index length (Ny), not mesh length (Ly)
    x_column = 1  # can be anything, ideally u is the same for all x for a given y
    y_1 = 2
    y_2 = 4
    y_3 = 10

    # Numerical solutions
    n_1 = u[Ny - y_1, x_column]
    n_2 = u[Ny - y_2, x_column]
    n_3 = u[Ny - y_3, x_column]

    # Analytical solutions
    a_1 = (poiseuille_force / (2 * alpha)) * ((y_1 * dy) * (Ly - (y_1 * dy)))
    a_2 = (poiseuille_force / (2 * alpha)) * ((y_2 * dy) * (Ly - (y_2 * dy)))
    a_3 = (poiseuille_force / (2 * alpha)) * ((y_3 * dy) * (Ly - (y_3 * dy)))

    # Print the numerical solution and the analytical solution (what it should equal)
    print(f"u(x, {y_1 * dy}) = {n_1}, which should equal {a_1}")
    print(f"u(x, {y_2 * dy}) = {n_2}, which should equal {a_2}")
    print(f"u(x, {y_3 * dy}) = {n_3}, which should equal {a_3}")

    # Print error between numerical and analytical
    # for reference, the error is relative to the U_t velocity, which is 1.0 for this simulation
    print(f"Error: {abs((n_1 - a_1) / a_1 * 100)} %")
    print(f"Error: {abs((n_2 - a_2) / a_2 * 100)} %")
    print(f"Error: {abs((n_3 - a_3) / a_3 * 100)} %")

    error_global = np.linalg.norm(u - u_old, 2)
    print(f"Global Error (L2 Norm): {error_global}")

    return [im, quiver]


# -----------------------------------------
# Visuals
# -----------------------------------------

# time set-up
t = 0
n_steps = 500

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(T[1:-1, 1:-1], extent=[0, Lx, 0, Ly], origin='upper', cmap=cm.jet, vmin=0, vmax=100)
fig.colorbar(im, ax=ax)
ax.set_title('Temperature Field')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')

# Create a coarser grid for quiver plot to avoid clutter
skip = 2  # Skip every 4 data points
Xq = X[1:-1, 1:-1][::skip, ::skip]
Yq = Y[1:-1, 1:-1][::skip, ::skip]

# Initialize quiver plot
quiver = ax.quiver(Xq, Ly - Yq, np.zeros_like(Xq), np.zeros_like(Yq), color='white', scale=50)

# Create the animation
anim = FuncAnimation(fig, animate, frames=n_steps, interval=50, blit=True)

# Show the animation
plt.show()
