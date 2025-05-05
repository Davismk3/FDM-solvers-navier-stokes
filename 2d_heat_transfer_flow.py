import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

# Navier-Stokes Fluid Model | Finite Volume Method | Staggered Grid | Origin Upper Left Corner

# Two meshes, where heat can diffuse across their shared boundary, but fluid can't flow across it

# Array Notation:
# array[0] - first item
# array[-1] - last item
# array[1] - second item
# array[-2] - second-to-last item
# array[1: -1] - second item to second-to-last item (does not include -1)

# -----------------------------------------
# Simulation Parameters
# -----------------------------------------

# Global 'Mesh'
Nx = 100
Ny = 100
Lx = 30.0
Ly = 30.0
dx = Lx / Nx
dy = Ly / Ny
cell_area = dx * dy

# Mantle Mesh
Nx_m = Nx
Ny_m = Ny // 2  # be sure this is an integer
Lx_m = Lx
Ly_m = Ly // 2  # be sure this is the same denominator as for Ny_m
x_m = np.linspace(0, Lx_m, Nx_m + 2)
y_m = np.linspace(0, Ly_m, Ny_m + 2)
X_m, Y_m = np.meshgrid(x_m, y_m)

# Core Mesh
Nx_c = Nx
Ny_c = Ny - Ny_m  # be sure this is an integer
Lx_c = Lx
Ly_c = Ly - Ly_m
x_c = np.linspace(0, Lx_c, Nx_c + 2)
y_c = np.linspace(0, Ly_c, Ny_c + 2)
X_c, Y_c = np.meshgrid(x_c, y_c)

# Mantle Constants
alpha_m = 0.1  # velocity diffusion speed (viscosity)
beta_m = 0.01  # temperature diffusion speed
kappa_m = 0.0001  # temperature buoyancy constant
g_m = - 10.0
rho_m = 1.0  # fluid density

# Core Constants
alpha_c = 0.1  # velocity diffusion speed (viscosity)
beta_c = 0.0001  # temperature diffusion speed
kappa_c = 0.00001  # temperature buoyancy constant
g_c = - 10.0
rho_c = 1.0  # fluid density

# Pressure
tolerance = 0.00001
max_iteration = 100
mu = 0.1

# Time Step Control
buffer = 0.9

# -----------------------------------------
# ICs
# -----------------------------------------

# Mantle # -----------------------------------------

# Scalars
P_m = np.ones([Ny_m + 2, Nx_m + 2]) * 0.0  # numpy is row-based, so Ny before Nx
u_m = np.zeros([Ny_m + 2, Nx_m + 2])
v_m = np.zeros([Ny_m + 2, Nx_m + 2])
T_m = np.ones([Ny_m + 2, Nx_m + 2]) * 0.0

P_star_m = np.ones_like(P_m)
u_star_m = np.zeros_like(u_m)  # for the intermediate step with pressure to conserve mass
v_star_m = np.zeros_like(v_m)  # for the intermediate step with pressure to conserve mass

T_new_m = np.zeros_like(T_m)

# Core # -----------------------------------------

# Scalars
P_c = np.ones([Ny_c + 2, Nx_c + 2]) * 0.0  # numpy is row-based, so Ny before Nx
u_c = np.zeros([Ny_c + 2, Nx_c + 2])
v_c = np.zeros([Ny_c + 2, Nx_c + 2])
T_c = np.ones([Ny_c + 2, Nx_c + 2]) * 0.0

P_star_c = np.ones_like(P_c)
u_star_c = np.zeros_like(u_c)  # for the intermediate step with pressure to conserve mass
v_star_c = np.zeros_like(v_c)  # for the intermediate step with pressure to conserve mass

T_new_c = np.zeros_like(T_c)

# Create a circular hot spot in the center
center_i = int(Nx / 2)
center_j = int(Ny / 10) + 20
radius = int(min(Nx, Ny) / 5)

for i in range(1, Nx + 1):
    for j in range(1, Ny + 1):
        # Calculate the distance from the center
        distance_squared = (i - center_i) ** 2 + (j - center_j) ** 2
        if distance_squared <= radius ** 2:
            T_c[j, i] = 100.0  # Initial hot temperature

T_c[40:, 1:-1] = 100.0

# -----------------------------------------
# The Algorithms
# -----------------------------------------


def animate(frame):
    global u_m, v_m, P_m, T_m, u_c, v_c, P_c, T_c, t

    # Step 0: Time Step # -----------------------------------------

    # Compute maximum velocities for CFL condition
    u_max_m = np.max(np.abs(u_m)) + 1e-5
    v_max_m = np.max(np.abs(v_m)) + 1e-5

    u_max_c = np.max(np.abs(u_c)) + 1e-5
    v_max_c = np.max(np.abs(v_c)) + 1e-5

    # compute term time steps
    dt_velocity_m = min(dx / u_max_m, dy / v_max_m) * buffer
    dt_advection_m = min(dx / alpha_m, dy / alpha_m) * buffer
    dt_diffusion_m = min((dx ** 2) / (beta_m * 2), (dy ** 2) / (beta_m * 2)) * buffer

    dt_velocity_c = min(dx / u_max_c, dy / v_max_c) * buffer
    dt_advection_c = min(dx / alpha_c, dy / alpha_c) * buffer
    dt_diffusion_c = min((dx ** 2) / (beta_c * 2), (dy ** 2) / (beta_c * 2)) * buffer

    # Choose the smallest dt to ensure stability
    dt = min(dt_advection_m, dt_diffusion_m, dt_velocity_m, dt_advection_c, dt_diffusion_c, dt_velocity_c)
    print(f"Dynamic Time Step: {dt}")

    # Step 1: BCs # -----------------------------------------

    # BC Types:
    # U or V = #.# if BC is #.#
    # U or V = 0.0 if BC is no slip
    # U or V = neighbor if BC is slip
    # U or V = opposite wall if BC is periodic

    # wall velocities are (where not explicitly defined) averaged of nearest two values

    # Mantle BCs # -----------------------------------------

    # u velocity at tangential boundaries:
    U_t_m = u_m[1, :]  # slip
    U_b_m = u_m[-2, :]  # slip

    # v velocity at tangential boundaries:
    V_l_m = v_m[:, 1]  # slip
    V_r_m = v_m[:, -2]  # slip

    # u velocities:
    u_m[:, 0] = 0.0  # Left: must be 0.0 to conserve mass
    u_m[:, -1] = 0.0  # Right: must be 0.0 to conserve mass
    u_m[-1, :] = 2 * U_b_m - u_m[-2, :]  # Bottom: u_b
    u_m[0, :] = 2 * U_t_m - u_m[1, :]  # Top: u_t

    # v velocities:
    v_m[-1, :] = 0.0  # Bottom: must be 0.0 to conserve mass
    v_m[0, :] = 0.0  # Top: must be 0.0 to conserve mass
    v_m[:, 0] = 2 * V_l_m - v_m[:, 1]  # Left: v_l
    v_m[:, -1] = 2 * V_r_m - v_m[:, -2]  # Right: v_r

    # Core BCs # -----------------------------------------

    # u velocity at tangential boundaries:
    U_t_c = u_c[1, :]  # slip
    U_b_c = u_c[-2, :]  # slip

    # v velocity at tangential boundaries:
    V_l_c = v_c[:, 1]  # slip
    V_r_c = v_c[:, -2]  # slip

    # u velocities:
    u_c[:, 0] = 0.0  # Left: must be 0.0 to conserve mass
    u_c[:, -1] = 0.0  # Right: must be 0.0 to conserve mass
    u_c[-1, :] = 2 * U_b_c - u_c[-2, :]  # Bottom: u_b
    u_c[0, :] = 2 * U_t_c - u_c[1, :]  # Top: u_t

    # v velocities:
    v_c[-1, :] = 0.0  # Bottom: must be 0.0 to conserve mass
    v_c[0, :] = 0.0  # Top: must be 0.0 to conserve mass
    v_c[:, 0] = 2 * V_l_c - v_c[:, 1]  # Left: v_l
    v_c[:, -1] = 2 * V_r_c - v_c[:, -2]  # Right: v_r

    # Temperature # -----------------------------------------

    # The BCs for temperature allow for heat diffusion between the mantle and core

    # mantle temperature
    T_m[0, :] = T_m[1, :]  # Top: No Flux
    T_m[-1, :] = T_c[1, :]  # Bottom: heat transfer between mantle and core
    T_m[:, 0] = T_m[:, 1]  # Left: No Flux
    T_m[:, -1] = T_m[:, -2]  # Right: No Flux

    # core temperature
    T_c[0, :] = T_m[-2, :]  # Top: heat transfer between mantle and core
    T_c[-1, :] = T_c[-2, :]  # Bottom: No Flux
    T_c[:, 0] = T_c[:, 1]  # Left: No Flux
    T_c[:, -1] = T_c[:, -2]  # Right: No Flux

    # Step 2.1: Intermediate X-Velocities # -----------------------------------------

    # Mantle Intermediate X-Velocities # -----------------------------------------

    # Pre-computations
    u_l_m = 0.5 * (u_m[1:-1, 1:-1] + u_m[1:-1, :-2])  # self + left average
    u_r_m = 0.5 * (u_m[1:-1, 1:-1] + u_m[1:-1, 2:])  # self + right average
    u_t_m = 0.5 * (u_m[1:-1, 1:-1] + u_m[:-2, 1:-1])  # self + top average
    u_b_m = 0.5 * (u_m[1:-1, 1:-1] + u_m[2:, 1:-1])  # self + right average

    v_b_m = 0.5 * (v_m[2:, 1:-1] + v_m[2:, :-2])  # see staggered grid, average of corner vs
    v_t_m = 0.5 * (v_m[1:-1, 1:-1] + v_m[1:-1, :-2])  # see staggered grid, average of corner vs

    u_p_m = u_m[1:-1, 1:-1]

    # Advection/Convection terms for u
    u_conv_m = - ((u_r_m ** 2 - u_l_m ** 2) * dy + (u_t_m * v_t_m - u_b_m * v_b_m) * dx)

    # Diffusion terms for u
    u_diff_m = 2 * alpha_m * (
            (u_t_m + u_b_m - 2 * u_p_m) * (dx / dy) + (u_l_m + u_r_m - 2 * u_p_m) * (dy / dx)
    )

    # Update intermediate u_star
    u_star_m[1:-1, 1:-1] = u_m[1:-1, 1:-1] + (dt / cell_area) * (u_conv_m + u_diff_m)

    # Core Intermediate X-Velocities # -----------------------------------------

    # Pre-computations
    u_l_c = 0.5 * (u_c[1:-1, 1:-1] + u_c[1:-1, :-2])  # self + left average
    u_r_c = 0.5 * (u_c[1:-1, 1:-1] + u_c[1:-1, 2:])  # self + right average
    u_t_c = 0.5 * (u_c[1:-1, 1:-1] + u_c[:-2, 1:-1])  # self + top average
    u_b_c = 0.5 * (u_c[1:-1, 1:-1] + u_c[2:, 1:-1])  # self + right average

    v_b_c = 0.5 * (v_c[2:, 1:-1] + v_c[2:, :-2])  # see staggered grid, average of corner vs
    v_t_c = 0.5 * (v_c[1:-1, 1:-1] + v_c[1:-1, :-2])  # see staggered grid, average of corner vs

    u_p_c = u_c[1:-1, 1:-1]

    # Advection/Convection terms for u
    u_conv_c = - ((u_r_c ** 2 - u_l_c ** 2) * dy + (u_t_c * v_t_c - u_b_c * v_b_c) * dx)

    # Diffusion terms for u
    u_diff_c = 2 * alpha_c * (
            (u_t_c + u_b_c - 2 * u_p_c) * (dx / dy) + (u_l_c + u_r_c - 2 * u_p_c) * (dy / dx)
    )

    # Update intermediate u_star
    u_star_c[1:-1, 1:-1] = u_c[1:-1, 1:-1] + (dt / cell_area) * (u_conv_c + u_diff_c)

    # Step 2.2: Intermediate Y-Velocities # -----------------------------------------

    # Mantle Intermediate Y-Velocities # -----------------------------------------

    # Pre-computations
    v_l_m = 0.5 * (v_m[1:-1, 1:-1] + v_m[1:-1, :-2])  # self + left average
    v_r_m = 0.5 * (v_m[1:-1, 1:-1] + v_m[1:-1, 2:])  # self + right average
    v_b_m = 0.5 * (v_m[1:-1, 1:-1] + v_m[2:, 1:-1])  # self + top average
    v_t_m = 0.5 * (v_m[1:-1, 1:-1] + v_m[:-2, 1:-1])  # self + right average

    u_l_m = 0.5 * (u_m[:-2, 1:-1] + u_m[1:-1, 1:-1])  # see staggered grid, average of corner vs
    u_r_m = 0.5 * (u_m[:-2, 2:] + u_m[1:-1, 2:])  # see staggered grid, average of corner vs

    v_p_m = v_m[1:-1, 1:-1]

    # Advection/Convection terms for v
    v_conv_m = - ((u_r_m * v_r_m - u_l_m * v_l_m) * dy + (v_t_m ** 2 - v_b_m ** 2) * dx)

    # Diffusion terms for v
    v_diff_m = 2 * alpha_m * (
            (v_t_m + v_b_m - 2 * v_p_m) * (dx / dy) + (v_l_m + v_r_m - 2 * v_p_m) * (dy / dx)
    )

    # Buoyancy term for v
    buoyancy_m = g_m * kappa_m * (T_m[1:-1, 1:-1] - 0.0)

    # Update intermediate v_star
    v_star_m[1:-1, 1:-1] = v_m[1:-1, 1:-1] + (dt / cell_area) * (v_conv_m + v_diff_m + buoyancy_m)

    # Core Intermediate Y-Velocities # -----------------------------------------

    # Pre-computations
    v_l_c = 0.5 * (v_c[1:-1, 1:-1] + v_c[1:-1, :-2])  # self + left average
    v_r_c = 0.5 * (v_c[1:-1, 1:-1] + v_c[1:-1, 2:])  # self + right average
    v_b_c = 0.5 * (v_c[1:-1, 1:-1] + v_c[2:, 1:-1])  # self + top average
    v_t_c = 0.5 * (v_c[1:-1, 1:-1] + v_c[:-2, 1:-1])  # self + right average

    u_l_c = 0.5 * (u_c[:-2, 1:-1] + u_c[1:-1, 1:-1])  # see staggered grid, average of corner vs
    u_r_c = 0.5 * (u_c[:-2, 2:] + u_c[1:-1, 2:])  # see staggered grid, average of corner vs

    v_p_c = v_c[1:-1, 1:-1]

    # Advection/Convection terms for v
    v_conv_c = - ((u_r_c * v_r_c - u_l_c * v_l_c) * dy + (v_t_c ** 2 - v_b_c ** 2) * dx)

    # Diffusion terms for v
    v_diff_c = 2 * alpha_c * (
            (v_t_c + v_b_c - 2 * v_p_c) * (dx / dy) + (v_l_c + v_r_c - 2 * v_p_c) * (dy / dx)
    )

    # Buoyancy term for v
    buoyancy_c = g_c * kappa_c * (T_c[1:-1, 1:-1] - 0.0)

    # Update intermediate v_star
    v_star_c[1:-1, 1:-1] = v_c[1:-1, 1:-1] + (dt / cell_area) * (v_conv_c + v_diff_c + buoyancy_c)

    # Step 3: Pressure Correction # -----------------------------------------

    # We want to solve for P, so that we can then use it for calculating its gradient

    # Mantle Pressure # -----------------------------------------

    # Pre-computations
    u_star_r_m = u_star_m[1:-1, 2:]
    u_star_p_m = u_star_m[1:-1, 1:-1]

    # Pre-computations
    v_star_b_m = v_star_m[2:, 1:-1]
    v_star_p_m = v_star_m[1:-1, 1:-1]

    # Compute divergence of velocity vector
    # because of the staggered grid, the velocity components are on either side of the cell center, allowing for
    # the divergence of velocity to be calculated at exactly the cell center, rather than forward or central
    # differences, for example
    div_velocity_star_m = np.zeros([Ny_m + 2, Nx_m + 2])
    div_velocity_star_m[1:-1, 1:-1] = (u_star_r_m - u_star_p_m) / dx + (v_star_b_m - v_star_p_m) / dy

    # Recall the right hand side (rhs) of the pressure poisson equation:
    poisson_rhs_m = rho_m * div_velocity_star_m / dt

    # Pre-computations
    P_t_m = P_m[:-2, 1:-1]
    P_b_m = P_m[2:, 1:-1]
    P_l_m = P_m[1:-1, :-2]
    P_r_m = P_m[1:-1, 2:]

    # The left hand side of the pressure equation can be expanded using central difference for the second derivative
    # poisson_lhs = (P_t + P_b - 2 * P_p) / (dy ** 2) + (P_l + P_r - 2 * P_p) / (dx ** 2)
    # note that the poisson_lhs is not used as a variable, rather, it is needed to solve for P_p
    # we solve the poisson equation for P, which is denoted as P_p

    # Iteration
    iteration_m = 0
    error_m = 1e10
    while error_m > tolerance and iteration_m < max_iteration:
        P_copy_m = P_m.copy()

        # BCs
        P_m[1: -1, 0] = P_m[1: -1, 1]
        P_m[1: -1, -1] = P_m[1: -1, -2]
        P_m[0, 1: -1] = P_m[1, 1: -1]
        P_m[-1, 1: -1] = P_m[-2, 1: -1]

        # Solving for P
        P_star_m[1:-1, 1:-1] = (
                (dx ** 2 * (P_t_m + P_b_m) + dy ** 2 * (P_l_m + P_r_m) - poisson_rhs_m[1:-1, 1:-1] * dx ** 2 * dy ** 2)
                / (2 * (dx ** 2 + dy ** 2))
        )

        # This is just saying P = P_star, but mu is there to control the 'speed of pressure flow'
        P_m[1:-1, 1:-1] = P_copy_m[1:-1, 1:-1] + mu * (P_star_m[1:-1, 1:-1] - P_copy_m[1:-1, 1:-1])

        error_m = np.linalg.norm(P_m.ravel() - P_copy_m.ravel(), 2)
        iteration_m += 1

    # print out correction note
    print(f'The mantle pressure was corrected with an error of: {error_m}')

    # correct the velocities with the known pressure gradient
    # include only the velocities that are between interior pressure cells, so not those along the 'red' boundary on
    # the staggered grid, hence the funny array shapes
    u_m[1:-1, 2:-1] = u_star_m[1:-1, 2:-1] - (dt / rho_m) * (P_m[1:-1, 2:-1] - P_m[1:-1, 1:-2]) / dx
    v_m[2:-1, 1:-1] = v_star_m[2:-1, 1:-1] - (dt / rho_m) * (P_m[2:-1, 1:-1] - P_m[1:-2, 1:-1]) / dy
    # cell area (the integration over cell volume) cancels out for both transient and pressure gradient terms, so a
    # standard FDM is appropriate here due to the structured nature of the grid

    # Core Pressure # -----------------------------------------

    # Pre-computations
    u_star_r_c = u_star_c[1:-1, 2:]
    u_star_p_c = u_star_c[1:-1, 1:-1]

    # Pre-computations
    v_star_b_c = v_star_c[2:, 1:-1]
    v_star_p_c = v_star_c[1:-1, 1:-1]

    # Compute divergence of velocity vector
    # because of the staggered grid, the velocity components are on either side of the cell center, allowing for
    # the divergence of velocity to be calculated at exactly the cell center, rather than forward or central
    # differences, for example
    div_velocity_star_c = np.zeros([Ny_c + 2, Nx_c + 2])
    div_velocity_star_c[1:-1, 1:-1] = (u_star_r_c - u_star_p_c) / dx + (v_star_b_c - v_star_p_c) / dy

    # Recall the right hand side (rhs) of the pressure poisson equation:
    poisson_rhs_c = rho_c * div_velocity_star_c / dt

    # Pre-computations
    P_t_c = P_c[:-2, 1:-1]
    P_b_c = P_c[2:, 1:-1]
    P_l_c = P_c[1:-1, :-2]
    P_r_c = P_c[1:-1, 2:]

    # The left hand side of the pressure equation can be expanded using central difference for the second derivative
    # poisson_lhs = (P_t + P_b - 2 * P_p) / (dy ** 2) + (P_l + P_r - 2 * P_p) / (dx ** 2)
    # note that the poisson_lhs is not used as a variable, rather, it is needed to solve for P_p
    # we solve the poisson equation for P, which is denoted as P_p

    # Iteration
    iteration_c = 0
    error_c = 1e10
    while error_c > tolerance and iteration_c < max_iteration:
        P_copy_c = P_c.copy()

        # BCs
        P_c[1: -1, 0] = P_c[1: -1, 1]
        P_c[1: -1, -1] = P_c[1: -1, -2]
        P_c[0, 1: -1] = P_c[1, 1: -1]
        P_c[-1, 1: -1] = P_c[-2, 1: -1]

        # Solving for P
        P_star_c[1:-1, 1:-1] = (
                (dx ** 2 * (P_t_c + P_b_c) + dy ** 2 * (P_l_c + P_r_c) - poisson_rhs_c[1:-1, 1:-1] * dx ** 2 * dy ** 2)
                / (2 * (dx ** 2 + dy ** 2))
        )

        # This is just saying P = P_star, but mu is there to control the 'speed of pressure flow'
        P_c[1:-1, 1:-1] = P_copy_c[1:-1, 1:-1] + mu * (P_star_c[1:-1, 1:-1] - P_copy_c[1:-1, 1:-1])

        error_c = np.linalg.norm(P_c.ravel() - P_copy_c.ravel(), 2)
        iteration_c += 1

    # print out correction note
    print(f'The core pressure was corrected with an error of: {error_c}')

    # correct the velocities with the known pressure gradient
    # include only the velocities that are between interior pressure cells, so not those along the 'red' boundary on
    # the staggered grid, hence the funny array shapes
    u_c[1:-1, 2:-1] = u_star_c[1:-1, 2:-1] - (dt / rho_c) * (P_c[1:-1, 2:-1] - P_c[1:-1, 1:-2]) / dx
    v_c[2:-1, 1:-1] = v_star_c[2:-1, 1:-1] - (dt / rho_c) * (P_c[2:-1, 1:-1] - P_c[1:-2, 1:-1]) / dy
    # cell area (the integration over cell volume) cancels out for both transient and pressure gradient terms, so a
    # standard FDM is appropriate here due to the structured nature of the grid

    # Step 4: Temperature # -----------------------------------------

    # Mantle Temperature # -----------------------------------------

    # bring velocities at faces to the cell centers for calculations
    u_c_m = 0.5 * (u_m[1:-1, 1:-1] + u_m[1:-1, 2:])  # self + left average
    v_c_m = 0.5 * (v_m[1:-1, 1:-1] + v_m[2:, 1:-1])  # self + bottom average

    # Upwind scheme is required for advection/convection terms
    # For u (x-direction)
    T_adv_x_m = np.where(
        u_c_m > 0,
        u_c_m * (T_m[1:-1, 1:-1] - T_m[1:-1, :-2]) / dx,  # Positive velocity
        u_c_m * (T_m[1:-1, 2:] - T_m[1:-1, 1:-1]) / dx  # Negative velocity
    )

    # For v (y-direction)
    T_conv_y_m = np.where(
        v_c_m > 0,
        v_c_m * (T_m[1:-1, 1:-1] - T_m[:-2, 1:-1]) / dy,  # Positive velocity
        v_c_m * (T_m[2:, 1:-1] - T_m[1:-1, 1:-1]) / dy  # Negative velocity
    )

    T_conv_m = - (T_adv_x_m + T_conv_y_m)

    # Diffusion
    T_diff_m = beta_m * (
            (T_m[1:-1, 2:] + T_m[1:-1, :-2] - 2 * T_m[1:-1, 1:-1]) / dx ** 2 +
            (T_m[2:, 1:-1] + T_m[:-2, 1:-1] - 2 * T_m[1:-1, 1:-1]) / dy ** 2
    )

    # Update temperature
    T_new_m[1:-1, 1:-1] = T_m[1:-1, 1:-1] + dt * (T_conv_m + T_diff_m)

    T_m[1:-1, 1:-1] = T_new_m[1:-1, 1:-1]

    # Core Temperature # -----------------------------------------

    # bring velocities at faces to the cell centers for calculations
    u_c_c = 0.5 * (u_c[1:-1, 1:-1] + u_c[1:-1, 2:])  # self + left average
    v_c_c = 0.5 * (v_c[1:-1, 1:-1] + v_c[2:, 1:-1])  # self + bottom average

    # Upwind scheme is required for advection/convection terms
    # For u (x-direction)
    T_adv_x_c = np.where(
        u_c_c > 0,
        u_c_c * (T_c[1:-1, 1:-1] - T_c[1:-1, :-2]) / dx,  # Positive velocity
        u_c_c * (T_c[1:-1, 2:] - T_c[1:-1, 1:-1]) / dx  # Negative velocity
    )

    # For v (y-direction)
    T_conv_y_c = np.where(
        v_c_c > 0,
        v_c_c * (T_c[1:-1, 1:-1] - T_c[:-2, 1:-1]) / dy,  # Positive velocity
        v_c_c * (T_c[2:, 1:-1] - T_c[1:-1, 1:-1]) / dy  # Negative velocity
    )

    T_conv_c = - (T_adv_x_c + T_conv_y_c)

    # Diffusion
    T_diff_c = beta_c * (
            (T_c[1:-1, 2:] + T_c[1:-1, :-2] - 2 * T_c[1:-1, 1:-1]) / dx ** 2 +
            (T_c[2:, 1:-1] + T_c[:-2, 1:-1] - 2 * T_c[1:-1, 1:-1]) / dy ** 2
    )

    # Update temperature
    T_new_c[1:-1, 1:-1] = T_c[1:-1, 1:-1] + dt * (T_conv_c + T_diff_c)

    T_c[1:-1, 1:-1] = T_new_c[1:-1, 1:-1]

    # Visuals # -----------------------------------------

    t += dt

    # Update mantle visualization
    im_m.set_data(T_m[1:-1, 1:-1])

    # Update core visualization
    im_c.set_data(T_c[1:-1, 1:-1])

    # Update quiver plots
    u_center_m = 0.5 * (u_m[1:-1, 1:-1] + u_m[1:-1, 2:])
    v_center_m = 0.5 * (v_m[1:-1, 1:-1] + v_m[2:, 1:-1])

    u_center_c = 0.5 * (u_c[1:-1, 1:-1] + u_c[1:-1, 2:])
    v_center_c = 0.5 * (v_c[1:-1, 1:-1] + v_c[2:, 1:-1])

    quiver_m.set_UVC(u_center_m[::skip, ::skip], -v_center_m[::skip, ::skip])
    quiver_c.set_UVC(u_center_c[::skip, ::skip], -v_center_c[::skip, ::skip])

    return [im_m, im_c, quiver_m, quiver_c]


# -----------------------------------------
# Visuals
# -----------------------------------------

# time set-up
t = 0
n_steps = 500

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figure size for better aspect ratio

# Plot mantle temperature field
im_m = ax.imshow(
    T_m[1:-1, 1:-1],
    extent=[0, Lx_m, Ly_m, Ly],  # Adjust extent to place mantle above core
    origin='upper',
    cmap=cm.jet,
    vmin=0,
    vmax=100,
    alpha=0.8  # Add transparency to see both layers if needed
)

# Plot core temperature field
im_c = ax.imshow(
    T_c[1:-1, 1:-1],
    extent=[0, Lx_c, 0, Ly_m],  # Adjust extent to place core below mantle
    origin='upper',
    cmap=cm.jet,
    vmin=0,
    vmax=100,
    alpha=0.8  # Add transparency to see both layers if needed
)

# Add a color bar
fig.colorbar(im_m, ax=ax, label='Temperature')

# Add labels and title
ax.set_title('Mantle and Core Temperature Field')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')

# Prepare quiver plots for both layers
skip = 2  # Skip every 4 data points
Xq_m = X_m[1:-1, 1:-1][::skip, ::skip]
Yq_m = Y_m[1:-1, 1:-1][::skip, ::skip]

Xq_c = X_c[1:-1, 1:-1][::skip, ::skip]
Yq_c = Y_c[1:-1, 1:-1][::skip, ::skip]

# Initialize quiver plots
quiver_m = ax.quiver(
    Xq_m, Ly - Yq_m, np.zeros_like(Xq_m), np.zeros_like(Yq_m),
    color='cyan', scale=50, label='Mantle Velocity'
)
quiver_c = ax.quiver(
    Xq_c, Ly_c - Yq_c, np.zeros_like(Xq_c), np.zeros_like(Yq_c),
    color='yellow', scale=50, label='Core Velocity'
)

# Create the animation
anim = FuncAnimation(fig, animate, frames=n_steps, interval=2, blit=False)

# Show the plot
plt.show()
