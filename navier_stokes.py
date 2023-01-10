import numpy as np
import matplotlib.pyplot as plt

# Set the fluid properties and flow conditions
rho = 1.0  # density
mu = 1.0   # viscosity
u = 1.0    # velocity
L = 1.0    # length

# Set the number of grid points and the grid spacing
nx = 50
dx = L / nx

# Set the timestep size and the number of timesteps
dt = 0.01
nt = 100

# Set the initial velocity and pressure
u_initial = np.zeros(nx)
p_initial = np.zeros(nx)

# Set the boundary conditions
u_left = 1.0
u_right = 10.0
p_left = 1.0
p_right = 10.0

# Create arrays to store the solution at each timestep
u_solution = np.zeros((nt, nx))
p_solution = np.zeros((nt, nx))

# Set the initial conditions
u_solution[0] = u_initial
p_solution[0] = p_initial

# Solve the Navier-Stokes equation using the finite difference method
for t in range(nt - 1):
    for i in range(1, nx - 1):
        u_solution[t + 1, i] = (
            u_solution[t, i]
            - dt / (2 * rho * dx) * (p_solution[t, i + 1] - p_solution[t, i - 1])
            + mu * dt / dx**2 * (u_solution[t, i + 1] - 2 * u_solution[t, i] + u_solution[t, i - 1])
        )
    for i in range(1, nx - 1):
        p_solution[t + 1, i] = (
            p_solution[t, i]
            - rho * dt / dx * (u_solution[t + 1, i + 1] - u_solution[t + 1, i - 1])
        )
        
    # Apply the boundary conditions
    u_solution[t + 1, 0] = u_left
    u_solution[t + 1, -1] = u_right
    p_solution[t + 1, 0] = p_left
    p_solution[t + 1, -1] = p_right


#Solution at the i-th grid point and the t-th timestep, 
velocity = u_solution[t, i] 
pressure = p_solution[t, i]
print(velocity)
print(pressure)
# Plot the velocity solution
plt.plot(u_solution[t])
plt.xlabel("Grid point")
plt.ylabel("Velocity")
plt.show()

# Plot the pressure solution
plt.plot(p_solution[t])
plt.xlabel("Grid point")
plt.ylabel("Pressure")
plt.show()