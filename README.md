# NavierStokes1DSolver
Python Navier-Stokes 1d solver
The Navier-Stokes equation is a set of partial differential equations that describes the motion of fluids and the forces acting on them. It is used to model a wide variety of fluid flow phenomena, including the flow of liquids and gases. Here is a simple Python script that solves the Navier-Stokes equation for a one-dimensional flow using the finite difference method. 
This script solves the Navier-Stokes equation for a one-dimensional flow in a domain of length L, using a finite difference method with nx grid points and a timestep size of dt. The density, viscosity, and velocity of the fluid are specified by the variables rho, mu, and u, respectively. The boundary conditions for the velocity and pressure are set by the variables u_left, u_right, p_left, and p_right.
The solutions for the velocity and pressure at each timestep are stored in the two-dimensional arrays u_solution and p_solution, respectively. These arrays have dimensions nt x nx, where nt is the number of timesteps and nx is the number of grid points.
This will create two plots, one showing the velocity solution at the t-th timestep, and one showing the pressure solution at the same timestep. You can adjust the value of t to visualize the solution at different times.
