""" Solves the optimal control problem for the heat equation """
from fenics import *
from mshr import *
from dolfin_adjoint import *

# Setup
n = 30
domain = Rectangle(
   Point(-1, -1),
   Point(1, 1))
mesh = generate_mesh(domain, n)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="State")
m = Function(V, name="Control")
v = TestFunction(V)

# Run the forward model once to create the simulation record
F = (inner(grad(u), grad(v)) - m*v)*dx
bc = DirichletBC(V, 0.0, 'on_boundary')
solve(F == 0, u, bc)

# The functional of interest is the normed difference between desired
# and simulated temperature profile
u_desired = interpolate(Expression('exp(-3/(1-x[0]*x[0])-1/(1-x[1]*x[1]))', degree=2), V)
J = Functional((inner(u-u_desired, u-u_desired))*dx*dt[FINISH_TIME])

# Run the optimisation
reduced_functional = ReducedFunctional(J, Control(m, value=m))
# Make sure you have scipy >= 0.11 installed
m_opt = minimize(reduced_functional,
   method = "L-BFGS-B",
   tol=2E-11,
   bounds = (-1, 1),
   options = {"disp": True})

solve(F == 0, u, bc)
plot(u, title='Temperature')
plot(u_desired, title='Desired Temperature')
plot(m, title='controlFunction')
interactive()
