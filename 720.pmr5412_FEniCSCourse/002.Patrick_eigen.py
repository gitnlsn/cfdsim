from dolfin import *
from mshr   import *

part1 = Rectangle(
   Point(0,0),
   Point(2,2)     )
part2 = Rectangle(
   Point(1,0),
   Point(2,1)     )
domain = part1 -part2
mesh = generate_mesh(domain, 100)

elem = FiniteElement('Lagrange', 'triangle', 1)
V = FunctionSpace(mesh, elem)

u = TrialFunction(V)
v = TestFunction(V)
dummy = inner(Constant(1), v)*dx
bc = DirichletBC(V, 0, "on_boundary")
a = inner(grad(u), grad(v))*dx
asm = SystemAssembler(a, dummy, bc)
A = PETScMatrix(); asm.assemble(A)

b = inner(u, v)*dx
asm = SystemAssembler(b, dummy)
B = PETScMatrix(); asm.assemble(B)
bc.zero(B)

solver = SLEPcEigenSolver(A, B)
solver.parameters["solver"] = "krylov-schur"
solver.parameters["spectrum"] = "target magnitude"
solver.parameters["problem_type"] = "gen_hermitian"
solver.parameters["spectral_transform"] = "shift-and-invert"
solver.parameters["spectral_shift"] = 10.
solver.solve(1)

eigenmodes = File("eigenmodes.pvd")
eigenfunction = Function(V, name="Eigenfunction")
for i in range(solver.get_number_converged()):
   (r, _, rv, _) = solver.get_eigenpair(i)
   eigenfunction.vector().zero()
   eigenfunction.vector().axpy(1, rv)
   eigenmodes << eigenfunction
