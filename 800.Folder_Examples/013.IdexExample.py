c = 10

from fenics import *

mesh = UnitSquareMesh(10,10)
U = FunctionSpace(mesh, 'P', 1)

u = TrialFunction(U)
v = TestFunction(U)
s = Function(U)

F = Dx(u,0)*Dx(v,0)*dx +c*Dx(u,1)*Dx(v,1)*dx
bil,lin = lhs(F), rhs(F)


left  = 'near(x[0],0)'
right = 'near(x[0],1)'
up    = 'near(x[1],1)'
down  = 'near(x[1],0)'
bc1 = DirichletBC(U, Constant(1), up)
bc2 = DirichletBC(U, Constant(2), right)
BC = [bc1, bc2]

solve(bil==lin, s, BC)
plot(s, title='Solution')
interactive()

