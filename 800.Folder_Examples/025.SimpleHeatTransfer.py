res = 30
c_k = 1E-3
q0 = 1
from fenics import *

mesh = UnitSquareMesh(res,res)
FE_T = FiniteElement('P', 'triangle', 5)
U = FunctionSpace(mesh, FE_T)

T = TrialFunction(U)
t = TestFunction(U)
k = Constant(c_k)
q0 = Constant(q0)
q = Expression('q0*exp(-((x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5))*100)', q0=q0, degree=2)

F = k*dot(grad(T),grad(t))*dx \
   -q*t*dx

left  = 'near(x[0],0)'
right = 'near(x[0],1)'
down  = 'near(x[1],0)'
up    = 'near(x[1],1)'
canto = 'on_boundary && near(x[0],0.5) && near(x[1],0.5)'

BC = [
   DirichletBC(U, Constant(0), 'on_boundary'),
   ]

ans = Function(U)
solve(lhs(F)==rhs(F), ans, BC)
#solve(F==0, ans, BC)

plot(ans, title='Temperature')
plot(interpolate(q,U), title='Heat Source')
#plot(-k*grad(ans), title='Heat Rate')
plot(-k*div(grad(ans))-q, title='Energy Conservation')
interactive()
