# ------ SIMULATION PARAMETERS ------ #
res = 10
t_beg = 0.
t_end = 0.1
N_steps = 100
dtt = dt = (t_end -t_beg)/N_steps

# ------ PHYSICAL PARAMETERS ------ #
rho = 7850
kc = 79.5
c = 4.5
h = 100
Tif = 300
T0 = 400

# ------ PROBLEM DEFINITION AND SOLVE ------ #
from fenics import *

mesh = UnitSquareMesh(res,res)
FE_T = FiniteElement('P', 'triangle', 1)
U = FunctionSpace(mesh, FE_T)

T2 = Function(U)
T1 = interpolate(Expression('T0', T0=T0, degree=2), U)
tt = TestFunction(U)

rho = Constant(rho)
c   = Constant(c)
kc  = Constant(kc)
dt  = Constant(dt)
Tif = Constant(Tif)
h   = Constant(h)

F = rho*c*(T2-T1)*tt*dx \
   -div(kc*grad(T1))*dt*tt*dx \
   -h*(Tif-T1)*dt*tt*ds

plot(T1, title='Initial Temperature')
t = t_beg
while t<t_end:
   print('PROGRESS %: {:.2f}'.format(100*t/t_end))
   BC = DirichletBC(U, Constant(T0), 'near(x[0],0)')
   solve(F==0, T2, BC)
   T1.assign(T2)
   t += dtt
plot(T2, title='Final Temperature')
interactive()