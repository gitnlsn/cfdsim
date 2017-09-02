'''

NELSON KENZO TAMASHIRO

19.07.2017

'''

# ------ LIBRARIES ------ #
from fenics          import *
from mshr            import *
from dolfin_adjoint  import *

# ------ SIMULATION PARAMETERS ------ #
filename = 'results_simple_transient_simulation'
mesh_0   = 0.0
mesh_D   = 0.020
mesh_L   = 0.060
mesh_res = 35

cons_dt  = 0.00001
cons_rho = 1E+3
cons_mu  = 1E-3

simple_tol  = 1E-7
simple_Nmax = 1000

# ------ MESH ------ #
part1 = Rectangle(
   Point(mesh_0, mesh_0),
   Point(mesh_L, mesh_D)   )
channel = part1
mesh = generate_mesh(channel, mesh_res)

# ------ BOUNDARIES ------ #
walls  = '(x[1]=='+str(1.0*mesh_D)+') || (x[1]=='+str(0.0*mesh_D)+')'
inlet  = '(x[0]=='+str(0.0*mesh_L)+')'
outlet = '(x[0]=='+str(1.0*mesh_L)+')'

# ------ VARIATIONAL FORMULATION ------ #
FE_P  = FiniteElement('P', 'triangle', 1)
FE_V  = VectorElement('P', 'triangle', 2)
U_prs = FunctionSpace(mesh, FE_P)
U_vel = FunctionSpace(mesh, FE_V)

u     = Function(U_vel)

u1    = Function(U_vel)
u3    = Function(U_vel)

p2    = Function(U_prs)
p4    = Function(U_prs)

v     = TestFunction(U_vel)
q     = TestFunction(U_prs)

DT    = Constant(cons_dt)
RHO   = Constant(cons_rho)
MU    = Constant(cons_mu)

F1    = RHO *inner( u1-u,v )/DT              *dx \
      + RHO *inner( dot(u1,grad(u1).T), v )  *dx \
      + MU  *inner( grad(u1), grad(v) )      *dx

F2    =      inner(grad(p2), grad(q))  *dx \
      + RHO *div(u1)/DT *q             *dx

F3    = RHO *inner( u3-u,v )/DT              *dx \
      + RHO *inner( dot(u3,grad(u3).T), v )  *dx \
      + MU  *inner( grad(u3), grad(v) )      *dx \
      -      div(v)*p2                       *dx

F4    =      inner(grad(p4), grad(q))  *dx \
      + RHO *div(u3)/DT *q             *dx

#F5: u.assign(project(u3 -DT*grad(p4)/RHO), U_vel)

# ------ BOUNDARY CONDITIONS ------ #
BC_u = [ DirichletBC(U_vel, Constant((0,0)), walls)   ]
BC_p = [ DirichletBC(U_prs, Constant(1E5), inlet ),
         DirichletBC(U_prs, Constant(0E0), outlet ),   ]

# ------ SAVE FILECONFIGURATIONS ------ #
vtk_uu  = File(filename+'/velocity.pvd')
vtk_pp  = File(filename+'/pressure.pvd')

def save_flow(u_tosave, p_tosave):
   ui = project(u_tosave,FunctionSpace(mesh,FE_V))
   pi = project(p_tosave,FunctionSpace(mesh,FE_P))
   ui.rename('velocity','velocity')
   pi.rename('pressure','pressure')
   vtk_uu << ui
   vtk_pp << pi

# ------ TRANSIENT SIMULATION ------ #
u = project(Constant((1E-4,0)), U_vel)
count_iteration   = 0
flag_converged    = False
while( not flag_converged and count_iteration < simple_Nmax ):
   solve(F1==0, u1, BC_u)
   solve(F2==0, p2, BC_p)
   solve(F3==0, u3, BC_u)
   solve(F4==0, p4, BC_p)
   u_next = project(u3 -DT*grad(p4)/RHO, U_vel)
   residual = assemble(inner(u-u_next, u-u_next)*dx)
   print ('Residual: {}'.format(residual) )
   if  residual < simple_tol:
      flag_converged = True
   u.assign(u_next)
   save_flow(u,p4)

plot(u , title='velocity')
plot(p4, title='pressure')
interactive()
