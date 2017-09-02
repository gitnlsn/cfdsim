'''

NELSON KENZO TAMASHIRO

19.07.2017

'''

# ------ LIBRARIES ------ #
from fenics          import *
from mshr            import *

# ------ SIMULATION PARAMETERS ------ #
filename = 'results_simple_transient_simulation'
mesh_0   = 0.0
mesh_D   = 0.020
mesh_L   = 0.060
mesh_Cx     = 0.020
mesh_Cy     = 0.010
mesh_Radius = 0.002
mesh_res = 100

cons_dt  = 0.000002
cons_rho = 1E+3
cons_mu  = 1E-3
cons_vin = 1E1

simple_tol  = 1E-7
simple_Nmax = 5000

# ------ MESH ------ #
part1 = Rectangle(
   Point(mesh_0, mesh_0),
   Point(mesh_L, mesh_D)   )
part2 = Circle(
   Point(mesh_Cx, mesh_Cy),
   mesh_Radius)
channel = part1 - part2
mesh = generate_mesh(channel, mesh_res)

# ------ BOUNDARIES ------ #
#walls  = '(x[1]=='+str(1.0*mesh_D)+') || (x[1]=='+str(0.0*mesh_D)+')'
inlet  = '(x[0]=='+str(0.0*mesh_L)+')'
outlet = '(x[0]=='+str(1.0*mesh_L)+')'
updown = '((x[1]=='+str(1.0*mesh_D)+') || (x[1]=='+str(0.0*mesh_D)+'))'
walls  = 'on_boundary && !'+inlet +' && !'+outlet+' && !'+updown

# ------ VARIATIONAL FORMULATION ------ #
FE_P  = FiniteElement('P', 'triangle', 1)
FE_V  = VectorElement('P', 'triangle', 2)
U_prs = FunctionSpace(mesh, FE_P)
U_vel = FunctionSpace(mesh, FE_V)

u = project(Constant((cons_vin,0)), U_vel)

u1    = Function(U_vel)
u3    = Function(U_vel)

p2    = Function(U_prs)
p4    = Function(U_prs)

v     = TestFunction(U_vel)
q     = TestFunction(U_prs)

n     = FacetNormal(mesh)

DT    = Constant(cons_dt)
RHO   = Constant(cons_rho)
MU    = Constant(cons_mu)

def sigma (uu, pp):
   return MU*(grad(uu)+grad(uu).T) -pp*Identity(len(uu))

def tau (uu):
   return MU*(grad(uu)+grad(uu).T)

F1    = RHO *inner( u1-u,v )/DT              *dx \
      + RHO *inner( dot(u1,grad(u1).T), v )  *dx \
      + MU  *inner( grad(u1), grad(v) )      *dx \
      - inner( dot(tau(u1),n),v )            *ds

F2    =      inner(grad(p2), grad(q))  *dx \
      + RHO *div(u1)/DT *q             *dx \

F3    = RHO *inner( u3-u,v )/DT              *dx \
      + RHO *inner( dot(u3,grad(u3).T), v )  *dx \
      +      inner( sigma(u3,p2), grad(v) )  *dx \
      - inner( dot( sigma(u3,p2),n),v )      *ds

F4    =      inner(grad(p4), grad(q))  *dx \
      + RHO *div(u3)/DT *q             *dx

#F5: u.assign(project(u3 -DT*grad(p4)/RHO), U_vel)

# ------ BOUNDARY CONDITIONS ------ #
#in_profile = (str(cons_vin)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', '0')
BC_u = [ 
         DirichletBC(U_vel, Constant((cons_vin,0)), inlet),
         DirichletBC(U_vel, Constant((0,0)), walls),
                                                                     ]
BC_p = [ DirichletBC(U_prs, Constant(0E0), outlet ),   ]

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
dF1 = derivative(F1, u1)
dF2 = derivative(F2, p2)
dF3 = derivative(F3, u3)
dF4 = derivative(F4, p4)
nlProblem1 = NonlinearVariationalProblem(F1, u1, BC_u, dF1)
nlProblem2 = NonlinearVariationalProblem(F2, p2, BC_p, dF2)
nlProblem3 = NonlinearVariationalProblem(F3, u3, BC_u, dF3)
nlProblem4 = NonlinearVariationalProblem(F4, p4, BC_p, dF4)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
nlSolver2  = NonlinearVariationalSolver(nlProblem2)
nlSolver3  = NonlinearVariationalSolver(nlProblem3)
nlSolver4  = NonlinearVariationalSolver(nlProblem4)
prm1 = nlSolver1.parameters["newton_solver"]
prm2 = nlSolver2.parameters["newton_solver"]
prm3 = nlSolver3.parameters["newton_solver"]
prm4 = nlSolver4.parameters["newton_solver"]
for prm in [prm1, prm2, prm3, prm4]:
   prm["maximum_iterations"      ] = 10
   prm["absolute_tolerance"      ] = 5E-11
   prm["relative_tolerance"      ] = 5E-11

#prm["convergence_criterion"   ] = "residual"
#prm["linear_solver"           ] = "mumps"
#prm["method"                  ] = "full"
#prm["preconditioner"          ] = "none"
#prm["error_on_nonconvergence" ] = True
#prm["relaxation_parameter"    ] = 1.0
#prm["report"                  ] = True
#set_log_level(PROGRESS)

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
count_iteration   = 0
flag_converged    = False
while( count_iteration < simple_Nmax ):
   count_iteration = count_iteration +1
   nlSolver1.solve()
   nlSolver2.solve()
   nlSolver3.solve()
   nlSolver4.solve()
   u_next = project(u3 -DT*grad(p4)/RHO, U_vel)
   residual = assemble(inner(u-u_next, u-u_next)*dx)/assemble(inner(u,u)*dx)
   print ('Residual: {}'.format(residual) )
   print ('Iteration: {}'.format(count_iteration) )
   u.assign(u_next)
   save_flow(u,p4)

#plot(u , title='velocity')
#plot(p4, title='pressure')
#interactive()
