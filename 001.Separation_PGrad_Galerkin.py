'''
DESCRIPTION:

ANALYSIS:

AUTOR: NELSON KENZO TAMASHIRO

DATE: 03.05.2017

'''

# ------ LIBRARIES IMPORT ------ #
from fenics import *
from mshr import *

########################################################
# ------ ------ 01) FOWARD PROBLEM SOLVE ------ ------ #
########################################################

# ------ GEOMETRICAL PARAMETERS ------ #
resolution  = 50

dim_0       = 0.0
dim_L       = 3.0E-1
dim_D       = 1.0E-1

# ------ SIMULATION PARAMETERS CONFIGURATION ------ #
cons_rh1 = 1.0E+3
cons_rh2 = 1.1E+3
cons_mu1 = 1.0E-3
cons_mu2 = 1.1E-3
cons_gg  = 9.8E-4
cons_dl  = 2E-4

v_0 = 1E-3

# ------ MESH CONFIGURATION ------ #
part1 = Rectangle(
   Point( dim_0, dim_0),
   Point( dim_L, dim_D),
   )
domain = part1
mesh = generate_mesh(domain, resolution)

# ------ VARIATIONAL FORMULATION ------ #
FE_u = FiniteElement('P', mesh.ufl_cell(), 2)
FE_p = FiniteElement('P', mesh.ufl_cell(), 1)
FE_a = FiniteElement('P', mesh.ufl_cell(), 1)
elem = MixedElement([FE_u, FE_u, FE_u, FE_u, FE_p, FE_p, FE_a])
U    = FunctionSpace(mesh, elem)

ans = Function(U)
ux1,uy1,ux2,uy2,p1,p2,a1 = split(ans)
vx1,vy1,vx2,vy2,q1,q2,b1 = TestFunctions(U)

u1 = as_vector([ux1,uy1])
u2 = as_vector([ux2,uy2])
v1 = as_vector([vx1,vy1])
v2 = as_vector([vx2,vy2])

N1  = Constant(1)
RH1 = Constant(cons_rh1)
RH2 = Constant(cons_rh2)
MU1 = Constant(cons_mu1)
MU2 = Constant(cons_mu2)
Bx  = Constant(0.0)
By  = Constant(-cons_gg)
BB  = as_vector([Bx, By])
POS = Expression(('x[0]','x[1]'), degree=2)
he  = CellSize(mesh)
dl  = Constant(cons_dl)

a2    = N1-a1
u_int = (u1*MU1*a1**(1.0/3) + u2*MU2*a2**(1.0/3))/(MU1*a1**(1.0/3) + MU2*a2**(1.0/3))
p_int = p1*a1 + p2*a2
F12   = MU1*MU2/(MU1+MU2)*(u2-u1)/dl +a1*a2*grad(p2-p1)
F21   = MU1*MU2/(MU1+MU2)*(u1-u2)/dl +a1*a2*grad(p1-p2)
SIGMA1    = MU1*(grad(u1)+grad(u1).T) -p1*Identity(len(u1))
SIGMA2    = MU2*(grad(u2)+grad(u2).T) -p2*Identity(len(u2))
SIGMA1_DS = as_tensor([  [-RH1*a1*inner(BB,POS),  Constant(0)],
                         [ Constant(0), -RH1*a1*inner(BB,POS)]  ])
SIGMA2_DS = as_tensor([  [-RH2*a2*inner(BB,POS),  Constant(0)],
                         [ Constant(0), -RH2*a2*inner(BB,POS)]  ])
NN    = FacetNormal(mesh)
x,y = 0,1

F  = inner(u_int, grad(a1))*b1*dx               \
   + inner(grad(p1-p2),grad(b1))*a1*a2*dl*dx    \
                                                \
   + div(u1*a1)*q1*dx                           \
                                                \
   + div(u2*a2)*q2*dx                           \
                                                \
   + inner( a1*RH1*dot(u1,grad(u1).T), v1 ) *dx \
   + inner( a1*SIGMA1, grad(v1) ) *dx           \
   - inner( BB*RH1*a1, v1) *dx                  \
   - inner( dot( SIGMA1_DS,NN ),v1 ) *ds        \
   - inner( F12,v1 ) *dx                        \
                                                \
   + inner( a2*RH2*dot(u2,grad(u2).T), v2 ) *dx \
   + inner( a2*SIGMA2, grad(v2) ) *dx           \
   - inner( BB*RH2*a2, v2) *dx                  \
   - inner( dot( SIGMA2_DS,NN ),v2 ) *ds        \
   - inner( F21,v2 ) *dx

# ------ BOUNDARY CONDITIONS AND SOLVE ------ #
inlet  = '( on_boundary && near(x[0],'+str(dim_0)+') && (x[1]>'+str(dim_0)+') && (x[1]<'+str(dim_D)+') )'
outlet = '( on_boundary && near(x[0],'+str(dim_L)+') && (x[1]>'+str(dim_0)+') && (x[1]<'+str(dim_D)+') )'
walls  = 'on_boundary && !'+inlet+'&& !'+outlet
a_in   = Constant(0.5)
#v_in   = Expression('4*v_in*x[1]*(1-x[1])', v_in=v_0, degree=2)
v_in   = Constant(v_0)
v_null = Constant(0)
p_ux1,p_uy1,p_ux2,p_uy2,p_p1,p_p2,p_aa = 0,1,2,3,4,5,6
BC = [
      DirichletBC(U.sub(p_aa) , a_in  , inlet  ),
      DirichletBC(U.sub(p_ux1), v_in  , inlet  ),
      DirichletBC(U.sub(p_uy1), v_null, inlet  ),
      DirichletBC(U.sub(p_ux2), v_in  , inlet  ),
      DirichletBC(U.sub(p_uy2), v_null, inlet  ),
      DirichletBC(U.sub(p_ux1), v_null, walls  ),
      DirichletBC(U.sub(p_uy1), v_null, walls  ),
      DirichletBC(U.sub(p_ux2), v_null, walls  ),
      DirichletBC(U.sub(p_uy2), v_null, walls  ),
      #DirichletBC(U.sub(p_uy1), v_null, outlet ),
      #DirichletBC(U.sub(p_uy2), v_null, outlet ),
      ]

assign(ans.sub(p_ux1), project(Constant(v_0 ), FunctionSpace(mesh, FE_u)))
assign(ans.sub(p_uy1), project(Constant(0   ), FunctionSpace(mesh, FE_u)))
assign(ans.sub(p_ux2), project(Constant(v_0 ), FunctionSpace(mesh, FE_u)))
assign(ans.sub(p_uy2), project(Constant(0   ), FunctionSpace(mesh, FE_u)))
assign(ans.sub(p_p1) , project(Constant(1e-3), FunctionSpace(mesh, FE_p)))
assign(ans.sub(p_p2) , project(Constant(1e-3), FunctionSpace(mesh, FE_p)))
assign(ans.sub(p_aa) , project(Constant(0.5 ), FunctionSpace(mesh, FE_a)))

v_max = 1.0E0
p_max = 1.0E5
a_min = 0.0
a_max = 1.0

lowBound = project(Constant((-v_max, -v_max, -v_max, -v_max, -p_max, -p_max, a_min)), U)
uppBound = project(Constant((+v_max, +v_max, +v_max, +v_max, +p_max, +p_max, a_max)), U)

dF = derivative(F, ans)
problem = NonlinearVariationalProblem(F, ans, BC, dF)
problem.set_bounds(lowBound,uppBound)
solver = NonlinearVariationalSolver(problem)
solver.parameters["nonlinear_solver"] = "snes"
# #prm = solver.parameters["newton_solver"]
# #prm["convergence_criterion"   ] = "residual"
# #prm["linear_solver"           ] = "mumps"
# #prm["method"                  ] = "full"
# #prm["preconditioner"          ] = "none"
# #prm["krylov_solver"           ] = 
# #prm["lu_solver"               ] = 
# prm["error_on_nonconvergence" ] = True
# prm["maximum_iterations"      ] = 10
# prm["absolute_tolerance"      ] = 6E-14
# prm["relative_tolerance"      ] = 8E-18
# prm["relaxation_parameter"    ] = 1.0
# #prm["report"                  ] = True


prm = solver.parameters["snes_solver"]
prm["error_on_nonconvergence"       ] = True
prm["solution_tolerance"            ] = 1.0E-16
prm["maximum_iterations"            ] = 1000
prm["maximum_residual_evaluations"  ] = 20000
prm["sign"                          ] = "default"
prm["absolute_tolerance"            ] = 6.0E-13
prm["relative_tolerance"            ] = 6.0E-13
prm["linear_solver"                 ] = "mumps"
#prm["method"                        ] = "vinewtonssls"
#prm["line_search"                   ] = "bt"
#prm["preconditioner"                ] = "none"
#prm["report"                        ] = True
#prm["krylov_solver"                 ]
#prm["lu_solver"                     ]

#set_log_level(PROGRESS)

def plot_all():
   plot(a1, title='Concentration 1')
   plot(grad(a1), title='Concentration Gradient 1')
   #plot(u1, title='Velocity Intrinsic 1')
   #plot(u2, title='Velocity Intrinsic 2')
   #plot(p1, title='Pressure Intrinsic 1')
   #plot(p2, title='Pressure Intrinsic 2')
   plot(u1*a1, title='Velocity Mean 1')
   plot(u2*a2, title='Velocity Mean 2')
   plot(p1*a1, title='Pressure Mean 1')
   #plot(p2*a2, title='Pressure Mean 2')
   interactive()

def save_results():
   foldername = 'results.Pgrad_Separation_Galerkin'
   vtk_ua1 = File(foldername+'/velocity_intrinsic1.pvd')
   vtk_ua2 = File(foldername+'/velocity_intrinsic2.pvd')
   vtk_u1  = File(foldername+'/velocity1.pvd')
   vtk_u2  = File(foldername+'/velocity2.pvd')
   vtk_p1  = File(foldername+'/pressure1.pvd')
   vtk_p2  = File(foldername+'/pressure2.pvd')
   vtk_aa  = File(foldername+'/fraction.pvd')
   FE_vector = FunctionSpace(mesh, VectorElement('P', mesh.ufl_cell(), 1))
   FE_scalar = FunctionSpace(mesh, FiniteElement('P', mesh.ufl_cell(), 1))
   ua1_viz = project(u1*a1, FE_vector); vtk_ua1 << ua1_viz
   ua2_viz = project(u2*a2, FE_vector); vtk_ua2 << ua2_viz
   u1_viz  = project(u1   , FE_vector); vtk_u1  << u1_viz
   u2_viz  = project(u2   , FE_vector); vtk_u2  << u2_viz
   p1_viz  = project(p1   , FE_scalar); vtk_p1  << p1_viz
   p2_viz  = project(p2   , FE_scalar); vtk_p2  << p2_viz
   aa_viz  = project(a1   , FE_scalar); vtk_aa  << aa_viz

lowBound = [-v_max, -v_max, -v_max, -v_max, -p_max, -p_max, a_min]
uppBound = [+v_max, +v_max, +v_max, +v_max, +p_max, +p_max, a_max]

solver.solve()
plot_all()

# g_exp_init = -12.0
# delta_exp  = 0.05
# steps      = 1000
# gg_list = [10**(g_exp_init+i*delta_exp) for i in range(steps)]
# for g_value in gg_list:
#    print('Solving for g={}'.format(g_value))
#    By.assign( -g_value )
#    solver.solve()
#    #save_results()

