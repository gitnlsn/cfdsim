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
resolution  = 70

dim_0       = 0.0
dim_L       = 5.0
dim_D       = 1.0

# ------ SIMULATION PARAMETERS CONFIGURATION ------ #
cons_rh1 = 1.0E+3
cons_rh2 = 1.3E+3
cons_mu1 = 1.0E-3
cons_mu2 = 1.0E-3
cons_gg  = 1E-12

v_0 = 1E-6

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
elem = MixedElement([FE_u, FE_u, FE_u, FE_u, FE_p, FE_a])
U    = FunctionSpace(mesh, elem)

ans = Function(U)
ux1,uy1,ux2,uy2,p1,a1 = split(ans)
vx1,vy1,vx2,vy2,q1,b1 = TestFunctions(U)

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

a2    = N1-a1
TAU_1 = MU1*(grad(u1)+grad(u1).T)
TAU_2 = MU2*(grad(u2)+grad(u2).T)
x,y = 0,1

F  = div(u1*a1)*q1*dx                           \
   + div(u2*a2)*b1*dx                           \
   + inner( a1*RH1*dot(u1,grad(u1).T), v1 ) *dx \
   + inner( a1*TAU_1, grad(v1) ) *dx            \
   - a1*p1*div(v1) *dx                          \
   - inner( BB*RH1*a1, v1) *dx                  \
   + inner( a2*RH2*dot(u2,grad(u2).T), v1 ) *dx \
   + inner( a2*TAU_2, grad(v2) ) *dx            \
   - a2*p1*div(v2) *dx                          \
   - inner( BB*RH2*a2, v2) *dx

# ------ BOUNDARY CONDITIONS AND SOLVE ------ #
inlet  = '(near(x[0],'+str(dim_0)+') && on_boundary)'
outlet = '(near(x[0],'+str(dim_L)+') && on_boundary)'
walls  = 'on_boundary && !'+inlet+'&& !'+outlet
p_ref  = 'x[0]==0 && x[1]==0'
a_in   = Constant(0.5)
#v_in   = Expression('4*v_in*x[1]*(1-x[1])', v_in=v_0, degree=2)
v_in   = Constant(v_0)
v_null = Constant(0)
p_ux1,p_uy1,p_ux2,p_uy2,p_p1,p_aa = 0,1,2,3,4,5
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
      #DirichletBC(U.sub(p_p1), Constant(0), p_ref, method='pointwise'),
      ]

assign(ans.sub(p_ux1), project(Constant(v_0 ), FunctionSpace(mesh, FE_u)))
assign(ans.sub(p_uy1), project(Constant(0   ), FunctionSpace(mesh, FE_u)))
assign(ans.sub(p_ux2), project(Constant(v_0 ), FunctionSpace(mesh, FE_u)))
assign(ans.sub(p_uy2), project(Constant(0   ), FunctionSpace(mesh, FE_u)))
assign(ans.sub(p_p1) , project(Constant(1e-5), FunctionSpace(mesh, FE_p)))
assign(ans.sub(p_aa) , project(Constant(0.5 ), FunctionSpace(mesh, FE_a)))

dF = derivative(F, ans)
problem = NonlinearVariationalProblem(F, ans, BC, dF)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters["newton_solver"]
#prm["convergence_criterion"   ] = "residual"
#prm["linear_solver"           ] = "mumps"
#prm["method"                  ] = "full"
#prm["preconditioner"          ] = "none"
prm["error_on_nonconvergence" ] = True
prm["maximum_iterations"      ] = 8
prm["absolute_tolerance"      ] = 6E-16
prm["relative_tolerance"      ] = 8E-16
prm["relaxation_parameter"    ] = 1.0
#prm["report"                  ] = True
#set_log_level(PROGRESS)

foldername = 'results.Pgrad_Separation_Galerkin'
vtk_ua1 = File(foldername+'/velocity_intrinsic1.pvd')
vtk_ua2 = File(foldername+'/velocity_intrinsic2.pvd')
vtk_u1  = File(foldername+'/velocity1.pvd')
vtk_u2  = File(foldername+'/velocity1.pvd')
vtk_pp  = File(foldername+'/pressure.pvd')
vtk_aa  = File(foldername+'/fraction.pvd')
FE_vector = FunctionSpace(mesh, VectorElement('P', mesh.ufl_cell(), 1))
FE_scalar = FunctionSpace(mesh, FiniteElement('P', mesh.ufl_cell(), 1))

def save_results():
   ua1_viz = project(u1*a1, FE_vector); vtk_ua1 << ua1_viz
   ua2_viz = project(u2*a2, FE_vector); vtk_ua2 << ua2_viz
   u1_viz  = project(u1   , FE_vector); vtk_u1  << u1_viz
   u2_viz  = project(u2   , FE_vector); vtk_u1  << u1_viz
   pp_viz  = project(p1   , FE_scalar); vtk_pp  << pp_viz
   aa_viz  = project(a1   , FE_scalar); vtk_aa  << aa_viz

g_exp_init = -12.0
delta_exp  = 0.05
steps      = 1000
gg_list = [10**(g_exp_init+i*delta_exp) for i in range(steps)]
for g_value in gg_list:
   print('Solving for g={}'.format(g_value))
   By.assign( -g_value )
   solver.solve()
   save_results()

