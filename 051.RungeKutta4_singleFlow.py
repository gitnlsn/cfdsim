'''

NELSON KENZO TAMASHIRO

19.07.2017

'''

# ------ LIBRARIES ------ #
from fenics          import *
from mshr            import *
#from dolfin_adjoint  import *

# ------ SIMULATION PARAMETERS ------ #
filename = 'results_Goda'
mesh_0   = 0.0
mesh_D   = 0.020
mesh_L   = 0.060
mesh_H   = 0.001
mesh_Cx     = 0.010
mesh_Cy     = 0.010
mesh_Radius = 0.002
mesh_res = 50

cons_dt  = 0.01
cons_rho = 1E+3
cons_mu  = 1E-3
cons_dif = 1E-8
cons_v1  = 1E-1
cons_v2  = 1E-4
cons_g   = 9.8E0
cons_pin    = 60
cons_pout   = 0
GENERAL_TOL = 1E-6

TRANSIENT_MAX_ITE = 200

# ------ MESH ------ #
part1 = Rectangle(
   Point(mesh_0, mesh_0),
   Point(mesh_L, mesh_D)   )
part2 = Circle(
   Point(mesh_Cx, mesh_Cy),
   mesh_Radius             )
channel = part1 -part2
mesh = generate_mesh(channel, mesh_res)

# ------ BOUNDARIES ------ #
#walls  = '(x[1]=='+str(1.0*mesh_D)+') || (x[1]=='+str(0.0*mesh_D)+')'
inlet  = 'near(x[0],'+str(0.0*mesh_L)+')'
inlet1 = 'near(x[0],'+str(0.0*mesh_L)+') && x[1]>='+str(mesh_D/2.0)
inlet2 = 'near(x[0],'+str(0.0*mesh_L)+') && x[1]<='+str(mesh_D/2.0)
outlet = 'near(x[0],'+str(1.0*mesh_L)+')'
walls  = 'on_boundary && !'+inlet +' && !'+outlet

ds_inlet, ds_walls, ds_outlet = 1,2,3

boundaries     = FacetFunction ('size_t', mesh)
side_inlet     = CompiledSubDomain( inlet  )
side_walls     = CompiledSubDomain( walls  )
side_outlet    = CompiledSubDomain( outlet )
boundaries.set_all(0)
side_inlet.mark   (boundaries, ds_inlet  )
side_walls.mark   (boundaries, ds_walls  )
side_outlet.mark  (boundaries, ds_outlet )
ds = Measure( 'ds', subdomain_data=boundaries )

# ------ VARIATIONAL FORMULATION ------ #
FE_u  = VectorElement('P', 'triangle', 2)
FE_p  = FiniteElement('P', 'triangle', 1)
FE_a  = FiniteElement('P', 'triangle', 1)
U_prs = FunctionSpace(mesh, FE_p)
U_vel = FunctionSpace(mesh, FE_u)
U     = FunctionSpace(mesh, MixedElement([FE_u, FE_p, FE_a]) )

ans1   = Function(U)
ans2   = Function(U)

u1,p1,a1 = split(ans1)
u2,p2,a2 = split(ans2)

v,q,b = TestFunctions(U)

DT       = Constant(cons_dt  )
RHO      = Constant(cons_rho )
MU       = Constant(cons_mu  )
DD       = Constant(cons_dif )
u_inlet  = Constant(cons_v1  )
GG       = as_vector([ Constant(0), Constant(cons_g) ])
n        = FacetNormal(mesh)

cons_pe = mesh_D*cons_v1/cons_dif

#in_profile1 = Expression(str(cons_v1)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', degree=2)
#in_profile2 = Expression(str(cons_v2)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', degree=2)

u_in  = as_vector([ u_inlet    , Constant(0) ])
u_wl  = as_vector([ Constant(0), Constant(0) ])
p_in  = Constant(cons_pin  )
p_out = Constant(cons_pout )

he = CellSize(mesh)
u_md  = (u1+u2)*0.5
p_md  = (p1+p2)*0.5
a_md  = (a1+a2)*0.5
sigma = MU*(grad(u_md)+grad(u_md).T) -p_md*Identity(len(u_md))
Tsupg = (4/(cons_pe*he**2) +2*sqrt(GENERAL_TOL+inner(u_md,u_md))/he)**-1

F1 = RHO/DT *inner( u2-u1,v )                   *dx \
   + RHO    *inner( dot(u_md,grad(u_md).T),v )  *dx \
   +         inner( sigma,grad(v) )             *dx \
   + q*div(u2)                                  *dx \
   +         inner( a2-a1,b ) /DT               *dx \
   +         inner( inner(u_md,grad(a_md)),b )  *dx \
   + DD     *inner(grad(a_md),grad(b))          *dx \
   + inner( dot(u_md,grad(b)),
            dot(u_md,grad(a_md)) )*Tsupg           *dx

# ------ BOUNDARY CONDITIONS ------ #
p_uu,p_pp,p_aa = 0,1,2
BC1 = [
         DirichletBC(U.sub(p_uu), Constant((cons_v1,0)), inlet),
         DirichletBC(U.sub(p_uu), Constant((      0,0)), walls),
         DirichletBC(U.sub(p_aa), Constant(       1   ), inlet1),
         DirichletBC(U.sub(p_aa), Constant(       0   ), inlet2),
      ] # end - BC #

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
dF1 = derivative(F1, ans2 )
# dF2 = derivative(F2, p_nxt)
# dF3 = derivative(F3, u_nxt)
nlProblem1 = NonlinearVariationalProblem(F1, ans2, BC1, dF1)
# nlProblem2 = NonlinearVariationalProblem(F2, p_nxt, BC2, dF2)
# nlProblem3 = NonlinearVariationalProblem(F3, u_nxt, [], dF3)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
# nlSolver2  = NonlinearVariationalSolver(nlProblem2)
# nlSolver3  = NonlinearVariationalSolver(nlProblem3)
prm1 = nlSolver1.parameters["newton_solver"]
# prm2 = nlSolver2.parameters["newton_solver"]
# prm3 = nlSolver3.parameters["newton_solver"]
for prm in [prm1]:
   prm["maximum_iterations"      ] = 10
   prm["absolute_tolerance"      ] = 9E-13
   prm["relative_tolerance"      ] = 8E-13

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
vtk_aa  = File(filename+'/concentration.pvd')

def save_flow(u_tosave, p_tosave, a_tosave):
   ui = project(u_tosave,FunctionSpace(mesh,FE_u))
   pi = project(p_tosave,FunctionSpace(mesh,FE_p))
   ai = project(a_tosave,FunctionSpace(mesh,FE_a))
   ui.rename('velocity','velocity')
   pi.rename('pressure','pressure')
   ai.rename('concentration','concentration')
   vtk_uu << ui
   vtk_pp << pi
   vtk_aa << ai

# ------ TRANSIENT SIMULATION ------ #
count_iteration   = 0
#flag_converged    = False
#assign(ans.sub(p_ux ), project(Constant(1E-2), FunctionSpace(mesh, FE_1) ) )
#assign(ans.sub(p_uy ), project(Constant(0E-2), FunctionSpace(mesh, FE_1) ) )
#assign(ans.sub(p_pp ), project(Constant(1E-2), FunctionSpace(mesh, FE_1) ) )
#u_inlet.assign(cons_v2)
#u_next.assign(project(u, U_vel))
while( count_iteration < TRANSIENT_MAX_ITE ):
   count_iteration = count_iteration +1
   nlSolver1.solve()
   # nlSolver2.solve()
   # nlSolver3.solve()
   residual = assemble(inner(u2-u1, u2-u1)*dx)
   print ('Residual : {}'.format(residual) )
   print ('Iteration: {}'.format(count_iteration) )
   ans1.assign(ans2)
   #save_flow(u2,p2,a2)

#plot(u, title='velocity')
#plot(p, title='pressure')
#interactive()

