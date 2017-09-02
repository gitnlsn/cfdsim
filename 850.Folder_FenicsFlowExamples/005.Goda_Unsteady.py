'''

NELSON KENZO TAMASHIRO

19.07.2017

'''

# ------ LIBRARIES ------ #
from fenics          import *
from mshr            import *
from dolfin_adjoint  import *

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
cons_v1  = 1E-1
cons_v2  = 1E-4
cons_pin  = 60
cons_pout = 0

simple_tol  = 1E-7
simple_Nmax = 5000

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
FE_p  = FiniteElement('P', 'triangle', 1)
FE_u  = VectorElement('P', 'triangle', 2)
U_prs = FunctionSpace(mesh, FE_p)
U_vel = FunctionSpace(mesh, FE_u)

u_aux    = project( Constant((cons_v1,0)), U_vel)
u_nxt    = project( Constant((cons_v1,0)), U_vel)
p_nxt    = project( Constant(    0      ), U_prs)

v = TestFunction(U_vel)
q = TestFunction(U_prs)

DT       = Constant(cons_dt)
RHO      = Constant(cons_rho)
MU       = Constant(cons_mu)
u_inlet  = Constant(cons_v1)
n        = FacetNormal(mesh)

#in_profile1 = Expression(str(cons_v1)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', degree=2)
#in_profile2 = Expression(str(cons_v2)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', degree=2)

u_in  = as_vector([ u_inlet    , Constant(0) ])
u_wl  = as_vector([ Constant(0), Constant(0) ])
p_in  = Constant(cons_pin  )
p_out = Constant(cons_pout )

u_md = (u_aux+u_nxt)*0.5

F1 = RHO*inner( u_aux -u_nxt, v )/DT         *dx \
   + RHO*inner( dot(u_md,grad(u_md).T), v )  *dx \
   + MU *inner( grad(u_md),grad(v) )         *dx \
   #- MU *inner( dot(grad(u_md),n),v )        *ds

F2 = inner( grad(p_nxt),grad(q) )   *dx \
   + inner( div(u_aux), q)*RHO/DT   *dx \
   #- inner( grad(p_nxt), n)*q       *ds

F3 = inner( u_nxt -u_aux,v )        *dx \
   + inner( grad(p_nxt),v) *DT/RHO  *dx

# ------ BOUNDARY CONDITIONS ------ #
p_ux,p_uy,p_pp,p_ww = 0,1,2,3
BC1 = [
         DirichletBC(U_vel, u_in, inlet),
         DirichletBC(U_vel, u_wl, walls),
      ] # end - BC #

BC2 = [
         #DirichletBC(U_prs, p_in,   inlet),
         DirichletBC(U_prs, p_out, outlet),
      ] # end - BC #

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
dF1 = derivative(F1, u_aux)
dF2 = derivative(F2, p_nxt)
dF3 = derivative(F3, u_nxt)
nlProblem1 = NonlinearVariationalProblem(F1, u_aux, BC1, dF1)
nlProblem2 = NonlinearVariationalProblem(F2, p_nxt, BC2, dF2)
nlProblem3 = NonlinearVariationalProblem(F3, u_nxt, [], dF3)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
nlSolver2  = NonlinearVariationalSolver(nlProblem2)
nlSolver3  = NonlinearVariationalSolver(nlProblem3)
prm1 = nlSolver1.parameters["newton_solver"]
prm2 = nlSolver2.parameters["newton_solver"]
prm3 = nlSolver3.parameters["newton_solver"]
for prm in [prm1, prm2, prm3]:
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

def save_flow(u_tosave, p_tosave):
   ui = project(u_tosave,FunctionSpace(mesh,FE_u))
   pi = project(p_tosave,FunctionSpace(mesh,FE_p))
   ui.rename('velocity','velocity')
   pi.rename('pressure','pressure')
   vtk_uu << ui
   vtk_pp << pi

# ------ TRANSIENT SIMULATION ------ #
count_iteration   = 0
#flag_converged    = False
#assign(ans.sub(p_ux ), project(Constant(1E-2), FunctionSpace(mesh, FE_1) ) )
#assign(ans.sub(p_uy ), project(Constant(0E-2), FunctionSpace(mesh, FE_1) ) )
#assign(ans.sub(p_pp ), project(Constant(1E-2), FunctionSpace(mesh, FE_1) ) )
#u_inlet.assign(cons_v2)
#u_next.assign(project(u, U_vel))
while( count_iteration < simple_Nmax ):
   count_iteration = count_iteration +1
   nlSolver1.solve()
   nlSolver2.solve()
   nlSolver3.solve()   
   residual = assemble(inner(u_nxt -u_aux, u_nxt -u_aux)*dx)
   print ('Residual : {}'.format(residual) )
   print ('Iteration: {}'.format(count_iteration) )
   if  residual < simple_tol:
      flag_converged = True
   save_flow(u_nxt,p_nxt)

#plot(u, title='velocity')
#plot(p, title='pressure')
#interactive()

