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
mesh_H   = 0.001
mesh_Cx     = 0.020
mesh_Cy     = 0.010
mesh_Radius = 0.002
mesh_res = 50

cons_dt  = 0.001
cons_rho = 1E+3
cons_mu  = 1E-3
cons_v1  = 10**(-4.8)
cons_v2  = 1E-4

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
FE_1  = FiniteElement('P', 'triangle', 2)
elem  = MixedElement([FE_1, FE_1, FE_1, FE_1, FE_1, FE_1, FE_1])
U     = FunctionSpace(mesh, elem)
U_vel = FunctionSpace(mesh, MixedElement([FE_1, FE_1]))

u_next   = project(Constant((cons_v1,0)), U_vel)
ans      = Function(U)
ux,uy,p,uxx,uxy,uyx,uyy = split(ans)

u        = as_vector(   [ ux, uy ]  )
grad_u   = as_tensor([  [uxx, uxy],
                        [uyx, uyy] ])

DT       = Constant(cons_dt)
RHO      = Constant(cons_rho)
MU       = Constant(cons_mu)
u_inlet  = Constant(cons_v1)

div_u = uxx +uyy
sigma = MU*(grad_u+grad_u.T) -p*Identity(len(u))

si_ou = as_tensor([  [Constant(0), Constant(0)],
                     [Constant(0), Constant(0)], ])

#in_profile1 = Expression(str(cons_v1)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', degree=2)
#in_profile2 = Expression(str(cons_v2)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', degree=2)

u_in  = as_vector([ u_inlet      , Constant(0) ])
u_wl  = as_vector([ Constant(0  ), Constant(0) ])

F_st  =                   RHO*dot(u,grad_u.T) -div(sigma)
F_mt  = u -u_next*RHO/DT +RHO*dot(u,grad_u.T) -div(sigma)
F_ct  = div_u
F_gu  = grad_u -grad(u)
F_in  = u -u_in
F_wl  = u -u_wl
F_ou  = sigma -si_ou

F1 = derivative(     inner( F_mt,F_mt )*dx \
                  +  inner( F_ct,F_ct )*dx \
                  +  inner( F_gu,F_gu )*dx \
                  +  inner( F_in,F_in )*ds(ds_inlet) \
                  +  inner( F_wl,F_wl )*ds(ds_walls) \
                  +  inner( F_ou,F_ou )*ds(ds_outlet)
                  , ans, TestFunction(U) )

F2 = derivative(     inner( F_st,F_st )*dx \
                  +  inner( F_ct,F_ct )*dx \
                  +  inner( F_gu,F_gu )*dx \
                  +  inner( F_in,F_in )*ds(ds_inlet) \
                  +  inner( F_wl,F_wl )*ds(ds_walls) \
                  +  inner( F_ou,F_ou )*ds(ds_outlet)
                  , ans, TestFunction(U) )

# ------ BOUNDARY CONDITIONS ------ #
p_ux,p_uy,p_pp,p_ww = 0,1,2,3
BC1 = [
         DirichletBC(U.sub(p_ux), in_profile1, inlet),
         DirichletBC(U.sub(p_uy), Constant(0), inlet),
         DirichletBC(U.sub(p_ux), Constant(0), walls),
         DirichletBC(U.sub(p_uy), Constant(0), walls),
                                                                     ] # end - BC #

BC2 = [
         DirichletBC(U.sub(p_ux), in_profile2, inlet),
         DirichletBC(U.sub(p_uy), Constant(0), inlet),
         DirichletBC(U.sub(p_ux), Constant(0), walls),
         DirichletBC(U.sub(p_uy), Constant(0), walls),
                                                                     ] # end - BC #

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
dF1 = derivative(F1, ans)
dF2 = derivative(F2, ans)
nlProblem1 = NonlinearVariationalProblem(F1, ans, [], dF1)
nlProblem2 = NonlinearVariationalProblem(F2, ans, [], dF2)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
nlSolver2  = NonlinearVariationalSolver(nlProblem2)
prm1 = nlSolver1.parameters["newton_solver"]
prm2 = nlSolver2.parameters["newton_solver"]
for prm in [prm1, prm2]:
   prm["maximum_iterations"      ] = 10
   prm["absolute_tolerance"      ] = 5E-16
   prm["relative_tolerance"      ] = 5E-16

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
assign(ans.sub(p_ux ), project(Constant(1E-2), FunctionSpace(mesh, FE_1) ) )
assign(ans.sub(p_uy ), project(Constant(0E-2), FunctionSpace(mesh, FE_1) ) )
assign(ans.sub(p_pp ), project(Constant(1E-2), FunctionSpace(mesh, FE_1) ) )
u_inlet.assign(cons_v2)
nlSolver2.solve()
u_next.assign(project(u, U_vel))
u_inlet.assign(cons_v1)
while( count_iteration < simple_Nmax ):
   count_iteration = count_iteration +1
   nlSolver1.solve()
   residual = assemble(inner(u-u_next, u-u_next)*dx)/assemble(inner(u,u)*dx)
   print ('Residual : {}'.format(residual) )
   print ('Iteration: {}'.format(count_iteration) )
   if  residual < simple_tol:
      flag_converged = True
   u_next.assign(project(u, U_vel))
   save_flow(u,p)

#plot(u, title='velocity')
#plot(p, title='pressure')
#interactive()
