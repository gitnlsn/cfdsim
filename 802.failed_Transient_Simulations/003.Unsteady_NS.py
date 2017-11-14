'''

 

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
mesh_H   = 0.001
mesh_L   = 0.080
mesh_Cx     = 0.030
mesh_Cy     = 0.010
mesh_Radius = 0.002
mesh_res = 50

cons_dt  = 0.00001
cons_rho = 1E+3
cons_mu  = 1E-3
cons_vin = 1E-0

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

#print assemble(CellSize(mesh)*dx)**(1/2.0)

# ------ BOUNDARIES ------ #
#walls  = '(x[1]=='+str(1.0*mesh_D)+') || (x[1]=='+str(0.0*mesh_D)+')'
inlet  = '( (x[0]=='+str(0.0*mesh_L)+') && (x[1]> '+str(mesh_0)+') && (x[1]< '+str(mesh_D)+') )'
outlet = '( (x[0]=='+str(1.0*mesh_L)+') && (x[1]> '+str(mesh_0)+') && (x[1]< '+str(mesh_D)+') )'
walls  = 'on_boundary && !'+inlet +' && !'+outlet

# ------ VARIATIONAL FORMULATION ------ #
FE_P  = FiniteElement('P', 'triangle', 1)
FE_V  = FiniteElement('P', 'triangle', 2)
elem  = MixedElement([FE_V, FE_V, FE_P])
U     = FunctionSpace(mesh, elem)
U_vel = FunctionSpace(mesh, MixedElement([FE_V, FE_V]))

u_next   = project(Constant((cons_vin,0)), U_vel)
ans      = Function(U)
ux,uy,p      = split(ans)
vx,vy,q      = TestFunctions(U)

u = as_vector([ ux,uy ])
v = as_vector([ vx,vy ])

DT    = Constant(cons_dt)
RHO   = Constant(cons_rho)
MU    = Constant(cons_mu)

F1 = RHO *inner( u -u_next,v )/DT      *dx \
   + RHO *inner( dot(u,grad(u).T), v ) *dx \
   + MU  *inner( grad(u), grad(v) )    *dx \
   - p*div(v)                          *dx \
   + q*div(u)                          *dx

F2 = RHO *inner( dot(u,grad(u).T), v ) *dx \
   + MU  *inner( grad(u), grad(v) )    *dx \
   - p*div(v)                          *dx \
   + q*div(u)                          *dx

# ------ BOUNDARY CONDITIONS ------ #
in_profile1 = (str(cons_vin)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')')
in_profile2 = (str(1.0E-2  )+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')')
p_ux,p_uy,p_pp = 0,1,2
BC1 = [
         DirichletBC(U.sub(p_ux), Expression(in_profile1, degree=2), inlet),
         DirichletBC(U.sub(p_uy), Constant(0), inlet),
         DirichletBC(U.sub(p_ux), Constant(0), walls),
         DirichletBC(U.sub(p_uy), Constant(0), walls),
                                                                     ] # end - BC #

BC2 = [
         DirichletBC(U.sub(p_ux), Expression(in_profile2, degree=2), inlet),
         DirichletBC(U.sub(p_uy), Constant(0), inlet),
         DirichletBC(U.sub(p_ux), Constant(0), walls),
         DirichletBC(U.sub(p_uy), Constant(0), walls),
                                                                     ] # end - BC #

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
dF1 = derivative(F1, ans)
nlProblem1 = NonlinearVariationalProblem(F1, ans, BC1, dF1)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
prm1 = nlSolver1.parameters["newton_solver"]
for prm in [prm1]:
   prm["maximum_iterations"      ] = 10
   prm["absolute_tolerance"      ] = 5E-10
   prm["relative_tolerance"      ] = 5E-10

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
solve(F2==0, ans, BC2)
u_next.assign(project(u, U_vel))
while( count_iteration < simple_Nmax ):
   count_iteration = count_iteration +1
   print ('Iteration: {}'.format(count_iteration) )
   nlSolver1.solve()
   residual = assemble(inner(u-u_next, u-u_next)*dx)/assemble(inner(u,u)*dx)
   print ('Residual : {}'.format(residual) )
   if  residual < simple_tol:
      flag_converged = True
   u_next.assign(project(u, U_vel))
   save_flow(u,p)

#plot(u, title='velocity')
#plot(p, title='pressure')
#interactive()
