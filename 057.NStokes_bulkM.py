'''

NELSON KENZO TAMASHIRO

19.07.2017

'''

# ------ LIBRARIES ------ #
from fenics    import *
from mshr      import *

# ------ SIMULATION PARAMETERS ------ #
filename = 'results_NStokes_bulk'

# ------ TMIXER GEOMETRY PARAMETERS ------ #
mesh_res = 100
mesh_0   = 0.0
mesh_D   = 0.020
mesh_L   = 0.060
mesh_H   = 0.001
mesh_Cx     = 0.010
mesh_Cy     = 0.010
mesh_Radius = 0.002

cons_dt  = 0.001
cons_rho = 1E+3
cons_mu  = 1E-3
cons_dif = 1E-8
cons_v1  = 1E-1
cons_g   = 9.8E0
cons_kb  = 2.2E9
GENERAL_TOL = 1E-6

TRANSIENT_MAX_ITE = 2000

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
obstcl = '( on_boundary && (x[0]>'+str(mesh_0)+') && (x[0]<'+str(mesh_L)+') '\
                     + '&& (x[1]>'+str(mesh_0)+') && (x[1]<'+str(mesh_D)+')   )'
walls  = '( on_boundary && ((x[1]=='+str(mesh_D)+') || (x[1]=='+str(mesh_0)+'))  ) || '+obstcl

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
U_prs = FunctionSpace(mesh, FE_p)
U_vel = FunctionSpace(mesh, FE_u)
U     = FunctionSpace(mesh, MixedElement( [FE_u, FE_p] ))

ans1  = Function(U)
ans2  = Function(U)

u1,p1 = split(ans1)
u2,p2 = split(ans2)

v,q = TestFunctions(U)

DT       = Constant(cons_dt   )
RHO      = Constant(cons_rho  )
MU       = Constant(cons_mu   )
DD       = Constant(cons_dif  )
Kb       = Constant(cons_kb   )
u_inlet  = Constant(cons_v1   )
GG       = as_vector([ Constant(0), Constant(cons_g) ])
n        = FacetNormal(mesh)

#in_profile1 = Expression(str(cons_v1)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', degree=2)
#in_profile2 = Expression(str(cons_v2)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', degree=2)

u_in  = as_vector([ u_inlet    , Constant(0) ])
u_wl  = as_vector([ Constant(0), Constant(0) ])

u_md  = (u1+u2)*0.5
p_md  = (p1+p2)*0.5
sigma = MU*(grad(u_md)+grad(u_md).T) -p_md*Identity(len(u_md))

F1 = (p2-p1)/DT *q                        *dx \
   + Kb*div(u_md) *q                      *dx \
   \
   + RHO/DT*inner(u2-u1, v)               *dx \
   + RHO*inner(div(outer(u_md,u_md)),v)   *dx \
   + inner(sigma,grad(v))                 *dx

# ------ BOUNDARY CONDITIONS ------ #
p_uu,p_pp = 0,1
BC1 = [
         DirichletBC(U.sub(p_uu), Constant((cons_v1,0)), inlet    ),
         DirichletBC(U.sub(p_uu), Constant((      0,0)), walls    ),
         DirichletBC(U.sub(p_pp), Constant(0          ), outlet   ),
      ] # end - BC #

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
dF1 = derivative(F1, ans2)
nlProblem1 = NonlinearVariationalProblem(F1, ans2, BC1, dF1)
#nlProblem1.set_bounds(lowBound,uppBound)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
nlSolver1.parameters["nonlinear_solver"] = "snes"

prm = nlSolver1.parameters["snes_solver"]
prm["error_on_nonconvergence"       ] = True
prm["solution_tolerance"            ] = 1.0E-10
prm["absolute_tolerance"            ] = 6.0E-10
prm["relative_tolerance"            ] = 6.0E-10
prm["maximum_iterations"            ] = 15
prm["maximum_residual_evaluations"  ] = 20000
prm["linear_solver"                 ] = "mumps"
#prm["sign"                          ] = "default"
#prm["method"                        ] = "vinewtonssls"
#prm["line_search"                   ] = "bt"
#prm["preconditioner"                ] = "none"
#prm["report"                        ] = True
#prm["krylov_solver"                 ]
#prm["lu_solver"                     ]

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

def RungeKutta4(ans_now, ans_nxt, nlSolver, DT):
   ans_aux  = Function(U)
   ans_aux.assign(ans_now)
   RK1      = Function(U)
   RK2      = Function(U)
   RK3      = Function(U)
   RK4      = Function(U)
   # 1st iteration
   ans_now.assign( ans_aux )
   nlSolver.solve()
   RK1.assign( ans_nxt -ans_now )
   # 2nd iteration
   ans_now.assign( ans_aux+RK1/2.0 )
   nlSolver.solve()
   RK2.assign( ans_nxt -ans_now )
   # 3rd iteration
   ans_now.assign( ans_aux+RK2/2.0 )
   nlSolver.solve()
   RK3.assign( ans_nxt -ans_now )
   # 4th iteration
   ans_now.assign( ans_aux+RK3 )
   nlSolver.solve()
   RK4.assign( ans_nxt -ans_now )
   # return RungeKutta estimate
   ans_now.assign(ans_aux)
   ans_nxt.assign(project( ans_aux+ DT/6.0*(RK1+RK2*2.0+RK3*2.0+RK4)/DT, U))

# ------ TRANSIENT SIMULATION ------ #
count_iteration   = 0
while( count_iteration < TRANSIENT_MAX_ITE ):
   count_iteration = count_iteration +1
   RungeKutta4(ans1, ans2, nlSolver1, DT)
   residual = assemble(inner(ans2 -ans1,ans2 -ans1)*dx)
   print ('Residual : {}'.format(residual) )
   print ('Iteration: {}'.format(count_iteration) )
   ans1.assign(ans2)
   save_flow(u2,p2)

#plot(u, title='velocity')
#plot(p, title='pressure')
#interactive()
