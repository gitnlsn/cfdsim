'''

NELSON KENZO TAMASHIRO

19.07.2017

'''

# ------ LIBRARIES ------ #
from fenics    import *
from mshr      import *

# ------ SIMULATION PARAMETERS ------ #
foldername = 'results_Axissimetric'

# ------ TMIXER GEOMETRY PARAMETERS ------ #
mesh_res  = 50
mesh_P0   = 0.0
mesh_R    = 0.4          # largura
mesh_H    = 0.6          # comprimento
mesh_tol  = mesh_R*0.01

# ------ TMIXER GEOMETRY PARAMETERS ------ #
cons_dt  = 1.0E-2
cons_rho = 1.0E+3
cons_mu  = 1.0E-3
cons_ome = 1.0E-2
cons_g   = 9.8E-0

v_max = cons_ome*mesh_R*50
p_max = 1.0E5

GENERAL_TOL = 1E-6
TRANSIENT_MAX_ITE = 1000

# ------ MESH ------ #
part1 = Rectangle(
   Point(mesh_P0, mesh_P0),
   Point(mesh_R , mesh_H )    )
channel = part1
mesh = generate_mesh(channel, mesh_res)

# ------ BOUNDARIES DEFINITION ------ #
upper  = '( (x[1]=='+str(mesh_H )+') )'
bottom = '( (x[1]=='+str(mesh_P0)+') )'
walls  = '( (x[0]=='+str(mesh_R )+') )'
middle = '( (x[0]=='+str(mesh_P0)+') )'

ds_upper, ds_bottom, ds_middle, ds_walls = 1,2,3,4

boundaries     = FacetFunction ('size_t', mesh)
side_upper     = CompiledSubDomain( upper    )
side_bottom    = CompiledSubDomain( bottom   )
side_walls     = CompiledSubDomain( walls    )
side_middle    = CompiledSubDomain( middle   )
boundaries.set_all(0)
side_upper.mark   (boundaries, ds_upper  )
side_walls.mark   (boundaries, ds_walls  )
side_bottom.mark  (boundaries, ds_bottom )
side_middle.mark  (boundaries, ds_middle )
ds = Measure( 'ds', subdomain_data=boundaries )


# ------ VARIATIONAL FORMULATION ------ #
FE_u  = FiniteElement('P', 'triangle', 2)
FE_p  = FiniteElement('P', 'triangle', 1)

U_prs = FunctionSpace(mesh, FE_p)

U_vel1 = FunctionSpace(mesh, FE_u )
U_vel2 = FunctionSpace(mesh, MixedElement([FE_u, FE_u          ]) )
U_vel3 = FunctionSpace(mesh, MixedElement([FE_u, FE_u, FE_u    ]) )

U     = FunctionSpace(mesh, MixedElement([FE_u, FE_u, FE_u, FE_p  ]) )

ansn  = Function(U)
ansm  = Function(U)

ur1,ut1,uw1,p1 = split(ans1)
ur2,ut2,uw2,p2 = split(ans2)
vr,vt,vw,q = split(ans1)

u1 = as_vector( [ur1,ut1,uw1] )
u2 = as_vector( [ur2,ut2,uw2] )
V  = as_vector( [ vr, vt, vw] )

dr,dw = 0,1
r = Expression('x[0]', degree=1)

def grad_cyl(ur,ut,uw):
   return as_tensor([   [Dx(ur,dr), -ut/r, Dx(ur,dw)],
                        [Dx(ut,dr), +ur/r, Dx(ut,dw)],
                        [Dx(uw,dr),     0, Dx(uw,dw)]  ])

def div_cyl(ur,ut,uw):
   return ur/r +Dx(ur,dr) +Dx(uw,dw)

div_u2  = div_cyl (ur2,ut2,uw2)
grad_u1 = grad_cyl(ur1,ut1,uw1)
grad_u2 = grad_cyl(ur2,ut2,uw2)
grad_v  = grad_cyl( vr,vt,vw  )

sigma1 = MU*(grad_u1+grad_u1.T) -p1*Identity(len(u2))
sigma2 = MU*(grad_u2+grad_u2.T) -p2*Identity(len(u1))

DT       = Constant(cons_dt   )
RHO      = Constant(cons_rho  )
MU       = Constant(cons_mu   )
u_inlet  = Constant(cons_vin  )
GG       = as_vector(   [Constant(0), Constant(0), Constant(-cons_g)           ]  )
HH       = as_vector(   [Constant(0), Constant(0), Expression('x[1]',degree=2) ]  )
NN       = FacetNormal(mesh)

SIGMA1_DS = as_tensor([  [-RH1*inner(GG,HH),  Constant(0)],
                         [ Constant(0), -RH1*inner(GG,HH)]  ])
SIGMA2_DS = as_tensor([  [-RH2*inner(GG,HH),  Constant(0)],
                         [ Constant(0), -RH2*inner(GG,HH)]  ])

F1 = div_u2*q                          *dx \
   + RHO/DT*inner(u2-u1, v)            *dx \
   + RHO*inner(sigma1+sigma2, grad_v)  *dx

# ------ BOUNDARY CONDITIONS ------ #
p_ur,p_uw,p_ut,p_pp = 0,1,2,3
BC1 = [
         DirichletBC(U.sub(p_ut), Constant(0), middle ),
         DirichletBC(U.sub(p_ur), Constant(0), bottom ),
         DirichletBC(U.sub(p_ut), Constant(0), bottom ),
         DirichletBC(U.sub(p_uw), Constant(0), bottom ),
         DirichletBC(U.sub(p_ur), Constant(0), walls  ),
         DirichletBC(U.sub(p_ut), Constant(0), walls  ),
         DirichletBC(U.sub(p_uw), Constant(0), walls  ),
         DirichletBC(U.sub(p_ur), Constant(0), upper  ),
         DirichletBC(U.sub(p_ut), ut_upper   , upper  ),
         DirichletBC(U.sub(p_uw), Constant(0), upper  ),
      ] # end - BC #

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
lowBound = project(Constant((a_min, -v_max, -v_max, -v_max, -v_max, -p_max, -p_max)), U)
uppBound = project(Constant((a_max, +v_max, +v_max, +v_max, +v_max, +p_max, +p_max)), U)

dF1 = derivative(F1, ans2)
nlProblem1 = NonlinearVariationalProblem(F1, ans2, BC1, dF1)
nlProblem1.set_bounds(lowBound,uppBound)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
nlSolver1.parameters["nonlinear_solver"] = "snes"

prm = nlSolver1.parameters["snes_solver"]
prm["error_on_nonconvergence"       ] = False
prm["solution_tolerance"            ] = 1.0E-16
prm["maximum_iterations"            ] = 15
prm["maximum_residual_evaluations"  ] = 20000
prm["sign"                          ] = "default"
prm["absolute_tolerance"            ] = 8.0E-12
prm["relative_tolerance"            ] = 6.0E-12
prm["linear_solver"                 ] = "mumps"
#prm["method"                        ] = "vinewtonssls"
#prm["line_search"                   ] = "bt"
#prm["preconditioner"                ] = "none"
#prm["report"                        ] = True
#prm["krylov_solver"                 ]
#prm["lu_solver"                     ]

#set_log_level(PROGRESS)

# ------ SAVE FILECONFIGURATIONS ------ #
vtk_uu   = File(foldername+'/velocity_stream.pvd')
vtk_ur   = File(foldername+'/velocity_radial.pvd')
vtk_ut   = File(foldername+'/velocity_tangencial.pvd')
vtk_uw   = File(foldername+'/velocity_axial.pvd')
vtk_pp   = File(foldername+'/pressure.pvd')

def save_results(ur,ut,uw,pp):
   ur_viz = project(ur , U_vel1); ur_viz.rename('velocity radial','velocity radial');           vtk_ur << ur_viz
   ut_viz = project(ut , U_vel1); ut_viz.rename('velocity tangencial','velocity tangencial');   vtk_ut << ut_viz
   uw_viz = project(uw , U_vel1); uw_viz.rename('pressure axial','pressure axial');             vtk_uw << uw_viz
   pp_viz = project(pp , U_prs ); pp_viz.rename('pressure','pressure intrinsic 2');             vtk_pp << pp_viz
   uu_viz = project(as_vector([ur,uw]) , U_vel2); uu_viz.rename('velocity','velocity');         vtk_uu << uu_viz

def plot_all():
   plot(a1,title='volume_fraction')
   plot(u1,title='velocity_intrinsic1')
   plot(u2,title='velocity_intrinsic2')
   plot(p1,title='pressure_intrinsic1')
   plot(p2,title='pressure_intrinsic2')
   interactive()


# ------ TRANSIENT SIMULATION ------ #
assign(ans1.sub(p_ur), project(Constant(0.0 ), U_vel1 ))
assign(ans1.sub(p_ut), project(Constant(0.0 ), U_vel1 ))
assign(ans1.sub(p_uw), project(Constant(0.0 ), U_vel1 ))
assign(ans1.sub(p_pp), project(Constant(0.0 ), U_prs  ))

assign(ans2.sub(p_ur), project(Constant(0.0 ), U_vel1 ))
assign(ans2.sub(p_ut), project(Constant(0.0 ), U_vel1 ))
assign(ans2.sub(p_uw), project(Constant(0.0 ), U_vel1 ))
assign(ans2.sub(p_pp), project(Constant(0.0 ), U_prs  ))

def RungeKutta4(ans_now, ans_nxt, nlSolver):
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
   ans_nxt.assign(project( ans_aux+ (RK1+RK2*2.0+RK3*2.0+RK4)/6.0, U))

count_iteration   = 0
while( count_iteration < TRANSIENT_MAX_ITE ):
   count_iteration = count_iteration +1
   #nlSolver1.solve()
   RungeKutta4(ans1, ans2, nlSolver1)
   residual = assemble(inner(ans2 -ans1,ans2 -ans1)*dx)
   print ('Iteration: {}'.format(count_iteration) )
   print ('Residual : {}'.format(residual) )
   ans1.assign(ans2)
   save_results(an1,an2,un1,un2,pn1,pn2)




