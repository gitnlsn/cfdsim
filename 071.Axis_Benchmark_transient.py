'''

NELSON KENZO TAMASHIRO

19.07.2017

'''

# ------ LIBRARIES ------ #
from fenics    import *
from mshr      import *

# ------ SIMULATION PARAMETERS ------ #
foldername = 'results_AxisFlowBenchmark'

# ------ TMIXER GEOMETRY PARAMETERS ------ #
mesh_res  = 200
mesh_P0   = 0.00
mesh_A    = 2.0
mesh_R    = 1.0             # Raio
mesh_H    = mesh_R*mesh_A   # Altura

# ------ TMIXER GEOMETRY PARAMETERS ------ #
cons_rho = 1.0E+3
cons_mu  = 1.0E-3
cons_ome = 2.7E-3
cons_dt  = 1/(20*cons_ome)
cons_gg  = 0.0
cons_u_00   = 0

TRANSIENT_MAX_ITE = 2000

# ------ MESH ------ #
part1 = Rectangle(
   Point(mesh_P0, mesh_P0),
   Point(mesh_R , mesh_H )    )
channel = part1
mesh = generate_mesh(channel, mesh_res)

# ------ BOUNDARIES DEFINITION ------ #
bottom = '( (x[1]=='+str(mesh_P0)+') )'
middle = '( (x[0]=='+str(mesh_P0)+') )'
upper  = '( (x[1]=='+str(mesh_H )+') )'
walls  = '( (x[0]=='+str(mesh_R )+') )'

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
elem  = MixedElement([FE_u, FE_u, FE_u, FE_p  ])
U     = FunctionSpace(mesh, elem)

U_prs  = FunctionSpace(mesh, FE_p)

U_vel1 = FunctionSpace(mesh, FE_u )
U_vel2 = FunctionSpace(mesh, MixedElement([FE_u, FE_u          ]) )
U_vel3 = FunctionSpace(mesh, MixedElement([FE_u, FE_u, FE_u    ]) )

ans_next = Function(U)
ans_last = Function(U)

ur_n,ut_n,uw_n,pp_n = split(ans_next)
ur_l,ut_l,uw_l,pp_l = split(ans_last)
vr,vt,vw,qq = TestFunctions(U)

uu_n = as_vector( [ur_n,ut_n,uw_n] )
uu_l = as_vector( [ur_l,ut_l,uw_l] )
vv   = as_vector( [vr  ,vt  ,vw  ] )

dr,dw = 0,1
r = Expression('x[0]', degree=2)

def grad_cyl(uu):
   p_radial       = 0
   p_tangencial   = 1
   p_axial        = 2
   u_rad = uu[ p_radial    ]
   u_tan = uu[ p_tangencial]
   u_axe = uu[ p_axial     ]
   return as_tensor([   [Dx(u_rad,dr), -u_tan/r, Dx(u_rad,dw)],
                        [Dx(u_tan,dr), +u_rad/r, Dx(u_tan,dw)],
                        [Dx(u_axe,dr),        0, Dx(u_axe,dw)]  ])

def div_cyl(uu):
   p_radial       = 0
   p_tangencial   = 1
   p_axial        = 2
   u_rad = uu[ p_radial    ]
   u_tan = uu[ p_tangencial]
   u_axe = uu[ p_axial     ]
   return u_rad/r +Dx(u_rad,dr) +Dx(u_axe,dw)

def eyed(pp):
   return as_tensor([   [pp,          Constant(0), Constant(0)],
                        [Constant(0), pp,          Constant(0)],
                        [Constant(0), Constant(0), pp         ],  ])

OMEGA    = Constant( cons_ome )
RHO      = Constant( cons_rho )
MU       = Constant( cons_mu  )
gravity  = Constant(-cons_gg  ) 
DT       = Constant( cons_dt  )

GG = as_vector([ Constant(0), Constant(0), gravity ])

div_uu_n  = div_cyl (uu_n)
grad_uu_n = grad_cyl(uu_n)
grad_uu_l = grad_cyl(uu_l)
grad_vv   = grad_cyl(vv)

sigma_n  = MU*(grad_uu_n+grad_uu_n.T) -eyed(pp_n)
sigma_l  = MU*(grad_uu_l+grad_uu_l.T) -eyed(pp_l)

uu_df       = (uu_n -uu_l)
uu_md       = (uu_n +uu_l)*Constant(0.5)
grad_uu_md  = (grad_uu_n+grad_uu_l)*Constant(0.5)
sigma_md    = (sigma_n+sigma_l)*Constant(0.5)

F1    = inner(RHO*uu_df/DT,vv)                  *dx \
      + inner(RHO*dot(uu_md,grad_uu_md.T), vv)  *dx \
      + inner(sigma_md, grad_vv)                *dx \
      - inner(RHO*GG, vv)                       *dx \
      + div_uu_n*qq                             *dx

F2    = inner(RHO*dot(uu_n,grad_uu_n.T), vv)  *dx \
      + inner(sigma_n, grad_vv)                *dx \
      - inner(RHO*GG, vv)                       *dx \
      + div_uu_n*qq                             *dx

u_00     = Constant(cons_u_00)
ut_up    = Expression('omega*x[0]', omega=OMEGA, degree=2)

# ------ BOUNDARY CONDITIONS ------ #
p_ur,p_ut,p_uw,p_pp = 0,1,2,3
BC1 = [
         DirichletBC(U.sub(p_ur), u_00,   upper  ),
         DirichletBC(U.sub(p_ut), ut_up,  upper  ),
         DirichletBC(U.sub(p_uw), u_00,   upper  ),
         DirichletBC(U.sub(p_ur), u_00,   walls  ),
         DirichletBC(U.sub(p_ut), u_00,   walls  ),
         DirichletBC(U.sub(p_uw), u_00,   walls  ),
         DirichletBC(U.sub(p_ur), u_00,   bottom ),
         DirichletBC(U.sub(p_ut), u_00,   bottom ),
         DirichletBC(U.sub(p_uw), u_00,   bottom ),
         # DirichletBC(U.sub(p_ur), u_00, middle ),
         # DirichletBC(U.sub(p_ut), u_00, middle ),
      ] # end - BC #

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
# lowBound = project(Constant((a_min, -v_max, -v_max, -v_max, -v_max, -p_max, -p_max)), U)
# uppBound = project(Constant((a_max, +v_max, +v_max, +v_max, +v_max, +p_max, +p_max)), U)

# solve(F1==0, ans, BC1)

vorticity  = Dx(ur_n, dw) -Dx(uw_n, dr)

M  = inner(vorticity, vorticity) *dx

dF1 = derivative(F1, ans_next)
nlProblem1 = NonlinearVariationalProblem(F1, ans_next, BC1, dF1)
# nlProblem1.set_bounds(lowBound,uppBound)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
nlSolver1.parameters["nonlinear_solver"] = "snes"

dF2 = derivative(F2, ans_next)
nlProblem2 = NonlinearVariationalProblem(F2, ans_next, BC1, dF2)
# nlProblem2.set_bounds(lowBound,uppBound)
nlSolver2  = NonlinearVariationalSolver(nlProblem2)
nlSolver2.parameters["nonlinear_solver"] = "snes"

# OMEGA.assign(1E-4)
# nlSolver2.solve(1E-16)

prm1 = nlSolver1.parameters["snes_solver"]
prm2 = nlSolver2.parameters["snes_solver"]
for prm in [prm1, prm2]:
   prm["error_on_nonconvergence"       ] = False
   prm["solution_tolerance"            ] = 1.0E-16
   prm["maximum_iterations"            ] = 15
   prm["maximum_residual_evaluations"  ] = 20000
   prm["absolute_tolerance"            ] = 8.0E-12
   prm["relative_tolerance"            ] = 6.0E-12
   prm["linear_solver"                 ] = "mumps"
   # prm["sign"                          ] = "default"
   # prm["method"                        ] = "vinewtonssls"
   # prm["line_search"                   ] = "bt"
   # prm["preconditioner"                ] = "none"
   # prm["report"                        ] = True
   # prm["krylov_solver"                 ]
   # prm["lu_solver"                     ]

#set_log_level(PROGRESS)

# ------ SAVE FILECONFIGURATIONS ------ #
vtk_uu   = File(foldername+'/velocity_surface.pvd')
vtk_ur   = File(foldername+'/velocity_radial.pvd')
vtk_ut   = File(foldername+'/velocity_tangencial.pvd')
vtk_uw   = File(foldername+'/velocity_axial.pvd')
vtk_pp   = File(foldername+'/pressure.pvd')

def save_results(uu,pp,Re):
   p_radial       = 0
   p_tangencial   = 1
   p_axial        = 2
   u_rad = uu[ p_radial    ]
   u_tan = uu[ p_tangencial]
   u_axe = uu[ p_axial     ]
   p_prs = pp
   uu_viz = project(as_vector([u_rad,u_axe]), U_vel2); uu_viz.rename('velocity','velocity');       vtk_uu << (uu_viz,Re)
   ut_viz = project(u_tan , U_vel1); ut_viz.rename('velocity tangencial','velocity tangencial');   vtk_ut << (ut_viz,Re)
   pp_viz = project(p_prs , U_prs ); pp_viz.rename('pressure','pressure intrinsic 2');             vtk_pp << (pp_viz,Re)
   # ur_viz = project(u_rad , U_vel1); ur_viz.rename('velocity radial','velocity radial');           vtk_ur << (ur_viz,Re)
   # uw_viz = project(u_axe , U_vel1); uw_viz.rename('pressure axial','pressure axial');             vtk_uw << (uw_viz,Re)

def plot_all(uu,pp):
   p_radial       = 0
   p_tangencial   = 1
   p_axial        = 2
   u_rad = uu[ p_radial    ]
   u_tan = uu[ p_tangencial]
   u_axe = uu[ p_axial     ]
   p_prs = pp
   plot(as_vector([u_rad,axial]),title='velocity_surface'   )
   plot(u_rad,                   title='velocity_radial'    )
   plot(u_tan,                   title='velocity_tangencial')
   plot(u_axe,                   title='velocity_axial'     )
   plot(p_prs,                   title='pressure'           )
   interactive()

def RungeKutta2(ans_now, ans_nxt, nlSolver):
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
   # return RungeKutta estimate
   ans_now.assign(ans_aux)
   ans_nxt.assign(project( ans_aux+ RK2, U))

# ------ TRANSIENT SIMULATION ------ #
# assign(ans.sub(p_ur), project(Constant(0.0 ), U_vel1 ))
# assign(ans.sub(p_ut), project(Constant(0.0 ), U_vel1 ))
# assign(ans.sub(p_uw), project(Constant(0.0 ), U_vel1 ))
# assign(ans.sub(p_pp), project(Constant(0.0 ), U_prs  ))

# gravity.assign(0.0E-3)
# for val_omega in [ 5E-4, 8E-4, 1.5E-3 ]: # cons_ome: desired omega value
#    print ('Continuation Method val_omega: {}'.format(val_omega))
#    OMEGA.assign(val_omega)
#    nlSolver2.solve()

# ans_last.assign(ans_next)

OMEGA.assign(cons_ome)
count_iteration   = 0
val_time = 0
while( count_iteration < TRANSIENT_MAX_ITE ):
   count_iteration = count_iteration +1
   val_time = val_time + cons_dt
   print ('Iteration: {}'.format(count_iteration) )
   #RungeKutta2(ans_last, ans_next, nlSolver1)
   nlSolver1.solve()
   residual = assemble(inner(ans_next -ans_last,ans_next -ans_last)*dx)
   print ('Residual : {}'.format(residual) )
   ans_last.assign(ans_next)
   save_results(uu_n,pp_n,val_time)


