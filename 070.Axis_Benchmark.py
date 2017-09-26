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
mesh_res  = 150
mesh_P0   = 0.00
mesh_A    = 1.5
mesh_R    = 1.0             # Raio
mesh_H    = mesh_R*mesh_A   # Altura

# ------ TMIXER GEOMETRY PARAMETERS ------ #
cons_rho = 1.0E+3
cons_mu  = 1.0E-3
cons_ome = 0.99E-4
cons_u_00   = 0

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

ans   = Function(U)

ur,ut,uw,pp = split(ans)
vr,vt,vw,qq = TestFunctions(U)

uu = as_vector( [ur,ut,uw] )
vv = as_vector( [vr,vt,vw] )

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

div_uu  = div_cyl (uu)
grad_uu = grad_cyl(uu)
grad_vv = grad_cyl(vv)

OMEGA    = Constant(cons_ome  )
RHO      = Constant(cons_rho  )
MU       = Constant(cons_mu   )

sigma    = MU*(grad_uu+grad_uu.T) -pp*Identity(len(uu))

F1    = \
        div_uu*qq                         *dx \
      + inner(RHO*dot(uu,grad_uu), vv)  *dx \
      + inner(sigma, grad_vv)             *dx

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

dF1 = derivative(F1, ans)
nlProblem1 = NonlinearVariationalProblem(F1, ans, BC1, dF1)
# nlProblem1.set_bounds(lowBound,uppBound)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
nlSolver1.parameters["nonlinear_solver"] = "snes"

prm = nlSolver1.parameters["snes_solver"]
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

def save_results(Re):
   uu_viz = project(as_vector([ur,uw]) , U_vel2); uu_viz.rename('velocity','velocity');         vtk_uu << (uu_viz,Re)
   ur_viz = project(ur , U_vel1); ur_viz.rename('velocity radial','velocity radial');           vtk_ur << (ur_viz,Re)
   ut_viz = project(ut , U_vel1); ut_viz.rename('velocity tangencial','velocity tangencial');   vtk_ut << (ut_viz,Re)
   uw_viz = project(uw , U_vel1); uw_viz.rename('pressure axial','pressure axial');             vtk_uw << (uw_viz,Re)
   pp_viz = project(pp , U_prs ); pp_viz.rename('pressure','pressure intrinsic 2');             vtk_pp << (pp_viz,Re)

def plot_all():
   plot(ur,                   title='velocity_radial'    )
   plot(ut,                   title='velocity_tangencial')
   plot(uw,                   title='velocity_axial'     )
   plot(as_vector([ur,uw]),   title='velocity_surface'   )
   plot(pp,                   title='pressure'           )
   interactive()

# ------ TRANSIENT SIMULATION ------ #
# assign(ans.sub(p_ur), project(Constant(0.0 ), U_vel1 ))
# assign(ans.sub(p_ut), project(Constant(0.0 ), U_vel1 ))
# assign(ans.sub(p_uw), project(Constant(0.0 ), U_vel1 ))
# assign(ans.sub(p_pp), project(Constant(0.0 ), U_prs  ))

OMEGA.assign(8E-4)
nlSolver1.solve()
for val_omega in [ 1E-3+n*5E-5 for n in range(80)]:
   val_Re = (val_omega*mesh_R**2)*cons_rho/cons_mu
   print ('Solving for Re = {}'.format(val_Re))
   OMEGA.assign(val_omega)
   nlSolver1.solve()
   save_results(val_Re)

# plot_all()
# save_results()
# count_iteration   = 0
# while( count_iteration < TRANSIENT_MAX_ITE ):
#    count_iteration = count_iteration +1
#    nlSolver1.solve()
#    residual = assemble(inner(ans2 -ans1,ans2 -ans1)*dx)
#    print ('Iteration: {}'.format(count_iteration) )
#    print ('Residual : {}'.format(residual) )
#    save_results(an1,an2,un1,un2,pn1,pn2)


