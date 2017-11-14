'''

 

19.07.2017

'''

# ------ LIBRARIES ------ #
from fenics    import *
from mshr      import *
from math      import pi, tan

# ------ SIMULATION PARAMETERS ------ #
foldername = 'results_AxisFlowBenchmark'

# ------ TMIXER GEOMETRY PARAMETERS ------ #
mesh_res = 100
mesh_P0  = 0.0
mesh_Dc  = 102.0E-3
mesh_Di  =   8.7E-3
mesh_Dv  =  23.7E-3
mesh_Ds  =  20.0E-3
mesh_th  =  10.0        *pi/180.0 #graus
mesh_Lv  =  60.0E-3
mesh_Lc  = 100.4E-3
mesh_es  =   1.0E-3

# ------ TMIXER GEOMETRY PARAMETERS ------ #
cons_rho    = 1.0E+3             # densidade agua
cons_rhoP   = 5.0E+3             # densidade particulado
cons_mu     = 1.0E-3             # viscosidade agua
cons_ome    = 1.5E-3             # velocidade angular tampa
cons_dt     = 1.0E-3             # intervalo de tempo
cons_gg     = 0.0#9.8                # gravidade
cons_vin    = 1.00E-4
cons_dp     = 1.0E-6            # tamanho da particula: 3.35; 10.25; 19.37; 28.27; 38; 63
cons_u_00   = 0                  # velocidade nula

TRANSIENT_MAX_ITE = 10000

# ------ MESH ------ #
part1 = Polygon([
   Point(mesh_P0,                mesh_P0                                                      ),
   Point(mesh_Ds/2.0,            mesh_P0                                                      ),
   Point(mesh_Ds/2.0,            mesh_Lv/2.0                                                  ),
   Point(mesh_Dc/2.0,            mesh_Lv/2.0 +(mesh_Dc -mesh_Ds)/(2.0*tan(mesh_th))           ),
   Point(mesh_Dc/2.0,            mesh_Lv/2.0 +(mesh_Dc -mesh_Ds)/(2.0*tan(mesh_th)) +mesh_Lc  ),
   Point(mesh_Dv/2.0 +mesh_es,   mesh_Lv/2.0 +(mesh_Dc -mesh_Ds)/(2.0*tan(mesh_th)) +mesh_Lc  ),
   Point(mesh_Dv/2.0 +mesh_es,                (mesh_Dc -mesh_Ds)/(2.0*tan(mesh_th)) +mesh_Lc  ),
   Point(mesh_Dv/2.0,                         (mesh_Dc -mesh_Ds)/(2.0*tan(mesh_th)) +mesh_Lc  ),
   Point(mesh_Dv/2.0,            mesh_Lv/2.0 +(mesh_Dc -mesh_Ds)/(2.0*tan(mesh_th)) +mesh_Lc  ),
   Point(mesh_P0,                mesh_Lv/2.0 +(mesh_Dc -mesh_Ds)/(2.0*tan(mesh_th)) +mesh_Lc  ),       ])
channel = part1
mesh = generate_mesh(channel, mesh_res)
# plot(mesh); interactive()

mesh_tol = mesh.hmax()/3.0

# ------ BOUNDARIES DEFINITION ------ #
outlet1  = '(  on_boundary && (x[0]< '+str(mesh_Dv/2.0 +mesh_tol)                            +') '\
         + '&& (x[1]>='+str(mesh_Lv/2.0 +(mesh_Dc -mesh_Ds)/(2.0*tan(mesh_th)) +mesh_Lc -mesh_tol )            +')  )'
outlet2  = '(  near(x[1],'+str(mesh_P0)                                                      +') '\
         + '&& (x[0]< '+str(mesh_Ds/2.0 +mesh_tol )                                                            +')  )'
inlet    = '(  on_boundary && (x[0]=='+str(mesh_Dc/2.0)                                      +') '\
         + '&& (x[1]>='+str(mesh_Lv/2.0 +(mesh_Dc -mesh_Ds)/(2.0*tan(mesh_th)) +mesh_Lc -mesh_tol -mesh_Di )   +')  )'
middle   = '(   (x[0]=='+str(mesh_P0)+')  )'
walls    = 'on_boundary && (x[0]>'+str(mesh_P0)+')' \
         + '&& !( (x[1]=='+str(mesh_P0)+') && (x[0]<'+str(mesh_Ds/2.0)+') )'\
         + '&& !( (x[1]>='+str(mesh_Lv/2.0 +(mesh_Dc -mesh_Ds)/(2.0*tan(mesh_th)) +mesh_Lc -mesh_tol)+') '\
         +        '&& (x[0]<'+str(mesh_Dv/2.0)+') )'\

ds_outlet1, ds_outlet2, ds_middle, ds_walls, ds_inlet = 1,2,3,4,5

boundaries     = FacetFunction ('size_t', mesh)
side_walls     = CompiledSubDomain( walls    )
side_middle    = CompiledSubDomain( middle   )
side_outlet1   = CompiledSubDomain( outlet1  )
side_outlet2   = CompiledSubDomain( outlet2  )
side_inlet     = CompiledSubDomain( inlet    )
boundaries.set_all(0)
side_walls.mark   (boundaries, ds_walls   )
side_middle.mark  (boundaries, ds_middle  )
side_outlet1.mark (boundaries, ds_outlet1 )
side_outlet2.mark (boundaries, ds_outlet2 )
side_inlet.mark   (boundaries, ds_inlet   )
ds = Measure( 'ds', subdomain_data=boundaries )

dx_inlet = 1
domain   = CellFunction  ('size_t', mesh)
inlet    = '('                                                                                           \
         +  '     (x[0]<'+str(mesh_Dc/2.0          +mesh_tol)+')'                                                 \
         +  '  && (x[0]>'+str(mesh_Dc/2.0 -mesh_Di -mesh_tol)+')'                                                 \
         +  '  && (x[1]<'+str(mesh_Lv/2.0 +(mesh_Dc -mesh_Ds)/(2.0*tan(mesh_th)) +mesh_Lc          +mesh_tol)+')' \
         +  '  && (x[1]>'+str(mesh_Lv/2.0 +(mesh_Dc -mesh_Ds)/(2.0*tan(mesh_th)) +mesh_Lc -mesh_Di -mesh_tol)+')' \
         +' )'
CompiledSubDomain( inlet ).mark( domain, dx_inlet )
dx = Measure('dx', subdomain_data=domain )

# plot(domain); interactive()

# ------ VARIATIONAL FORMULATION ------ #
FE_u  = FiniteElement('P', 'triangle', 2)
FE_p  = FiniteElement('P', 'triangle', 1)
FE_a  = FiniteElement('P', 'triangle', 1)
elem  = MixedElement([FE_u, FE_u, FE_u, FE_p])
U     = FunctionSpace(mesh, elem)

U_prs  = FunctionSpace(mesh, FE_p)
U_con  = FunctionSpace(mesh, FE_a)

U_vel1 = FunctionSpace(mesh, FE_u )
U_vel2 = FunctionSpace(mesh, MixedElement([FE_u, FE_u          ]) )
U_vel3 = FunctionSpace(mesh, MixedElement([FE_u, FE_u, FE_u    ]) )

ans_next = Function(U)
ans_last = Function(U)

aa_n  = Function(U_con)
aa_l  = Function(U_con)
bb    = TestFunction(U_con)

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
   return as_tensor([   [Dx(u_rad,dr),    -u_tan/r, Dx(u_rad,dw)],
                        [Dx(u_tan,dr),    +u_rad/r, Dx(u_tan,dw)],
                        [Dx(u_axe,dr),           0, Dx(u_axe,dw)]    ])

def grad_scalar(aa):
   return as_vector([ Dx(aa,dr), 0, Dx(aa,dw) ])

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
RHO_p    = Constant( cons_rhoP)
MU       = Constant( cons_mu  )
gravity  = Constant(-cons_gg  )
DT       = Constant( cons_dt  )
dP       = Constant( cons_dp  )
N2       = Constant( 2.0      )
VI_IN    = Constant( cons_vin )
AA_IN    = Constant( 1.0      )
PI       = Constant( pi       )
GG = as_vector([ Constant(0), Constant(0), gravity ])

div_uu_n  = div_cyl (uu_n)
grad_uu_n = grad_cyl(uu_n)
grad_uu_l = grad_cyl(uu_l)
grad_vv   = grad_cyl(vv)
grad_aa_n = grad_scalar(aa_n)
grad_aa_l = grad_scalar(aa_l)
grad_bb   = grad_scalar(bb)

sigma_n  = MU*(grad_uu_n+grad_uu_n.T) -eyed(pp_n)
sigma_l  = MU*(grad_uu_l+grad_uu_l.T) -eyed(pp_l)

uu_df       = (uu_n -uu_l)
aa_df       = (aa_n -aa_l)
uu_md       = Constant(0.5)*(uu_n+uu_l)
ur_md       = Constant(0.5)*(ur_n+ur_l)
ut_md       = Constant(0.5)*(ut_n+ut_l)
uw_md       = Constant(0.5)*(uw_n+uw_l)
aa_md       = Constant(0.5)*(aa_n+aa_l)
pp_md       = Constant(0.5)*(pp_n+pp_l)
grad_uu_md  = Constant(0.5)*(grad_uu_n+grad_uu_l)
grad_aa_md  = Constant(0.5)*(grad_aa_n+grad_aa_l)
sigma_md    = Constant(0.5)*(sigma_n+sigma_l)

NORMAL      = FacetNormal(mesh)
# SIGMA_DS    = as_tensor([  [-RHO*gravity*Constant(-mesh_H), Constant(0)],
#                            [Constant(0),                    Constant(0)],
#                            [Constant(0), -RHO*gravity*Constant(-mesh_H)],  ] )

vorticity  = as_vector([  -Dx(ut_md,dw),
                           Dx(ur_n, dw) -Dx(uw_n, dr), 
                           ut_n/r +Dx(ut_n, dr)          ])

u_pmi = dP*(RHO*GG +grad_scalar(pp_n))

F1    = \
      + inner(RHO*dot(uu_n,grad_uu_n.T), vv)    *dx \
      + inner(sigma_n, grad_vv)                 *dx \
      + div_uu_n*qq                             *dx \
      - VI_IN/(2*PI*r)*qq                       *dx(dx_inlet) \
      - VI_IN*VI_IN/(2*PI*r)*vt                 *dx(dx_inlet) \
      # + inner(RHO*uu_df/DT,vv)                  *dx \
      # + inner(aa_df/DT, bb)                     *dx \
      # + div_cyl((uu_md)*aa_md) *bb              *dx \
      # + inner(grad_aa_md, grad_bb)*Constant(1E-8)*dx \
      # + inner(dot(uu_md, grad_aa_md),      dot(uu_md, grad_bb))   *DT/N2 *dx
      # - inner(RHO*GG, vv)                       *dx \
      # - inner(dot(SIGMA_DS, NORMAL), vv)        *ds(ds_inlet) \
      # + inner(grad_scalar(pp_md),          dot(uu_md,grad_vv.T))  *DT/N2 *dx \
      # + inner(RHO*dot(uu_md,grad_uu_md.T), dot(uu_md,grad_vv.T))  *DT/N2 *dx \

F2    = inner(aa_df/DT, bb)                                                         *dx \
      + div_cyl((uu_n+u_pmi)*aa_md) *bb                                             *dx \
      + inner(grad_aa_md, grad_bb)*Constant(1E-8)                                   *dx \
      - VI_IN*AA_IN/(2*PI*r)*bb                                                     *dx(dx_inlet)
      # + inner(dot(uu_n+u_pmi, grad_aa_md),      dot(uu_n+u_pmi, grad_bb))    *DT/N2 *dx \

u_00     = Constant( cons_u_00 )
ur_in    = Constant(-cons_vin  )
ut_in    = Constant( cons_vin*10.0  )
ut_up    = Expression('omega*x[0]', omega=OMEGA, degree=2)
a_in1    = Constant(1.0)
a_in2    = Constant(0.0)
a_in     = Constant(0.5)

# ------ BOUNDARY CONDITIONS ------ #
p_ur,p_ut,p_uw,p_pp,p_aa = 0,1,2,3,4
BC1 = [
         DirichletBC(U.sub(p_ur), u_00,   walls  ),
         DirichletBC(U.sub(p_ut), u_00,   walls  ),
         DirichletBC(U.sub(p_uw), u_00,   walls  ),
         # DirichletBC(U.sub(p_ur), u_00,   middle ),
         # DirichletBC(U.sub(p_ut), u_00,   middle ),
         # DirichletBC(U.sub(p_aa), a_in,   inlet  ),
         # DirichletBC(U.sub(p_ut), ut_in,  inlet  ),
         # DirichletBC(U.sub(p_ur), ur_in,  inlet  ),
         # DirichletBC(U.sub(p_uw), u_00,   inlet  ),
      ] # end - BC #

BC2 = [
         # DirichletBC(U.sub(p_ur), u_00,   walls  ),
         # DirichletBC(U.sub(p_ut), u_00,   walls  ),
         # DirichletBC(U.sub(p_uw), u_00,   walls  ),
         # DirichletBC(U.sub(p_ur), u_00,   middle ),
         # DirichletBC(U.sub(p_ut), u_00,   middle ),
         # DirichletBC(U_con, a_in,   inlet  ),
         # DirichletBC(U.sub(p_ut), ut_in,  inlet  ),
         # DirichletBC(U.sub(p_ur), ur_in,  inlet  ),
         # DirichletBC(U.sub(p_uw), u_00,   inlet  ),
      ] # end - BC #

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
# lowBound = project(Constant((a_min, -v_max, -v_max, -v_max, -v_max, -p_max, -p_max)), U)
# uppBound = project(Constant((a_max, +v_max, +v_max, +v_max, +v_max, +p_max, +p_max)), U)

# solve(F1==0, ans, BC1)

dF1 = derivative(F1, ans_next)
nlProblem1 = NonlinearVariationalProblem(F1, ans_next, BC1, dF1)
# nlProblem1.set_bounds(lowBound,uppBound)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
nlSolver1.parameters["nonlinear_solver"] = "snes"

dF2 = derivative(F2, aa_n)
nlProblem2 = NonlinearVariationalProblem(F2, aa_n, BC2, dF2)
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
   prm["maximum_iterations"            ] = 10
   prm["maximum_residual_evaluations"  ] = 20000
   prm["absolute_tolerance"            ] = 8.0E-14
   prm["relative_tolerance"            ] = 4.0E-15
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
vtk_ut   = File(foldername+'/velocity_tangencial.pvd')
vtk_pp   = File(foldername+'/pressure.pvd')
vtk_aa   = File(foldername+'/concentration.pvd')
# vtk_ur   = File(foldername+'/velocity_radial.pvd')
# vtk_uw   = File(foldername+'/velocity_axial.pvd')

def save_results(ans,ans_con,Re):
   p_radial          = 0
   p_tangencial      = 1
   p_axial           = 2
   p_pressure        = 3
   # p_concentration   = 4
   u_rad = ans[ p_radial         ]
   u_tan = ans[ p_tangencial     ]
   u_axe = ans[ p_axial          ]
   p_prs = ans[ p_pressure       ]
   a_con = ans_con
   uu_viz = project(as_vector([u_rad,u_axe]), U_vel2); uu_viz.rename('velocity','velocity');       vtk_uu << (uu_viz,Re)
   ut_viz = project(u_tan , U_vel1); ut_viz.rename('velocity tangencial','velocity tangencial');   vtk_ut << (ut_viz,Re)
   pp_viz = project(p_prs , U_prs ); pp_viz.rename('pressure','pressure');                         vtk_pp << (pp_viz,Re)
   aa_viz = project(a_con , U_con ); aa_viz.rename('concentration','concentration');               vtk_aa << (aa_viz,Re)
   # ur_viz = project(u_rad , U_vel1); ur_viz.rename('velocity radial','velocity radial');           vtk_ur << (ur_viz,Re)
   # uw_viz = project(u_axe , U_vel1); uw_viz.rename('pressure axial','pressure axial');             vtk_uw << (uw_viz,Re)

def plot_all(ans,ans_con):
   p_radial          = 0
   p_tangencial      = 1
   p_axial           = 2
   p_pressure        = 3
   # p_concentration   = 4
   u_rad = ans[ p_radial         ]
   u_tan = ans[ p_tangencial     ]
   u_axe = ans[ p_axial          ]
   p_prs = ans[ p_pressure       ]
   a_con = ans_con
   plot(as_vector([u_rad,u_axe]),title='velocity_surface'   )
   plot(u_tan,                   title='velocity_tangencial')
   plot(p_prs,                   title='pressure'           )
   plot(a_con,                   title='concentration'      )
   interactive()
   # plot(u_axe,                   title='velocity_axial'     )
   # plot(u_rad,                   title='velocity_radial'    )

def RungeKutta2(ans_now, ans_nxt, nlSolver, U):
   ans_aux  = Function(U)
   ans_aux.assign(ans_now)
   RK1      = Function(U)
   RK2      = Function(U)
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

def RungeKutta4(ans_now, ans_nxt, nlSolver, U):
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

# ------ TRANSIENT SIMULATION ------ #
# assign(ans_next.sub(p_ur), project(Constant(0.0 ), U_vel1 ))
# assign(ans_next.sub(p_ut), project(Constant(0.0 ), U_vel1 ))
# assign(ans_next.sub(p_uw), project(Constant(0.0 ), U_vel1 ))
# assign(ans_next.sub(p_pp), project(Constant(0.0 ), U_prs  ))
# assign(ans_next.sub(p_aa), project(Constant(0.0 ), U_con  ))

# assign(ans_last.sub(p_ur), project(Constant(0.0 ), U_vel1 ))
# assign(ans_last.sub(p_ut), project(Constant(0.0 ), U_vel1 ))
# assign(ans_last.sub(p_uw), project(Constant(0.0 ), U_vel1 ))
# assign(ans_last.sub(p_pp), project(Constant(0.0 ), U_prs  ))
# assign(ans_last.sub(p_aa), project(Constant(0.0 ), U_con  ))

# gravity.assign(0.0E-3)                   # benchmark problem
# for val_omega in [ 5E-4, 8E-4, 1.5E-3 ]: # cons_ome: desired omega value
#    print ('Continuation Method val_omega: {}'.format(val_omega))
#    OMEGA.assign(val_omega)
#    nlSolver1.solve()

# ans_last.assign(ans_next)

for u_val in [ 10**(-2+exp*0.2) for exp in range(11) ]:
   print ('Velocity: {}'.format(u_val))
   VI_IN.assign(  u_val  )
   nlSolver1.solve()

save_results(ans_next, aa_n, 0.0)

count_iteration   = 0
val_time = 0
while( count_iteration < TRANSIENT_MAX_ITE ):
   count_iteration = count_iteration +1
   val_time = val_time + cons_dt
   print ('Iteration: {}'.format(count_iteration) )
   # RungeKutta2(ans_last, ans_next, nlSolver1, U     )
   RungeKutta2(aa_l,     aa_n,     nlSolver2, U_con )
   #nlSolver1.solve()
   residual = assemble(inner(aa_n -aa_l,aa_n -aa_l)*dx)
   print ('Residual : {}'.format(residual) )
   ans_last.assign(ans_next)
   aa_l.assign(aa_n)
   save_results(ans_next, aa_n, val_time)

