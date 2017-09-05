'''

NELSON KENZO TAMASHIRO

19.07.2017

'''

# ------ LIBRARIES ------ #
from fenics    import *
from mshr      import *

# ------ SIMULATION PARAMETERS ------ #
foldername = 'results_BaerNunziato'

# ------ TMIXER GEOMETRY PARAMETERS ------ #
mesh_res  = 50
mesh_P0   = 0.0
mesh_DD   = 0.03          # largura
mesh_L    = 0.10          # comprimento
mesh_Cx   = mesh_L *0.25       # initial circle
mesh_Cy   = mesh_DD*0.5
mesh_Rad  = mesh_DD*0.2
mesh_tol  = mesh_DD*0.01

# ------ TMIXER GEOMETRY PARAMETERS ------ #
cons_dt  = 0.0001
cons_kk  = 1.0E+2
cons_rh1 = 1.0E+3
cons_rh2 = 2.0E+3
cons_mu1 = 1.0E-3
cons_mu2 = 1.1E-3
cons_vin = 1.0E-2
cons_g   = 9.8E-0
cons_dl  = 1.0E-3

a_min = 0
a_max = 1
v_max = cons_vin*50
p_max = 1.0E5

GENERAL_TOL = 1E-6
TRANSIENT_MAX_ITE = 1000

# ------ MESH ------ #
part1 = Rectangle(
   Point(mesh_P0, mesh_P0),
   Point(mesh_L , mesh_DD)    )
part2 = Circle(
   Point(mesh_Cx, mesh_Cy),
   mesh_Rad                   )
channel = part1 #-part2
mesh = generate_mesh(channel, mesh_res)

# ------ BOUNDARIES DEFINITION ------ #
inlet  = '( (x[0]=='+str(mesh_P0)+') && (x[1]>='+str(0.0*mesh_DD)+') && (x[1]<='+str(1.0*mesh_DD)+') )'
outlet = '( (x[0]=='+str(mesh_L )+') && (x[1]>='+str(0.0*mesh_DD)+') && (x[1]<='+str(1.0*mesh_DD)+') )'
walls  = '( (x[1]=='+str(mesh_P0)+') || (x[1]=='+str(mesh_DD)+') )'

ds_inlet, ds_walls, ds_outlet = 1,2,3

boundaries     = FacetFunction ('size_t', mesh)
side_inlet     = CompiledSubDomain( inlet    )
side_outlet    = CompiledSubDomain( outlet   )
side_walls     = CompiledSubDomain( walls    )
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
U_vol = FunctionSpace(mesh, FE_a)
U     = FunctionSpace(mesh, MixedElement([FE_a, FE_u, FE_u, FE_p, FE_p]) )

ansn  = Function(U)
ansm  = Function(U)

an1,un1,un2,pn1,pn2 = split(ansn)
am1,um1,um2,pm1,pm2 = split(ansm)

b1,v1,v2,q1,q2 = TestFunctions(U)

N1       = Constant(1         )
KK       = Constant(cons_kk   )
DT       = Constant(cons_dt   )
RH1      = Constant(cons_rh1  )
RH2      = Constant(cons_rh2  )
MU1      = Constant(cons_mu1  )
MU2      = Constant(cons_mu2  )
u_inlet  = Constant(cons_vin  )
dl       = Constant(cons_dl   )
GG       = as_vector(   [Constant(0), Constant(-cons_g)]  )
HH       = as_vector(   [Constant(0), Expression('x[1]',degree=2)]   )
NN       = FacetNormal(mesh)

u_in  = as_vector(   [Constant(cons_vin), Constant(0)]   )
u_wl  = as_vector(   [Constant(0),        Constant(0)]   )

he = CellSize(mesh)

an2 = N1 -an1
am2 = N1 -am1

a1_md  = (an1+am1)*0.5
a2_md  = (an2+am2)*0.5
u1_md  = (un1+um1)*0.5
u2_md  = (un2+um2)*0.5
p1_md  = (pn1+pm1)*0.5
p2_md  = (pn2+pm2)*0.5

a1_df  = (an1-am1)/DT
a2_df  = (an2-am2)/DT
au1df  = (an1*un1-am1*um1)/DT
au2df  = (an2*un2-am2*um2)/DT
u1_df  = (un1-um1)/DT
u2_df  = (un2-um2)/DT
p1_df  = (pn1-pm1)/DT
p2_df  = (pn2-pm2)/DT

u_intn = (un1*RH1*an1 + un2*RH2*an2)/(RH1*an1 + RH2*an2)
u_intm = (um1*RH1*am1 + um2*RH2*am2)/(RH1*am1 + RH2*am2)
p_int  = p1_md*a1_md + p2_md*a2_md

#F12   = MU1*MU2/(MU1+MU2)*(u2_md -u1_md)/dl +(an1*an2*grad(pn2 -pn1)+am1*am2*grad(pm2 -pm1))*0.5
#F21   = MU1*MU2/(MU1+MU2)*(u1_md -u2_md)/dl +(an1*an2*grad(pn1 -pn2)+am1*am2*grad(pm1 -pm2))*0.5

F12   = MU1*MU2/(MU1+MU2)*(u2_md -u1_md)/dl +p_int*grad(a1_md)
F21   = MU1*MU2/(MU1+MU2)*(u1_md -u2_md)/dl +p_int*grad(a2_md)

F1 = a1_df *b1                                     *dx \
   + inner(u_intn, grad(an1))*0.5 *b1              *dx \
   + inner(u_intm, grad(am1))*0.5 *b1              *dx \
   + an1*an2*(pn1 -pn2)*b1*0.5*KK                  *dx \
   + am1*am2*(pm1 -pm2)*b1*0.5*KK                  *dx \
   \
   + a1_df *q1                                     *dx \
   + div(an1*un1)*0.5 *q1                          *dx \
   + div(am1*um1)*0.5 *q1                          *dx \
   \
   + a2_df *q2                                     *dx \
   + div(an2*un2)*0.5 *q2                          *dx \
   + div(am2*um2)*0.5 *q2                          *dx \
   \
   + RH1*inner(au1df,v1)                           *dx \
   + RH1*inner(div(an1*outer(un1,un1)),v1)*0.5     *dx \
   + RH1*inner(div(am1*outer(um1,um1)),v1)*0.5     *dx \
   - inner(F12,v1)                                 *dx \
   - inner(RH1*a1_md*GG,v1)                        *dx \
   \
   + RH2*inner(au2df,v2)                           *dx \
   + RH2*inner(div(an2*outer(un2,un2)),v2)*0.5     *dx \
   + RH2*inner(div(am2*outer(um2,um2)),v2)*0.5     *dx \
   - inner(F21,v2)                                 *dx \
   - inner(RH2*a2_md*GG,v2)                        *dx \
   \

# ------ BOUNDARY CONDITIONS ------ #
p_out1 = Expression('-rho*g*x[1]', rho=0.5*(cons_rh1+cons_rh2), g=cons_g, degree=1)
p_out2 = Expression('-rho*g*x[1]', rho=0.5*(cons_rh1+cons_rh2), g=cons_g, degree=1)

p_aa,p_u1,p_u2,p_p1,p_p2 = 0,1,2,3,4
BC1 = [
         DirichletBC(U.sub(p_aa), Constant(     0.5   ), inlet   ),
         DirichletBC(U.sub(p_u1), Constant((u_inlet,0)), inlet   ),
         DirichletBC(U.sub(p_u2), Constant((u_inlet,0)), inlet   ),
         #DirichletBC(U.sub(p_u1), Constant((0,0)), inlet    ),
         #DirichletBC(U.sub(p_u2), Constant((0,0)), inlet    ),
         DirichletBC(U.sub(p_u1), Constant((0,0)), walls    ),
         DirichletBC(U.sub(p_u2), Constant((0,0)), walls    ),
         #DirichletBC(U.sub(p_u1), Constant((0,0)), outlet   ),
         #DirichletBC(U.sub(p_u2), Constant((0,0)), outlet   ),
         DirichletBC(U.sub(p_p1), p_out1, outlet   ),
         DirichletBC(U.sub(p_p2), p_out2, outlet   ),
      ] # end - BC #

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
lowBound = project(Constant((a_min, -v_max, -v_max, -v_max, -v_max, -p_max, -p_max)), U)
uppBound = project(Constant((a_max, +v_max, +v_max, +v_max, +v_max, +p_max, +p_max)), U)

dF1 = derivative(F1, ansn)
nlProblem1 = NonlinearVariationalProblem(F1, ansn, BC1, dF1)
nlProblem1.set_bounds(lowBound,uppBound)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
nlSolver1.parameters["nonlinear_solver"] = "snes"

prm = nlSolver1.parameters["snes_solver"]
prm["error_on_nonconvergence"       ] = False
prm["solution_tolerance"            ] = 1.0E-16
prm["maximum_iterations"            ] = 15
prm["maximum_residual_evaluations"  ] = 20000
prm["sign"                          ] = "default"
prm["absolute_tolerance"            ] = 6.0E-10
prm["relative_tolerance"            ] = 6.0E-10
prm["linear_solver"                 ] = "mumps"
#prm["method"                        ] = "vinewtonssls"
#prm["line_search"                   ] = "bt"
#prm["preconditioner"                ] = "none"
#prm["report"                        ] = True
#prm["krylov_solver"                 ]
#prm["lu_solver"                     ]

#set_log_level(PROGRESS)

# ------ SAVE FILECONFIGURATIONS ------ #
vtk_aa  = File(foldername+'/volume_fraction.pvd')
vtk_ui1 = File(foldername+'/velocity_intrinsic1.pvd')
vtk_ui2 = File(foldername+'/velocity_intrinsic2.pvd')
vtk_pi1 = File(foldername+'/pressure_intrinsic1.pvd')
vtk_pi2 = File(foldername+'/pressure_intrinsic2.pvd')
vtk_u1  = File(foldername+'/velocity_mean1.pvd')
vtk_u2  = File(foldername+'/velocity_mean2.pvd')
vtk_p1  = File(foldername+'/pressure_mean1.pvd')
vtk_p2  = File(foldername+'/pressure_mean2.pvd')
vtk_dp  = File(foldername+'/pressure_difference.pvd')

def save_results(a1,a2,u1,u2,p1,p2):
   aa_viz  = project(a1   , U_vol); aa_viz.rename('Fraction','Fraction');  vtk_aa  << aa_viz
   ui1_viz = project(u1   , U_vel); ui1_viz.rename('velocity intrinsic 1','velocity intrinsic 1'); vtk_ui1 << ui1_viz
   ui2_viz = project(u2   , U_vel); ui2_viz.rename('velocity intrinsic 2','velocity intrinsic 2'); vtk_ui2 << ui2_viz
   pi1_viz = project(p1   , U_prs); pi1_viz.rename('pressure intrinsic 1','pressure intrinsic 1'); vtk_pi1 << pi1_viz
   pi2_viz = project(p2   , U_prs); pi2_viz.rename('pressure intrinsic 2','pressure intrinsic 2'); vtk_pi2 << pi2_viz
   u1_viz  = project(u1*a1, U_vel); u1_viz.rename('velocity mean 1','velocity mean 1');  vtk_u1  << u1_viz
   u2_viz  = project(u2*a2, U_vel); u2_viz.rename('velocity mean 2','velocity mean 2');  vtk_u2  << u2_viz
   p1_viz  = project(p1*a1, U_prs); p1_viz.rename('pressure mean 1','pressure mean 1');  vtk_p1  << p1_viz
   p2_viz  = project(p2*a2, U_prs); p2_viz.rename('pressure mean 2','pressure mean 2');  vtk_p2  << p2_viz
   dp_viz  = project(p1-p2, U_prs); dp_viz.rename('pressure difference','pressure difference');  vtk_dp  << dp_viz

def plot_all():
   plot(a1,title='volume_fraction')
   plot(u1,title='velocity_intrinsic1')
   plot(u2,title='velocity_intrinsic2')
   plot(p1,title='pressure_intrinsic1')
   plot(p2,title='pressure_intrinsic2')
   interactive()


# ------ TRANSIENT SIMULATION ------ #
count_iteration   = 0

#p1_init = project( -RH1*inner(GG,HH), U_prs)
#p2_init = project( -RH2*inner(GG,HH), U_prs)

assign(ansn.sub(p_aa), project(Constant(0.5         ), U_vol))
assign(ansn.sub(p_u1), project(Constant((0,0)), U_vel))
assign(ansn.sub(p_u2), project(Constant((0,0)), U_vel))
assign(ansn.sub(p_p1), project(Constant(0.0E-1      ), U_prs))
assign(ansn.sub(p_p2), project(Constant(0.0E-1      ), U_prs))
#assign(ansn.sub(p_p1), p1_init)
#assign(ansn.sub(p_p2), p2_init)

assign(ansm.sub(p_aa), project(Constant(0.5         ), U_vol))
assign(ansm.sub(p_u1), project(Constant((0,0)), U_vel))
assign(ansm.sub(p_u2), project(Constant((0,0)), U_vel))
assign(ansm.sub(p_p1), project(Constant(0.0E-1      ), U_prs))
assign(ansm.sub(p_p2), project(Constant(0.0E-1      ), U_prs))
#assign(ansm.sub(p_p1), p1_init)
#assign(ansm.sub(p_p2), p2_init)

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
   ans_nxt.assign(project( ans_aux+ (RK1+RK2*2.0+RK3*2.0+RK4)/6.0, U))

while( count_iteration < TRANSIENT_MAX_ITE ):
   count_iteration = count_iteration +1
   #nlSolver1.solve()
   RungeKutta4(ansm, ansn, nlSolver1, DT)
   residual = assemble(inner(ansn -ansm, ansn -ansm)*dx)
   print ('Iteration: {}'.format(count_iteration) )
   print ('Residual : {}'.format(residual) )
   ansm.assign(ansn)
   save_results(an1,an2,un1,un2,pn1,pn2)


