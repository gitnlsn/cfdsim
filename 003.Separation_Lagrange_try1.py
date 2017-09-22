'''
NELSON KENZO TAMASHIRO
22 SEPTEMBER 2017
TENTATIVA DE SIMULAR SEPARACAO POR EQUACIONAMENTO DERIVADO DE LAGRANGEANO

'''

from fenics import *
from mshr   import *

cons_vin = 1.0E-3
cons_rh1 = 1.0E+3
cons_rh2 = 1.0E+3
cons_mu1 = 1.0E-3
cons_mu2 = 1.0E-3
cons_dd  = 1.0E-8
cons_gg  = 1.0E-1

mesh_res = 50
mesh_P0  = 0.0
mesh_LL  = 3.0E-2
mesh_DD  = 1.0E-2

part1 = Rectangle(
   Point(mesh_P0, mesh_P0),
   Point(mesh_LL, mesh_DD),   )
channel= part1 #-part2
mesh = generate_mesh(channel, mesh_res)

inlet  = '(x[0]=='+str(mesh_P0)+')'
walls  = '(x[1]=='+str(mesh_P0)+') || (x[1]=='+str(mesh_DD)+')'
obsts  = 'on_boundary && (x[0]>'+str(mesh_P0)+') && (x[1]>'+str(mesh_P0)+') && (x[0]<'+str(mesh_LL)+') && (x[1]<'+str(mesh_DD)+')'
outlet = '(x[0]=='+str(mesh_LL)+')'

FE_1  = FiniteElement('P', 'triangle', 1)
FE_2  = FiniteElement('P', 'triangle', 2)

elem  = MixedElement([FE_2, FE_2, FE_1, FE_1])
U     = FunctionSpace(mesh, elem)
U_vel = FunctionSpace(mesh, MixedElement([FE_1,FE_1]))
U_pre = FunctionSpace(mesh, FE_1)
U_con = FunctionSpace(mesh, FE_1)

ans = Function(U)
ux,uy,pp,aa = split(ans)
vx,vy,qq,bb = TestFunctions(U)
uu = as_vector ([ux,uy])
vv = as_vector ([vx,vy])

RH1   = Constant(cons_rh1)
RH2   = Constant(cons_rh2)
MU1   = Constant(cons_mu1)
MU2   = Constant(cons_mu2)
DD    = Constant(cons_dd)
N1    = Constant(1.0)
N05   = Constant(0.5)

rho   = RH1*aa +RH2*(N1-aa)
mu    = MU1*aa +MU2*(N1-aa)

sym_gradu   = grad(uu)+grad(uu).T
sigma       = mu*sym_gradu +pp*Identity(len(uu))
GG          = as_vector([ Constant(0), Constant(-cons_gg)])

F  = inner(sigma,grad(vv))                            *dx \
   + inner(rho*GG,vv)                                 *dx \
   + inner(div(uu),qq)                                *dx \
   + inner((RH2 -RH1)*GG,uu) *bb                      *dx \
   + inner((MU2 -MU1)*sym_gradu, sym_gradu)*N05 *bb   *dx

p_ux,p_uy,p_pp,p_aa = 0,1,2,3
BC = [
      DirichletBC(U.sub(p_ux), Constant(cons_vin ), inlet),
      DirichletBC(U.sub(p_uy), Constant(0        ), inlet),
      DirichletBC(U.sub(p_aa), Constant(0.5      ), inlet),
      DirichletBC(U.sub(p_ux), Constant(0        ), walls),
      DirichletBC(U.sub(p_uy), Constant(0        ), walls),
      # DirichletBC(U.sub(p_ux), Constant(0        ), obsts),
      # DirichletBC(U.sub(p_uy), Constant(0        ), obsts),
      # DirichletBC(U.sub(p_pp), Constant(0        ), outlet),
      ]

dF = derivative(F, ans)
nlProblem = NonlinearVariationalProblem(F, ans, BC, dF)
nlSolver  = NonlinearVariationalSolver(nlProblem)
nlSolver.parameters["nonlinear_solver"] = "snes"

prm = nlSolver.parameters["snes_solver"]
prm["error_on_nonconvergence"       ] = False
prm["solution_tolerance"            ] = 1.0E-16
prm["maximum_iterations"            ] = 15
prm["maximum_residual_evaluations"  ] = 20000
prm["sign"                          ] = "default"
prm["absolute_tolerance"            ] = 8.0E-13
prm["relative_tolerance"            ] = 6.0E-13
prm["linear_solver"                 ] = "mumps"
#prm["method"                        ] = "vinewtonssls"
#prm["line_search"                   ] = "bt"
#prm["preconditioner"                ] = "none"
#prm["report"                        ] = True
#prm["krylov_solver"                 ]
#prm["lu_solver"                     ]

#set_log_level(PROGRESS)

# assign(ans.sub(p_ux ), project(Constant(1E-5), FunctionSpace(mesh, FE) ) )
# assign(ans.sub(p_uy ), project(Constant(1E-5), FunctionSpace(mesh, FE) ) )
# assign(ans.sub(p_pp ), project(Constant(0E-2), FunctionSpace(mesh, FE) ) )
assign(ans.sub(p_aa ), project(Constant(0.5), FunctionSpace(mesh, FE_1) ) )

nlSolver.solve()
plot(uu, title='velocity')
plot(pp, title='pressure')
plot(aa, title='concentration')
interactive()

