'''
NELSON KENZO TAMASHIRO
22 SEPTEMBER 2017
TENTATIVA DE SIMULAR SEPARACAO POR EQUACIONAMENTO DERIVADO DE LAGRANGEANO

'''

from fenics import *
from mshr   import *

cons_vin = 1.0E-2
cons_rh1 = 1.0E+3
cons_mu1 = 1.0E-3
cons_gg  = 1.0E-1

mesh_res = 100
mesh_P0  = 0.0
mesh_LL  = 3.0E-2
mesh_DD  = 1.0E-2
mesh_Cx     = mesh_LL*0.2
mesh_Cy     = mesh_DD*0.5
mesh_Radius = mesh_DD*0.1

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

elem  = MixedElement([FE_2, FE_2, FE_1])
U     = FunctionSpace(mesh, elem)
U_vel = FunctionSpace(mesh, MixedElement([FE_2,FE_2]))
U_pre = FunctionSpace(mesh, FE_1)
U_por = FunctionSpace(mesh, FE_1)

ans = Function(U)
ux,uy,pp = split(ans)
vx,vy,qq = TestFunctions(U)
uu    = as_vector ([ux,uy])
vv    = as_vector ([vx,vy])

class InitialPorosity(Expression):
   def eval(self, value, x):
      tol = 1E-9
      is_obstacle = (x[0] -mesh_Cx)**2 + (x[1] -mesh_Cy)**2 <= mesh_Radius**2
      if is_obstacle:
         value[0] = 1.0 -tol
      else:
         value[0] = 0.0 +tol

gam   = project(InitialPorosity(degree=1),U_por)

RH1   = Constant(cons_rh1)
MU1   = Constant(cons_mu1)
N1    = Constant(1.0)
N05   = Constant(0.5)

sym_gradu   = grad(uu)+grad(uu).T
sigma       = MU1*sym_gradu +pp*Identity(len(uu))
GG          = as_vector([ Constant(0), Constant(-cons_gg) ])

F  = inner(MU1*gam*grad(uu),grad(vv))                *dx \
   + inner(MU1*N05*outer(uu,grad(gam)),grad(vv))      *dx \
   - inner(pp,div(vv*gam))                            *dx \
   + inner(MU1*dot(grad(uu),grad(gam)), vv)         *dx \
   + inner(div(uu*gam),qq)                            *dx

p_ux,p_uy,p_pp = 0,1,2
BC = [
      DirichletBC(U.sub(p_ux), Constant(cons_vin ), inlet),
      DirichletBC(U.sub(p_uy), Constant(0        ), inlet),
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

nlSolver.solve()
plot(uu, title='velocity')
plot(pp, title='pressure')
interactive()

