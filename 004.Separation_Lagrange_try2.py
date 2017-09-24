'''
NELSON KENZO TAMASHIRO
22 SEPTEMBER 2017
TENTATIVA DE SIMULAR SEPARACAO POR EQUACIONAMENTO DERIVADO DE LAGRANGEANO

'''

from fenics import *
from mshr   import *

cons_vin = 1.0E-2
cons_rh1 = 1.0E+3
cons_rh2 = 1.0E+3
cons_mu1 = 1.0E-3
cons_mu2 = 1.0E-3
cons_gg  = 0.0E-1

mesh_res = 100
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
obsts  = '(on_boundary && (x[0]>'+str(mesh_P0)+') && (x[1]>'+str(mesh_P0)+') && (x[0]<'+str(mesh_LL)+') && (x[1]<'+str(mesh_DD)+') )'
outlet = '(x[0]=='+str(mesh_LL)+')'

ds_inlet, ds_walls, ds_outlet = 1,2,3

boundaries     = FacetFunction ('size_t', mesh)
side_inlet     = CompiledSubDomain( inlet                )
side_outlet    = CompiledSubDomain( outlet               )
side_walls     = CompiledSubDomain( walls+' || '+obsts   )
boundaries.set_all(0)
side_inlet.mark   (boundaries, ds_inlet  )
side_walls.mark   (boundaries, ds_walls  )
side_outlet.mark  (boundaries, ds_outlet )
ds = Measure( 'ds', subdomain_data=boundaries )

FE_T  = TensorElement('P', 'triangle', 1)
FE_V  = VectorElement('P', 'triangle', 1)
FE_P  = FiniteElement('P', 'triangle', 1)
FE_A  = FiniteElement('P', 'triangle', 1)

elem  = MixedElement([FE_V, FE_P, FE_T])
U     = FunctionSpace(mesh, elem)
U_ten = FunctionSpace(mesh, FE_T)
U_vel = FunctionSpace(mesh, FE_V)
U_pre = FunctionSpace(mesh, FE_P)
U_con = FunctionSpace(mesh, FE_A)

ans = Function(U)
u,p,t = split(ans)

RH1   = Constant(cons_rh1)
RH2   = Constant(cons_rh2)
MU1   = Constant(cons_mu1)
MU2   = Constant(cons_mu2)
N1    = Constant(1.0)
N2    = Constant(2.0)
N4    = Constant(4.0)

u_inn  = Constant( (cons_vin, 0) )
u_wal  = Constant( (0, 0) )
t_out  = Constant( ((0,0),(0,0)) )

G     = Constant( -cons_gg )
GG    = as_vector([ Constant(0), G])

F1    = MU1*inner(t,t)/N2 *dx \
      + inner(grad(p), u) *dx

F2    = inner( grad(u)-t,grad(u)-t )   *dx \
      + inner( u -u_inn,u -u_inn )     *ds(ds_inlet) \
      + inner( u -u_wal,u -u_wal )     *ds(ds_walls) \
      + inner( t -t_out,t -t_out )     *ds(ds_outlet)

F = derivative(F1, ans)
F = derivative(inner(F,F)+F2, ans, TestFunction(U))

p_uu,p_pp = 0,1
BC = [
      DirichletBC(U.sub(p_uu), u_in , inlet),
      DirichletBC(U.sub(p_uu), u_00 , walls),
      # DirichletBC(U.sub(p_ux), Constant(0        ), obsts),
      # DirichletBC(U.sub(p_uy), Constant(0        ), obsts),
      # DirichletBC(U.sub(p_pp), Constant(0        ), outlet),
      ]

cons_tol = 1E-3
v_max = cons_vin*20
p_max = 1E2
a_min = 0.0 +cons_tol
a_max = 1.0 -cons_tol

# lowBound = project(Constant((-v_max, -v_max, -v_max, -v_max, -p_max, -p_max, a_min)), U)
# uppBound = project(Constant((+v_max, +v_max, +v_max, +v_max, +p_max, +p_max, a_max)), U)

dF = derivative(F, ans)
nlProblem = NonlinearVariationalProblem(F, ans, BC, dF)
# nlProblem.set_bounds(lowBound,uppBound)
nlSolver  = NonlinearVariationalSolver(nlProblem)
nlSolver.parameters["nonlinear_solver"] = "snes"

prm = nlSolver.parameters["snes_solver"]
prm["error_on_nonconvergence"       ] = False
prm["solution_tolerance"            ] = 1.0E-16
prm["maximum_iterations"            ] = 15
prm["maximum_residual_evaluations"  ] = 20000
prm["absolute_tolerance"            ] = 8.0E-13
prm["relative_tolerance"            ] = 6.0E-13
# prm["method"                        ] = "vinewtonrsls" # vinewtonrsls, vinewtonssls
prm["linear_solver"                 ] = "mumps"
# bicgstab       |  Biconjugate gradient stabilized method                      
# cg             |  Conjugate gradient method                                   
# default        |  default linear solver                                       
# gmres          |  Generalized minimal residual method                         
# minres         |  Minimal residual method                                     
# mumps          |  MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)
# petsc          |  PETSc built in LU solver                                    
# richardson     |  Richardson method                                           
# superlu        |  SuperLU                                                     
# tfqmr          |  Transpose-free quasi-minimal residual method                
# umfpack        |  UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)  
# prm["line_search"                   ] = "basic" # "basic", "bt", "l2", "cp", "nleqerr"
#prm["sign"                          ] = "default"
#prm["preconditioner"                ] = "none"
#prm["report"                        ] = True
#prm["krylov_solver"                 ]
#prm["lu_solver"                     ]

#set_log_level(PROGRESS)

# a_init = Expression('0.5 +0.01*cos(x[0]*1000)+0.01*cos(x[1]*1000)', degree=2)

# assign(ans.sub(p_u1 ), project(Constant((cons_vin*1.1,0)), FunctionSpace(mesh, FE_V) ) )
# assign(ans.sub(p_u2 ), project(Constant((cons_vin*0.9,0)), FunctionSpace(mesh, FE_V) ) )
# assign(ans.sub(p_p1 ), project(Constant(0.0E+0), FunctionSpace(mesh, FE_P) ) )
# assign(ans.sub(p_p2 ), project(Constant(0.0E+0), FunctionSpace(mesh, FE_P) ) )
# assign(ans.sub(p_aa ), project(a_init, FunctionSpace(mesh, FE_A) ) )

# nlSolver.solve()

# plot(u1, title='velocity')
# plot(p1, title='pressure')
# plot(a1, title='concentration')
# interactive()


