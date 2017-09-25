'''
NELSON KENZO TAMASHIRO
22 SEPTEMBER 2017
TENTATIVA DE SIMULAR SEPARACAO POR EQUACIONAMENTO DERIVADO DE LAGRANGEANO

'''

from fenics import *
from mshr   import *

cons_vin = 1.0E-3
cons_rh1 = 1.0E+3
cons_rh2 = 1.1E+3
cons_mu1 = 1.0E-3
cons_mu2 = 1.0E-3
cons_gg  = 1.0E-9

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

FE_V  = VectorElement('P', 'triangle', 2)
FE_P  = FiniteElement('P', 'triangle', 1)
FE_A  = FiniteElement('P', 'triangle', 1)

elem  = MixedElement([FE_V, FE_V, FE_P, FE_P, FE_A])
U     = FunctionSpace(mesh, elem)
U_vel = FunctionSpace(mesh, FE_V)
U_pre = FunctionSpace(mesh, FE_P)
U_con = FunctionSpace(mesh, FE_A)

ans = Function(U)
u1,u2,p1,p2,a1 = split(ans)
v1,v2,q1,q2,b1 = TestFunctions(U)

RH1   = Constant(cons_rh1)
RH2   = Constant(cons_rh2)
MU1   = Constant(cons_mu1)
MU2   = Constant(cons_mu2)
N1    = Constant(1.0)
N2    = Constant(2.0)

a2    = N1 -a1

G     = Constant( -cons_gg )
GG    = as_vector([ Constant(0), G])

def F_cct(u1,a1,u2,a2,tt):
   return div( u1*a1 +u1*a1*a2 +u2*a2*a2) *tt *dx

def F_mmt(u1,a1,u2,a2,p1,p2,mu1,mu2,rh1,rh2,tt):
   return  inner( mu1*grad(u1)*a1,              grad(tt) )     *dx \
         + inner( mu1*outer(u1,grad(a1) ),      grad(tt) )/N2  *dx \
         + inner( mu1*grad(u1)*a1*a2,           grad(tt) )     *dx \
         + inner( mu1*outer(u1,grad(a1*a2) ),   grad(tt) )/N2  *dx \
         + inner( mu1*grad(u2)*a2*a2,           grad(tt) )/N2  *dx \
         + inner( mu1*outer(u2,grad(a2*a2) ),   grad(tt) )/N2  *dx \
         + inner( mu1*dot(grad(a1),grad(u1).T),      tt  )/N2  *dx \
         + inner( mu1*dot(grad(a1*a2),grad(u1).T),   tt  )/N2  *dx \
         + inner( mu2*grad(u2)*a1*a1,           grad(tt) )/N2  *dx \
         + inner( mu2*dot(grad(a1*a1),grad(u2).T),   tt  )/N2  *dx \
         + inner( p1, div(tt*a1*(N1+a2))                 )     *dx \
         + inner( rh1*GG*a1*(N1+a2),                 tt  )     *dx \
         + inner( p2, div(tt*a1*a1)                      )     *dx \
         + inner( rh2*GG*a1*a1,                      tt  )     *dx

def F_eng(u1,a1,u2,a2,p1,p2,mu1,mu2,rh1,rh2,tt):
   return  inner(  mu1*grad(u1),grad(u1) )/N2 *tt                       *dx \
         + inner( -mu2*grad(u2),grad(u2) )/N2 *tt                       *dx \
         + inner(  mu1*dot(u1,grad(u1).T)
                  -mu2*dot(u2,grad(u2).T), grad(tt) )/N2                *dx \
         + inner(  mu1*grad(u1),grad(u1)*(N1 -N2*a1) )/N2 *tt           *dx \
         + inner(  mu2*grad(u2),grad(u2)*(N1 -N2*a1) )/N2 *tt           *dx \
         + inner(  mu1*dot(u1,grad(u1).T)
                  +mu2*dot(u2,grad(u2).T), grad(tt)*(N1 -N2*a1) )/N2    *dx \
         + inner(  mu1*grad(u1),grad(u2)*(-N2 -N2*a1) )/N2 *tt          *dx \
         + inner(  mu2*grad(u1),grad(u2)*(     N2*a1) )/N2 *tt          *dx \
         + inner(  mu1*dot(u2,grad(u1).T), grad(tt)*(-N2 -N2*a1) )/N2   *dx \
         + inner(  mu2*dot(u1,grad(u2).T), grad(tt)*(     N2*a1) )/N2   *dx \
         + inner( (grad(p1)+rh1*GG),(u1-u2)*(N2 -N2*a1) )*tt            *dx \
         + inner( (grad(p2)+rh2*GG),(u1-u2)*(    N2*a1) )*tt            *dx \
         + inner( grad(a1), grad(tt) ) *1E-6 *dx

F  = F_cct(u1,a1,u2,a2,q1) \
   + F_cct(u2,a2,u1,a1,q2) \
   + F_mmt(u1,a1,u2,a2,p1,p2,MU1,MU2,RH1,RH2,v1) \
   + F_mmt(u2,a2,u1,a1,p2,p1,MU2,MU1,RH2,RH1,v2) \
   + F_eng(u1,a1,u2,a2,p1,p2,MU1,MU2,RH1,RH2,b1)

u_in = Constant((cons_vin, 0))
u_00 = Constant((0, 0))
a_in = Constant(0.5)

p_u1,p_u2,p_p1,p_p2,p_aa = 0,1,2,3,4
BC = [
      DirichletBC(U.sub(p_u1), u_in, inlet),
      DirichletBC(U.sub(p_u2), u_in, inlet),
      DirichletBC(U.sub(p_aa), a_in, inlet),
      DirichletBC(U.sub(p_u1), u_00, walls),
      DirichletBC(U.sub(p_u2), u_00, walls),
      # DirichletBC(U.sub(p_ux), Constant(0        ), obsts),
      # DirichletBC(U.sub(p_uy), Constant(0        ), obsts),
      DirichletBC(U.sub(p_p1), Constant(0        ), outlet),
      DirichletBC(U.sub(p_p2), Constant(0        ), outlet),
      ]

a_init = Expression('0.5 +0.001*cos(x[0]*1000)+0.001*cos(x[1]*1000)', degree=2)

assign(ans.sub(p_u1 ), project(Constant((cons_vin*1.1,-cons_vin)), FunctionSpace(mesh, FE_V) ) )
assign(ans.sub(p_u2 ), project(Constant((cons_vin*0.9,+cons_vin)), FunctionSpace(mesh, FE_V) ) )
assign(ans.sub(p_p1 ), project(Constant(0.0E+0), FunctionSpace(mesh, FE_P) ) )
assign(ans.sub(p_p2 ), project(Constant(0.0E+0), FunctionSpace(mesh, FE_P) ) )
assign(ans.sub(p_aa ), project(a_init, FunctionSpace(mesh, FE_A) ) )

# solve(F==0, ans, BC)

cons_tol = 1.0E-9
v_max = cons_vin*20
p_max = 1E3
a_min = 0.0 +cons_tol
a_max = 1.0 -cons_tol

lowBound = project(Constant((-v_max, -v_max, -v_max, -v_max, -p_max, -p_max, a_min)), U)
uppBound = project(Constant((+v_max, +v_max, +v_max, +v_max, +p_max, +p_max, a_max)), U)

dF = derivative(F, ans)
nlProblem = NonlinearVariationalProblem(F, ans, BC, dF)
nlProblem.set_bounds(lowBound,uppBound)
nlSolver  = NonlinearVariationalSolver(nlProblem)
nlSolver.parameters["nonlinear_solver"] = "snes"

prm = nlSolver.parameters["snes_solver"]
prm["error_on_nonconvergence"       ] = False
prm["solution_tolerance"            ] = 1.0E-16
prm["maximum_iterations"            ] = 15
prm["maximum_residual_evaluations"  ] = 20000
prm["absolute_tolerance"            ] = 8.0E-13
prm["relative_tolerance"            ] = 6.0E-13
prm["method"                        ] = "vinewtonrsls" # vinewtonrsls, vinewtonssls
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
prm["line_search"                   ] = "basic" # "basic", "bt", "l2", "cp", "nleqerr"
#prm["sign"                          ] = "default"
#prm["preconditioner"                ] = "none"
#prm["report"                        ] = True
#prm["krylov_solver"                 ]
#prm["lu_solver"                     ]

#set_log_level(PROGRESS)

nlSolver.solve()
# plot(u1, title='velocity')
# plot(p1, title='pressure')
# plot(a1, title='concentration')
# interactive()


