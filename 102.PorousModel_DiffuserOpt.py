'''
DESCRIPTION:

ANALYSIS:

AUTOR:  

DATE: 03.05.2017

'''

# ------ LIBRARIES IMPORT ------ #
from fenics import *
from mshr import *
from dolfin_adjoint import *
import pyipopt

########################################################
# ------ ------ 01) FOWARD PROBLEM SOLVE ------ ------ #
########################################################

# ------ GEOMETRICAL PARAMETERS ------ #
resolution  = 100
fe_max_ite  = 25
fe_abs_tol  = 5E-13
fe_rel_tol  = 5E-13

dim_0       = 0.0
dim_L       = 1.0
dim_delta   = 1.0
dim_out     = 0.1

# ------ SIMULATION PARAMETERS CONFIGURATION ------ #
cons_rho = 1.0E+3
cons_mu  = 1.0E-3
cons_C23 = 1.0E+3
cons_amp = 1.0E+9

v_ct = 1E-5

limLower = 0.0
limUpper = 1.0
mass_maximum = 0.4

# ------ MESH CONFIGURATION ------ #
part1 = Rectangle(
   Point(     dim_0, dim_0),
   Point( dim_delta, dim_L),
   )
domain = part1
mesh = generate_mesh(domain, resolution)

inlet  = '(near(x[0],0) && on_boundary)'
outlet = '(near(x[0],1) && (x[1]>'+str(dim_delta/2.0-dim_out)\
                      +' && x[1]<'+str(dim_delta/2.0+dim_out)+') && on_boundary)'
walls  = 'on_boundary && !'+inlet+'&& !'+outlet
p_ref  = 'x[0]==0 && x[1]==0'

# ------ VARIATIONAL FORMULATION ------ #
FE_u = VectorElement('P', mesh.ufl_cell(), 2)
FE_p = FiniteElement('P', mesh.ufl_cell(), 1)
elem = MixedElement([FE_u, FE_p])
U_TH = FunctionSpace(mesh, elem)
U_AA = FunctionSpace(mesh, FE_p)

class aa_init(Expression):
   def eval(self, value, x):
      y1 = ( dim_delta -x[0]*((dim_delta/2.0)-dim_out)/dim_L )
      y2 = (            x[0]*((dim_delta/2.0)-dim_out)/dim_L )
      is_porous = x[1]>y2 and x[1]<y1
      if is_porous:
         value[0] = 1
      else:
         value[0] = 0

ans = Function(U_TH)
u,p = split(ans)
v,q = TestFunctions(U_TH)
#aa  = project(aa_init(degree=1), U_AA)
aa  = project(Constant(1.0), U_AA)

N1  = Constant( 1.0)
N12 = Constant(12.0)
C23 = Constant(cons_C23)
RHO = Constant(cons_rho)
MU  = Constant(cons_mu)

sig_a = MU*aa*(grad(u)+grad(u).T) -p*aa*Identity(len(u))
tau_a = MU*   (grad(u)+grad(u).T)
tau_I = -N12*MU*(N1-aa)*C23*u
x,y = 0,1

F  = div( u*aa )*q*dx                        \
   + inner( aa*RHO*dot(u,grad(u).T), v ) *dx \
   + inner( sig_a, grad(v) ) *dx          \
   - inner( tau_I, v ) *dx

# ------ BOUNDARY CONDITIONS AND SOLVE ------ #
v_in   = Expression(('4*v_in*x[1]*(1-x[1])','0'), v_in=v_ct, degree=2)
BC = [
      DirichletBC(U_TH.sub(0), v_in, inlet  ),
      DirichletBC(U_TH.sub(0), Constant((0,0)), walls  ),
      DirichletBC(U_TH.sub(1), Constant(0), p_ref, method='pointwise'),
      ]

solve(F==0, ans, BC,
      solver_parameters={'newton_solver':
      {'maximum_iterations' : fe_max_ite,
      'absolute_tolerance'  : fe_abs_tol,
      'relative_tolerance'  : fe_rel_tol,
      'relaxation_parameter': 1.0
      } })

# ------ PLOTING AND SAVING ------ #
# plot(u, title='Velocity')
# plot(p, title='Pressure')
# interactive()

########################################################
# ------ ------ 02) ADJOINT OPTIMIZATION ------ ------ #
########################################################

# ------ OTIMIZATION STEP POS EVALUATION ------ #
foldername = 'opt011_M'+str(mass_maximum)+'_R'+str(resolution)\
           + '_V'+str(v_ct)+'_C'+str(cons_C23)+'Out'+str(dim_out)+'A'+str(cons_amp)
vtk_aa = File(foldername+'/porosity.pvd')
aa_viz = Function(U_AA)
def post_eval(j, alpha):
   aa_viz.assign(alpha)
   vtk_aa << aa_viz

# ------ FUNCTIONAL DEFINITION ------ #
AMP = Constant(cons_amp)
J = Functional(
        AMP*inner( tau_a, tau_a ) *dx \
      + AMP*inner( tau_I, tau_I ) *dx \
      )
m = Control(aa)
J_reduced = ReducedFunctional(J, m, eval_cb_post=post_eval)

# ------ VOLUME CONSTRAINT DEFINITION ------ #
class MassConstraint(InequalityConstraint):
   def __init__(self, MaxMass):
      self.MaxMass = float(MaxMass)
      self.smass = assemble(TestFunction(U_AA)*Constant(1)*dx)
      self.temp = Function(U_AA)
   def function(self, m):
      print("Evaluting constraint residual")
      self.temp.vector()[:] = m
      integral = self.smass.inner(self.temp.vector())
      integral = integral/(dim_L*dim_delta)
      print("Current control integral: ", integral)
      return [self.MaxMass -integral]
   def jacobian(self, m):
      print("Computing constraint Jacobian")
      return [-self.smass]
   def output_workspace(self):
      return [0.0]

# ------ OPTIMIZATION PROBLEM DEFINITION ------ #
problem = MinimizationProblem(
   J_reduced,
   bounds         = (limLower, limUpper),
   constraints    = MassConstraint(mass_maximum))
parameters = {'maximum_iterations': 10000}
solver = IPOPTSolver(
   problem,
   parameters     = parameters)
aa_opt = solver.solve()

aa.assign(aa_opt)
solve(F==0, ans, BC,
      solver_parameters={'newton_solver':
      {'maximum_iterations' : fe_max_ite,
      'absolute_tolerance'  : fe_abs_tol,
      'relative_tolerance'  : fe_rel_tol,
      'relaxation_parameter': 1.0
      } })
plot(u,        title='Velocity Intrinsic')
plot(p,        title='Pressure')
plot(aa_opt,   title='Porosity')
interactive()


