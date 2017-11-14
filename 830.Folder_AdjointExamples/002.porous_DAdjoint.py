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
resolution  = 20

dim_0       = 0.0
dim_L       = 1.0
dim_delta   = 1.0

# ------ SIMULATION PARAMETERS CONFIGURATION ------ #
cons_rho = 1.0E3
cons_mu  = 1.0E-3
cons_re  = 1E-1

v_in = 1E-2

limLower = 0.0
limUpper = 1.0
mass_maximum = 0.3

# ------ MESH CONFIGURATION ------ #
part1 = Rectangle(
   Point(     dim_0, dim_0),
   Point( dim_delta, dim_L),
   )
domain = part1
mesh = generate_mesh(domain, resolution)

inlet  = '(near(x[0],0) && on_boundary)'
outlet = '(near(x[0],1) && (x[1]>0.4 && x[1]<0.6) && on_boundary)'
walls  = 'on_boundary && !'+inlet+'&& !'+outlet
p_ref  = 'x[0]==0 && x[1]==0'

side_inlet  = CompiledSubDomain(inlet)
side_outlet = CompiledSubDomain(outlet)
boundaries = FacetFunction('size_t', mesh)
boundaries.set_all(0)
ds_inlet, ds_outlet = 0,1
side_inlet.mark (boundaries, ds_inlet  )
side_outlet.mark(boundaries, ds_outlet )
ds = Measure('ds', subdomain_data=boundaries)

# ------ VARIATIONAL FORMULATION ------ #
FE_u = VectorElement('P', mesh.ufl_cell(), 2)
FE_p = FiniteElement('P', mesh.ufl_cell(), 1)
elem = MixedElement([FE_u, FE_p])
U_TH = FunctionSpace(mesh, elem)
U_AA = FunctionSpace(mesh, FE_p)

ans = Function(U_TH)
u,p = split(ans)
v,q = TestFunctions(U_TH)
aa = project(Constant(mass_maximum), U_AA)

N1  = Constant(1.0)
RHO = Constant(cons_rho)
MU  = Constant(cons_mu)
RE  = Constant(cons_re)

sigma = MU*(grad(u)+grad(u).T) -p*Identity(len(u))

x,y = 0,1

F  = div( u*aa )*q*dx \
   + inner( aa*RHO*dot(u,grad(u).T), v ) *dx \
   + inner( sigma, grad(v) ) *dx

# ------ BOUNDARY CONDITIONS AND SOLVE ------ #
inlet  = '(near(x[0],0) && on_boundary)'
outlet = '(near(x[0],1) && (x[1]>0.3 && x[1]<0.7) && on_boundary)'
walls  = 'on_boundary && !'+inlet+'&& !'+outlet
p_ref  = 'x[0]==0 && x[1]==0'
v_in   = Expression(('4*v_in*x[1]*(1-x[1])','0'), v_in=v_in, degree=2)
BC = [
      #DirichletBC(U_TH.sub(0), InflowOutflow(degree=2), 'on_boundary'),
      DirichletBC(U_TH.sub(0), v_in, inlet  ),
      DirichletBC(U_TH.sub(0), Constant((0,0)), walls  ),
      DirichletBC(U_TH.sub(1), Constant(0), p_ref, method='pointwise'),
      ]

solve(F==0, ans, BC,
      solver_parameters={'newton_solver':
      {'maximum_iterations' : 15,
      'absolute_tolerance'  : 6E-12,
      'relative_tolerance'  : 8E-13,
      'relaxation_parameter': 1.0
      } })

u,p = split(ans)

# ------ PLOTING AND SAVING ------ #
#plot(uu, title='Velocity')
#plot(pp, title='Pressure')
#plot(alpha, title='Mass Fraction')
#plot(inner(R_CT,R_CT), title='Continuty Conservation')
#plot(inner(R_NS,R_NS), title='Momentum Conservation')
#interactive()

########################################################
# ------ ------ 02) ADJOINT OPTIMIZATION ------ ------ #
########################################################

# ------ OTIMIZATION STEP POS EVALUATION ------ #
vtk_aa = File("results/porosity.pvd")
aa_viz = Function(U_AA)
def post_eval(j, alpha):
   aa_viz.assign(alpha)
   vtk_aa << aa_viz

# ------ FUNCTIONAL DEFINITION ------ #
J = Functional(
      inner(   MU*(grad(u)+grad(u).T),
               MU*(grad(u)+grad(u).T)  )*dx 
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
parameters = {'maximum_iterations': 1000}
solver = IPOPTSolver(
   problem, 
   parameters     = parameters)
aa_opt = solver.solve()

plot(u,        title='Velocity Intrinsic')
plot(p,        title='Pressure')
plot(inner(tau,tau),     title='Interface Tension')
plot(aa_opt,   title='Porosity')
interactive()
