'''
DESCRIPTION:

ANALYSIS:

AUTOR: NELSON KENZO TAMASHIRO

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
resolution  = 30
dim_0       = 0.0
dim_L       = 5.0
dim_H       = 1.0
dim_to_opt  = 0.1

# ------ SIMULATION PARAMETERS CONFIGURATION ------ #
cons_gg  = 1.0E-9
cons_rh1 = 1.2E+3
cons_rh2 = 1.0E+3
cons_mu1 = 1.0E-3
cons_mu2 = 1.0E-3
v_ct     = 1.0E-4

limLower = 0.0
limUpper = 1.0
mass_maximum = 0.5

# ------ MESH CONFIGURATION ------ #
part1 = Rectangle(
   Point( dim_0, dim_0 ),
   Point( dim_L, dim_H ),
   )
domain = part1
mesh = generate_mesh(domain, resolution)

inlet  = '('                  \
       + '  near(x[0],'+str(dim_0)+') '\
       + '  && x[1] > '+str(dim_0)+'  '\
       + '  && x[1] < '+str(dim_H)+'      )'
outlet = '('                  \
       + '  near(x[0],'+str(dim_L)+') '\
       + '  && x[1] > '+str(dim_0)+'  '\
       + '  && x[1] < '+str(dim_H)+'      )'
walls  = 'on_boundary && !'+inlet+' && !'+outlet
p_ref  = 'x[0]==0 && x[1]==0'
to_opt      = 'x[0] >  '+str(dim_0+dim_to_opt)
not_to_opt = 'x[0] <= '+str(dim_0+dim_to_opt)

boundaries        = FacetFunction ('size_t', mesh)
side_inlet        = CompiledSubDomain( inlet  )
side_outlet       = CompiledSubDomain( outlet )
boundaries.set_all(0)
boundaries.set_all(0)
ds_inlet, ds_outlet = 0,1
ds = Measure('ds', subdomain_data=boundaries  )
side_inlet.mark   (boundaries, ds_inlet  )
side_outlet.mark  (boundaries, ds_outlet )

domain            = CellFunction  ('size_t', mesh)
domain_to_opt     = CompiledSubDomain( to_opt     )
domain_not_to_opt = CompiledSubDomain( not_to_opt )
dx_to_opt, dx_not_to_opt = 0,1
domain_to_opt.mark     (domain, dx_to_opt    )
domain_not_to_opt.mark (domain, dx_not_to_opt)
dx = Measure('dx', subdomain_data=domain      )

def solve_optimization(cons_c2=0, cons_c3=0, cons_c4=0):
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

   x,y = 0,1

   N1  = Constant(1.0)
   N23 = Constant(2.0/3.0)
   GG  = as_vector([Constant(0), -Constant(cons_gg)])
   RHO = aa*Constant(cons_rh1) +(N1-aa)*Constant(cons_rh2)
   MU  = aa*Constant(cons_mu1) +(N1-aa)*Constant(cons_mu2)

   sigma = MU*(grad(u)+grad(u).T)         \
         + MU*N23*div(u)*Identity(len(u)) \
         - p*Identity(len(u))

   F  = div(   RHO*u )*q *dx \
      + inner( RHO*dot(u,grad(u).T), v ) *dx \
      + inner( sigma, grad(v) ) *dx \
      - inner( RHO*GG, v) *dx

   # ------ BOUNDARY CONDITIONS AND SOLVE ------ #
   v_in  = Expression(('4*v_ct*x[1]*(1-x[1])','0'), v_ct=v_ct, degree=2)
   BC = [
         DirichletBC(U_TH.sub(0), v_in, inlet ),
         DirichletBC(U_TH.sub(0), Constant((0,0)), walls ),
         DirichletBC(U_TH.sub(1), Constant(0), p_ref, method='pointwise'),
         ]

   solve(F==0, ans, BC,
         solver_parameters={'newton_solver':
         {'maximum_iterations' : 25,
         'absolute_tolerance'  : 6E-11,
         'relative_tolerance'  : 8E-11,
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
   foldername = 'opt010_R'+str(resolution)+'_C2.'+str(cons_c2)\
              + '_C3.'+str(cons_c3)+'_C4.'+str(cons_c4)
   vtk_aa = File(foldername+'/mass_fraction.pvd')
   aa_viz = Function(U_AA)
   def post_eval(j, alpha):
      aa_viz.assign(alpha)
      vtk_aa << aa_viz

   # ------ FUNCTIONAL DEFINITION ------ #
   position = as_vector([  Expression('x[0]', degree=2),
                           Expression('x[1]', degree=2)  ])
   v_versor = u/inner(u,u)
   C0 = Constant(9.8E9)
   C2 = Constant(cons_c2)
   C3 = Constant(cons_c3)
   C4 = Constant(cons_c4)
   J1 = inner(                         RHO*GG, v_versor)*dx(dx_to_opt)    # GRAVITY
   J2 = inner(           RHO*dot(u,grad(u).T), v_versor)*dx(dx_to_opt)    # INERTIA
   J3 = inner( div(                             \
               MU*N23*div(u)*Identity(len(u))   \
                     +MU*(grad(u)+grad(u).T)), v_versor)*dx(dx_to_opt)    # VISCOSITY
   J4 = inner(                       -grad(p), v_versor)*dx(dx_to_opt)    # PRESSURE
   m  = Control(aa)
   J_reduced = ReducedFunctional(
         Functional( C0*J1 +C0*C2*J2 +C0*C3*J3 +C0*C4*J4 ),
         m, eval_cb_post=post_eval  )

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
         integral = integral/(dim_L*dim_H)
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

#solve_optimization(cons_c2=1,cons_c3=1,cons_c4=1)

c_range_list = [0, 0.1, 1, 10]
for c2 in c_range_list:
   for c3 in c_range_list:
      for c4 in c_range_list:
         solve_optimization(
            cons_c2=c2,
            cons_c3=c3,
            cons_c4=c4  )

#plot(u,        title='Velocity Intrinsic')
#plot(p,        title='Pressure')
#plot(aa_opt,   title='Porosity')
#interactive()

