'''
Filename: 001.Tmixer_Opt.py
Author  : Nelson Kenzo Tamashiro
Date    : 24 Jun 2017

dolfin-version: 2017.1

Description:

'''


# ------ LIBRARIES ------ #
from fenics import *
from mshr import *
from dolfin_adjoint import *
import pyipopt

########################################################
# ------ ------ 01) FOWARD PROBLEM SOLVE ------ ------ #
########################################################

# ------ SIMULATION PARAMETERS ------ #
res      = 100
opt_par  = 1.0E5
cons_D   = 8.8E-10       # 8.8E-6 cm**2/s
cons_rho = 1.0E3         # 1kg/m**3
cons_mu  = 8.5E-4        # 0.00089 N*s/m**2
cons_g   = 9.8
cons_vin = 2E-3

limLower = 0.0
limUpper = 1.0
mass_maximum = 0.5

# ------ TMIXER GEOMETRY PARAMETERS ------ #
mesh_d   = 0.010         # 10mm
mesh_L   = 3.0*mesh_d
mesh_P0  = 0.0

# ------ SIMULATION PARAMETERS ------ #
Re = cons_rho*cons_vin*mesh_d/cons_mu
Pe = mesh_d*cons_vin/cons_D
print('Velocity: {:.2e}'.format(cons_vin))
print('Reynolds: {:.2e}'.format(Re))
print('Peclet  : {:.2e}'.format(Pe))

# ------ MESH ------ #
part1 = Rectangle(
   Point( mesh_P0, mesh_P0 ),
   Point( mesh_L , mesh_d  )   )
channel = part1
mesh = generate_mesh(channel, res)

# ------ BOUNDARIES DEFINITION ------ #
inlet_1 = '( x[0]=='+str(1.0*mesh_P0)+' && x[1]>='+str(0.5*mesh_d )+' )'
inlet_2 = '( x[0]=='+str(1.0*mesh_P0)+' && x[1]<='+str(0.5*mesh_d )+' )'
outlet  = '( x[0]=='+str(1.0*mesh_L )+' && x[1]> '+str(1.0*mesh_P0)+' && x[1]<'+str(1.0*mesh_d)+' )'
walls   = 'on_boundary'    \
        + ' && !'+inlet_1  \
        + ' && !'+inlet_2  \
        + ' && !'+outlet

ds_inlet1, ds_inlet2, ds_outlet = 0,1,2

boundaries        = FacetFunction ('size_t', mesh)
side_inlet_1      = CompiledSubDomain( inlet_1  )
side_inlet_2      = CompiledSubDomain( inlet_2  )
side_outlet       = CompiledSubDomain( outlet )
boundaries.set_all(0)
side_inlet_1.mark (boundaries, ds_inlet1 )
side_inlet_2.mark (boundaries, ds_inlet2 )
side_outlet.mark  (boundaries, ds_outlet )
ds = Measure('ds', subdomain_data=boundaries  )

# ------ FUNCTION SPACES ------ #
FE_V = FiniteElement('P', 'triangle', 2)
FE_P = FiniteElement('P', 'triangle', 1)
FE_U = VectorElement('P', 'triangle', 2)
elem = MixedElement([FE_V, FE_V, FE_P, FE_P])
U    = FunctionSpace(mesh, elem)

FE_A = FiniteElement('P', 'triangle', 1)
U_AA = FunctionSpace(mesh, FE_A)

# ------ FORMULACAO VARIACIONAL ------ #
x,y = 0,1
ans = Function(U)   
ux,uy,p,a = split(ans)
vx,vy,q,b = TestFunctions(U)
gam       = project(Constant(0.0), U_AA)

u = as_vector([ux,uy])
v = as_vector([vx,vy])

N1  = Constant(1.0)
N12 = Constant(12.0)
N56 = Constant(5.0/6.0)
PAR = Constant(opt_par)
HH  = Constant((mesh_d/10.0)**2.0)
RHO = Constant(cons_rho)
MU  = Constant(cons_mu)
DD  = Constant(cons_D)

F_ctt = div(u)*q                                *dx
F_mtt = inner( RHO*N56*dot(u,grad(u).T), v )    *dx \
      + inner( MU*(grad(u)+grad(u).T), grad(v)) *dx \
      - div(v)*p                                *dx \
      + inner( (N12*MU/HH)*(N1+gam*PAR)*u,v )   *dx
F_cnv = inner( DD*grad(a),grad(b) )             *dx \
      + inner( u,grad(a))*b                     *dx

# Formulacao Final
F1 = F_ctt +F_mtt +F_cnv

# ------ CONDICOES DE CONTORNO ------ #
u_in  = Expression('v_ct*x[1]*(Lx-x[1])/K', v_ct=cons_vin, Lx=mesh_d, K=(mesh_d**2.0)/6.0, degree=2)
p_ux,p_uy,p_pp,p_aa = 0,1,2,3
BC = [
      DirichletBC(U.sub(p_ux), Constant(cons_vin),    inlet_1),
      DirichletBC(U.sub(p_uy), Constant(0 ),    inlet_1),
      DirichletBC(U.sub(p_aa), Constant(1 ),    inlet_1),
      DirichletBC(U.sub(p_ux), Constant(cons_vin),    inlet_2),
      DirichletBC(U.sub(p_uy), Constant(0 ),    inlet_2),
      DirichletBC(U.sub(p_aa), Constant(0 ),    inlet_2),
      DirichletBC(U.sub(p_ux), Constant(0 ),    walls),
      DirichletBC(U.sub(p_uy), Constant(0 ),    walls),
      ]

# ------ FOWARD PROBLEM ------ #
solve(F1==0, ans, BC,
   solver_parameters={'newton_solver':
   {'maximum_iterations' : 10,
   'absolute_tolerance'  : 5E-13,
   'relative_tolerance'  : 5E-14
   } })

# uu = project(u,FunctionSpace(mesh,FE_U))
# pp = project(p,FunctionSpace(mesh,FE_P))
# aa = project(a,FunctionSpace(mesh,FE_P))

# vm  = assemble( ux*ds(ds_outlet) )/mesh_d
# Qm  = vm*mesh_d*mesh_d/10.0
# eta = assemble( (a-0.5)*(a-0.5)*ds(ds_outlet))/mesh_d
# dp  = assemble( p*ds(ds_inlet1) )/(mesh_d*0.5)    \
#     + assemble( p*ds(ds_inlet2) )/(mesh_d*0.5)    \
#     - assemble( p*ds(ds_outlet) )/ mesh_d
# h20 = dp/(cons_rho*cons_g)

# print ('V media ( m/s): {}'.format(vm      ) )
# print ('Vazao   (ml/s): {}'.format(Qm*1E6  ) )
# print ('dP      (Pa  ): {}'.format(dp      ) )
# print ('h20     (mm  ): {}'.format(h20/1000) )
# print ('Dispersao(%) ): {}'.format(eta*100 ) )

# plot(uu,title='velocity')
# plot(pp,title='pressure')
# plot(aa,title='concentration')
# interactive()

########################################################
# ------ ------ 02) ADJOINT OPTIMIZATION ------ ------ #
########################################################

# ------ OTIMIZATION STEP POS EVALUATION ------ #
foldername = 'opt010_R'+str(res)
vtk_aa = File(foldername+'/porosity.pvd')
aa_viz = Function(U_AA)
def post_eval(j, alpha):
   aa_viz.assign(alpha)
   vtk_aa << aa_viz

# ------ FUNCTIONAL DEFINITION ------ #
position = as_vector([  Expression('x[0]', degree=2),
                        Expression('x[1]', degree=2)  ])
v_versor = u/inner(u,u)
a_obj = Constant(0.5)
J = (a -a_obj)*(a -a_obj)*ds(ds_outlet)
m  = Control(gam)
J_reduced = ReducedFunctional(
      Functional( J ),
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
      integral = integral/(mesh_d*mesh_L)
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

