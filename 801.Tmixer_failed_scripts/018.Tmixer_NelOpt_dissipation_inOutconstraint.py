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
from math import pi

########################################################
# ------ ------ 01) FOWARD PROBLEM SOLVE ------ ------ #
########################################################

# ------ SIMULATION PARAMETERS ------ #
res      = 30
cons_D   = 8.5E-9        # 8.8E-6 cm**2/s
cons_rho = 1.0E3         # 1kg/m**3
cons_mu  = 8.5E-4        # 0.00089 N*s/m**2
cons_g   = 9.8
cons_vin = 2E-4

limLower       = 0.0
limUpper       = 1.0
inletFract     = 1.0
volumFract     = 0.4
max_opt        = 100000
noOpt_len      = 0.025

# ------ TMIXER GEOMETRY PARAMETERS ------ #
mesh_d   = 0.010          # 10mm
mesh_DD  = 0.030          # largura para otimizacao
mesh_L   = 0.060          # 100mm
mesh_P0  = 0.0
mesh_tol = mesh_L *0.001

# ------ SIMULATION PARAMETERS ------ #
Re = cons_rho*cons_vin*mesh_d/cons_mu
Pe = mesh_d*cons_vin/cons_D
print('Velocity: {:.2e}'.format(cons_vin))
print('Reynolds: {:.2e}'.format(Re))
print('Peclet  : {:.2e}'.format(Pe))

# ------ MESH ------ #
part1 = Rectangle(
   Point( mesh_P0, mesh_P0 ),
   Point( mesh_L , mesh_DD )   )
channel = part1
mesh = generate_mesh(channel, res)

# ------ BOUNDARIES DEFINITION ------ #
inlet_1 = '( on_boundary && x[0]=='+str(0.00*mesh_L )                               \
                     + ' && x[1]>='+str(0.50*mesh_DD             -mesh_tol )        \
                     + ' && x[1]<='+str(0.50*mesh_DD +mesh_d/2.0 +mesh_tol )+' )'
inlet_2 = '( on_boundary && x[0]=='+str(0.00*mesh_L )                               \
                     + ' && x[1]<='+str(0.50*mesh_DD             +mesh_tol )        \
                     + ' && x[1]>='+str(0.50*mesh_DD -mesh_d/2.0 -mesh_tol )+' )'
outlet  = '( on_boundary && x[0]=='+str(1.00*mesh_L )                               \
                     + ' && x[1]>='+str(0.50*mesh_DD -mesh_d/2.0 -mesh_tol )        \
                     + ' && x[1]<='+str(0.50*mesh_DD +mesh_d/2.0 +mesh_tol )+' )'
walls   = 'on_boundary'    \
        + ' && !'+inlet_1  \
        + ' && !'+inlet_2  \
        + ' && !'+outlet

ds_inlet1, ds_inlet2, ds_outlet = 1,2,3

boundaries        = FacetFunction ('size_t', mesh)
side_inlet_1      = CompiledSubDomain( inlet_1  )
side_inlet_2      = CompiledSubDomain( inlet_2  )
side_outlet       = CompiledSubDomain( outlet )
boundaries.set_all(0)
side_inlet_1.mark (boundaries, ds_inlet1 )
side_inlet_2.mark (boundaries, ds_inlet2 )
side_outlet.mark  (boundaries, ds_outlet )
ds = Measure('ds', subdomain_data=boundaries  )

not_to_opt      = '( 3*(x[0]-'+str(0.0*mesh_L )+')*(x[0]-'+str(0.0*mesh_L )+') '\
            + '      + (x[1]-'+str(mesh_DD/2.0)+')*(x[1]-'+str(mesh_DD/2.0)+') <= '+str((mesh_d/2.0)**2)+'    ) '\
            + ' || ( 3*(x[0]-'+str(1.0*mesh_L )+')*(x[0]-'+str(1.0*mesh_L )+') '\
            + '      + (x[1]-'+str(mesh_DD/2.0)+')*(x[1]-'+str(mesh_DD/2.0)+') <= '+str((mesh_d/2.0)**2)+'    )'
dx_not_to_opt           = 1
domain                  = CellFunction  ('size_t', mesh)
domain_not_to_opt       = CompiledSubDomain( not_to_opt )
domain_not_to_opt.mark( domain, dx_not_to_opt )
dx = Measure('dx', subdomain_data=domain )

# ------ FUNCTION SPACES ------ #
FE_V = FiniteElement('P', 'triangle', 2)
FE_P = FiniteElement('P', 'triangle', 1)
FE_U = VectorElement('P', 'triangle', 1)
elem = MixedElement([FE_V, FE_V, FE_P, FE_P])
U    = FunctionSpace(mesh, elem)

FE_A = FiniteElement('P', 'triangle', 1)
U_AA = FunctionSpace(mesh, FE_A)

# ------ FORMULACAO VARIACIONAL ------ #
x,y = 0,1
ans = Function(U)
ux,uy,p,a = split(ans)
vx,vy,q,b = TestFunctions(U)

class gam_wave(Expression):
   def eval(self, value, x):
      N = 2.0
      A = mesh_d*0.5
      y1 = A*sin(2*pi*N*x[0] /mesh_L) +mesh_d/1.8 +mesh_DD/2.0
      y2 = A*sin(2*pi*N*x[0] /mesh_L) -mesh_d/1.8 +mesh_DD/2.0
      is_permeable = x[1]>y2 and x[1]<y1
      if is_permeable:
         value[0] = 0.99
      else:
         value[0] = 0.01

gam = project(gam_wave(degree=1), U_AA)
#gam = project(Constant(volumFract), U_AA)

u = as_vector([ux,uy])
v = as_vector([vx,vy])

RHO = Constant(cons_rho)
MU  = Constant(cons_mu)
DD  = Constant(cons_D)
N3  = Constant(3.0)

he = CellSize(mesh)
Tsupg = Constant(cons_vin*mesh_d/(4*cons_D))*he**2

def material(f, n=1):
   if n==0:
      return f
   for i in range(n):
      f = sin(f*pi-pi/2.0)/2.0+1.0/2.0
   return f

GAM = material(f=gam, n=0)

F1    = inner( MU*grad(u), grad(v))                      *dx \
      + inner( MU/(N3*GAM)*outer(grad(GAM),u), grad(v))  *dx \
      - div(v)*p                                         *dx \
      + div(u*GAM)*q                                     *dx \
      + inner( DD*GAM*grad(a),grad(b) )                  *dx \
      + inner( u*GAM,grad(a))*b                          *dx \
      + inner( dot(u*GAM,grad(b)),
               dot(u*GAM,grad(a)) )*Tsupg                *dx

# ------ CONDICOES DE CONTORNO ------ #
p_ux,p_uy,p_pp,p_aa = 0,1,2,3
BC = [
      DirichletBC(U.sub(p_ux), Constant(cons_vin),    inlet_1),
      DirichletBC(U.sub(p_uy), Constant(0       ),    inlet_1),
      DirichletBC(U.sub(p_aa), Constant(0       ),    inlet_1),
      DirichletBC(U.sub(p_ux), Constant(cons_vin),    inlet_2),
      DirichletBC(U.sub(p_uy), Constant(0       ),    inlet_2),
      DirichletBC(U.sub(p_aa), Constant(1       ),    inlet_2),
      DirichletBC(U.sub(p_ux), Constant(0       ),    walls  ),
      DirichletBC(U.sub(p_uy), Constant(0       ),    walls  ),
      ]

assign(ans.sub(p_ux), project(Constant(cons_vin ), FunctionSpace(mesh, FE_V)))
assign(ans.sub(p_uy), project(Constant(0        ), FunctionSpace(mesh, FE_V)))
assign(ans.sub(p_pp), project(Constant(1e-5     ), FunctionSpace(mesh, FE_P)))
assign(ans.sub(p_aa), project(Constant(0.5      ), FunctionSpace(mesh, FE_P)))

# ------ FOWARD PROBLEM ------ #
dF = derivative(F1, ans)
nlProblem = NonlinearVariationalProblem(F1, ans, BC, dF)
nlSolver  = NonlinearVariationalSolver(nlProblem)
prm = nlSolver.parameters["newton_solver"]
#prm["convergence_criterion"   ] = "residual"
#prm["linear_solver"           ] = "mumps"
#prm["method"                  ] = "full"
#prm["preconditioner"          ] = "none"
#prm["error_on_nonconvergence" ] = True
prm["maximum_iterations"      ] = 20
prm["absolute_tolerance"      ] = 5E-13
prm["relative_tolerance"      ] = 5E-14
#prm["relaxation_parameter"    ] = 1.0
#prm["report"                  ] = True
#set_log_level(PROGRESS)

foldername = 'results_opt_R'+str(res)
vtk_uu  = File(foldername+'/velocity_mean.pvd')
vtk_pp  = File(foldername+'/pressure_mean.pvd')
vtk_aa  = File(foldername+'/concentration_mean.pvd')
vtk_ui  = File(foldername+'/velocity_intrinsic.pvd')
vtk_pi  = File(foldername+'/pressure_intrinsic.pvd')
vtk_ai  = File(foldername+'/concentration_intrinsic.pvd')
vtk_gam = File(foldername+'/porosity.pvd')

def save_flow():
   ui = project(u,FunctionSpace(mesh,FE_U))
   pi = project(p,FunctionSpace(mesh,FE_P))
   ai = project(a,FunctionSpace(mesh,FE_P))
   ui.rename('velocity_intrinsic','velocity_intrinsic')
   pi.rename('pressure_intrinsic','pressure_intrinsic')
   ai.rename('concentration_intrinsic','concentration_intrinsic')
   vtk_ui << ui
   vtk_pi << pi
   vtk_ai << ai
   uu = project(u*gam,FunctionSpace(mesh,FE_U))
   pp = project(p*gam,FunctionSpace(mesh,FE_P))
   aa = project(a*gam,FunctionSpace(mesh,FE_P))
   uu.rename('velocity_mean','velocity_mean')
   pp.rename('pressure_mean','pressure_mean')
   aa.rename('concentration_mean','concentration_mean')
   vtk_uu << uu
   vtk_pp << pp
   vtk_aa << aa

def plot_all():
   plot(u*GAM,title='velocity_mean')
   plot(p*GAM,title='pressure_mean')
   plot(a*GAM,title='concentration_mean')
   interactive()

def print_status():
   vm  = assemble( ux*ds(ds_outlet) )/mesh_d
   Qm  = vm*mesh_d*mesh_d/10.0
   eta = assemble( (a-0.5)*(a-0.5)*ds(ds_outlet))/mesh_d
   dp  = assemble( p*ds(ds_inlet1) )/(mesh_d*0.5)    \
       + assemble( p*ds(ds_inlet2) )/(mesh_d*0.5)    \
       - assemble( p*ds(ds_outlet) )/ mesh_d
   h20 = dp/(cons_rho*cons_g)
   print ('V media ( m/s): {}'.format(vm      ) )
   print ('Vazao   (ml/s): {}'.format(Qm*1E6  ) )
   print ('dP      (Pa  ): {}'.format(dp      ) )
   print ('h20     (mm  ): {}'.format(h20/1000) )
   print ('Dispersao(%)  : {}'.format(eta*100 ) )

nlSolver.solve()

#print_status()

#plot_all()

#save_flow()

########################################################
# ------ ------ 02) ADJOINT OPTIMIZATION ------ ------ #
########################################################

# ------ OTIMIZATION STEP POS EVALUATION ------ #
gam_viz = Function(U_AA)
def post_eval(j, gamma):
   gam_viz.assign(gamma)
   vtk_gam << gam_viz

# ------ FUNCTIONAL DEFINITION ------ #
a_obj    = Constant(0.5)
gam_high = Constant(1.0)
gam_low  = Constant(0.0)
AMP_a = Constant(1.0E6)
AMP_u = Constant(1.0E4)
AMP_g = Constant(1.0E-2)
J  = AMP_a*(a*GAM -a_obj)*(a*GAM -a_obj)*ds(ds_outlet)                \
   + AMP_u*inner( grad(u)+outer(grad(GAM),u)/(N3*GAM), grad(u) )  *dx \
   + AMP_g*4*(gam_high -GAM)*(GAM -gam_low)                       *dx

m  = Control(gam)
J_reduced = ReducedFunctional(
      Functional( J ),
      m, eval_cb_post=post_eval  )

# ------ INLET VOLUME CONSTRAINT ------ #
class InletOutletConstraint(InequalityConstraint):
   def __init__(self, MinFract):
      self.MinFract  = MinFract
      self.refValue  = assemble( project(Constant(1),U_AA)        *dx(dx_not_to_opt) )
      self.smass     = assemble( TestFunction(U_AA)*Constant(1)   *dx(dx_not_to_opt) )
      self.temp      = Function(U_AA)
   def function(self, m):
      print("Evaluting constraint residual")
      self.temp.vector()[:] = m
      integral = self.smass.inner(self.temp.vector())
      print("Current control integral: ", integral)
      return [integral -self.refValue*self.MinFract]
   def jacobian(self, m):
      print("Computing constraint Jacobian")
      return [self.smass]
   def output_workspace(self):
      return [0.0]

class VolumeConstraint(InequalityConstraint):
   def __init__(self, MinFract):
      self.MinFract  = MinFract
      self.refValue  = assemble( project(Constant(1),U_AA)        *dx )
      self.smass     = assemble( TestFunction(U_AA)*Constant(1)   *dx )
      self.temp      = Function(U_AA)
   def function(self, m):
      print("Evaluting constraint residual")
      self.temp.vector()[:] = m
      integral = self.smass.inner(self.temp.vector())
      print("Current control integral: ", integral)
      return [-integral +self.refValue*self.MinFract]
   def jacobian(self, m):
      print("Computing constraint Jacobian")
      return [-self.smass]
   def output_workspace(self):
      return [0.0]

# ------ OPTIMIZATION PROBLEM DEFINITION ------ #
adjProblem = MinimizationProblem(
   J_reduced,
   bounds         = (limLower, limUpper),
   constraints    = [   InletOutletConstraint   ( inletFract ),
                        VolumeConstraint        ( volumFract )   ])
parameters = {'maximum_iterations': max_opt}
adjSolver = IPOPTSolver(
   adjProblem,
   parameters     = parameters)

# TESTE PARA EXPOENTE 10 DA PENALIZACAO
nlSolver.solve()
save_flow()
gam_opt = adjSolver.solve()
gam.assign(gam_opt)

# for amp_value in [10**(exp) for exp in [1, 10, 30, 100, 300, 1000]]:
#    AMP_g.assign(amp_value)
#    adj_reset();
#    nlSolver.solve();
#    save_flow()
#    gam_opt = adjSolver.solve()
#    gam.assign(gam_opt)

nlSolver.solve()
save_flow()
print_status()
plot_all()
