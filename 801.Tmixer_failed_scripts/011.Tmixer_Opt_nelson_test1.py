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
res      = 50
cons_D   = 8.5E-9        # 8.8E-6 cm**2/s
cons_rho = 1.0E3         # 1kg/m**3
cons_mu  = 8.5E-4        # 0.00089 N*s/m**2
cons_g   = 9.8
cons_vin = 2E-4

limLower       = 0.0
limUpper       = 1.0
mass_maximum   = 1.0
max_opt        = 100000

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
      A = mesh_d/2.0
      pos_y1 = 1.0*mesh_d
      pos_y2 = 0.0*mesh_d
      y1 = A*sin(2*pi*N*x[0] /mesh_L) + pos_y1
      y2 = A*sin(2*pi*N*x[0] /mesh_L) + pos_y2
      is_permeable = x[1]>y2 and x[1]<y1
      if is_permeable:
         value[0] = 0.99
      else:
         value[0] = 0.01

#gam = project(gam_wave(degree=1), U_AA)
gam = project(Constant(0.5), U_AA)

u = as_vector([ux,uy])
v = as_vector([vx,vy])

RHO = Constant(cons_rho)
MU  = Constant(cons_mu)
DD  = Constant(cons_D)
N3  = Constant(3.0)

he = CellSize(mesh)
Tsupg = Constant(cons_vin*mesh_d/(4*cons_D))*he**2

def material(f, n=1):
   for i in range(n):
      f = sin(f*pi-pi/2.0)/2.0+1.0/2.0
   return f

GAM = material(f=gam, n=2)

F1    = inner( MU*grad(u), grad(v))                      *dx \
      + inner( MU/(N3*gam)*outer(u,grad(gam)), grad(v))  *dx \
      - div(v)*p                                         *dx \
      + div(u*GAM)*q                                     *dx \
      + inner( DD*GAM*grad(a),grad(b) )                  *dx \
      + inner( u*GAM,grad(a))*b                          *dx \
      + inner( dot(u*GAM,grad(b)),
               dot(u*GAM,grad(a)) )*Tsupg                *dx

# ------ CONDICOES DE CONTORNO ------ #
u_in  = Expression('v_ct*x[1]*(Lx-x[1])/K', v_ct=cons_vin, Lx=mesh_d, K=(mesh_d**2.0)/6.0, degree=2)
p_ux,p_uy,p_pp,p_aa = 0,1,2,3
BC = [
      DirichletBC(U.sub(p_ux), u_in,    inlet_1),
      DirichletBC(U.sub(p_uy), Constant(0 ),    inlet_1),
      DirichletBC(U.sub(p_aa), Constant(0 ),    inlet_1),
      DirichletBC(U.sub(p_ux), u_in,    inlet_2),
      DirichletBC(U.sub(p_uy), Constant(0 ),    inlet_2),
      DirichletBC(U.sub(p_aa), Constant(1 ),    inlet_2),
      DirichletBC(U.sub(p_ux), Constant(0 ),    walls),
      DirichletBC(U.sub(p_uy), Constant(0 ),    walls),
      ]

assign(ans.sub(p_ux), project(Constant(cons_vin ), FunctionSpace(mesh, FE_V)))
assign(ans.sub(p_uy), project(Constant(0        ), FunctionSpace(mesh, FE_V)))
assign(ans.sub(p_pp), project(Constant(1e-5     ), FunctionSpace(mesh, FE_P)))
assign(ans.sub(p_aa), project(Constant(0.5      ), FunctionSpace(mesh, FE_P)))

# ------ FOWARD PROBLEM ------ #
solve(F1==0, ans, BC,
   solver_parameters={'newton_solver':
   {'maximum_iterations' : 10,
   'absolute_tolerance'  : 5E-13,
   'relative_tolerance'  : 5E-14
   } })

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
   plot(u*gam,title='velocity_mean')
   plot(p*gam,title='pressure_mean')
   plot(a*gam,title='concentration_mean')
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

#print_status()

#plot_all()

save_flow()

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
AMP_a = Constant(1/cons_vin)
AMP_g = Constant(1000)
AMP_u = Constant(1)
J  = AMP_a*(a*gam -a_obj)*(a*gam -a_obj)*ux/mesh_d     *ds(ds_outlet) \
   #- AMP_g*(gam -gam_high)*(gam -gam_low)/(mesh_d*mesh_L)   *dx
   #+ AMP_u*inner( grad(u) +outer(u,grad(gam))/(N3*gam),
   #               grad(u)  )                    *dx \

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
parameters = {'maximum_iterations': 100}
solver = IPOPTSolver(
   problem,
   parameters     = parameters)
gam_opt = solver.solve()

J  = AMP_a*(a*gam -a_obj)*(a*gam -a_obj)*ux/mesh_d     *ds(ds_outlet) \
   - AMP_g*(gam -gam_high)*(gam -gam_low)/(mesh_d*mesh_L)   *dx
   #+ AMP_u*inner( grad(u) +outer(u,grad(gam))/(N3*gam),
   #               grad(u)  )                    *dx \

J_reduced = ReducedFunctional(
      Functional( J ),
      m, eval_cb_post=post_eval  )
problem = MinimizationProblem(
   J_reduced,
   bounds         = (limLower, limUpper),
   constraints    = MassConstraint(mass_maximum))
parameters = {'maximum_iterations': max_opt}
solver = IPOPTSolver(
   problem,
   parameters     = parameters)
gam_opt = solver.solve()

gam.assign(gam_opt)
solve(F1==0, ans, BC,
   solver_parameters={'newton_solver':
   {'maximum_iterations' : 10,
   'absolute_tolerance'  : 5E-13,
   'relative_tolerance'  : 5E-14
   } })

save_flow()
print_status()
plot_all()
