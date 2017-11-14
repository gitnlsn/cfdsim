'''

 

19.07.2017

'''

# ------ LIBRARIES ------ #
from fenics          import *
from mshr            import *
from dolfin_adjoint  import *

# ------ SIMULATION PARAMETERS ------ #
filename = 'results_Goda'

# ------ TMIXER GEOMETRY PARAMETERS ------ #
mesh_res  = 50
mesh_P0   = 0.0
mesh_DD   = 0.030          # channel width
mesh_L    = 0.080          # 100mm
mesh_obX  = 0.2*mesh_L     # obstruction for optimization dimensions
mesh_obY  = 0.5*mesh_DD
mesh_obD  = 0.010
mesh_OPT  = 0.8*mesh_DD
mesh_fuX  = 0.8*mesh_L     # functional domain dimensions
mesh_fuY  = 0.5*mesh_DD
mesh_fuD  = 0.3*mesh_DD
mesh_Cx   = 0.2*mesh_L     # fixed circle obstruction dimensions
mesh_Cy   = 0.5*mesh_DD
mesh_Rad  = 0.004
mesh_tol  = mesh_DD*0.05

# ------ TMIXER GEOMETRY PARAMETERS ------ #
cons_dt  = 0.01
cons_rho = 1E+3
cons_mu  = 1E-3
cons_dif = 1E-8
cons_v1  = 5E-2
cons_g   = 9.8E0

cons_bor       = 0.01      # parameter for \alpha
alphaunderbar  = 0.0       # parameter for \alpha
alphabar       = 1.0E5     # parameter for \alpha

limLower       = 0.0
limUpper       = 1.0
MinFract       = 1.0
max_opt        = 100000

GENERAL_TOL = 1E-6
TRANSIENT_MAX_ITE = 300

# ------ MESH ------ #
part1 = Rectangle(
   Point(mesh_P0, mesh_P0),
   Point(mesh_L , mesh_DD)   )
part2 = Circle(
   Point(mesh_Cx, mesh_Cy),
   mesh_Rad             )
channel = part1 -part2
mesh = generate_mesh(channel, mesh_res)

# ------ BOUNDARIES DEFINITION ------ #
inlet_1 = '( on_boundary && (x[0]=='+str(0.00*mesh_L )+ ') && (x[1]<='+str(mesh_obY )+')  )'
inlet_2 = '( on_boundary && (x[0]=='+str(0.00*mesh_L )+ ') && (x[1]>='+str(mesh_obY )+')  )'
outlet  = '( on_boundary && (x[0]=='+str(1.00*mesh_L )+ ')                                )'
walls   = 'on_boundary'    \
        + ' && !'+inlet_1  \
        + ' && !'+inlet_2  \
        + ' && !'+outlet

ds_inlet1, ds_inlet2, ds_walls, ds_outlet = 1,2,3,4

boundaries     = FacetFunction ('size_t', mesh)
side_inlet1    = CompiledSubDomain( inlet_1 )
side_inlet2    = CompiledSubDomain( inlet_2 )
side_outlet    = CompiledSubDomain( outlet  )
side_walls     = CompiledSubDomain( walls   )
boundaries.set_all(0)
side_inlet1.mark  (boundaries, ds_inlet1 )
side_inlet2.mark  (boundaries, ds_inlet2 )
side_walls.mark   (boundaries, ds_walls  )
side_outlet.mark  (boundaries, ds_outlet )
ds = Measure( 'ds', subdomain_data=boundaries )

domain = CellFunction ('size_t', mesh)
functional_domain = '( (x[0]>'+str(mesh_fuX          )+') && '\
                  + '  (x[1]>'+str(mesh_fuY -mesh_fuD)+') && '\
                  + '  (x[1]<'+str(mesh_fuY +mesh_fuD)+')      )'
dx_not_to_opt  = 1
CompiledSubDomain( functional_domain ).mark( domain, dx_not_to_opt )
dx = Measure('dx', subdomain_data=domain )

# ------ VARIATIONAL FORMULATION ------ #
FE_u  = VectorElement('P', 'triangle', 2)
FE_p  = FiniteElement('P', 'triangle', 1)
FE_a  = FiniteElement('P', 'triangle', 1)
FE_g  = FiniteElement('P', 'triangle', 1)
U_prs = FunctionSpace(mesh, FE_p)
U_vel = FunctionSpace(mesh, FE_u)
U_gam = FunctionSpace(mesh, FE_g)
U     = FunctionSpace(mesh, MixedElement([FE_u, FE_p, FE_a]) )

class straigthChannel(Expression):
   def eval(self, value, x):
      is_wall     = x[1]>mesh_chY+mesh_d or x[1]<mesh_chY-mesh_d
      is_obstacle = (x[0]-mesh_obX)**2 + (x[1]-mesh_obY)**2 < mesh_Rad**2
      if is_wall or is_obstacle:
         value[0] = 0.0
      else:
         value[0] = 1.0

ans1  = Function(U)
ans2  = Function(U)

u1,p1,a1 = split(ans1)
u2,p2,a2 = split(ans2)

v,q,b = TestFunctions(U)

gam   = project(Constant(1), U_gam)

DT       = Constant(cons_dt  )
RHO      = Constant(cons_rho )
MU       = Constant(cons_mu  )
DD       = Constant(cons_dif )
u_inlet  = Constant(cons_v1  )
n        = FacetNormal(mesh)

cons_pe = mesh_DD*cons_v1/cons_dif

#in_profile1 = Expression(str(cons_v1)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', degree=2)
#in_profile2 = Expression(str(cons_v2)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', degree=2)

u_in  = as_vector([ u_inlet    , Constant(0) ])
u_wl  = as_vector([ Constant(0), Constant(0) ])

he = CellSize(mesh)
u_md  = (u1+u2)*0.5
p_md  = (p1+p2)*0.5
a_md  = (a1+a2)*0.5
sigma = MU*(grad(u_md)+grad(u_md).T) -p_md*Identity(len(u_md))
Tsupg = (4/(cons_pe*he**2) +2*sqrt(GENERAL_TOL+inner(u_md,u_md))/he)**-1

# ------ MATERIAL MODEL ------ #
bor_q = Constant(cons_bor) # q value that controls difficulty/discrete-valuedness of solution
def alpha(gam):
   return alphabar + (alphaunderbar - alphabar) * gam * (1 + bor_q) / (gam + bor_q)

F1 = RHO/DT *inner( u2-u1,v )                   *dx \
   + RHO    *inner( dot(u_md,grad(u_md).T),v )  *dx \
   +         inner( sigma,grad(v) )             *dx \
   + q*div(u2)                                  *dx \
   +         inner( a2-a1,b ) /DT               *dx \
   +         inner( inner(u_md,grad(a_md)),b )  *dx \
   + DD     *inner(grad(a_md),grad(b))          *dx \
   + inner( dot(u_md,grad(b)),
            dot(u_md,grad(a_md)) )*Tsupg        *dx \
   + inner(u_md,v)*alpha(gam)                   *dx

# ------ BOUNDARY CONDITIONS ------ #
p_uu,p_pp,p_aa = 0,1,2
BC1 = [
         DirichletBC(U.sub(p_uu), Constant((cons_v1,0)), inlet_1),
         DirichletBC(U.sub(p_uu), Constant((cons_v1,0)), inlet_2),
         DirichletBC(U.sub(p_aa), Constant(       1   ), inlet_1),
         DirichletBC(U.sub(p_aa), Constant(       0   ), inlet_2),
         DirichletBC(U.sub(p_uu), Constant((      0,0)), walls),
      ] # end - BC #

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
dF1 = derivative(F1, ans2 )
# dF2 = derivative(F2, p_nxt)
# dF3 = derivative(F3, u_nxt)
nlProblem1 = NonlinearVariationalProblem(F1, ans2, BC1, dF1)
# nlProblem2 = NonlinearVariationalProblem(F2, p_nxt, BC2, dF2)
# nlProblem3 = NonlinearVariationalProblem(F3, u_nxt, [], dF3)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
# nlSolver2  = NonlinearVariationalSolver(nlProblem2)
# nlSolver3  = NonlinearVariationalSolver(nlProblem3)
prm1 = nlSolver1.parameters["newton_solver"]
# prm2 = nlSolver2.parameters["newton_solver"]
# prm3 = nlSolver3.parameters["newton_solver"]
for prm in [prm1]:
   prm["maximum_iterations"      ] = 10
   prm["absolute_tolerance"      ] = 9E-13
   prm["relative_tolerance"      ] = 8E-13

#prm["convergence_criterion"   ] = "residual"
#prm["linear_solver"           ] = "mumps"
#prm["method"                  ] = "full"
#prm["preconditioner"          ] = "none"
#prm["error_on_nonconvergence" ] = True
#prm["relaxation_parameter"    ] = 1.0
#prm["report"                  ] = True
#set_log_level(PROGRESS)

# ------ SAVE FILECONFIGURATIONS ------ #
class Transient_flow_save():
   def __init__(self, folderName):
      self.vtk_uu  = File(folderName+'/velocity.pvd')
      self.vtk_pp  = File(folderName+'/pressure.pvd')
      self.vtk_aa  = File(folderName+'/concentration.pvd')
   def save_flow(self, u_tosave, p_tosave, a_tosave):
      ui = project(u_tosave,FunctionSpace(mesh,FE_u))
      pi = project(p_tosave,FunctionSpace(mesh,FE_p))
      ai = project(a_tosave,FunctionSpace(mesh,FE_a))
      ui.rename('velocity','velocity')
      pi.rename('pressure','pressure')
      ai.rename('concentration','concentration')
      self.vtk_uu << ui
      self.vtk_pp << pi
      self.vtk_aa << ai

vtk_gam = File(filename+'/porosity.pvd')
vtk_uu  = File(filename+'/velocity.pvd')
vtk_pp  = File(filename+'/pressure.pvd')
vtk_aa  = File(filename+'/concentration.pvd')

def save_flow(u_tosave, p_tosave, a_tosave):
   ui = project(u_tosave,FunctionSpace(mesh,FE_u))
   pi = project(p_tosave,FunctionSpace(mesh,FE_p))
   ai = project(a_tosave,FunctionSpace(mesh,FE_a))
   ui.rename('velocity','velocity')
   pi.rename('pressure','pressure')
   ai.rename('concentration','concentration')
   vtk_uu << ui
   vtk_pp << pi
   vtk_aa << ai

def plot_all():
   plot(u*GAM,title='velocity_mean')
   plot(p*GAM,title='pressure_mean')
   plot(a*GAM,title='concentration_mean')
   interactive()

# ------ TRANSIENT SIMULATION ------ #
#flag_converged    = False
#assign(ans.sub(p_ux ), project(Constant(1E-2), FunctionSpace(mesh, FE_1) ) )
#assign(ans.sub(p_uy ), project(Constant(0E-2), FunctionSpace(mesh, FE_1) ) )
#assign(ans.sub(p_pp ), project(Constant(1E-2), FunctionSpace(mesh, FE_1) ) )
#u_inlet.assign(cons_v2)
#u_next.assign(project(u, U_vel))
def foward_solve(folderName):
   flowSto = Transient_flow_save(folderName)
   #adj_start_timestep(time=0.0)
   count_iteration   = 0
   t = 0.0
   while( count_iteration < TRANSIENT_MAX_ITE ):
      count_iteration   = count_iteration +1
      t                 = t               +cons_dt
      nlSolver1.solve()
      # nlSolver2.solve()
      # nlSolver3.solve()
      residual = assemble(inner(u2-u1, u2-u1)*dx)
      print ('Residual : {}'.format(residual) )
      print ('Iteration: {}'.format(count_iteration) )
      ans1.assign(ans2)
      flowSto.save_flow(u2,p2,a2)
      #if count_iteration==TRANSIENT_MAX_ITE:
      #   adj_inc_timestep(time=t, finished=True)
      #else:
      #   adj_inc_timestep(time=t, finished=False)

foward_solve('initialSimulation')

#plot(u, title='velocity')
#plot(p, title='pressure')
#interactive()

########################################################
# ------ ------ 02) ADJOINT OPTIMIZATION ------ ------ #
########################################################

# ------ OTIMIZATION STEP POS EVALUATION ------ #
gam_viz = Function(U_gam)
def post_eval(j, gamma):
   gam_viz.assign(gamma)
   vtk_gam << gam_viz

# ------ FUNCTIONAL DEFINITION ------ #
a_obj    = Constant(0.5)
AMP_a  = Constant(1.0E0)
J  = \
   + AMP_a* (a1 -a_obj)*(a1 -a_obj)*gam  *dx*dt[FINISH_TIME]

m  = Control(gam)
J_reduced = ReducedFunctional(
      Functional( J ),
      m, eval_cb_post=post_eval  )

class VolConstraint(InequalityConstraint):
   def __init__(self, MinFract, dx_position):
      self.MinFract  = MinFract
      self.refValue  = assemble( project(Constant(1),U_gam)        *dx(dx_position) )
      self.smass     = assemble( TestFunction(U_gam)*Constant(1)   *dx(dx_position) )
      self.temp      = Function(U_gam)
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

class LowerBound(Expression):
   def __init__(self,degree):
      self.degree = degree
   def eval_cell(self, values, x, ufc_cell):
      values[0] = Constant(1.0)
      in_opt_domain = x[1]> mesh_chY +mesh_OPT or x[1]<mesh_chY- mesh_OPT
      if in_opt_domain:
         values[0] = Constant(0.0)

class UpperBound(Expression):
   def __init__(self,degree):
      self.degree = degree
   def eval_cell(self, values, x, ufc_cell):
      values[0] = Constant(1.0)

adjProblem = MinimizationProblem(
   J_reduced,
   bounds         = [   interpolate(LowerBound(degree=1), U_gam),
                        interpolate(UpperBound(degree=1), U_gam)  ],
   constraints    = [],
   )
parameters = {'maximum_iterations': max_opt}
adjSolver = IPOPTSolver(
   adjProblem,
   parameters     = parameters)

gam_opt = adjSolver.solve()
gam.assign(gam_opt)
foward_solve('finalSimulation')
