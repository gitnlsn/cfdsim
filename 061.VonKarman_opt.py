'''

NELSON KENZO TAMASHIRO

19.07.2017

'''

# ------ LIBRARIES ------ #
from fenics          import *
from mshr            import *
from dolfin_adjoint  import *

# ------ SIMULATION PARAMETERS ------ #
filename = 'results_VonKarman'
mesh_res = 80
mesh_0   = 0.0
mesh_D   = 0.020
mesh_L   = 0.060
mesh_H   = 0.001
mesh_Cx     = 0.010
mesh_Cy     = 0.5*mesh_D
mesh_Radius = 0.1*mesh_D
obstr_size  = mesh_D*1.0/3.0

cons_dt   = 0.002
cons_rho  = 1.0E+3
cons_mu   = 1.0E-3
cons_dd   = 1.0E-8
cons_v1   = 1.0E-1
cons_pout = 0.0
cons_tol  = 1.0E-3

a_min = 0
a_max = 1.00
v_max = cons_v1*50
p_min = -1.0E5
p_max =  1.0E5

k_mat = 3
FMAX  = 1.0E5

TRANSIENT_MAX_TIME = 1.00
TRANSIENT_MAX_ITE  = int(TRANSIENT_MAX_TIME/cons_dt)
OPTIMIZAT_MAX_ITE  = 100000

# ------ MESH ------ #
part1 = Rectangle(
   Point(mesh_0, mesh_0),
   Point(mesh_L, mesh_D)   )
part2 = Circle(
   Point(mesh_Cx, mesh_Cy),
   mesh_Radius             )
channel = part1 -part2
mesh = generate_mesh(channel, mesh_res)

# ------ BOUNDARIES ------ #
inlet  = '( x[0]=='+str(0.0*mesh_L)+' )'
inlet1 = '( x[0]=='+str(0.0*mesh_L)+' && x[1]>='+str(mesh_D/2.0)+' )'
inlet2 = '( x[0]=='+str(0.0*mesh_L)+' && x[1]<='+str(mesh_D/2.0)+' )'
outlet = '( x[0]=='+str(1.0*mesh_L)+' )'
obstcl = '( on_boundary && (x[0]>'+str(mesh_0)+') && (x[0]<'+str(mesh_L)+') '\
                     + '&& (x[1]>'+str(mesh_0)+') && (x[1]<'+str(mesh_D)+')   )'
walls  = '( on_boundary && ((x[1]=='+str(mesh_D)+') || (x[1]=='+str(mesh_0)+'))  ) || '+obstcl

ds_inlet, ds_walls, ds_outlet = 1,2,3

boundaries     = FacetFunction ('size_t', mesh)
side_walls     = CompiledSubDomain( walls  )
side_inlet     = CompiledSubDomain( inlet  )
side_outlet    = CompiledSubDomain( outlet )
boundaries.set_all(0)
side_walls.mark   (boundaries, ds_walls  )
side_inlet.mark   (boundaries, ds_inlet  )
side_outlet.mark  (boundaries, ds_outlet )
ds = Measure( 'ds', subdomain_data=boundaries )

domain = CellFunction ('size_t', mesh)
functional_domain = '(x[0]>'+str(mesh_Cx+obstr_size*2.0)+')'
dx_to_opt  = 1
CompiledSubDomain( functional_domain ).mark( domain, dx_to_opt )
dx = Measure('dx', subdomain_data=domain )

# ------ VARIATIONAL FORMULATION ------ #
FE_u  = VectorElement('P', 'triangle', 2)
FE_p  = FiniteElement('P', 'triangle', 1)
FE_a  = FiniteElement('P', 'triangle', 1)
FE_m  = FiniteElement('P', 'triangle', 1)
U_vel = FunctionSpace(mesh, FE_u)
U_prs = FunctionSpace(mesh, FE_p)
U_alp = FunctionSpace(mesh, FE_a)
U_mat = FunctionSpace(mesh, FE_m)

class initChannel(Expression):
   def eval(self, value, x):
      is_obstacle = (x[0] -mesh_Cx)**2 + (x[1] -mesh_Cy)**2 <= mesh_Radius**2
      if is_obstacle:
         value[0] = 1.0
      else:
         value[0] = 0.0

class initChannel2(Expression):
   def eval(self, value, x):
      is_obstacle  = (x[0] -mesh_Cx) < +obstr_size \
                 and (x[0] -mesh_Cx) > -obstr_size \
                 and (x[1] -mesh_Cy) < +obstr_size \
                 and (x[1] -mesh_Cy) > -obstr_size
      if is_obstacle:
         value[0] = 0.5
      else:
         value[0] = 0.0

TOL      = Constant(cons_tol  )

def mat(x,k):
   return 1.0/2.0+ (1.0 -2.0*TOL)*tanh((x*2.0-1.0)*k)/(tanh(k)*2.0)

alpha    = project( initChannel(degree=1), U_mat, name='alpha')
#alpha    = project( Constant(0), U_mat, annotate=True)
u_lst    = Function(U_vel) #project( Constant((cons_v1,0)), U_vel)
u_aux    = Function(U_vel) #project( Constant((cons_v1,0)), U_vel)
u_nxt    = Function(U_vel) #project( Constant((cons_v1,0)), U_vel)
p_nxt    = Function(U_prs) #project( Constant(    0      ), U_prs)
a_lst    = Function(U_alp) #project( Constant(    0      ), U_alp)
a_nxt    = Function(U_alp) #project( Constant(    0      ), U_alp)

v = TestFunction(U_vel)
q = TestFunction(U_prs)
b = TestFunction(U_alp)

DT       = Constant(cons_dt   )
RHO      = Constant(cons_rho  )
MU       = Constant(cons_mu   )
DD       = Constant(cons_dd   )
u_inlet  = Constant(cons_v1   )
n        = FacetNormal(mesh)

#in_profile1 = Expression(str(cons_v1)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', degree=2)
#in_profile2 = Expression(str(cons_v2)+'*x[1]*('+str(mesh_D)+'-x[1])/('+str((mesh_D**2.0) /6.0)+')', degree=2)

u_in  = as_vector([ u_inlet    , Constant(0) ])
u_wl  = as_vector([ Constant(0), Constant(0) ])
p_out = Constant(cons_pout )

u_md = (u_aux+u_lst)*0.5
u_cv = (u_nxt+u_lst)*0.5
a_md = (a_nxt+a_lst)*0.5

def compl(x):
   return (2.0*TOL-1.0)*x +(1.0 -TOL)

# plot(alpha)
# plot(compl(alpha))
# plot(mat(alpha,k_mat))
# interactive()

F1 = RHO*inner( u_aux -u_lst, v )/DT       *compl(alpha) *dx \
   + RHO*inner( dot(u_md,grad(u_md).T), v )*compl(alpha) *dx \
   + MU *inner( grad(u_md)*compl(alpha)+outer(u_md,grad(compl(alpha))),grad(v) )        *dx \
   + FMAX*inner( u_md*mat(alpha,k_mat), v )  *dx

F2 = inner( grad(p_nxt),grad(q) )*compl(alpha)  *dx \
   + inner( div(u_aux*compl(alpha)), q)*RHO/DT  *dx

F3 = inner( u_nxt -u_aux,v )      *compl(alpha) *dx \
   + inner( grad(p_nxt),v)*compl(alpha) *DT/RHO *dx

F4 = inner(a_nxt -a_lst,b) /DT                              *compl(alpha) *dx \
   + inner(u_cv,grad(a_md))*b                               *compl(alpha) *dx \
   + inner( grad(a_md), grad(b))*DD                         *compl(alpha) *dx \
   + inner( dot(u_cv,grad(a_md)), dot(u_cv,grad(b)))*DT/2.0 *compl(alpha) *dx

F5 = inner(u_lst,v)*dx
F6 = inner(a_lst,b)*dx

# ------ BOUNDARY CONDITIONS ------ #
p_ux,p_uy,p_pp,p_ww = 0,1,2,3
BC1 = [
         DirichletBC(U_vel, u_in, inlet),
         DirichletBC(U_vel, u_wl, walls),
      ] # end - BC #

BC2 = [
         #DirichletBC(U_prs, p_in,   inlet),
         DirichletBC(U_prs, p_out, outlet),
         #DirichletBC(U_prs, p_out, 'x[0]=='+str(mesh_L)+' && x[1]=='+str(mesh_D*0.5), method='pointwise'),
      ] # end - BC #

BC4 = [
         DirichletBC(U_alp, Constant(1.0), inlet1),
         DirichletBC(U_alp, Constant(0.0), inlet2),
      ] # end - BC #

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
dF1 = derivative(F1, u_aux)
dF2 = derivative(F2, p_nxt)
dF3 = derivative(F3, u_nxt)
dF4 = derivative(F4, a_nxt)
dF5 = derivative(F5, u_lst)
dF6 = derivative(F6, a_lst)

nlProblem1 = NonlinearVariationalProblem(F1, u_aux, BC1, dF1)
nlProblem2 = NonlinearVariationalProblem(F2, p_nxt, BC2, dF2)
nlProblem3 = NonlinearVariationalProblem(F3, u_nxt,  [], dF3)
nlProblem4 = NonlinearVariationalProblem(F4, a_nxt, BC4, dF4)
nlProblem5 = NonlinearVariationalProblem(F5, u_lst,  [], dF5)
nlProblem6 = NonlinearVariationalProblem(F6, a_lst,  [], dF6)

#nlProblem2.set_bounds( project(Constant(p_min),U_alp),project(Constant(p_max),U_alp) )
#nlProblem4.set_bounds( project(Constant(a_min),U_alp),project(Constant(a_max),U_alp) )

nlSolver1  = NonlinearVariationalSolver(nlProblem1)
nlSolver2  = NonlinearVariationalSolver(nlProblem2)
nlSolver3  = NonlinearVariationalSolver(nlProblem3)
nlSolver4  = NonlinearVariationalSolver(nlProblem4)
nlSolver5  = NonlinearVariationalSolver(nlProblem5)
nlSolver6  = NonlinearVariationalSolver(nlProblem6)

nlSolver1.parameters["nonlinear_solver"] = "snes"
nlSolver2.parameters["nonlinear_solver"] = "snes"
nlSolver3.parameters["nonlinear_solver"] = "snes"
nlSolver4.parameters["nonlinear_solver"] = "snes"
nlSolver5.parameters["nonlinear_solver"] = "snes"
nlSolver6.parameters["nonlinear_solver"] = "snes"

prm1 = nlSolver1.parameters["snes_solver"]
prm2 = nlSolver2.parameters["snes_solver"]
prm3 = nlSolver3.parameters["snes_solver"]
prm4 = nlSolver4.parameters["snes_solver"]
prm5 = nlSolver5.parameters["snes_solver"]
prm6 = nlSolver6.parameters["snes_solver"]

for prm in [prm1, prm2, prm3, prm4, prm5, prm6]:
   prm["error_on_nonconvergence"       ] = False
   prm["solution_tolerance"            ] = 1.0E-16
   prm["maximum_iterations"            ] = 15
   prm["maximum_residual_evaluations"  ] = 20000
   prm["absolute_tolerance"            ] = 9.0E-12
   prm["relative_tolerance"            ] = 8.0E-12
   prm["linear_solver"                 ] = "mumps"
   #prm["krylov_solver"                 ] = "amg"
   #prm["sign"                          ] = "default"
   #prm["method"                        ] = "vinewtonssls"
   #prm["line_search"                   ] = "bt"
   #prm["preconditioner"                ] = "none"
   #prm["report"                        ] = True
   #prm["lu_solver"                     ]

#set_log_level(PROGRESS)

# # ------ SAVE FILECONFIGURATIONS ------ #
# vtk_uu  = File(filename+'/velocity.pvd')
# vtk_pp  = File(filename+'/pressure.pvd')
# vtk_aa  = File(filename+'/concentration.pvd')

# def save_flow(u_tosave, p_tosave, a_tosave,time):
#    ui = project(u_tosave,FunctionSpace(mesh,FE_u), annotate=False)
#    pi = project(p_tosave,FunctionSpace(mesh,FE_p), annotate=False)
#    ai = project(a_tosave,FunctionSpace(mesh,FE_a), annotate=False)
#    ui.rename('velocity','velocity')
#    pi.rename('pressure','pressure')
#    ai.rename('concentration','concentration')
#    vtk_uu << (ui,time)
#    vtk_pp << (pi,time)
#    vtk_aa << (ai,time)

class Transient_flow_save():
   def __init__(self, folderName):
      self.vtk_uu  = File(folderName+'/velocity.pvd')
      self.vtk_pp  = File(folderName+'/pressure.pvd')
      self.vtk_aa  = File(folderName+'/concentration.pvd')
   def save_flow(self, u_tosave, p_tosave, a_tosave):
      ui = project(u_tosave,FunctionSpace(mesh,FE_u), annotate=False)
      pi = project(p_tosave,FunctionSpace(mesh,FE_p), annotate=False)
      ai = project(a_tosave,FunctionSpace(mesh,FE_a), annotate=False)
      ui.rename('velocity','velocity')
      pi.rename('pressure','pressure')
      ai.rename('concentration','concentration')
      self.vtk_uu << ui
      self.vtk_pp << pi
      self.vtk_aa << ai

# ------ TRANSIENT SIMULATION ------ #
def foward(folderName, annotate=True, MAX_ITERATIONS=TRANSIENT_MAX_ITE):
   t                 = 0.0
   count_iteration   = 0
   flowSto = Transient_flow_save(folderName)
   nlSolver5.solve(annotate=annotate)
   nlSolver6.solve(annotate=annotate)
   #if annotate: adj_start_timestep()
   while( count_iteration < MAX_ITERATIONS ):
      count_iteration   = count_iteration +1
      t                 = t               +cons_dt
      print ('Iteration: {}'.format(count_iteration) )
      nlSolver1.solve(annotate=annotate)
      nlSolver2.solve(annotate=annotate)
      nlSolver3.solve(annotate=annotate)
      nlSolver4.solve(annotate=annotate)
      # residual = assemble( inner(a_nxt -a_lst,a_nxt -a_lst)*dx
      #                     +inner(u_nxt -u_lst,u_nxt -u_lst)*dx )
      #print ('Residual : {}'.format(residual) )
      flowSto.save_flow(u_nxt,p_nxt,a_nxt)
      u_lst.assign(u_nxt, annotate=annotate)
      a_lst.assign(a_nxt, annotate=annotate)
      # if annotate:
      #    if count_iteration==TRANSIENT_MAX_ITE:
      #       print ('Annotate: time {}, finished'.format(count_iteration))
      #       adj_inc_timestep(time=t, finished=True)
      #    else:
      #       print ('Annotate: time {}, not finished'.format(count_iteration))
      #       adj_inc_timestep(time=t, finished=False)

foward('01.InitialGuess', annotate=True)

########################################################
# ------ ------ 02) ADJOINT OPTIMIZATION ------ ------ #
########################################################

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")

# ------ OTIMIZATION STEP POS EVALUATION ------ #
vtk_gam = File(filename+'/porosity.pvd')
vtk_dj  = File(filename+'/gradient.pvd')
gam_viz = Function(U_mat); gam_viz.rename('porosity','porosity')
djj_viz = Function(U_mat); djj_viz.rename('gradient','gradient')

#fig     = plot(alpha, title='Gradient', mode='color')

# ------ FUNCTIONAL DEFINITION ------ #
# a_obj    = Constant(0.5)
# J        = inner(a_nxt -a_obj,a_nxt -a_obj)*dx(dx_to_opt)
m        = Control(alpha)
J        = inner(grad(a_nxt),grad(a_nxt))*dx(dx_to_opt)

def post_eval(j, m):
   print ('Post_eval')
   gam_viz.assign(m, annotate=False)
   vtk_gam << gam_viz
   alpha.assign(m, annotate=False)

def derivative_cb(j, dj, m):
   #fig.plot(dj)
   print ('calculating djj_viz assign')
   gam_viz.assign(m, annotate=False)
   vtk_gam << gam_viz
   djj_viz.assign(dj, annotate=False)
   vtk_dj  << djj_viz
   print ("j = %f" % (j))
   foward('post_eval', annotate=False, MAX_ITERATIONS=300)

J_reduced = ReducedFunctional(
      functional  = Functional( J ),
      controls    = m,
      #eval_cb_post       = post_eval,
      derivative_cb_post = derivative_cb
      )

class LowerBound(Expression):
   def __init__(self,degree):
      self.degree = degree
   def eval_cell(self, values, x, ufc_cell):
      values[0] = Constant(0.0)

class UpperBound(Expression):
   def __init__(self,degree):
      self.degree = degree
   def eval_cell(self, values, x, ufc_cell):
      values[0] = Constant(0.0)
      can_be_solid = (x[0] -mesh_Cx) < +obstr_size \
                 and (x[0] -mesh_Cx) > -obstr_size
                 # and (x[1] -mesh_Cy) < +obstr_size \
                 # and (x[1] -mesh_Cy) > -obstr_size
      if can_be_solid:
         values[0] = Constant(1.0)

# ------ OPTIMIZATION PROBLEM DEFINITION ------ #
# adjProblem = MinimizationProblem(
#    J_reduced,
#    #bounds = (0.0,1.0),
#    bounds         = [   interpolate(LowerBound(degree=1), U_mat),
#                         interpolate(UpperBound(degree=1), U_mat)  ],
#    #constraints    = [],
#    )
# parameters = {'maximum_iterations': OPTIMIZAT_MAX_ITE}
# adjSolver = IPOPTSolver(
#    adjProblem,
#    parameters     = parameters)
# alpha_opt = adjSolver.solve()
alpha_opt = minimize(J_reduced,
   method   =  'L-BFGS-B',
   tol      =  2e-08,
   bounds   =  [  interpolate(LowerBound(degree=1), U_mat),
                  interpolate(UpperBound(degree=1), U_mat)  ],
   options  =  {"disp": True}                                     )
alpha.assign(alpha_opt, annotate=False)
foward('02.solution', annotate=False)

