'''

NELSON KENZO TAMASHIRO

19.07.2017

'''

# ------ LIBRARIES ------ #
from fenics          import *
from mshr            import *

# ------ SIMULATION PARAMETERS ------ #
filename = 'results_Goda'
mesh_res = 100
mesh_0   = 0.0
mesh_D   = 0.020
mesh_L   = 0.060
mesh_H   = 0.001
mesh_Cx     = 0.010
mesh_Cy     = 0.5*mesh_D
mesh_Radius = 0.1*mesh_D

cons_dt  = 0.001
cons_rho = 1E+3
cons_mu  = 1E-3
cons_dd  = 1E-8
cons_v1  = 1E-1
cons_pout = 0

a_min = 0
a_max = 1.01
v_max = cons_v1*50
p_min = -1.0E5
p_max =  1.0E5

TRANSIENT_MAX_ITE = 2000

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

# ------ VARIATIONAL FORMULATION ------ #
FE_u  = VectorElement('P', 'triangle', 2)
FE_p  = FiniteElement('P', 'triangle', 1)
FE_a  = FiniteElement('P', 'triangle', 1)
U_vel = FunctionSpace(mesh, FE_u)
U_prs = FunctionSpace(mesh, FE_p)
U_alp = FunctionSpace(mesh, FE_a)

u_lst    = project( Constant((cons_v1,0)), U_vel)
u_aux    = project( Constant((cons_v1,0)), U_vel)
u_nxt    = project( Constant((cons_v1,0)), U_vel)
p_nxt    = project( Constant(    0      ), U_prs)
a_lst    = project( Constant(    0      ), U_alp)
a_nxt    = project( Constant(    0      ), U_alp)

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

F1 = RHO*inner( u_aux -u_lst, v )/DT         *dx \
   + RHO*inner( dot(u_md,grad(u_md).T), v )  *dx \
   + MU *inner( grad(u_md),grad(v) )         *dx \

F2 = inner( grad(p_nxt),grad(q) )   *dx \
   + inner( div(u_aux), q)*RHO/DT   *dx \

F3 = inner( u_nxt -u_aux,v )        *dx \
   + inner( grad(p_nxt),v) *DT/RHO  *dx

F4 = inner(a_nxt -a_lst,b) /DT                              *dx \
   + inner(u_cv,grad(a_md))*b                               *dx \
   + inner( grad(a_md), grad(b))*DD                         *dx \
   + inner( dot(u_cv,grad(a_md)), dot(u_cv,grad(b)))*DT/2.0 *dx

# ------ BOUNDARY CONDITIONS ------ #
p_ux,p_uy,p_pp,p_ww = 0,1,2,3
BC1 = [
         DirichletBC(U_vel, u_in, inlet),
         DirichletBC(U_vel, u_wl, walls),
      ] # end - BC #

BC2 = [
         #DirichletBC(U_prs, p_in,   inlet),
         DirichletBC(U_prs, p_out, outlet),
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

nlProblem1 = NonlinearVariationalProblem(F1, u_aux, BC1, dF1)
nlProblem2 = NonlinearVariationalProblem(F2, p_nxt, BC2, dF2)
nlProblem3 = NonlinearVariationalProblem(F3, u_nxt,  [], dF3)
nlProblem4 = NonlinearVariationalProblem(F4, a_nxt, BC4, dF4)

#nlProblem2.set_bounds( project(Constant(p_min),U_alp),project(Constant(p_max),U_alp) )
#nlProblem4.set_bounds( project(Constant(a_min),U_alp),project(Constant(a_max),U_alp) )

nlSolver1  = NonlinearVariationalSolver(nlProblem1)
nlSolver2  = NonlinearVariationalSolver(nlProblem2)
nlSolver3  = NonlinearVariationalSolver(nlProblem3)
nlSolver4  = NonlinearVariationalSolver(nlProblem4)

nlSolver1.parameters["nonlinear_solver"] = "snes"
nlSolver2.parameters["nonlinear_solver"] = "snes"
nlSolver3.parameters["nonlinear_solver"] = "snes"
nlSolver4.parameters["nonlinear_solver"] = "snes"

prm1 = nlSolver1.parameters["snes_solver"]
prm2 = nlSolver2.parameters["snes_solver"]
prm3 = nlSolver3.parameters["snes_solver"]
prm4 = nlSolver4.parameters["snes_solver"]
for prm in [prm1, prm2, prm3, prm4]:
   prm["error_on_nonconvergence"       ] = True
   prm["solution_tolerance"            ] = 1.0E-16
   prm["maximum_iterations"            ] = 15
   prm["maximum_residual_evaluations"  ] = 20000
   prm["absolute_tolerance"            ] = 9.0E-13
   prm["relative_tolerance"            ] = 8.0E-13
   prm["linear_solver"                 ] = "mumps"
   #prm["sign"                          ] = "default"
   #prm["method"                        ] = "vinewtonssls"
   #prm["line_search"                   ] = "bt"
   #prm["preconditioner"                ] = "none"
   #prm["report"                        ] = True
   #prm["krylov_solver"                 ]
   #prm["lu_solver"                     ]

#set_log_level(PROGRESS)

# ------ SAVE FILECONFIGURATIONS ------ #
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

# ------ TRANSIENT SIMULATION ------ #
count_iteration   = 0
while( count_iteration < TRANSIENT_MAX_ITE ):
   count_iteration = count_iteration +1
   nlSolver1.solve()
   nlSolver2.solve()
   nlSolver3.solve()
   nlSolver4.solve()
   residual = assemble( inner(a_nxt -a_lst,a_nxt -a_lst)*dx )
   print ('Residual : {}'.format(residual) )
   print ('Iteration: {}'.format(count_iteration) )
   save_flow( u_nxt,p_nxt,a_nxt )
   a_lst.assign(a_nxt)
   u_lst.assign(u_nxt)

#plot(u, title='velocity')
#plot(p, title='pressure')
#interactive()





