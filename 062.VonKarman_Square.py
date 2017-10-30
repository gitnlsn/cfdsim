'''

NELSON KENZO TAMASHIRO

19.07.2017

'''

# ------ LIBRARIES ------ #
from numpy           import *
from fenics          import *
from mshr            import *

# ------ SIMULATION PARAMETERS ------ #
filename = 'results_VonKarman'
mesh_res = 300
mesh_0   = 0.0
mesh_D   = 15E-3
mesh_L   = 60E-3
mesh_H   = 0.001
mesh_Cy     = 0.5*mesh_D
mesh_Cx     = (1.0/6.0)*mesh_D
mesh_obstr  = 2.5E-3
mesh_R      = 125E-6

cons_dt  = 5.0E-3
cons_rho = 1.0E+3
cons_mu  = 1.0E-3
cons_dd  = 1.0E-8
cons_v1  = 3.0/2.0*1E-1
cons_pout = 0

T_vk     = (mesh_obstr)/(0.2*cons_v1)
N_steps  = int(5*T_vk/cons_dt)

TRANSIENT_MAX_TIME = 3.0E-0

comm = mpi_comm_world()
rank = MPI.rank(comm)

# ------ MESH ------ #
part1 = Rectangle(
   Point(mesh_0, mesh_0),
   Point(mesh_L, mesh_D)   )
part2 = Rectangle(
   Point(mesh_Cx -mesh_obstr, mesh_Cy -mesh_obstr),
   Point(mesh_Cx +mesh_obstr, mesh_Cy +mesh_obstr)    )
part3 = Circle(
   Point(mesh_Cx, mesh_Cy),
   mesh_R                     )
channel = part1 -part2
mesh = generate_mesh(channel, mesh_res)
# plot(mesh); interactive()

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

def compl(x):
   return Constant(1) -x

F1 = RHO*inner( u_aux -u_lst, v )/DT        *dx \
   + RHO*inner( dot(u_md,grad(u_md).T), v ) *dx \
   + MU *inner( grad(u_md),grad(v) )        *dx

F2 = inner( grad(p_nxt),grad(q) )  *dx \
   + inner( div(u_aux), q)*RHO/DT  *dx

F3 = inner( u_nxt -u_aux,v )       *dx \
   + inner( grad(p_nxt),v) *DT/RHO *dx

F4 = inner(a_nxt -a_lst,b) /DT                               *dx \
   + inner(u_cv,grad(a_md))*b                                *dx \
   + inner( grad(a_md), grad(b))*DD                          *dx \
   + inner( dot(u_cv,grad(a_md)), dot(u_cv,grad(b)))*DT/2.0  *dx

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
   prm["error_on_nonconvergence"       ] = False
   prm["solution_tolerance"            ] = 1.0E-16
   prm["maximum_iterations"            ] = 15
   prm["maximum_residual_evaluations"  ] = 20000
   prm["absolute_tolerance"            ] = 9.0E-15
   prm["relative_tolerance"            ] = 8.0E-15
   prm["linear_solver"                 ] = "mumps"
   #prm["sign"                          ] = "default"
   #prm["method"                        ] = "vinewtonssls"
   #prm["line_search"                   ] = "bt"
   #prm["preconditioner"                ] = "none"
   #prm["report"                        ] = True5
   #prm["krylov_solver"                 ]
   #prm["lu_solver"                     ]

#set_log_level(PROGRESS)

# ------ SAVE FILECONFIGURATIONS ------ #
vtk_uu  = File(filename+'/velocity.pvd')
vtk_pp  = File(filename+'/pressure.pvd')
vtk_aa  = File(filename+'/concentration.pvd')

def save_flow(u_tosave, p_tosave, a_tosave,time):
   ui = project(u_tosave,FunctionSpace(mesh,FE_u))
   pi = project(p_tosave,FunctionSpace(mesh,FE_p))
   ai = project(a_tosave,FunctionSpace(mesh,FE_a))
   ui.rename('velocity','velocity')
   pi.rename('pressure','pressure')
   ai.rename('concentration','concentration')
   vtk_uu << (ui,time)
   vtk_pp << (pi,time)
   vtk_aa << (ai,time)

class SimulationRecord(object):
   
   def __init__(self, dt, T_vk):
      self.record = zeros(4)
      self.dt     = dt
      self.T_vk   = T_vk
   
   def add_step(self, u_torecord, p_torecord, a_torecord):
      new_step = hstack([ 
         self.calc_mixtureEfficiency   (u_torecord, p_torecord, a_torecord),
         self.calc_pressureDrop        (u_torecord, p_torecord, a_torecord),
         self.calc_flowRate            (u_torecord, p_torecord, a_torecord),
         self.calc_power               (u_torecord, p_torecord, a_torecord),
         ])
      if rank==0:
         self.record = vstack([ self.record, new_step ])
   
   def calc_mixtureEfficiency (self, u_torecord, p_torecord, a_torecord):
      a_opt = Constant(0.5)
      return 1.0 - assemble( (a_torecord -a_opt)**2*ds(ds_outlet) )\
                 /(assemble( (a_torecord -a_opt)**2*ds(ds_inlet) )*3.0/2.0)
   
   def calc_pressureDrop      (self, u_torecord, p_torecord, a_torecord):
      return  assemble( p_torecord*ds(ds_inlet ) )/(mesh_D*2.0/3.0) \
            - assemble( p_torecord*ds(ds_outlet) )/mesh_D
   
   def calc_flowRate          (self, u_torecord, p_torecord, a_torecord):
      return assemble( inner(u_torecord, FacetNormal(mesh))*ds(ds_outlet) )/mesh_D

   def calc_power             (self, u_torecord, p_torecord, a_torecord):
      return  assemble( (RHO*inner(u_torecord, u_torecord)/2.0+p_torecord)*inner(u_torecord,FacetNormal(mesh))*ds )

   def get_properties(self):
      if rank==0:
         prop_eta       = 0.0
         prop_deltaP    = 0.0
         prop_flowRate  = 0.0
         print self.record[-N_steps:,:]
         for i in range( N_steps ):
            vertical_position = self.record.shape[0]-1-i
            prop_eta       = prop_eta        + self.record[vertical_position][0]
            prop_deltaP    = prop_deltaP     + self.record[vertical_position][1]
            prop_flowRate  = prop_flowRate   + self.record[vertical_position][2]
            prop_power     = prop_power      + self.record[vertical_position][3]
         return prop_eta/N_steps, prop_deltaP/N_steps, prop_flowRate/N_steps, prop_power/N_steps

   def get_properties_instant(self):
      if rank==0:
         return self.record[-1][0], self.record[-1][1], self.record[-1][2], self.record[-1][3]

# ------ TRANSIENT SIMULATION ------ #
t                 = 0
count_iteration   = 0
tape              = SimulationRecord(cons_dt, T_vk)
while( t < TRANSIENT_MAX_TIME ):
   count_iteration = count_iteration +1
   t = t +cons_dt
   nlSolver1.solve()
   nlSolver2.solve()
   nlSolver3.solve()
   nlSolver4.solve()
   residual = assemble( inner(a_nxt -a_lst,a_nxt -a_lst)*dx
                       +inner(u_nxt -u_lst,u_nxt -u_lst)*dx )
   save_flow( u_nxt,p_nxt,a_nxt,t )
   u_lst.assign(u_nxt)
   a_lst.assign(a_nxt)
   tape.add_step(u_nxt,p_nxt,a_nxt)
   if rank==0:
      print ('Residual : {}'.format(residual) )
      print ('Iteration: {}'.format(count_iteration) )
   prop = tape.get_properties_instant()
   if rank==0:
      print ('Properties: {}, {}, {}, {}, {}'.format(t, prop[0], prop[1], prop[2], prop[3]))
