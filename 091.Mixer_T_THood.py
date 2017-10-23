'''
Filename: 026.TMixer_TaylorHood.py
Author  : Nelson Kenzo Tamashiro
Date    : 18 Jan 2017

dolfin-version: 2016.2

Description:

   This program simulates Navier-Stokes and
   Convection-Diffusion equations on TMixer
   topology.

'''


# ------ LIBRARIES ------ #
from numpy           import *
from fenics          import *
from mshr            import *

# ------ TMIXER GEOMETRY PARAMETERS ------ #
filename = 'results_Tmixer'
mesh_res    = 500
mesh_0      = 0.0
mesh_D      = 0.010
mesh_L      = 0.085

cons_vin    = 1.0E-2
cons_v00    = 0.0E-0
cons_a01    = 1.0
cons_a02    = 0.0
cons_DD     = 1.0E-8
cons_rho    = 1.0E+3
cons_mu     = 1.0E-3

comm = mpi_comm_world()
rank = MPI.rank(comm)

# ------ MESH ------ #
part1 = Rectangle(
   Point( mesh_0, mesh_0 ),
   Point( mesh_L, mesh_D )    )
channel = part1
mesh  = generate_mesh(channel, mesh_res)

# ------ BOUNDARIES ------ #
inlet  = '( x[0]=='+str(0.0*mesh_L)+' )'
inlet1 = '( x[0]=='+str(0.0*mesh_L)+' && x[1]>='+str(mesh_D/2.0)+' )'
inlet2 = '( x[0]=='+str(0.0*mesh_L)+' && x[1]<='+str(mesh_D/2.0)+' )'
outlet = '( x[0]=='+str(1.0*mesh_L)+' )'
walls  = '( on_boundary && ((x[1]=='+str(mesh_D)+') || (x[1]=='+str(mesh_0)+'))  )'

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

# ------ FORMULACAO VARIACIONAL ------ #
FE_U  = FiniteElement('P', 'triangle', 2)
FE_P  = FiniteElement('P', 'triangle', 1)
FE_A  = FiniteElement('P', 'triangle', 1)
elem  = MixedElement([FE_U, FE_U, FE_P, FE_A])
U     = FunctionSpace(mesh, elem)

ans   = Function(U)
ux,uy,p,a = split(ans)
vx,vy,q,b = TestFunctions(U)

u = as_vector([ux,uy])
v = as_vector([vx,vy])

RHO = Constant(cons_rho )
MU  = Constant(cons_mu  )
DD  = Constant(cons_DD  )

F_ctt = div(u)*q                                *dx
F_mtt = inner( RHO*dot(u,grad(u).T), v )        *dx \
      + inner( MU*(grad(u)+grad(u).T), grad(v)) *dx \
      - div(v)*p                                *dx
F_cnv = inner( DD*grad(a),grad(b) )             *dx \
      + inner(u,grad(a))*b                      *dx

# Formulacao Final
F1 = F_ctt +F_mtt +F_cnv

# ------ CONDICOES DE CONTORNO ------ #
u_in = Constant( cons_vin )
u_00 = Constant( cons_v00 )
a_01 = Constant( cons_a01 )
a_02 = Constant( cons_a02 )
p_ux,p_uy,p_pp,p_aa = 0,1,2,3
BC1 = [
         DirichletBC(U.sub(p_ux), u_in, inlet),
         DirichletBC(U.sub(p_uy), u_00, inlet),
         DirichletBC(U.sub(p_ux), u_00, walls),
         DirichletBC(U.sub(p_uy), u_00, walls),
         DirichletBC(U.sub(p_aa), a_01, inlet1),
         DirichletBC(U.sub(p_aa), a_02, inlet2),
      ]

# ------ NON LINEAR PROBLEM DEFINITIONS ------ #
dF1 = derivative(F1, ans)
nlProblem1 = NonlinearVariationalProblem(F1, ans, BC1, dF1)
nlSolver1  = NonlinearVariationalSolver(nlProblem1)
nlSolver1.parameters["nonlinear_solver"] = "snes"

prm = nlSolver1.parameters["snes_solver"]
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

# ------ SAVE FILECONFIGURATIONS ------ #
vtk_uu  = File(filename+'/velocity.pvd')
vtk_pp  = File(filename+'/pressure.pvd')
vtk_aa  = File(filename+'/concentration.pvd')

def save_flow(u_tosave, p_tosave, a_tosave, time):
   ui = project(u_tosave,FunctionSpace(mesh,MixedElement([FE_U,FE_U])))
   pi = project(p_tosave,FunctionSpace(mesh,FE_P))
   ai = project(a_tosave,FunctionSpace(mesh,FE_A))
   ui.rename('velocity','velocity')
   pi.rename('pressure','pressure')
   ai.rename('concentration','concentration')
   vtk_uu << (ui,time)
   vtk_pp << (pi,time)
   vtk_aa << (ai,time)

def calc_mixtureEfficiency (u_torecord, p_torecord, a_torecord):
   a_opt = Constant(0.5)
   return 1.0 - assemble( (a_torecord -a_opt)**2*ds(ds_outlet) )/assemble( (a_torecord -a_opt)**2*ds(ds_inlet) )

def calc_pressureDrop      (u_torecord, p_torecord, a_torecord):
   return  assemble( p_torecord*ds(ds_inlet ) )/mesh_D \
         - assemble( p_torecord*ds(ds_outlet) )/mesh_D

def calc_flowRate          (u_torecord, p_torecord, a_torecord):
   return assemble( inner(u_torecord, FacetNormal(mesh))*ds(ds_outlet) )/mesh_D

def get_properties(u_torecord, p_torecord, a_torecord):
   prop_eta       = calc_mixtureEfficiency   (u_torecord, p_torecord, a_torecord)
   prop_deltaP    = calc_pressureDrop        (u_torecord, p_torecord, a_torecord)
   prop_flowRate  = calc_flowRate            (u_torecord, p_torecord, a_torecord)
   return prop_eta, prop_deltaP, prop_flowRate


for vel in [10**-(2+0.01*exp) for exp in range(200)]:
   u_in.assign( vel )
   nlSolver1.solve()
   save_flow(u,p,a, vel)
   prop = get_properties(u,p,a)
   if rank==0:
      print ('Solved for velocity = {}'.format(vel))
      print ('Properties: {}, {}, {}, {}'.format(vel, prop[0], prop[1], prop[2]))

