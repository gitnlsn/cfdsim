'''
Date    : 18 Jan 2017

dolfin-version: 2016.2

Description:

   This program simulates Navier-Stokes and
   Convection-Diffusion equations on TMixer
   topology.

'''


# ------ LIBRARIES ------ #
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# ------ SIMULATION PARAMETERS ------ #
mesh_res = 100

# ------ GEOMETRICAL PARAMETERS ------ #
mesh_P0 = 0.0
mesh_DD = 1.0E-2
mesh_RR = mesh_DD *0.2
mesh_LL = mesh_DD *8.0
mesh_CX = mesh_DD *0.5
mesh_CY = mesh_DD *0.5

# ------ PHYSICAL PARAMETERS ------ #
cons_ro1 = 1E+3
cons_ro2 = 1E+4
cons_mu  = 1E-3
cons_gg  = 9.8E1
cons_dd  = 1E-8

# ------ MESH AND BOUNDARIES DEFINITION ------ #
part1 = Rectangle(
   Point(mesh_P0, mesh_P0),
   Point(mesh_LL, mesh_DD),
   )
part2 = Circle(
   Point(mesh_CX, mesh_CY),
   mesh_RR
   )
domain = part1 #-part2
mesh = generate_mesh(domain, mesh_res)

# ------ FUNCTION SPACES ------ #
FE_V = FiniteElement('P', 'triangle', 2)
FE_P = FiniteElement('P', 'triangle', 1)
FE_A = FiniteElement('P', 'triangle', 1)
elem = MixedElement([FE_V, FE_V, FE_P, FE_A])
U    = FunctionSpace(mesh, elem)

# ------ FORMULACAO VARIACIONAL ------ #
x,y = 0,1
ans = Function(U)   
ux,uy,p,a = split(ans)
vx,vy,q,b = TestFunctions(U)

u = as_vector([ux,uy])
v = as_vector([vx,vy])

H   = Expression( ('x[0]','x[1]'), degree=1 )
G   = Constant(cons_gg)
RH1 = Constant(cons_ro1)
RH2 = Constant(cons_ro2)
rho = RH1*a +RH2*(Constant(1)-a)
MU  = Constant(cons_mu)
DD  = Constant(cons_dd)
GG  = Constant( (0,-cons_gg) )
NN  = FacetNormal(mesh)

SIGMA_DS = as_tensor([  [ -rho*inner(GG,H), Constant(0) ],
                        [ Constant(0),  -rho*inner(GG,H)]  ])

F_ctt = div(rho*u)*q                            *dx
F_mtt = inner( rho*dot(u,grad(u).T), v )        *dx \
      + inner( MU*(grad(u)+grad(u).T), grad(v)) *dx \
      - div(v)*p                                *dx \
      - inner( rho*GG, v)                       *dx \
      - inner( dot(SIGMA_DS,NN), v)             *ds
F_dif = inner( u,grad(a) )*b                    *dx \
      + inner( grad(a), grad(b) ) *DD           *dx

# Formulacao Final
F1 = F_ctt +F_mtt +F_dif

# ------ BOUNDARIES DEFINITION ------ #
inlet   = '( x[0]=='+str(mesh_P0)+' && x[1]>'+str(mesh_P0)+' && x[1]<'+str(mesh_DD)+' )'
inlet1  = '( x[0]=='+str(mesh_P0)+' && x[1]>='+str(mesh_DD*0.5)+')'
inlet2  = '( x[0]=='+str(mesh_P0)+' && x[1]<='+str(mesh_DD*0.5)+')'
outlet  = '( x[0]=='+str(mesh_LL)+' && x[1]>'+str(mesh_P0)+' && x[1]<'+str(mesh_DD)+' )'
outlet1 = '( x[0]=='+str(mesh_LL)+' && x[1]>='+str(mesh_DD*0.3)+')'
outlet2 = '( x[0]=='+str(mesh_LL)+' && x[1]<='+str(mesh_DD*0.3)+')'
walls   = 'on_boundary' \
        + ' && !'+inlet \
        + ' && !'+outlet
v_out = Expression('x[1]*(1-x[1])*0.01/4.0', degree=2)
#aa_out = Expression('', degree=2)
p_ux,p_uy,p_pp,p_aa = 0,1,2,3
BC = [
      DirichletBC(U.sub(p_ux), Constant(1E-2 ), inlet),
      DirichletBC(U.sub(p_uy), Constant(0 ),    inlet),
      DirichletBC(U.sub(p_aa), Constant(0.3),   inlet),
      DirichletBC(U.sub(p_ux), Constant(0 ),    walls),
      DirichletBC(U.sub(p_uy), Constant(0 ),    walls),
      #DirichletBC(U.sub(p_ux), v_out,    outlet),
      #DirichletBC(U.sub(p_uy), Constant(0),    outlet),
      DirichletBC(U.sub(p_aa), Constant(0),    outlet1),
      DirichletBC(U.sub(p_aa), Constant(1),    outlet2),
      ]

solve(F1==0, ans, BC,
   solver_parameters={'newton_solver':
   {'maximum_iterations' : 15,
   'absolute_tolerance'  : 5E-13,
   'relative_tolerance'  : 5E-13
   } })

ctt_eq = inner(   div(rho*u),div(rho*u) )*dx
mmt_eq = inner(   rho*dot(u,grad(u).T) +grad(p) +MU*div(grad(u)),
                  rho*dot(u,grad(u).T) +grad(p) +MU*div(grad(u)) ) *dx
print assemble(ctt_eq)
print assemble(mmt_eq)

vtk_uu  = File('velocity_mean.pvd')
vtk_pp  = File('pressure_mean.pvd')

ui = project(u,FunctionSpace(mesh,MixedElement([FE_V, FE_V]) ))
pi = project(p,FunctionSpace(mesh,FE_P))
ui.rename('velocity_intrinsic','velocity_intrinsic')
pi.rename('pressure_intrinsic','pressure_intrinsic')
vtk_uu << ui
vtk_pp << pi

plot(u,title='velocity')
plot(p,title='pressure')
plot(a,title='pressure')
interactive()

def save_results():
   u1_viz  = project(u   , FE_vector); vtk_u1  << u1_viz
   pp_viz  = project(p   , FE_scalar); vtk_pp  << pp_viz
   aa_viz  = project(a   , FE_scalar); vtk_aa  << aa_viz

vtk_u1  = File('velocity1.pvd')
vtk_pp  = File('pressure.pvd')
vtk_aa  = File('fraction.pvd')
FE_vector = FunctionSpace(mesh, VectorElement('P', mesh.ufl_cell(), 1))
FE_scalar = FunctionSpace(mesh, FiniteElement('P', mesh.ufl_cell(), 1))

save_results()
