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
mesh_LL = mesh_DD *3.0
mesh_CX = mesh_DD *0.5
mesh_CY = mesh_DD *0.5

# ------ PHYSICAL PARAMETERS ------ #
cons_ro = 1E+3
cons_mu = 1E-3
cons_gg = 9.8

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
elem = MixedElement([FE_V, FE_V, FE_P])
U    = FunctionSpace(mesh, elem)

# ------ FORMULACAO VARIACIONAL ------ #
x,y = 0,1
ans = Function(U)   
ux,uy,p = split(ans)
vx,vy,q = TestFunctions(U)

u = as_vector([ux,uy])
v = as_vector([vx,vy])

H   = Expression( 'x[1]', degree=1 )
G   = Constant(cons_gg)
RHO = Constant(cons_ro)
MU  = Constant(cons_mu)
GG  = Constant( (0,-cons_gg) )
NN  = FacetNormal(mesh)

sigma = MU*(grad(u)+grad(u).T) -p*Identity(len(u))

SIGMA_DS = as_tensor([  [ +RHO*G*H, Constant(0) ],
                        [ Constant(0),  +RHO*G*H]  ])

F_ctt = div(u)*q                                *dx
F_mtt = inner( RHO*dot(u,grad(u).T), v )        *dx \
      + inner( sigma, grad(v))                  *dx \
      - inner( RHO*GG, v)                       *dx \
      - inner( dot(SIGMA_DS,NN), v)             *ds

# Formulacao Final
F1 = F_ctt +F_mtt

# ------ CONDICOES DE CONTORNO ------ #
# ------ BOUNDARIES DEFINITION ------ #
inlet   = '( x[0]=='+str(mesh_P0)+' && x[1]>'+str(mesh_P0)+' && x[1]<'+str(mesh_DD)+' )'
outlet  = '( x[0]=='+str(mesh_LL)+' && x[1]>'+str(mesh_P0)+' && x[1]<'+str(mesh_DD)+' )'
walls   = 'on_boundary' \
        + ' && !'+inlet \
        + ' && !'+outlet
p_ux,p_uy,p_pp = 0,1,2
BC = [
      DirichletBC(U.sub(p_ux), Constant(1E-2 ),    inlet),
      DirichletBC(U.sub(p_uy), Constant(0 ),    inlet),
      DirichletBC(U.sub(p_ux), Constant(0 ),    walls),
      DirichletBC(U.sub(p_uy), Constant(0 ),    walls),
      #DirichletBC(U.sub(p_pp), Constant(0 ),    walls)
      ]

solve(F1==0, ans, BC,
   solver_parameters={'newton_solver':
   {'maximum_iterations' : 15,
   'absolute_tolerance'  : 5E-13,
   'relative_tolerance'  : 5E-13
   } })

ctt_eq = inner(   div(u),div(u) )*dx
mmt_eq = inner(   RHO*dot(u,grad(u).T) +grad(p) +MU*div(grad(u)),
                  RHO*dot(u,grad(u).T) +grad(p) +MU*div(grad(u)) ) *dx
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
interactive()
