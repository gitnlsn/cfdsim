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
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# ------ SIMULATION PARAMETERS ------ #
mesh_res = 100

# ------ GEOMETRICAL PARAMETERS ------ #
mesh_P0 = 0.0
mesh_DD = 1.0
mesh_RR = mesh_DD *0.2
mesh_LL = mesh_DD *2.0
mesh_CX = mesh_DD *0.5
mesh_CY = mesh_DD *0.5
foldername = 'results_015'

# ------ PHYSICAL PARAMETERS ------ #
cons_ro = 1E+3
cons_mu = 1E-3

# ------ MESH AND BOUNDARIES DEFINITION ------ #
part1 = Rectangle(
   Point(mesh_P0, mesh_P0),
   Point(mesh_LL, mesh_DD),
   )
part2 = Circle(
   Point(mesh_CX, mesh_CY),
   mesh_RR
   )
domain = part1 -part2
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

RHO = Constant(cons_ro)
MU  = Constant(cons_mu)

F_ctt = div(u)*q                                *dx
F_mtt = inner( RHO*dot(u,grad(u).T), v )        *dx \
      + inner( MU*(grad(u)+grad(u).T), grad(v)) *dx \
      - div(v)*p                                *dx

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
      DirichletBC(U.sub(p_ux), Constant(1E-5 ),    inlet),
      DirichletBC(U.sub(p_uy), Constant(0 ),    inlet),
      DirichletBC(U.sub(p_ux), Constant(0 ),    walls),
      DirichletBC(U.sub(p_uy), Constant(0 ),    walls),
      ]

solve(F1==0, ans, BC,
   solver_parameters={'newton_solver':
   {'maximum_iterations' : 15,
   'absolute_tolerance'  : 5E-18,
   'relative_tolerance'  : 5E-18
   } })

ctt_eq = inner(   div(u),div(u) )*dx
mmt_eq = inner(   RHO*dot(u,grad(u).T) +grad(p) +MU*div(grad(u)),
                  RHO*dot(u,grad(u).T) +grad(p) +MU*div(grad(u)) ) *dx
print assemble(ctt_eq)
print assemble(mmt_eq)

plot(u,title='velocity')
plot(p,title='pressure')
interactive()

vtk_uu  = File(foldername+'/velocity.pvd')
vtk_pp  = File(foldername+'/pressure.pvd')

def save_flow():
   ui = project(u,FunctionSpace(mesh,MixedElement([FE_V, FE_V])))
   pi = project(p,FunctionSpace(mesh,FE_P))
   ui.rename('velocity','velocity')
   pi.rename('pressure','pressure')
   vtk_uu << ui
   vtk_pp << pi

save_flow()
