'''
DESCRIPTION:

ANALYSIS:

AUTHOR: Nelson Kenzo Tamashiro

DATE: 24.06.2017
'''

# ------ LIBRARIES IMPORT ------ #
from fenics import *
from mshr import *

# ------ SIMULATION PARAMETERS ------ #
mesh_res = 40

# ------ GEOMETRICAL PARAMETERS ------ #
mesh_P0 = 0.0
mesh_DD = 1.0
mesh_RR = mesh_DD *0.2
mesh_LL = mesh_DD *2.0
mesh_CX = mesh_DD *0.5
mesh_CY = mesh_DD *0.5
foldername = 'results_014'

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

boundaries = FacetFunction('size_t', mesh)

class boundary_inlet(SubDomain):
   def inside(self, x, on_boundary):
      return on_boundary and near(x[0], mesh_P0)

class boundary_outlet(SubDomain):
   def inside(self, x, on_boundary):
      return on_boundary and near(x[0], mesh_LL)

class boundary_walls(SubDomain):
   def inside(self, x, on_boundary):
      return on_boundary \
            and (near(x[1], mesh_DD) or near(x[1], -mesh_DD))

class boundary_osbtructions(SubDomain):
   def inside(self, x, on_boundary):
      return on_boundary \
            and x[0] >  mesh_P0 \
            and x[0] <  mesh_LL \
            and x[1] > -mesh_DD \
            and x[1] <  mesh_DD

b_inlet  = boundary_inlet()
b_outlet = boundary_outlet()
b_walls  = boundary_walls()
b_osbt   = boundary_osbtructions()

dsp_inlet,dsp_outlet,dsp_walls,dsp_obst,dsp_pRef = 1,2,3,4,5

b_inlet.mark  (boundaries, dsp_inlet  )
b_outlet.mark (boundaries, dsp_outlet )
b_walls.mark  (boundaries, dsp_walls  )
b_osbt.mark   (boundaries, dsp_obst   )

ds = Measure('ds', domain=mesh, subdomain_data=boundaries )

FE    = FiniteElement('P', 'triangle', 2)
elem  = MixedElement([FE, FE, FE, FE, FE, FE, FE])
U     = FunctionSpace(mesh, elem)

ans = Function(U)
p,ux,uy,uxx,uxy,uyx,uyy = split(ans)

RHO = Constant(cons_ro)
MU  = Constant(cons_mu)

u = as_vector([ux,uy])
grad_u = as_tensor([ [uxx, uxy],
                     [uyx, uyy] ])
div_u = uxx +uyy
sigma = MU*(grad_u+grad_u.T) -p*Identity(len(u))

u_inlet = as_vector([ Constant(1E-5), Constant(0E-3) ])
u_walls = as_vector([ Constant(0E-3), Constant(0E-3) ])
sigma_out = as_tensor([ [ Constant(0), Constant(0)], 
                        [ Constant(0), Constant(0)]   ])

x,y = 0,1

F  = inner( RHO*dot(u,grad_u.T) -div(sigma),
            RHO*dot(u,grad_u.T) -div(sigma)  )           *dx  \
   + inner( div_u,div_u )                                *dx  \
   + inner(grad_u-grad(u), grad_u-grad(u))               *dx  \
   + inner(u -u_inlet, u -u_inlet)              *ds(dsp_inlet) \
   + inner(u -u_walls, u -u_walls)              *ds(dsp_walls) \
   + inner(u -u_walls, u -u_walls)              *ds(dsp_obst) \
   + inner(sigma -sigma_out, sigma -sigma_out  )  *ds(dsp_outlet) \

J = derivative(F, ans, TestFunction(U))

# ------ SOLVING PROBLEM ------ #
p_pp,p_ux,p_uy,p_uxx,p_uxy,p_uyx,p_uyy = 0,1,2,3,4,5,6
pointRef = 'near(x[0], '+str(mesh_LL)+')' \
     + ' && near(x[1], '+str(mesh_P0)+')'
BC = DirichletBC(U.sub(p_pp), Constant(0), pointRef, method='pointwise')

assign(ans.sub(p_ux ), project(Constant(1E-5), FunctionSpace(mesh, FE) ) )
assign(ans.sub(p_uy ), project(Constant(1E-5), FunctionSpace(mesh, FE) ) )
assign(ans.sub(p_pp ), project(Constant(0E-2), FunctionSpace(mesh, FE) ) )

solve(J==0, ans, [],
   solver_parameters={'newton_solver':
   {'maximum_iterations' : 10,
   'absolute_tolerance'  : 5E-18,
   'relative_tolerance'  : 6E-18,
   'relaxation_parameter': 1.0
   } })

ctt_eq = inner(   div_u,div_u )*dx
mmt_eq = inner(   RHO*dot(u,grad_u.T) -div(sigma),
                  RHO*dot(u,grad_u.T) -div(sigma) ) *dx
print assemble(ctt_eq)
print assemble(mmt_eq)

#plot(u, title='velocity')
#plot(p, title='pressure')
#interactive()

vtk_uu  = File(foldername+'/velocity.pvd')
vtk_pp  = File(foldername+'/pressure.pvd')

def save_flow():
   ui = project(u,FunctionSpace(mesh,MixedElement([FE, FE])))
   pi = project(p,FunctionSpace(mesh,FE))
   ui.rename('velocity','velocity')
   pi.rename('pressure','pressure')
   vtk_uu << ui
   vtk_pp << pi

save_flow()
