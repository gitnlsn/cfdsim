from fenics import *

mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh, 'P', 1)
u = TrialFunction(V)
v = TestFunction(V)

boundaries = FacetFunction('size_t', mesh)

class boundary_inlet(SubDomain):
   def inside(self, x, on_boundary):
      return on_boundary and near(x[0],0)

class boundary_outlet(SubDomain):
   def inside(self, x, on_boundary):
      return on_boundary and near(x[0],1)

class boundary_upper_wall(SubDomain):
   def inside(self, x, on_boundary):
      return on_boundary and near(x[1],1)

class boundary_bottom_wall(SubDomain):
   def inside(self, x, on_boundary):
      return on_boundary and near(x[1],0)

b_Inlet = boundary_inlet()
b_Outlet = boundary_outlet()
b_Upper_wall = boundary_upper_wall()
b_bottom_wall = boundary_bottom_wall()

b_Inlet.mark(boundaries, 0)
b_Outlet.mark(boundaries, 1)
b_Upper_wall.mark(boundaries, 2)
b_bottom_wall.mark(boundaries, 3)

boundary_conditions = { 0: {'Neumann': 1},
                        1: {'Dirichlet': 0},
                        2: {'Neumann': 5},
                        3: {'Neumann': 5}}

ds = Measure('ds', domain=mesh, subdomain_data=boundaries )

integral_N = []
for i in boundary_conditions:
   if 'Neumann' in boundary_conditions[i]:
      if boundary_conditions[i]['Neumann'] !=0:
         g = boundary_conditions[i]['Neumann']
         integral_N.append(g*v*ds(i))

BC = []
for i in boundary_conditions:
   if 'Dirichlet' in boundary_conditions[i]:
      bc = DirichletBC(V, boundary_conditions[i]['Dirichlet'], boundaries, i)
      BC.append(bc)

f = Constant(0);
a = dot(grad(u), grad(v))*dx
l = f*v*dx -sum(integral_N)

s = Function(V)
solve(a==l, s, BC)
plot(mesh)
plot(s, interactive=True)

