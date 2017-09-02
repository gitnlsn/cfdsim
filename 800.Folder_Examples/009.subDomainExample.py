from fenics import *

k1 = 1; k2 = 10;
tol = 1E-14;
mat_type_1 = 1;
mat_type_2 = 2;

mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh, 'P', 1);

# primeira maneira de definir subdominios
#k = Expression('x[0]>=0.5+1E-14 ? k1 : k2',
#   k1=k1, k2=k2, degree=2)

materials = CellFunction('int', mesh)
class mat1(SubDomain):
   def inside(self, x, on_boundary):
      return x[0]>=0.5-tol
class mat2(SubDomain):
   def inside(self, x, on_boundary):
      return x[0]<=0.5+tol
k1_mat = mat1()
k2_mat = mat2()
k1_mat.mark(materials, 1)
k2_mat.mark(materials, 2)

class K(Expression):
   def __init__(self, materials, k1, k2, **kwargs):
      self.materials = materials
      self.k1 = k1
      self.k2 = k2
   def eval_cell(self, values, x, cell):
      if self.materials[cell.index]==1:
         values[0]=self.k1
      else:
         values[0]=self.k2

k = K(materials, k1, k2, degree=0)

bc1 = DirichletBC(V, Constant(0), 'near(x[0], 0)')
bc2 = DirichletBC(V, Constant(1), 'near(x[0], 1)')
bc = [bc1, bc2];

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0);
a = dot(grad(u),grad(v))*k*dx
l = f*v*dx;

u = Function(V)
solve(a==l, u, bc );

plot(u, interactive=True);