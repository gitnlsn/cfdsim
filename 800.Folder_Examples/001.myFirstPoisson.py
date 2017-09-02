# PROGRAMA TESTE
# SAO CAETANO DO SUL, 17 OUTUBRO 2016
# NELSON KENZO TAMASHIRO

# 1) IMPORTAR BIBLIOTECA
from fenics import *
import numpy as np

# 2) CRIAR MESH E FUNCTION SPACE
mesh = UnitSquareMesh( 10, 10)
V = FunctionSpace(mesh, 'P', 1)

# 3) CONDICOES DE CONTORNO
#u_D = Expression("(x[0]-.5)*(x[0]-.5) + (x[1]-.5)*(x[1]-.5)", degree=2)
u_D = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)
def boundary (x, on_boundary):
	return on_boundary
BC = DirichletBC(V, u_D, boundary)

# 4) DEFINICAO DO PROBLEMA VARIACIONAL
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6);
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# 5) RESOLUCAO DO PROBLEMA VARIACIONAL
u = Function(V)
solve(a==L, u, BC)

# 6) GRAFICO DA SOLUCAO
u.rename('u', 'solution')
#plot(u)
#plot(mesh)

# 7) SALVAR SOLUCAO EM ARQUIVO
vtkfile = File('poisson.pvd')
vtkfile << u

# 8) CALCULO DO ERRO DA SOLUCAO
error_L2 = errornorm(u_D, u, 'L2')
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u   =   u.compute_vertex_values(mesh)
error_max = np.max(np.abs(vertex_values_u - vertex_values_u_D))
print error_L2
print error_max

#interactive()
