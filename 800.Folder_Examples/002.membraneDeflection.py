# PROGRAMA TESTE DEFLEXAO EM MEMBRANA
# PDF: FENICS TUTORIAL 1.4
#  
# , 17 OUTUBRO 2016

# 1) IMPORTACAO DO FENICS
from fenics import *
from mshr import *

# 2) CRIACAO DO MESH
domain = Circle( Point(0.0, 0.0), 1.0)
mesh = generate_mesh(domain, 50)
V = FunctionSpace(mesh, 'P', 1)
#plot(mesh, interactive=True)

# 3) PRESSAO NA MEMBRANA
beta = 8
R0 = 0.6
p = Expression (
	'4*exp( -pow(beta,2) *( pow(x[0],2)+pow(x[1]-R0,2) )  )',
	beta=beta, R0=R0, degree=2)
def boundary (x, on_boundary):
	return on_boundary
BC = DirichletBC(V, p, boundary)


# 4) PROBLEMA VARIACIONAL
w = TrialFunction(V)
v = TestFunction(V)
a = dot( grad(w), grad(v) )*dx
L = p*v*dx

# 5) RESOLUCAO DO PROBLEMA VARIACIONAL
w = Function(V)
solve (a==L, w, BC)

# 6) INTERPOLACAO DA PRESSAO
p = interpolate(p, V)

# 7) GRAFICOS DE DEFLEXAO E PRESSAO
plot(w, title = 'Deflection')
plot(p, title = 'Pressure')

# 8) SAVAR ARQUIVOS
vtkfile = File('membrane_deflection.pvd')
vtkfile << w
vtkfile = File('membrane_deflection.pvd')
vtkfile << p


# 9) GRAFICO YZ AO LONGO DA CURVA
import numpy as np
import matplotlib.pyplot as plt
tol = 1E-8
y = np.linspace(-1+tol, +1-tol, 101)
points = [(0,y_) for y_ in y]
w_line = np.array( [w(point) for point in points] )
p_line = np.array( [p(point) for point in points] )
plt.plot(y, 100*w_line, 'r-', y, p_line, 'b-')
plt.legend(['100*deflection', 'pressure'], loc='uppper_left')
plt.xlabel('y-coordinate')
plt.ylabel('100*deflection and pressure')
plt.savefig('deflection.png')

# 99) MODO INTERATIVO
#interactive()
plt.show()
