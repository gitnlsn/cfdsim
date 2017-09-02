# PROGRAMA TREINO - TRANSFERENCIA DE CALOR TRANSIENTE
#
# DESCRICAO: ESSE PROBLEMA EXPLORA UMA TRANSFERENCIA DE CALOR
# EM REGIME TRANSIENTE; PARA AVALIAR A CAPACIDADE DE RESOLVER
# A SIMULACAO DO FENICS, PROPOE-SE CRIAR CONDICOES DE CONTORNO
# E DE EXCITACAO EXTERNAS CUJA SOLUCAO E CONHECIDA PARA PERMI-
# TIR A COMPARACAO DOS RESULTADOS NUMERICOS DA SIMULACAO COM O
# RESULTADO CONHECIDO.
#
# NELSON KENZO TAMASHIRO
# SAO PAULO, 20 OUTUBRO 2016.

from fenics import *
import numpy as np

# 1) DECLARACAO DE PARAMETROS FISICOS DO SISTEMA
alpha = 3;			# Parametros da solucao: 
beta = 5;	 		# 	u = 1 +x^2+ alpha*y^2 +beta*t
T = 1.				# Tempo de simulacao: 2 segundos
num_steps = 100		# Discretizacao temporal: 10 passos
dt = T / num_steps	# Intervalo de tempo de cada passo
nx = ny = 16		# Discretizacao espacial: 8 passos
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
	alpha=alpha, beta=beta, t=0, degree=2);
					# Solucao do problema
vtkfile = File('output.pvd')
### FIM: 1.PARAMETROS INICIAIS


# 2) CONSTRUCAO DO MESH
mesh = UnitSquareMesh(nx,ny)
###  FIM: 2.MESH


# 3) DECLARACAO DO PROBLEMA VARIACIONAL
V = FunctionSpace(mesh, 'P', 1);
u = TrialFunction(V);
v = TestFunction(V);
f = Constant(beta -2 -2*alpha);
u_n = interpolate(u_D, V); # Instante Inicial
#u_n = project(u_D, V)
#F = u*v*dx +dt*dot(grad(u),grad(v))*dx -(u_n+dt*f)*v*dx
a = (u*v + dt*dot(grad(u),grad(v)) )*dx
L = (u_n+dt*f)*v*dx
### FIM: 3.PROBLEMA VARIACIONAL


# 4) INSTANTE INICIAL 			>>> RETIRADO
### FIM: 4.INSTANTE INICIAL


# 5) CONDICOES DE CONTORNO
def boundary(x, on_boundary):
	return on_boundary
bc = DirichletBC(V, u_D, boundary)
### FIM: 5. CONDICOES DE CONTORNO


# 6) CALCULOS TRANSIENTES
u = Function(V)
t = 0
vtkfile << (u_n, float(t))
for n in range(num_steps):
	
	# 6.1) ATUALIZACAO DO TEMPO
	t += dt
	u_D.t = t

	# 6.2) RESOLUCAO DO PROBLEMA VARIACIONAL
	solve(a==L, u, bc)

	# 6.3) UPDATE PREVIOUS SOLUTION
	u_n.assign(u)

	# 6.4) CALCULO DO ERRO
	u_e = interpolate(u_D, V)
	error = np.abs(u_e.vector().array()-u.vector().array()).max()
	print('t=%2.2f error=%2.2g' % (t,error))
	vtkfile << (u_n, float(t))

### FIM: 6.SIMULACAO TRANSIENTE
