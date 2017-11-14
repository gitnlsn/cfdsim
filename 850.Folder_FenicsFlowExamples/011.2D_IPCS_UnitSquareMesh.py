# PROGRAMA TREINAMENTO: CFD IPCS
#
# IMPLEMENTA ALGORITMO IPCS
#   INCREMENTAL PRESSURE CORRECTION SCHEME
#
#  
# , 11 NOVEMBRO 2016

# IMPORTACOES DE BIBLIOTECAS
from fenics import *
import numpy as np

# CONFIGURACOES DE SIMULACAO
T           = 10.;
num_steps   = 500;
dt          = T / num_steps;
mu          = 1;
rho         = 1;
mesh_Lx     = 16;
mesh_Ly     = mesh_Lx;
p_in        = 8;
p_out       = 0;


# CONSTRUCAO DO MESH
mesh = UnitSquareMesh       (mesh_Lx, mesh_Ly);
V    = VectorFunctionSpace  (mesh, 'P', 2); # ESPACO VETORIAL
Q    = FunctionSpace        (mesh, 'P', 1); # ESPACO ESCALAR

# DEFINICAO DAS CONDICOES DE CONTORNO
bc_inflow   = 'near(x[0], 0)';
bc_outflow  = 'near(x[0], 1)';
bc_walls    = 'near(x[1], 0) || near(x[1], 1)';

bc_u_noslip  = DirichletBC(V, Constant((0,0)), bc_walls   );
bc_p_inflow  = DirichletBC(Q, Constant(p_in ), bc_inflow  );
bc_p_outflow = DirichletBC(Q, Constant(p_out), bc_outflow );

BC_u = [bc_u_noslip];               # Aderencia na parede
BC_p = [bc_p_inflow, bc_p_outflow]; # Pressoes entrada/saida

# PROBLEMA VARIACIONAL
u = TrialFunction (V);
v = TestFunction  (V);
p = TrialFunction (Q);
q = TestFunction  (Q);

u_n  = Function(V);
u_n1 = Function(V);
p_n  = Function(Q);
p_n1 = Function(Q);

U   = 0.5 *(u_n + u);
n   = FacetNormal ( mesh    );
f   = Constant    ( (0,0)   );
k   = Constant    ( dt      );
mu  = Constant    ( mu      );
rho = Constant    ( rho     );

def epsilon(u):     # ESPSILON
    return sym(nabla_grad(u));

def sigma(u, p):    # TENSOR TENSAO
    return (2*mu*epsilon(u) - p*Identity( len(u)) );

# 1o Passo
F =   rho*dot( (u-u_n)/k, v)*dx                         \
    + rho*dot( dot(u_n,nabla_grad(u_n)), v)*dx          \
    + inner(sigma(U, p_n), epsilon(v))*dx               \
    + dot(p_n*n, v)*ds -dot(mu*nabla_grad(U)*n, v)*ds   \
    - rho*dot(f,v)*dx;
a1 = lhs(F);
L1 = rhs(F);
# 2o Passo
a2 = dot(nabla_grad(p  ), nabla_grad(q))*dx;
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx         \
    - 1/k * div(u_n1)*q*dx;
# 3o Passo
a3 = dot(u,v)*dx;
L3 = dot(u_n1, v)*dx - k*dot(nabla_grad(p_n1-p_n), v)*dx;

# assemble matrices
A1 = assemble(a1);
A2 = assemble(a2);
A3 = assemble(a3);

# apply boundary conditions
[bc.apply(A1) for bc in BC_u];
[bc.apply(A2) for bc in BC_p];

# SIMULACAO DO MODELO
t = 0.0;
vtkfile = File('output.pvd')
vtkfile << (u_n1, t);
for n in range(num_steps):
    t += dt;
    # Passo 1
    b1 = assemble(L1);
    [bc.apply(b1) for bc in BC_u];
    solve(A1, u_n1.vector(), b1);
    # Passo 2
    b2 = assemble(L2);
    [bc.apply(b2) for bc in BC_p];
    solve(A2, p_n1.vector(), b2);
    # Passo 3
    b3 = assemble(L3);
    solve(A3, u_n1.vector(), b3);

    # visualizacao intermediaria
    vtkfile << (p_n1, t);

    # Compute error
    u_e = Expression(('4*x[1]*(1.0 - x[1])','0'), degree=2);
    u_e = interpolate(u_e, V);
    error = np.abs(u_e.vector().array() - u_n1.vector().array() ).max();
    print('t = %.2f: error = %.3g' % (t, error));
    print('max u:', u_n1.vector().array().max());
    print('min u:', u_n1.vector().array().min());

    # atualizar solucao anterior
    u_n.assign(u_n1);
    p_n.assign(p_n1);

plot(u_n1);
interactive();
