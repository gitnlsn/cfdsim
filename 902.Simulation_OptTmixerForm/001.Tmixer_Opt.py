'''
Filename: 001.Tmixer_Opt.py
Author  : Nelson Kenzo Tamashiro
Date    : 24 Jun 2017

dolfin-version: 2017.1

Description:

'''


# ------ LIBRARIES ------ #
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# ------ TMIXER GEOMETRY PARAMETERS ------ #
res     = 350
opt_par = 1.0E2
d       = 0.010         # 10mm
D       = 8.8E-10       # 8.8E-6 cm**2/s
rho     = 1.0E3         # 1kg/m**3
mu      = 8.5E-4        # 0.00089 N*s/m**2
g       = 9.8
L_in    = 0.5*d
v_in    = 2E-2

# ------ SIMULATION PARAMETERS ------ #
Re = rho*v_in*d/mu
Pe = d*v_in/D
print('Velocity: {:.2e}'.format(v_in))
print('Reynolds: {:.2e}'.format(Re))
print('Peclet  : {:.2e}'.format(Pe))

# ------ MESH ------ #
part1 = Rectangle(
   Point(   0*d, 0*d ),
   Point( 0.5*d, 3*d ))
part2 = Rectangle(
   Point( 0.5*d, 1*d ),
   Point( 9.0*d, 2*d ))
channel = part1 +part2

mesh = generate_mesh(channel, res)

# ------ BOUNDARIES DEFINITION ------ #
inlet_1 = '( x[1]=='+str(3.0*d)+' && x[0]>'+str(0.0*d)+' && x[0]<'+str(0.5*d)+' )'
inlet_2 = '( x[1]=='+str(0.0*d)+' && x[0]>'+str(0.0*d)+' && x[0]<'+str(0.5*d)+' )'
outlet  = '( x[0]=='+str(9.0*d)+' && x[1]>'+str(1.0*d)+' && x[1]<'+str(2.0*d)+' )'
walls   = 'on_boundary'\
        + ' && !'+inlet_1 \
        + ' && !'+inlet_2 \
        + ' && !'+outlet

ds_inlet1, ds_inlet2, ds_outlet = 0,1,2

boundaries        = FacetFunction ('size_t', mesh)
side_inlet_1      = CompiledSubDomain( inlet_1  )
side_inlet_2      = CompiledSubDomain( inlet_2  )
side_outlet       = CompiledSubDomain( outlet )
boundaries.set_all(0)
side_inlet_1.mark (boundaries, ds_inlet1 )
side_inlet_2.mark (boundaries, ds_inlet2 )
side_outlet.mark  (boundaries, ds_outlet )
ds = Measure('ds', subdomain_data=boundaries  )

# ------ FUNCTION SPACES ------ #
FE_V = FiniteElement('P', 'triangle', 2)
FE_P = FiniteElement('P', 'triangle', 1)
FE_U = VectorElement('P', 'triangle', 2)
elem = MixedElement([FE_V, FE_V, FE_P, FE_P])
U    = FunctionSpace(mesh, elem)

FE_A = FiniteElement('P', 'triangle', 1)
U_AA = FunctionSpace(mesh, FE_A)

# ------ FORMULACAO VARIACIONAL ------ #
x,y = 0,1
ans = Function(U)   
ux,uy,p,a = split(ans)
vx,vy,q,b = TestFunctions(U)
gam       = project(Constant(0.0), U_AA)

u = as_vector([ux,uy])
v = as_vector([vx,vy])

N1  = Constant(1.0)
N12 = Constant(12.0)
N56 = Constant(5.0/6.0)
PAR = Constant(opt_par)
HH  = Constant((d/10.0)**2.0)
RHO = Constant(rho)
MU  = Constant(mu)
DD  = Constant(D)

F_ctt = div(u)*q                                *dx
F_mtt = inner( RHO*N56*dot(u,grad(u).T), v )    *dx \
      + inner( MU*(grad(u)+grad(u).T), grad(v)) *dx \
      - div(v)*p                                *dx \
      + inner( (N12*MU/HH)*(N1+gam*PAR)*u,v )   *dx
F_cnv = inner( DD*grad(a),grad(b) )             *dx \
      + inner( u,grad(a))*b                     *dx

# Formulacao Final
F1 = F_ctt +F_mtt +F_cnv

# ------ CONDICOES DE CONTORNO ------ #
v_in1 = Expression('v_ct*x[0]*(Lx-x[0])/K', v_ct= -v_in, Lx=L_in, K=(L_in**2.0)/6.0, degree=2)
v_in2 = Expression('v_ct*x[0]*(Lx-x[0])/K', v_ct= +v_in, Lx=L_in, K=(L_in**2.0)/6.0, degree=2)
p_ux,p_uy,p_pp,p_aa = 0,1,2,3
BC = [
      DirichletBC(U.sub(p_ux), Constant(0 ),    inlet_1),
      DirichletBC(U.sub(p_uy), v_in1       ,    inlet_1),
      DirichletBC(U.sub(p_aa), Constant(1 ),    inlet_1),
      DirichletBC(U.sub(p_ux), Constant(0 ),    inlet_2),
      DirichletBC(U.sub(p_uy), v_in2       ,    inlet_2),
      DirichletBC(U.sub(p_aa), Constant(0 ),    inlet_2),
      DirichletBC(U.sub(p_ux), Constant(0 ),    walls),
      DirichletBC(U.sub(p_uy), Constant(0 ),    walls),
      ]

solve(F1==0, ans, BC,
   solver_parameters={'newton_solver':
   {'maximum_iterations' : 10,
   'absolute_tolerance'  : 5E-13,
   'relative_tolerance'  : 5E-14
   } })

uu = project(u,FunctionSpace(mesh,FE_U))
pp = project(p,FunctionSpace(mesh,FE_P))
aa = project(a,FunctionSpace(mesh,FE_P))

vm  = assemble( ux*ds(ds_outlet) )/d
Qm  = vm*d*d/10.0
eta = assemble( (a-0.5)*(a-0.5)*ds(ds_outlet))/d
dp  = assemble( p*ds(ds_inlet1) )/(d*0.5)    \
    + assemble( p*ds(ds_inlet2) )/(d*0.5)    \
    - assemble( p*ds(ds_outlet) )/ d
h20 = dp/(rho*g)

print ('V media ( m/s): {}'.format(vm      ) )
print ('Vazao   (ml/s): {}'.format(Qm*1E6  ) )
print ('dP      (Pa  ): {}'.format(dp      ) )
print ('h20     (mm  ): {}'.format(h20/1000) )
print ('Dispersao(%) ): {}'.format(eta*100 ) )

plot(uu,title='velocity')
plot(pp,title='pressure')
plot(aa,title='concentration')
interactive()
