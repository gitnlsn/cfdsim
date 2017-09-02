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
res     = 300
opt_par = 1.0E2
d       = 0.010         # 10mm
D       = 8.8E-10       # 8.8E-6 cm**2/s
rho     = 1.0E3         # 1kg/m**3
mu      = 8.5E-4        # 0.00089 N*s/m**2
g       = 9.8
L_in    = 0.5*d
v_in    = 2E-7

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
#inlet_1 = '( x[1]=='+str(3.0*d)+' && x[0]>'+str(0.0*d)+' && x[0]<'+str(0.5*d)+' )'
#inlet_2 = '( x[1]=='+str(0.0*d)+' && x[0]>'+str(0.0*d)+' && x[0]<'+str(0.5*d)+' )'
#outlet  = '( x[0]=='+str(9.0*d)+' && x[1]>'+str(1.0*d)+' && x[1]<'+str(2.0*d)+' )'
#walls   = 'on_boundary'\
#        + ' && !'+inlet_1 \
#        + ' && !'+inlet_2 \
#        + ' && !'+outlet
inlet_1 = '( near(x[1],'+str(3.0*d)+') )'
inlet_2 = '( near(x[1],'+str(0.0*d)+') )'
outlet  = '( near(x[0],'+str(9.0*d)+') )'
walls   = 'on_boundary && !('+inlet_1+' || '+inlet_2+' || '+outlet+')'

ds_inlet1, ds_inlet2, ds_walls, ds_outlet = 0,1,2,3

boundaries     = FacetFunction ('size_t', mesh)
side_inlet_1   = CompiledSubDomain( inlet_1  )
side_inlet_2   = CompiledSubDomain( inlet_2  )
side_walls     = CompiledSubDomain( walls )
side_outlet    = CompiledSubDomain( outlet )
boundaries.set_all(0)
side_inlet_1.mark (boundaries, ds_inlet1 )
side_inlet_2.mark (boundaries, ds_inlet2 )
side_walls.mark   (boundaries, ds_walls  )
side_outlet.mark  (boundaries, ds_outlet )
ds = Measure('ds', subdomain_data=boundaries  )

# ------ FUNCTION SPACES ------ #
FE_1 = FiniteElement('P', 'triangle', 1)
FE_2 = FiniteElement('P', 'triangle', 2)
FE_V = VectorElement('P', 'triangle', 1)
FE_S = FiniteElement('P', 'triangle', 1)
elem = MixedElement([FE_1, FE_1, FE_1, FE_1, FE_1, FE_1, FE_1, FE_1, FE_1, FE_1])
U    = FunctionSpace(mesh, elem)

FE_A = FiniteElement('P', 'triangle', 1)
U_AA = FunctionSpace(mesh, FE_A)

# ------ FORMULACAO VARIACIONAL ------ #
x,y = 0,1
ans = Function(U)   
ux,uy,p,a,uxx,uxy,uyx,uyy,ax,ay = split(ans)
gam       = project(Constant(0.0), U_AA)

N1  = Constant(1.0)
N12 = Constant(12.0)
N56 = Constant(5.0/6.0)
PAR = Constant(opt_par)
HH  = Constant((d/10.0)**2.0)
RHO = Constant(rho)
MU  = Constant(mu)
DD  = Constant(D)

u      = as_vector(  [ux,uy]  )
grad_a = as_vector(  [ax,ay]  )
grad_u = as_tensor([ [uxx,uxy],
                     [uyx,uyy], ])
div_u = uxx +uyy
sigma = MU*(grad_u+grad_u.T) -Identity(len(u))*p
Ja    = DD*grad_a

v_in1 = Expression('v_ct*x[0]*(Lx-x[0])/K', v_ct= -v_in, Lx=L_in, K=(L_in**2.0)/6.0, degree=2)
v_in2 = Expression('v_ct*x[0]*(Lx-x[0])/K', v_ct= +v_in, Lx=L_in, K=(L_in**2.0)/6.0, degree=2)
u_in1 = as_vector( [Constant(0), v_in1] )
u_in2 = as_vector( [Constant(0), v_in2] )
a_in1 = Constant(1.0)
a_in2 = Constant(0.0)
u_wal = as_vector( [Constant(0), Constant(0)] )
Ja_wl  = DD*as_vector( [Constant(0), Constant(0)] )
Ja_out = DD*as_vector( [Constant(0), Constant(0)] )
sigma_out = as_tensor([ [Constant(0), Constant(0)],
                        [Constant(0), Constant(0)] ])

# Formulacao Final
F  = inner( RHO*dot(u,grad_u.T) +grad(p) -div(sigma),
            RHO*dot(u,grad_u.T) +grad(p) -div(sigma) )  *dx \
   + inner( div_u, div_u   )                 *dx \
   + inner( div(Ja) -dot(u,grad_a),
            div(Ja) -dot(u,grad_a) )  *dx \
   + inner( grad(u)-grad_u, grad(u)-grad_u ) *dx \
   + inner( grad(a)-grad_a, grad(a)-grad_a ) *dx \
   + inner( u  -u_in1, u  -u_in1 )                 *ds(ds_inlet1) \
   + inner( a  -a_in1, a  -a_in1 )                 *ds(ds_inlet1) \
   + inner( u  -u_in2, u  -u_in2 )                 *ds(ds_inlet2) \
   + inner( a  -a_in2, a  -a_in2 )                 *ds(ds_inlet2) \
   + inner( u  -u_wal, u  -u_wal )                 *ds(ds_walls)  \
   + inner( Ja -Ja_wl, Ja -Ja_wl )                 *ds(ds_walls)  \
   + inner( sigma -sigma_out, sigma -sigma_out )   *ds(ds_outlet) \
   + inner( Ja -Ja_out, Ja -Ja_out )               *ds(ds_outlet)

J = derivative(F, ans, TestFunction(U))

solve(J==0, ans, [],
   solver_parameters={'newton_solver':
   {'maximum_iterations' : 7,
   'absolute_tolerance'  : 5E-12,
   'relative_tolerance'  : 5E-12
   } })

uu = project(u,FunctionSpace(mesh,FE_V))
pp = project(p,FunctionSpace(mesh,FE_S))
aa = project(a,FunctionSpace(mesh,FE_S))

ctt_eq = inner(   div_u,div_u )*dx
mmt_eq = inner(   RHO*dot(u,grad_u.T) +grad(p) +MU*div(grad_u),
                  RHO*dot(u,grad_u.T) +grad(p) +MU*div(grad_u) ) *dx
dif_eq = inner(   DD*div(grad_a) -dot(u,grad_a),
                  DD*div(grad_a) -dot(u,grad_a) ) *dx

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
print ('Res Eq ctt    : {}'.format(assemble(ctt_eq)))
print ('Res Eq mmt    : {}'.format(assemble(mmt_eq)))
print ('Res Eq dif    : {}'.format(assemble(dif_eq)))

plot(u,title='velocity')
plot(p,title='pressure')
plot(a,title='concentration')
interactive()
