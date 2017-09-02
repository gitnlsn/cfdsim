
from fenics import *
from mshr import *
from math import pi

# ------ PARAMETROS SIMULACAO ------ #
res     = 150
v_inlet = 1E-6

# ------ PARAMETROS FISICOS ------ #
mu = 8.9E-4
rho = 1E3
grav = 0

# ------ PARAMETROS HIDROCICLONE ------ #
R_1 =  1.0
R_2 =  0.5
R_3 =  4.0
H   = 10.0
L_1 =  6.0
L_2 =  1.0
L_3 =  1.0
H_3 =  2.0
esp =  0.3

# ------ MESH ------ #
part1 = Polygon([
   Point(0       , 0        ),
   Point(R_2     , 0        ),
   Point(R_2     , L_2      ),
   Point(R_3     , H -H_3  ),
   Point(R_3 +L_3, H -H_3  ),
   Point(R_3 +L_3, H        ),
   Point(0       , H        ),
   ])
part2 = Rectangle(
   Point(R_1     , H -L_1),
   Point(R_1 +esp, H     ))
domain = part1 -part2
mesh = generate_mesh(domain,res)

class domain_inlet(SubDomain):
   def inside(self, x, on_boundary):
      return x[0] > R_3          \
         and x[0] < R_3 +L_3     \
         and x[1] > H   -H_3     \
         and x[1] < H

dxp_inlet = 1

domain_mark = CellFunction('size_t', mesh)
d_inlet = domain_inlet()
d_inlet.mark(domain_mark, dxp_inlet)
dx = Measure('dx', domain=mesh, subdomain_data=domain_mark)

# ------ VARIATIONAL FORMULATION ------ #
FE_1 = FiniteElement('P', 'triangle', 1)
FE_2 = FiniteElement('P', 'triangle', 2)
elem = MixedElement([FE_1, FE_2, FE_2, FE_2])
U = FunctionSpace(mesh, elem)

ans = Function(U)
pp,ur,ut,uw = split(ans)
tp,tr,tt,tw = TestFunctions(U)
dr,dw = 0,1
N2 = Constant(2)
MU = Constant(mu)
R0 = Constant(rho)
GG = Constant(grav)
r = Expression('x[0]', degree=2)

v_in = Constant(v_inlet)
u = as_vector([ur,ut,uw])
w = as_vector([tr,tt,tw])

div_u = ur/r +Dx(ur,dr) +Dx(uw,dw)

nab_u = as_tensor([
   [Dx(ur,dr), -ut/r, Dx(ur,dw)],
   [Dx(ut,dr),  ur/r, Dx(ut,dw)],
   [Dx(uw,dr),     0, Dx(uw,dw)]])

nab_w = as_tensor([
   [Dx(tr,dr), -tt/r, Dx(tr,dw)],
   [Dx(tt,dr),  tr/r, Dx(tt,dw)],
   [Dx(tw,dr),     0, Dx(tw,dw)]])

T_tau = as_tensor([
   [ -pp+N2*MU*Dx(ur,dr),       MU*(Dx(ut,dr)-ut/r), MU*(Dx(uw,dr)+Dx(ur,dw,))],
   [ MU*(Dx(ut,dr)-ut/r),            -pp+N2*MU*ur/r,              MU*Dx(ut,dw)],
   [ MU*(Dx(uw,dr)+Dx(ur,dw,)),        MU*Dx(ut,dw),       -pp+N2*MU*Dx(uw,dw)]])

B = as_vector([0,0,-GG*R0])

Rc = div_u*tp*dx                          \
   - v_in/(2*pi*r)*tp*dx(dxp_inlet)

Rm = R0*inner(dot(u,nab_u.T),w)*dx        \
   + inner(T_tau, nab_w)*dx               \
   - v_in*v_in/(2*pi*r)*tt*dx(dxp_inlet)  \
   - inner(B,w)*dx

F = Rm +Rc

# ------ BOUNDARY CONDITIONS ------ #
p_pp,p_ur,p_ut,p_uw = 0,1,2,3
P1 ='('                             \
   +'near(x[1], '+str(H)+')   && '  \
   +'x[0]<'+str(R_1)+'           '  \
   +')'
P2 ='('                             \
   +'near(x[1], '+str(0)+')   && '  \
   +'x[0]<'+str(R_2)+'           '  \
   +')'
Center = '('                        \
   +'near(x[0],'+str(0)+') '        \
   +')'
out1 = 'on_boundary && '+P1
out2 = 'on_boundary && '+P2
walls = 'on_boundary && ! '+P1+' && !'+P2+' && ! '+Center
point = 'near(x[0], 0) && near(x[1],0)'
BC = [

      DirichletBC(U.sub(p_ur), Constant(0 ), walls),
      DirichletBC(U.sub(p_ut), Constant(0 ), walls),
      DirichletBC(U.sub(p_uw), Constant(0 ), walls),

      DirichletBC(U.sub(p_ur), Constant(0 ), out1 ),
      #DirichletBC(U.sub(p_ut), Constant(0 ), out1 ),
      #DirichletBC(U.sub(p_uw), Constant(0 ), out1 ),

      DirichletBC(U.sub(p_ur), Constant(0 ), out2),
      #DirichletBC(U.sub(p_ut), Constant(0 ), out2),
      #DirichletBC(U.sub(p_uw), Constant(0 ), out2),

      DirichletBC(U.sub(p_pp), Constant(0 ), point, method='pointwise'),
      ]

dF = derivative(F, ans)
problem = NonlinearVariationalProblem(F, ans, BC, dF)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters["newton_solver"]
prm["convergence_criterion"   ] = "residual"
prm["linear_solver"           ] = "mumps"
prm["method"                  ] = "full"
prm["preconditioner"          ] = "none"
prm["error_on_nonconvergence" ] = True
prm["maximum_iterations"      ] = 10
prm["absolute_tolerance"      ] = 1E-23
prm["relative_tolerance"      ] = 5E-13
prm["relaxation_parameter"    ] = 1.0
prm["report"                  ] = True
set_log_level(PROGRESS)
solver.solve()

# ------ SAVE SIMULATION RESULTS ------ #
mesh_viz = generate_mesh(domain, res+50)
U_viz  = FunctionSpace(mesh_viz, FE_1)
UU_viz = FunctionSpace(mesh_viz, FE_1*FE_1)
uu = as_vector([ur,uw])

to_save = [
         [pp, 'pressure'            ],
         [ur, 'velocity_radial'     ],
         [ut, 'velocity_tangencial' ],
         [uw, 'velocity_axial'      ],
         ]

pos_FunctionValue, pos_FunctionName = 0,1
for var in to_save:
   var_vtk = File('results.035/'+str(var[pos_FunctionName])+'.pvd')
   var_viz = project(var[pos_FunctionValue], U_viz)
   var_viz.rename(var[pos_FunctionName], var[pos_FunctionName])
   var_vtk << var_viz

uu_vtk = File('results.035/VelocityVector.pvd')
uu_viz = project(uu, UU_viz)
uu_viz.rename('VelocityVector','VelocityVector')
uu_vtk << uu_viz


plot(pp, title='pressure')
plot(uu, title='Velocity')
plot(ut, title='Tangencial')
plot(ur, title='Radial')
plot(uw, title='Axial')
interactive()


