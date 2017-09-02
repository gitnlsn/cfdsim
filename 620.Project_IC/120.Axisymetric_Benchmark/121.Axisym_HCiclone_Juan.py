
from fenics import *
from mshr import *
from math import pi

# ------ PARAMETROS SIMULACAO ------ #
res     = 300
v_in    = 1E-1

# ------ PARAMETROS FISICOS ------ #
mu    = 8.9E-4
rho   = 1E3
grav  = 0

# ------ PARAMETROS GEOMETRICOS ------ #
mesh_p10x, mesh_p10y =  0.00E-3,   0.00E-3
mesh_p20x, mesh_p20y =  6.75E-3,   0.00E-3
mesh_p30x, mesh_p30y =  6.75E-3,  25.00E-3
mesh_p40x, mesh_p40y = 37.50E-3, 211.00E-3
mesh_p50x, mesh_p50y = 37.50E-3, 286.00E-3
mesh_p60x, mesh_p60y = 12.50E-3, 286.00E-3
mesh_p70x, mesh_p70y = 13.50E-3, 287.00E-3
mesh_p80x, mesh_p80y = 13.50E-3, 290.00E-3
mesh_p90x, mesh_p90y = 00.00E-3, 290.00E-3
mesh_s10x, mesh_s10y = 12.50E-3, 236.00E-3
mesh_s20x, mesh_s20y = 13.50E-3, 286.00E-3

mesh_p43x, mesh_p43y = 37.50E-3, 243.00E-3
mesh_p45x, mesh_p45y = 37.50E-3, 211.00E-3
mesh_p47x, mesh_p47y = 37.50E-3, 211.00E-3

# ------ MESH ------ #
part1 = Polygon([
   Point(mesh_p10x, mesh_p10y),
   Point(mesh_p20x, mesh_p20y),
   Point(mesh_p30x, mesh_p30y),
   Point(mesh_p40x, mesh_p40y),
   Point(mesh_p50x, mesh_p50y),
   Point(mesh_p60x, mesh_p60y),
   Point(mesh_p70x, mesh_p70y),
   Point(mesh_p80x, mesh_p80y),
   Point(mesh_p90x, mesh_p90y),
   ])
part2 = Rectangle(
   Point(mesh_s10x, mesh_s10y)  ,
   Point(mesh_s20x, mesh_s20y)  )
domain = part1 -part2
mesh = generate_mesh(domain,res)

class domain_inlet(SubDomain):
   def inside(self, x, on_boundary):
      # 11.5: distancia para que as medidas 
      #     da area de entrada fiquem na
      #     razao aurea: 43x26
      return x[0] > 11.5E-3      \
         and x[0] < mesh_p50x    \
         and x[1] > mesh_p40y    \
         and x[1] < mesh_p50y

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

v_in = Constant(v_in)
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
out1  = '('\
      +'near(x[0],'+str(mesh_p40x)+') && ' \
      +'x[1]>'     +str(mesh_p40y)+'  && ' \
      +'x[1]<'     +str(mesh_p50y)         \
      + ')'
out3  = '('\
      +'near(x[1],'+str(mesh_p10y)+') && ' \
      +'x[0]>'     +str(mesh_p10x)+'  && ' \
      +'x[0]<'     +str(mesh_p20x)         \
      + ')'
out4  = '('\
      +'near(x[1],'+str(mesh_p90y)+') && ' \
      +'x[0]>'     +str(mesh_p90x)+'  && ' \
      +'x[0]<'     +str(mesh_p80x)         \
      + ')'
inlet = out1
walls = 'on_boundary && !'+out3+' && !'+out4
point = 'near(x[0], 0) && near(x[1],0)'
BC = [
      DirichletBC(U.sub(p_ur), Constant(0 ), walls),
      DirichletBC(U.sub(p_ut), Constant(0 ), walls),
      DirichletBC(U.sub(p_uw), Constant(0 ), walls),

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
prm["relative_tolerance"      ] = 8E-12
prm["relaxation_parameter"    ] = 1.0
prm["report"                  ] = True
set_log_level(PROGRESS)
solver.solve()

# ------ SAVE SIMULATION RESULTS ------ #
mesh_viz = generate_mesh(domain, res+50)
U_viz  = FunctionSpace(mesh, FE_1)
UU_viz = FunctionSpace(mesh, FE_1*FE_1)
uu = as_vector([ur,uw])

to_save = [
         [pp, 'pressure'            ],
         [ur, 'velocity_radial'     ],
         [ut, 'velocity_tangencial' ],
         [uw, 'velocity_axial'      ],
         ]

pos_FunctionValue, pos_FunctionName = 0,1
for var in to_save:
   var_vtk = File('results.010/'+str(var[pos_FunctionName])+'.pvd')
   var_viz = project(var[pos_FunctionValue], U_viz)
   var_viz.rename(var[pos_FunctionName], var[pos_FunctionName])
   var_vtk << var_viz

uu_vtk = File('results.010/VelocityVector.pvd')
uu_viz = project(uu, UU_viz)
uu_viz.rename('VelocityVector','VelocityVector')
uu_vtk << uu_viz


plot(pp, title='pressure')
plot(uu, title='Velocity')
plot(ut, title='Tangencial')
plot(ur, title='Radial')
plot(uw, title='Axial')
interactive()


