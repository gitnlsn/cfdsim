'''

EULER EQUATIONS SIMULATION

NELSON TAMASHIRO

20 SEPTEMBER 2017

'''


from fenics import *
from mshr   import *

foldername = 'results_002Euler'

mesh_res = 50
mesh_p0  = 0.0
mesh_l   = 3.0
mesh_d   = 1.0

cons_vin = 1.0E-0
cons_rho = 1.0E3
cons_dt  = 1.0E-3

TRANSIENT_MAX_ITE = 1000

part1 = Rectangle(
   Point(mesh_p0, mesh_p0),
   Point(mesh_l ,  mesh_d),       )
part2 = Circle(
   Point(mesh_l*0.2, mesh_d*0.5),
   mesh_d*0.1)
channel = part1 -part2
mesh  = generate_mesh(channel, mesh_res)

FE_1  = FiniteElement('P', 'triangle', 1)
FE_2  = FiniteElement('P', 'triangle', 2)

elem  = MixedElement([FE_2, FE_2, FE_1])
U     = FunctionSpace(mesh, elem)
U_vel = FunctionSpace(mesh, MixedElement([FE_2,FE_2]))
U_vol = FunctionSpace(mesh, FE_1)

ans_lst = Function(U)
ans_nxt = Function(U)
ux_nxt,uy_nxt,p_nxt = split(ans_nxt)
ux_lst,uy_lst,p_lst = split(ans_lst)
vx,vy,q = TestFunctions(U)

u_nxt = as_vector ([ux_nxt,uy_nxt])
u_lst = as_vector ([ux_lst,uy_lst])
v = as_vector ([vx,vy])

RHO = Constant(cons_rho)
DT = Constant(cons_dt)

u_df = (u_nxt -u_lst)
u_md = (u_nxt +u_lst)*0.5
p_md = (p_nxt +p_lst)*0.5

F  = inner(RHO*u_df/DT, v)                *dx \
   + inner(RHO*div(outer(u_md,u_md)), v)  *dx \
   + inner(grad(p_md), v)                 *dx \
   + div(u_nxt)*q *dx

inlet  = '(x[0]=='+str(mesh_p0)+')'
walls  = '(x[1]=='+str(mesh_p0)+') || (x[1]=='+str(mesh_d)+')'
obsts  = 'on_boundary && (x[0]>'+str(mesh_p0)+') && (x[1]>'+str(mesh_p0)+') && (x[0]<'+str(mesh_l)+') && (x[1]<'+str(mesh_d)+')'
outlet = '(x[0]=='+str(mesh_l)+')'

p_ux,p_uy,p_pp = 0,1,2
BC = [
      DirichletBC(U.sub(p_ux), Constant(cons_vin ), inlet),
      DirichletBC(U.sub(p_uy), Constant(0    ), inlet),
      DirichletBC(U.sub(p_ux), Constant(0    ), walls),
      DirichletBC(U.sub(p_uy), Constant(0    ), walls),
      DirichletBC(U.sub(p_ux), Constant(0    ), obsts),
      DirichletBC(U.sub(p_uy), Constant(0    ), obsts),
      DirichletBC(U.sub(p_pp), Constant(0    ), outlet),
      ]

dF = derivative(F, ans_nxt)
nlProblem = NonlinearVariationalProblem(F, ans_nxt, BC, dF)
nlSolver  = NonlinearVariationalSolver(nlProblem)
nlSolver.parameters["nonlinear_solver"] = "snes"

prm = nlSolver.parameters["snes_solver"]
prm["error_on_nonconvergence"       ] = False
prm["solution_tolerance"            ] = 1.0E-16
prm["maximum_iterations"            ] = 15
prm["maximum_residual_evaluations"  ] = 20000
prm["sign"                          ] = "default"
prm["absolute_tolerance"            ] = 8.0E-13
prm["relative_tolerance"            ] = 6.0E-13
prm["linear_solver"                 ] = "mumps"
#prm["method"                        ] = "vinewtonssls"
#prm["line_search"                   ] = "bt"
#prm["preconditioner"                ] = "none"
#prm["report"                        ] = True
#prm["krylov_solver"                 ]
#prm["lu_solver"                     ]

#set_log_level(PROGRESS)

def RungeKutta2(ans_now, ans_nxt, nlSolver, DT):
   ans_aux  = Function(U)
   ans_aux.assign(ans_now)
   RK1      = Function(U)
   RK2      = Function(U)
   RK3      = Function(U)
   RK4      = Function(U)
   # 1st iteration
   ans_now.assign( ans_aux )
   nlSolver.solve()
   RK1.assign( ans_nxt -ans_now )
   # 2nd iteration
   ans_now.assign( ans_aux+RK1/2.0 )
   nlSolver.solve()
   RK2.assign( ans_nxt -ans_now )
   # return RungeKutta estimate
   ans_now.assign(ans_aux)
   ans_nxt.assign(project( ans_aux+ RK2, U))

# ------ SAVE FILECONFIGURATIONS ------ #
vtk_u = File(foldername+'/velocity.pvd')
vtk_p = File(foldername+'/pressure.pvd')

def save_results(u,p):
   p_viz = project(p, U_vol); p_viz.rename('pressure','pressure'); vtk_p << p_viz
   u_viz = project(u, U_vel); u_viz.rename('velocity','velocity'); vtk_u << u_viz

count_iteration = 0
while( count_iteration < TRANSIENT_MAX_ITE ):
   count_iteration = count_iteration +1
   #nlSolver1.solve()
   #RungeKutta4(ansm, ansn, nlSolver1, DT)
   RungeKutta2(ans_lst, ans_nxt, nlSolver, DT)
   residual = assemble(inner(ans_nxt -ans_lst, ans_nxt -ans_lst)*dx)
   print ('Iteration: {}'.format(count_iteration) )
   print ('Residual : {}'.format(residual) )
   ans_lst.assign(ans_nxt)
   save_results(u_nxt,p_nxt)

