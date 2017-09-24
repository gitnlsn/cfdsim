'''

EULER EQUATIONS SIMULATION

NELSON TAMASHIRO

20 SEPTEMBER 2017

'''


from fenics import *
from mshr   import *

foldername = 'results_005EulerLSFEM'

mesh_res = 50
mesh_p0  = 0.0
mesh_l   = 4.0
mesh_d   = 2.0

cons_vin = 1.0E-3
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

# obsts  = '(on_boundary && (x[0]>'+str(mesh_p0)+') && (x[1]>'+str(mesh_p0)+') && (x[0]<'+str(mesh_l)+') && (x[1]<'+str(mesh_d)+') )'
walls  = '(x[1]=='+str(mesh_p0)+') || (x[1]=='+str(mesh_d)+')'
inlet  = '(x[0]=='+str(mesh_p0)+')'
outlet = '(x[0]=='+str(mesh_l )+')'
obsts  = 'on_boundary && !'+inlet+' && !'+outlet

ds_inlet, ds_walls, ds_outlet = 1,2,3

boundaries     = FacetFunction ('size_t', mesh)
side_inlet     = CompiledSubDomain( inlet                )
side_outlet    = CompiledSubDomain( outlet               )
side_walls     = CompiledSubDomain( walls+' || '+obsts   )
boundaries.set_all(0)
side_inlet.mark   (boundaries, ds_inlet  )
side_walls.mark   (boundaries, ds_walls  )
side_outlet.mark  (boundaries, ds_outlet )
ds = Measure( 'ds', subdomain_data=boundaries )

FE_P  = FiniteElement('P', 'triangle', 1)
FE_V  = FiniteElement('P', 'triangle', 1)

elem  = MixedElement([FE_V, FE_V, FE_P])
U     = FunctionSpace(mesh, elem)
U_vel = FunctionSpace(mesh, MixedElement([FE_V,FE_V]))
U_vol = FunctionSpace(mesh, FE_P)

ans_lst = Function(U)
ans_nxt = Function(U)
ux_nxt,uy_nxt,p_nxt = split(ans_nxt)
ux_lst,uy_lst,p_lst = split(ans_lst)
vx,vy,qq = TestFunctions(U)

u_nxt = as_vector ([ux_nxt,uy_nxt])
u_lst = as_vector ([ux_lst,uy_lst])
vv    = as_vector ([vx,vy])

RHO   = Constant(cons_rho)
DT    = Constant(cons_dt)
N     = FacetNormal(mesh)

u_df = (u_nxt -u_lst)
u_md = (u_nxt +u_lst)*0.5
p_md = (p_nxt +p_lst)*0.5

u_inn  = Constant( (cons_vin, 0) )
u_wal  = Constant( (0, 0) )

F  = inner( RHO*u_df/DT                      \
         +  RHO*dot(u_md,grad(u_md).T)       \
         +  grad(p_md)                 ,
            RHO*vv/DT                        \
         +  RHO*dot(vv,grad(u_nxt).T)        \
         +  RHO*dot(u_nxt,grad(vv).T)        \
         +  grad(qq)                )  *dx \
   + inner( div(u_nxt),div(vv)      )  *dx \
   + inner( u_nxt -u_inn,vv  )  *ds(ds_inlet) \
   + inner( u_nxt -u_wal,vv  )  *ds(ds_walls) \
   + inner( grad(u_nxt), grad(vv)     )  *ds(ds_outlet) \
   # + inner( grad(p_nxt), grad(qq)     )  *ds(ds_outlet) \

#F = derivative(F, ans_nxt, TestFunction(U))

p_ux,p_uy,p_pp = 0,1,2
BC = [
      # DirichletBC(U.sub(p_ux), Constant(cons_vin ), inlet),
      # DirichletBC(U.sub(p_uy), Constant(0    ), inlet),
      # DirichletBC(U.sub(p_ux), Constant(0    ), walls),
      # DirichletBC(U.sub(p_uy), Constant(0    ), walls),
      # DirichletBC(U.sub(p_ux), Constant(0    ), obsts),
      # DirichletBC(U.sub(p_uy), Constant(0    ), obsts),
      # DirichletBC(U.sub(p_pp), Constant(0    ), outlet),
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
prm["absolute_tolerance"            ] = 6.0E-13
prm["relative_tolerance"            ] = 8.0E-14
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

assign(ans_nxt.sub(p_ux), project(Constant(1E-5), FunctionSpace(mesh, FE_V) ) )

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

