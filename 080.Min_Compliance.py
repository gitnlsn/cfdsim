'''

NELSON KENZO TAMASHIRO

06 OUTUBRO 2017

PROBLEMA MIN COMPLIANCE TRIAL

'''



from fenics          import *
from mshr            import *
from dolfin_adjoint  import *

mesh_res = 50
mesh_0   = 0.0
mesh_L   = 5.0
mesh_D   = 2.0

cons_K   = 1E3

part1 = Rectangle(
   Point(mesh_0, mesh_0),
   Point(mesh_L, mesh_D),       )
channel = part1
mesh = generate_mesh(channel, mesh_res)

mesh_tol = mesh.hmax()/2.0

ds_f1 = 1
boundaries  = FacetFunction  ('size_t', mesh)
force_1     = '('                                \
            +  '     (x[0]=='+str(mesh_L              )+')'          \
            +  '  && (x[1]>='+str(mesh_D/2.0 -mesh_D/10.0 )+')'      \
            +  '  && (x[1]<='+str(mesh_D/2.0 +mesh_D/10.0 )+')'      \
            +' )'
CompiledSubDomain( force_1 ).mark( boundaries, ds_f1 )
ds = Measure('ds', subdomain_data=boundaries )

# plot(boundaries, interactive=True )

FE_dis   = VectorElement('P', 'triangle', 1)
FE_mat   = FiniteElement('P', 'triangle', 1)

U_dis = FunctionSpace(mesh, FE_dis)
U_mat = FunctionSpace(mesh, FE_mat)

u     = Function(U_dis)
du    = TestFunction(U_dis)
mat   = project(Constant(1), U_mat)

K = Constant(cons_K)

sigma = mat*K*( grad(u)+grad(u).T )
f_out = as_vector([ Constant(0), Constant(-10) ])

F  = inner(sigma, grad(du))   *dx \
   - inner(f_out, du)         *ds(ds_f1)

fixed_domain = '( x[0]== '+str(mesh_0)+')'
u_fixed      = as_vector([ Constant(0), Constant(0) ])
BC = [
      DirichletBC(U_dis, u_fixed, fixed_domain),
      ]

solve(F==0, u, BC)
J = Functional(inner(sigma, grad(u))*0.5*dx)
grad_J = compute_gradient(J, Control(mat))
mat.assign(project(mat +grad_J, U_mat))
fig_material = plot(mat, title='material')
adj_reset()


def my_funciton (x):
   return x -conditional(ge(x, 1), x, -1) -conditional(le(x, 0), x, 0)

INTERACTIONS = 100
for i in range(INTERACTIONS):
   solve(F==0, u, BC)
   J = Functional(inner(f_out, u)*dx)
   grad_J = compute_gradient(J, Control(mat))
   mat.assign(project(my_funciton(mat +grad_J), U_mat))
   fig_material.plot(mat)
   if (i%100==0):
      interactive()
   adj_reset()

# plot(u); interactive()

# plot(outer(u, derivative(K*mat, mat)*u))





