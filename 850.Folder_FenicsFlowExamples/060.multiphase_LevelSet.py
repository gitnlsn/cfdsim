from dolfin import *
from mshr   import *
import ufl
import time
import os

#set_log_level(PROGRESS)
# get file name
fileName = os.path.splitext(__file__)[0]
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 8
#parameters["form_compiler"]["quadrature_rule"] = 'auto'

comm = mpi_comm_world()
rank = MPI.rank(comm)
set_log_level(INFO if rank==0 else INFO+1)
ufl.set_level(ufl.INFO if rank==0 else ufl.INFO+1)
parameters["std_out_all_processes"] = False;
info_blue(dolfin.__version__)


center = Point(0.5, 0.5)
radius = 0.2

#parameters['linear_algebra_backend'] = 'uBLAS'


# Time stepping parameters
dt = 0.01    #casovy krok
t_end = 10.0  # cas pujde od nuly do t_end
theta=Constant(0.5)   # theta schema
k=Constant(1.0/dt)
g=Constant((0.0,-1.0))

# Mesh
channel = Rectangle(
   Point(0.0,0.0),
   Point(1.0,2.0)       )
mesh = generate_mesh(channel,50)

# pocatecni distance function
dist = Expression("sqrt((x[0]-A)*(x[0]-A) + (x[1]-B)*(x[1]-B))-r", A=center[0], B=center[1],r=radius, degree=2)

class InitialCondition(Expression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0
        values[3] = sqrt((x[0]-center[0])*(x[0]-center[0]) + (x[1]-center[1])*(x[1]-center[1]))-radius
    def value_shape(self):
        return (4,)

ic=InitialCondition(degree=2)

# Define function spaces
FE_V = VectorElement('P', 'triangle', 2)
FE_P = FiniteElement('P', 'triangle', 1)
FE_L = FiniteElement('P', 'triangle', 1)  # prostor pro transport distancni funkce
elem  = MixedElement([FE_V, FE_P, FE_L])
W     = FunctionSpace(mesh, elem)

# Define unknown and test function(s)
w = Function(W)
w0 = Function(W)

(v_, p_, l_) = TestFunctions(W)

(v,p,l)=split(w)
(v0,p0,l0)=split(w0)

bcs = list()
bcs.append( DirichletBC(W.sub(0), Constant((0.0, 0.0)), "near(x[0],0.0) || near(x[0],1.0) || near(x[1],0.0)") )

rho1=1e3
rho2=1e2
mu1=1e1
mu2=1e0
eps=1e-4

def Sign(q):
    return conditional(lt(abs(q),eps),q/eps,sign(q))

def Delta(q):
    return conditional(lt(abs(q),eps),(1.0/eps)*0.5*(1.0+cos(3.14159*q/eps)),Constant(0.0))

def rho(l):
    return(rho1 * 0.5* (1.0+ Sign(l)) + rho2 * 0.5*(1.0 - Sign(l)))

def nu(l):
   return(mu1 * 0.5* (1.0+ Sign(l)) + mu2 * 0.5*(1.0 - Sign(l)))

def EQ(v,p,l,v_,p_,l_):
    F_ls = inner(div(l*v),l_)*dx 
    T= -p*I + nu(l)*(grad(v)+grad(v).T)
    F_ns = inner(T,grad(v_))*dx + rho(l)*inner(grad(v)*v, v_)*dx - rho(l)*inner(g,v_)*dx
    F=F_ls+F_ns
    return(F)

n = FacetNormal(mesh)
I = Identity(FE_V.cell().geometric_dimension())    # Identity tensor
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
alpha=Constant(0.1)
r = alpha('+')*h_avg*h_avg*inner(jump(grad(l),n), jump(grad(l_),n))*dS

F=k*0.5*(theta*rho(l)+(1.0-theta)*rho(l0))*inner(v-v0,v_)*dx + k*inner(l-l0,l_)*dx + theta*EQ(v,p,l,v_,p_,l_) + (1.0-theta)*EQ(v0,p,l0,v_,p_,l_) + div(v)*p_*dx + r

J = derivative(F, w)
ffc_options = {"quadrature_degree": 4, "optimize": True, "eliminate_zeros": False}
problem=NonlinearVariationalProblem(F,w,bcs,J)
solver=NonlinearVariationalSolver(problem)

prm = solver.parameters
#info(prm, True)
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'umfpack'
prm['newton_solver']['lu_solver']['report'] = False
prm['newton_solver']['lu_solver']['same_nonzero_pattern']=True
prm['newton_solver']['absolute_tolerance'] = 1E-10
prm['newton_solver']['relative_tolerance'] = 1E-10
prm['newton_solver']['maximum_iterations'] = 20
prm['newton_solver']['report'] = True
#prm['newton_solver']['error_on_nonconvergence'] = False


w.assign(interpolate(ic,W))
w0.assign(interpolate(ic,W))

(v,p,l) = w.split()
(v0,p0,l0) = w0.split()


def reinit(l,mesh):
    #implement here:
    #   given mesh and function l on the mesh
    # reinitialize function l such that |grad(l)|=1
    # and the zero levelset does not change (too much)

    

    return l


#assign(l, interpolate (dist,L))
#assign(l0, interpolate (dist,L))

#plot(l0,interactive=True)
#plot(rho(l),interactive=True)
# Create files for storing solution
vfile = File("%s.results/velocity.pvd" % (fileName))
pfile = File("%s.results/pressure.pvd" % (fileName))
lfile = File("%s.results/levelset.pvd" % (fileName))

v.rename("v", "velocity") ; vfile << v
p.rename("p", "pressure") ; pfile << p
l.rename("l", "levelset") ; lfile << l

# Time-stepping
t = dt
while t < t_end:

   print "t =", t

   begin("Solving transport...")
   solver.solve()
   end()

   (v,p,l)=w.split(True)
   v.rename("v", "velocity") ; vfile << v
   p.rename("p", "pressure") ; pfile << p
   l.rename("l", "levelset") ; lfile << l

   V=assemble(conditional(lt(l,0.0),1.0,0.0)*dx)
   print "volume= %e"%V

   #plot(v,interactive=True)
   # Move to next time step
   l1=reinit(l,mesh)
   #assign(w.sub(2),interpolate(l1,L))

   w0.assign(w)
   t += dt  # t:=t+1
