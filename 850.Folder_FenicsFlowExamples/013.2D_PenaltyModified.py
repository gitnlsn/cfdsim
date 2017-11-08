'''

dolfin-version: 2016.2

Description:

   This program simulates Navier-Stokes equations
   on Straight Channel geometry with Least Squares
   Galerking stabilization methods.

   The results are printed on vtk format and can
   be analysed on Paraview.
'''


# ------ LIBRARIES ------ #
from fenics import *
from mshr import *

def ChannelSolver(Re, res=30, dt=5E-3):

   # ------ CHANNEL GEOMETRY ------ #
   d = 1
   rho = 1E3
   kb = 2.15E9   #dp/drho *(1/rho) = 2.15 [GPa] water

   print('\nBegining Simulation.')
   print('Resolution    : {:d}'.format(res ))
   print('Time Interval : {:.2e} [s]'.format(dt  ))
   print('Density       : {:.2e} [kg/m**3]'.format(rho ))
   print('Bulk Modulus  : {:.2e} [Pa]'.format(kb  ))
   print('Reynolds      : {:.2e}'.format(Re  ))
   print

   # ------ MESH ------ #
   part1 = Rectangle(
         Point(0*d,0*d),
         Point(5*d,5*d))
   part2 = Circle(
         Point(0.5*d,2.5*d),
         0.2*d)
   channel = part1 -part2

   mesh = generate_mesh(channel,res)
   FE_V = FiniteElement('P', 'triangle', 1)
   FE_P = FiniteElement('P', 'triangle', 1)
   FE_U = VectorElement('P', 'triangle', 1)
   elem = MixedElement([FE_P, FE_V, FE_V])
   U = FunctionSpace(mesh, elem)

   # ------ VARIATIONAL FORMULATION ------ #
   x,y = 0,1
   ans = Function(U)
   pp,ux,uy = split(ans)
   tq,tx,ty = TestFunctions(U)
   uu = as_vector([ux,uy])
   tu = as_vector([tx,ty])
   one = Constant(1)
   rho = Constant(rho)
   dt  = Constant(dt)
   Re  = Constant(Re)
   kb  = Constant(kb)

   F_std = div(uu)*tq                     \
         +inner(uu,grad(ux))*tx           \
         +inner(uu,grad(uy))*ty           \
         -pp*Dx(tx,x)/rho                 \
         -pp*Dx(ty,y)/rho                 \
         +inner(grad(ux),grad(tx))/(Re)   \
         +inner(grad(uy),grad(ty))/(Re)

   F_tq = dt*kb*div(tu)

   F_tx = dt*inner(uu,grad(tx)) \
         +dt*inner(tu,grad(ux)) \
         +dt/rho*Dx(tq,x)

   F_ty = dt*inner(uu,grad(ty)) \
         +dt*inner(tu,grad(uy)) \
         +dt/rho*Dx(tq,y)

   F_ctt = div(uu)

   F_mtx = inner(uu,grad(ux)) \
         +Dx(pp,x)/rho \

   F_mty = inner(uu,grad(uy)) \
         +Dx(pp,y)/rho \

   F = F_std*dx \
      +F_ctt*F_tq*dx \
      +F_mtx*F_tx*dx \
      +F_mty*F_ty*dx

   # ------ BOUNDARY CONDITIONS ------ #
   str_0d = str(0.0*d)
   str_md = str(2.5*d)
   str_1d = str(1.0*d)
   str_2d = str(2.0*d)
   str_3d = str(3.0*d)
   str_5d = str(5.0*d)
   left  = 'near(x[0], '+str_0d+')'
   right = 'near(x[0], '+str_5d+')'
   down  = 'near(x[1], '+str_0d+')'
   up    = 'near(x[1], '+str_5d+')'
   slids = 'on_boundary' \
         +'&& x[0]>'+str_0d \
         +'&& x[1]>'+str_0d \
         +'&& x[0]<'+str_5d \
         +'&& x[1]<'+str_5d
   walls = up+' || '+down
   point = 'near(x[0], '+str_5d+') && near(x[1], '+str_md+')'
   p_pp,p_ux,p_uy = 0,1,2
   BC = [
         DirichletBC(U.sub(p_ux), Constant(1 ), left),
         DirichletBC(U.sub(p_uy), Constant(0 ), left),

         #DirichletBC(U.sub(p_ux), Constant(0 ), walls),
         DirichletBC(U.sub(p_uy), Constant(0 ), walls),

         DirichletBC(U.sub(p_ux), Constant(0 ), slids),
         DirichletBC(U.sub(p_uy), Constant(0 ), slids),

         #DirichletBC(U.sub(p_ux), Constant(0 ), right),
         DirichletBC(U.sub(p_uy), Constant(0 ), right),

         DirichletBC(U.sub(p_pp), Constant(0 ), point, method='pointwise'),
         #DirichletBC(U.sub(p_pp), Constant(0 ), right),
         ]

   solve(F==0, ans, BC,
         solver_parameters={'newton_solver':
         {'maximum_iterations' : 25,
         'absolute_tolerance'  : 1E-13,
         'relative_tolerance'  : 1E-14
         } })

   pp,ux,uy = split(ans)
   uu = as_vector([ux,uy])
   pp = project(pp,FunctionSpace(mesh,FE_P))
   uu = project(uu,FunctionSpace(mesh,FE_U))
   return pp, uu

if __name__ == "__main__":
   res    = 40
   dt     = 2E-2
   exp_1  = -4.
   exp_2  = 5.
   nSteps = 1
   d_exp  = (exp_2 -exp_1)/(nSteps)

   vtkfile_pp = File('029.NStokes_ContinutyModified/pressure.pvd')
   vtkfile_vv = File('029.NStokes_ContinutyModified/velocity.pvd')
   for i in range(nSteps):
      exp = exp_1 +i*d_exp
      Re = 10**(exp)
      print
      print('029.NStokes_ContinutyModified: Simulation Progress = {:.2f}%.'.format((100.*i)/nSteps))
      try:
         pp,uu = ChannelSolver(Re ,res, dt)
      except Exception as e:
         print('029.NStokes_ContinutyModified: dt = dt/2.')
         dt /= 2
         pp,uu = ChannelSolver(Re ,res, dt)
      pp.rename('pp','pp')
      uu.rename('uu','uu')
      plot(pp, title='pressure')
      plot(uu, title='velocity')
      interactive()
      vtkfile_pp << (pp,Re)
      vtkfile_vv << (uu,Re)

