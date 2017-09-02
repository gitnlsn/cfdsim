'''
Filename: 027.TeslaMixer_TaylorHood.py
Author  : Nelson Kenzo Tamashiro
Date    : 18 Jan 2017

dolfin-version: 2016.2

Description:

   This program simulates Navier-Stokes and
   Convection-Diffusion equations on Tesla
   Mixer topology.

   The results are printed on vtk format and
   can be analysed on Paraview.
'''


# ------ LIBRARIES ------ #
from fenics import *
from mshr import *

def TeslaSolver(v_in, res):

   # ------ TMIXER GEOMETRY PARAMETERS ------ #
   d    = 0.005         # 5mm
   D    = 8.8E-10       # 8.8E-6 cm**2/s
   rho  = 1E3           # 1kg/m**3
   mu   = 8.9E-4        # 0.00089 N*s/m**2

   # ------ SIMULATION PARAMETERS ------ #
   Re = rho*v_in*d/mu
   Pe = d*v_in/D
   print('Velocity: {:.2e}'.format(v_in))
   print('Reynolds: {:.2e}'.format(Re))
   print('Peclet  : {:.2e}'.format(Pe))
   v_in = 1
   d = 1

   # ------ MESH ------ #
   def TeslaChannel(d, x, y):
      part_structure = Polygon(
         [  Point( 0*d+x, 0*d+y ),
            Point( 2*d+x, 0*d+y ),
            Point( 2*d+x, 2*d+y ),
            Point( 3.7*d+x, 2*d+y ),
            Point( 5*d+x, 0*d+y ),
            Point( 6*d+x, 0*d+y ),
            Point( 6*d+x, 1*d+y ),
            Point( 5*d+x, 1*d+y ),
            Point( 5*d+x, 3*d+y ),
            Point( 2*d+x, 3*d+y ),
            Point( 0.7*d+x, 1*d+y ),
            Point( 0*d+x, 1*d+y )])
      part_circles = \
           Circle( Point(2*d+x, 1*d+y), d ) \
         + Circle( Point(5*d+x, 2*d+y), d )
      part_drops = \
           Circle( Point(2*d+x, 5/4.*d+y), d/4.) \
         + Circle( Point(5*d+x, 7/4.*d+y), d/4.) \
         - Rectangle( Point(7/4.*d+x, d+y), Point(2*d+x, 3/2.*d+y)) \
         - Rectangle( Point(7/4.*d+x, d+y), Point(2*d+x, 3/2.*d+y)) \
         + Polygon(
            [  Point(1*d+x, 1*d+y  ),
               Point(2*d+x, 1*d+y  ),
               Point(2*d+x, 3/2.*d+y)]) \
         + Polygon(
            [  Point(4*d+x, 2*d+y),
               Point(5*d+x, 3/2.*d+y),
               Point(5*d+x, 2*d+y)])
      teslaChannel =     \
         part_circles    \
         +part_structure \
         -part_drops
      return teslaChannel

   def TChannel (d,L,x,y):
      part1 = Rectangle(
         Point(x     ,y    ),
         Point(x+d/2.,y+3*d))
      part2 = Rectangle(
         Point(x+d/2.  ,y+1*d),
         Point(x+d/2.+L,y+2*d))
      tChannel = part1+part2
      return tChannel

   channel = TeslaChannel(d,d*3/2.,d) +TChannel(d,d,0,0)
   mesh = generate_mesh(channel, res)
   FE_V = FiniteElement('P', 'triangle', 2)
   FE_P = FiniteElement('P', 'triangle', 1)
   FE_U = VectorElement('P', 'triangle', 2)
   elem = MixedElement([FE_P, FE_V, FE_V, FE_P])
   U = FunctionSpace(mesh, elem)


   # ------ VARIATIONAL FORMULATION ------ #
   x,y = 0,1
   ans1 = Function(U)
   ans2 = Function(U)
   pp,ux,uy,aa = split(ans1)
   t1,t2,t3,t4 = TestFunctions(U)
   uu = as_vector([ux,uy])
   tt = as_vector([t2,t3])
   Re = Constant(Re)
   Pe = Constant(Pe)

   F_ctt = div(uu)*t1
   F_mtx = inner(grad(ux),uu)*t2 -pp*Dx(t2,x) \
         +inner(grad(ux),grad(t2))/Re
   F_mty = inner(grad(uy),uu)*t3 -pp*Dx(t3,y) \
         +inner(grad(uy),grad(t3))/Re
   F_dif = +inner(uu,grad(aa))*t4 \
         +inner(grad(aa),grad(t4))/Pe

   F1 = F_ctt*dx +F_mtx*dx +F_mty*dx +F_dif*dx

   # ------ BOUNDARY CONDITIONS ------ #
   inlet_1 = 'near(x[1],'+str(3*d)+') && x[0]<'+str(d)
   inlet_2 = 'near(x[1],0) && x[0]<'+str(d)
   w_inlet = 'near(x[0],0)'
   w_channel = 'on_boundary && x[0]>='+str(d/2.) \
         +' && x[0]<'+str(15*d/2.)
   w_outlet = 'x[0]=='+str(15*d/2.)+' && x[1]=='+str(3*d)
   outlet = 'near(x[0],'+str(15*d/2.)+') && x[1]>'+str(d) \
         +' && x[1]<'+str(2*d)
   walls = w_inlet+' || '+w_channel+' || '+w_outlet
   p_point = 'near(x[1],'+str(3*d/2.)+') && near(x[0],'+str(15*d/2.)+')'
   p_pp,p_vx,p_vy,p_aa = 0,1,2,3
   BC = [
         DirichletBC(U.sub(p_vx), Constant(0 ), inlet_1),
         DirichletBC(U.sub(p_vy), Constant(-1), inlet_1),
         DirichletBC(U.sub(p_aa), Constant(1 ), inlet_1),
         DirichletBC(U.sub(p_vx), Constant(0 ), inlet_2),
         DirichletBC(U.sub(p_vy), Constant(1 ), inlet_2),
         DirichletBC(U.sub(p_aa), Constant(0 ), inlet_2),
         DirichletBC(U.sub(p_vx), Constant(0 ), walls),
         DirichletBC(U.sub(p_vy), Constant(0 ), walls),
         DirichletBC(U.sub(p_vy), Constant(0 ), outlet),
         DirichletBC(U.sub(p_pp), Constant(0 ), p_point, method='pointwise'),
         ]

   solve(F1==0, ans1, BC,
      solver_parameters={'newton_solver':
      {'maximum_iterations' : 100,
      'absolute_tolerance'  : 1E-12,
      'relative_tolerance'  : 1E-13
      } })

   pp,ux,uy,aa = split(ans1)
   uu = as_vector([ux,uy])
   pp = project(pp,FunctionSpace(mesh,FE_P))
   uu = project(uu,FunctionSpace(mesh,FE_U))
   aa = project(aa,FunctionSpace(mesh,FE_P))
   return pp, uu, aa

if __name__ == "__main__":
   res    = 32
   v_in_1 = 1E-6
   v_in_2 = 1E-5
   nSteps = 100
   delta_v_in = (v_in_2 -v_in_1)/(nSteps-1)

   vtkfile_pp = File('027.TeslaMixer_TaylorHood/pressure.pvd')
   vtkfile_vv = File('027.TeslaMixer_TaylorHood/velocity.pvd')
   vtkfile_aa = File('027.TeslaMixer_TaylorHood/concentration.pvd')
   for i in range(nSteps):
      v_in = v_in_1 +i*delta_v_in
      print
      print('TeslaMixer_TaylorHood: Progress = {:.2f}%.'.format((100.*i)/nSteps))
      pp,uu,aa = TeslaSolver(v_in,res)
      pp.rename('pp','pp')
      uu.rename('uu','uu')
      aa.rename('aa','aa')
      vtkfile_pp << (pp,v_in)
      vtkfile_vv << (uu,v_in)
      vtkfile_aa << (aa,v_in)
