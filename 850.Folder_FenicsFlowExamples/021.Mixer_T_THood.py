'''
Filename: 026.TMixer_TaylorHood.py
Author  : Nelson Kenzo Tamashiro
Date    : 18 Jan 2017

dolfin-version: 2016.2

Description:

   This program simulates Navier-Stokes and
   Convection-Diffusion equations on TMixer
   topology.

'''


# ------ LIBRARIES ------ #
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

def Tmixersolver(v_in,res):

   # ------ TMIXER GEOMETRY PARAMETERS ------ #
   d    = 0.010         # 10mm
   D    = 8.8E-10       # 8.8E-6 cm**2/s
   rho  = 1E3           # 1kg/m**3
   mu   = 8.9E-4        # 0.00089 N*s/m**2

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
      Point( 8.5*d, 2*d ))
   channel = part1 +part2
   mesh = generate_mesh(channel, res)
   FE_V = FiniteElement('P', 'triangle', 2)
   FE_P = FiniteElement('P', 'triangle', 1)
   FE_U = VectorElement('P', 'triangle', 2)
   elem = MixedElement([FE_V, FE_V, FE_P, FE_P])
   U = FunctionSpace(mesh, elem)

   # ------ FORMULACAO VARIACIONAL ------ #
   x,y = 0,1
   ans = Function(U)   
   ux,uy,p,a = split(ans)
   vx,vy,q,b = TestFunctions(U)

   u = as_vector([ux,uy])
   v = as_vector([vx,vy])

   RHO = Constant(rho)
   MU  = Constant(mu)
   DD  = Constant(D)

   F_ctt = div(u)*q                                *dx
   F_mtt = inner( RHO*dot(u,grad(u).T), v )        *dx \
         + inner( MU*(grad(u)+grad(u).T), grad(v)) *dx \
         - div(v)*p                                *dx
   F_cnv = inner( DD*grad(a),grad(b) )             *dx \
         + inner(u,grad(a))*b                      *dx

   # Formulacao Final
   F1 = F_ctt +F_mtt +F_cnv

   # ------ CONDICOES DE CONTORNO ------ #
   inlet_1 = '( x[1]=='+str(3*d)+' )'
   inlet_2 = '( x[1]=='+str(0*d)+' )'
   outlet  = '( x[0]=='+str(8.5*d)+' )'
   walls   = 'on_boundary'\
           + ' && !'+inlet_1 \
           + ' && !'+inlet_2 \
           + ' && !'+outlet
   p_point = 'x[0]=='+str(8.5*d)+' && x[1]=='+str(1.5*d)+''
   p_ux,p_uy,p_pp,p_aa = 0,1,2,3
   BC = [
         DirichletBC(U.sub(p_ux), Constant(0 ), inlet_1),
         DirichletBC(U.sub(p_uy), Constant(-v_in), inlet_1),
         DirichletBC(U.sub(p_aa), Constant(1 ), inlet_1),
         DirichletBC(U.sub(p_ux), Constant(0 ), inlet_2),
         DirichletBC(U.sub(p_uy), Constant(+v_in), inlet_2),
         DirichletBC(U.sub(p_aa), Constant(0 ), inlet_2),
         DirichletBC(U.sub(p_ux), Constant(0 ), walls),
         DirichletBC(U.sub(p_uy), Constant(0 ), walls),
         DirichletBC(U.sub(p_uy), Constant(0 ), outlet),
         DirichletBC(U.sub(p_pp), Constant(0 ), p_point, method='pointwise'),
         ]

   solve(F1==0, ans, BC,
      solver_parameters={'newton_solver':
      {'maximum_iterations' : 15,
      'absolute_tolerance'  : 5E-13,
      'relative_tolerance'  : 5E-14
      } })

   uu = project(u,FunctionSpace(mesh,FE_U))
   pp = project(p,FunctionSpace(mesh,FE_P))
   aa = project(a,FunctionSpace(mesh,FE_P))
   return uu,pp,aa

if __name__ == "__main__":
   res    = 50
   v_in_1 = 1E-4
   v_in_2 = 1E-5
   nSteps = 1
   if nSteps==1:
      delta_v_in=0
   else:
      delta_v_in = (v_in_2 -v_in_1)/(nSteps-1)

   foldername = 'results.026'
   vtkfile_pp = File(foldername+'/pressure.pvd')
   vtkfile_vv = File(foldername+'/velocity.pvd')
   vtkfile_aa = File(foldername+'/concentration.pvd')
   for i in range(nSteps):
      v_in = v_in_1 +i*delta_v_in
      print
      print('TMixer_TaylorHood: Progress = {:.2f}%.'.format((100.*i)/nSteps))
      uu,pp,aa = Tmixersolver(v_in,res)
      uu.rename('velocity','velocity')
      pp.rename('pressure','pressure')
      aa.rename('concentration','concentration')
      vtkfile_vv << (uu,v_in)
      vtkfile_pp << (pp,v_in)
      vtkfile_aa << (aa,v_in)
      plot(uu)
      plot(pp)
      plot(aa)
      interactive()
