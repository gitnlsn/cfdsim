from dolfin import *
from mshr   import *
import numpy as np
import matplotlib.pyplot as plt

res  = 100
domain = Rectangle(Point(0,0), Point(1,1))
mesh = generate_mesh(domain, res)

FE_1 = FiniteElement('Lagrange', 'triangle', 1)
FE_2 = FiniteElement('Lagrange', 'triangle', 2)
elem = MixedElement([FE_1, FE_1, FE_1])
U = FunctionSpace(mesh, elem)

ans = Function(U)
ux,uy,p = split(ans)
tx,ty,q = TestFunctions(U)

u = as_vector([ux,uy])
v = as_vector([tx,ty])

GA = Constant(1E0)
RE = Constant(100)

FF = p*q                         *dx \
   + GA*div(u)*q                 *dx \
   + inner( dot(u,grad(u).T),v ) *dx \
   - p*div(v)                    *dx \
   + inner(grad(u),grad(v))/RE   *dx

p_ux,p_uy,p_pp = 0,1,2
left  = '( x[0]==0 )'
right = '( x[0]==1 )'
up    = '( x[1]==1 )'
down  = '( x[1]==0 )'
walls  = left+'||'+right+'||'+down
driven = up
corner = '( x[0]==0 && x[1]==0 )'
bc = [
      DirichletBC(U.sub(p_ux), Constant(1), driven),
      DirichletBC(U.sub(p_uy), Constant(0), driven),

      DirichletBC(U.sub(p_ux), Constant(0), walls),
      DirichletBC(U.sub(p_uy), Constant(0), walls),

      DirichletBC(U.sub(p_pp), Constant(0), corner, method='pointwise'),
      ]

def plotxy_velocity(ans, foldername):
   ans_ux,ans_uy,ans_pp = ans.split()
   p_pos = np.linspace(0+1E-8, 1-1E-8, 101)
   # HORIZONTAL X-VELOCITY PLOT
   points = [(0.5, y_) for y_ in p_pos] # 2D points
   ux_vals = np.array([ux(point) for point in points])
   plt.figure(); plt.plot(ux_vals, p_pos, 'r-', ux_vals, p_pos, 'bo')
   plt.legend(['approximation', 'exact values'])
   plt.xlabel('x-Velocity'); plt.ylabel('y-Position')
   #plt.savefig(foldername+'\x-Velocity.pdf')
   plt.savefig(foldername+'_x-Velocity.png')
   # VERTICAL Y-VELOCITY PLOT
   points = [(x_, 0.5) for x_ in p_pos] # 2D points
   uy_vals = np.array([uy(point) for point in points])
   plt.figure(); plt.plot(p_pos, uy_vals, 'r-', p_pos, uy_vals, 'bo')
   plt.legend(['approximation', 'exact values'])
   plt.xlabel('x-Position'); plt.ylabel('y-Velocity')
   #plt.savefig(foldername+'_y-Velocity.pdf')
   plt.savefig(foldername+'_y-Velocity.png')

def save_velocity_txt(ans, foldername):
   print('saving')
   x_velocity_pos = [
      1.000, 0.990, 0.980, 0.970, 0.960, 0.950, 0.940, 0.930, 0.920, 0.910, 0.900, 0.500,
      0.200, 0.180, 0.160, 0.140, 0.120, 0.100, 0.080, 0.060, 0.040, 0.020, 0.000
   ] # end: x_velocity
   y_velocity_pos = [
      1.000, 0.985, 0.970, 0.955, 0.940, 0.925, 0.910, 0.895, 0.880, 0.865, 0.850,
      0.500, 0.150, 0.135, 0.120, 0.105, 0.090, 0.075, 0.060, 0.045, 0.030, 0.015, 0.000
   ] # end: y_velocity
   ux,uy,p = ans.split()
   # HORIZONTAL X-VELOCITY PLOT
   points = [(0.5, y_) for y_ in x_velocity_pos] # 2D points
   ux_vals = [(ux(point), point[1] ) for point in points]
   file = open(foldername+'_x-velocity.txt','w')
   for point in ux_vals:
      ux,y = point
      file.write(str(ux) +', ' +str(y)+'\n')
   file.close()
   # VERTICAL Y-VELOCITY PLOT
   points = [(x_, 0.5) for x_ in y_velocity_pos] # 2D points
   uy_vals = [(point[0], uy(point)) for point in points]
   file = open(foldername+'_y-velocity.txt','w')
   for point in uy_vals:
      x,uy = point
      file.write(str(x) +', ' +str(uy)+'\n')
   file.close()

def solve_re(cons_re, foldername):
   last_re = np.zeros(1)
   RE.eval(last_re, np.zeros(3))
   GA.assign(100)
   for re in range(int(last_re), cons_re+1, 100):
      print('Solving for Re = {}'.format(re))
      RE.assign(re);
      solve(FF==0, ans, bc)
   for ga_exp in range(3,5):
      print('Solving for gamma = {}'.format(10**ga_exp))
      GA.assign(10**(ga_exp))
      solve(FF==0, ans, bc)
   plotxy_velocity  (ans, foldername+'_Re'+str(cons_re))
   save_velocity_txt(ans, foldername+'_Re'+str(cons_re))

RE.assign(100)
GA.assign(100)
for cons_re in [1000, 2500, 5000, 7500, 10000]:
   print('Solving for Re = {}'.format(cons_re))
   solve_re(cons_re, 'Penalty_R'+str(res))

