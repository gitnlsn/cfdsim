from dolfin import *
from mshr   import *
import numpy as np
import matplotlib.pyplot as plt

res = 100
domain = Rectangle( Point(0,0), Point(1,1) )
mesh   = generate_mesh(domain, res)

FE_1 = FiniteElement('Lagrange', 'triangle', 1)
elem = MixedElement ([FE_1, FE_1, FE_1, FE_1])
U    = FunctionSpace(mesh, elem)

side_driven = CompiledSubDomain('on_boundary &&  (x[1]==1)')
side_walls  = CompiledSubDomain('on_boundary && !(x[1]==1)')
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
ds_driven, ds_walls = 0,1
side_driven.mark(boundaries, ds_driven)
side_walls.mark (boundaries, ds_walls )
ds = Measure("ds", subdomain_data=boundaries)

ans = Function(U)
w,psi,ux,uy = split(ans)
m,phi,tx,ty = TestFunctions(U)

RE = Constant(100)
x,y = 0,1

u        = as_vector([ux,uy])
v        = as_vector([tx,ty])
rot_psi  = as_vector([Dx(psi,y), -Dx(psi,x)])
v_driven = as_vector([Constant(1), Constant(0)])
grad_psi = as_vector([-v_driven[1], v_driven[0]])
normal = FacetNormal(mesh)

FF = w*m                               *dx \
   - inner( grad(psi), grad(m) )       *dx \
   + inner( grad_psi, normal)*m        *ds(ds_driven) \
   + inner( grad(w), rot_psi   )*phi   *dx \
   + inner( grad(w), grad(phi) )/RE    *dx \
   + inner( u -rot_psi, v )            *dx

p_ome,p_psi,p_ux,p_uy = 0,1,2,3
left  = '( x[0]==0 )'
right = '( x[0]==1 )'
up    = '( x[1]==1 )'
down  = '( x[1]==0 )'
walls  = left+'||'+right+'||'+down
driven = up
corner = 'x[0]==0 && x[1]==0'
bc = [
      DirichletBC(U.sub(p_psi), Constant(0), driven),
      DirichletBC(U.sub(p_psi), Constant(0), walls),
      DirichletBC(U.sub(p_ux ), Constant(0), walls),
      DirichletBC(U.sub(p_uy ), Constant(0), walls),
      DirichletBC(U.sub(p_ux ), Constant(1), driven),
      DirichletBC(U.sub(p_uy ), Constant(0), driven),
      #DirichletBC(U.sub(p_psi), Constant(0), corner, method='pointwise'),
      ]

solve(FF==0, ans, bc)
#plot(u); interactive()

#plot(rot_psi, title='velocity')
#plot(psi, title='streamfunction')
#plot(ome, title='vorticity')
#interactive()

def save_streamVorticy_txt(ans, foldername, re='1000'):
   print('saving')
   streamVorticity_pos = {
      '1000' : [  (0.5300, 0.5650), (0.8633, 0.1117), (0.0833, 0.0783), (0.9917, 0.0067), (0.0050, 0.0050)  ],
      '2500' : [  (0.5200, 0.5433), (0.8350, 0.0917), (0.0850, 0.1100), (0.9900, 0.0100), (0.0067, 0.0067)  ],
      '5000' : [  (0.5150, 0.5350), (0.8050, 0.0733), (0.0733, 0.1367), (0.9783, 0.0183), (0.0083, 0.0083)  ],
      '7500' : [  (0.5133, 0.5317), (0.7900, 0.0650), (0.0650, 0.1517), (0.9517, 0.0417), (0.0117, 0.0117)  ],
      '10000': [  (0.5117, 0.5300), (0.7767, 0.0600), (0.0583, 0.1633), (0.9350, 0.0667), (0.0167, 0.0200)  ], 
   } # end: streamVorticity
   psi,ome,ux,uy = ans.split(); x,y = 0,1
   # HORIZONTAL X-VELOCITY PLOT
   points = streamVorticity_pos[re]
   to_print_list = [(psi(point), ome(point), point[0], point[1] ) for point in points]
   file = open(foldername+'_streamV.txt','w')
   for to_print in to_print_list:
      print_psi, print_ome, print_px, print_py = to_print
      file.write(str(print_psi)+', ')
      file.write(str(print_ome)+', ')
      file.write(str(print_px) +', ')
      file.write(str(print_py) +', ')
      file.write('\n')
   file.close()

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
   psi,ome,ux,uy = ans.split(); x,y = 0,1
   ux = project( Dx(psi,y), FunctionSpace(mesh,FE_1))
   uy = project(-Dx(psi,x), FunctionSpace(mesh,FE_1))
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

def plotxy_velocity(ans, foldername):
   psi,ome,ux,uy = ans.split(); x,y = 0,1
   ux = project( Dx(psi,y), FunctionSpace(mesh,FE_1))
   uy = project(-Dx(psi,x), FunctionSpace(mesh,FE_1))
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

plotxy_velocity         (ans, 'fdsafas')
save_streamVorticy_txt  (ans, 'fdsafas')
save_velocity_txt       (ans, 'fdsafas')

def solve_re(cons_re, foldername):
   last_re = np.zeros(1)
   RE.eval(last_re, np.zeros(3))
   for re in range(int(last_re), cons_re+1, 100):
      print('Solving for Re = {}'.format(re))
      RE.assign(re)
      solve(FF==0, ans, bc)
   last_re = np.zeros(1)
   RE.eval(last_re, np.zeros(3))
   save_streamVorticy_txt(ans, foldername+'_Re'+str(cons_re), re=last_re)
   save_velocity_txt     (ans, foldername+'_Re'+str(cons_re))
   plotxy_velocity       (ans, foldername+'_Re'+str(cons_re))

RE.assign(100)
for cons_re in [1000, 2500, 5000, 7500, 10000]:
   print('Solving for Re = {}'.format(cons_re))
   solve_re(cons_re, 'WV_R'+str(res))

