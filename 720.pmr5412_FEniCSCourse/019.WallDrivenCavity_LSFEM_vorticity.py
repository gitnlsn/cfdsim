from dolfin import *
from mshr   import *
import matplotlib.pyplot as plt
import numpy             as np

res    = 40
domain = Rectangle(Point(0,0), Point(1,1))
mesh   = generate_mesh(domain, res)

side_driven = CompiledSubDomain('on_boundary &&  (x[1]==1)')
side_walls  = CompiledSubDomain('on_boundary && !(x[1]==1)')
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
ds_driven, ds_walls = 0,1
side_driven.mark(boundaries, ds_driven)
side_walls.mark (boundaries, ds_walls )
ds = Measure("ds", subdomain_data=boundaries)

FE_1 = FiniteElement('Lagrange', 'triangle', 2)
elem = MixedElement([FE_1, FE_1, FE_1, FE_1, FE_1])
U = FunctionSpace(mesh, elem)

ans = Function(U)
ux,uy,p,w,psi = split(ans)
x,y = 0,1

rot_u = Dx(ux,y)-Dx(uy,x)
u     = as_vector([ux,uy])
rot_w = as_vector([Dx(w,y), -Dx(w,x)])
r_psi = as_vector([Dx(psi,y), -Dx(psi,x)])
u_dr  = as_vector([Constant(1), Constant(0)])
u_wl  = as_vector([Constant(0), Constant(0)])

RE = Constant(1)

Lct = inner(   div(u), div(u)                         ) *dx
Lmt = inner(   dot(u,grad(u).T) +grad(p) +rot_w/RE,
               dot(u,grad(u).T) +grad(p) +rot_w/RE    ) *dx
Lbd = inner(   u -u_dr , u -u_dr                      ) *ds(ds_driven) \
    + inner(   u -u_wl , u -u_wl                      ) *ds(ds_walls)
Lww = inner(   w -rot_u, w -rot_u                     ) *dx
Lps = inner(   u +r_psi, u +r_psi                     ) *dx

MM = Lct +Lmt +Lww +Lbd +Lps

FF = derivative( MM, ans, TestFunction(U) )

p_ux,p_uy,p_pp,p_ww,p_si = 0,1,2,3,4
left  = '( x[0]==0 )'
right = '( x[0]==1 )'
up    = '( x[1]==1 )'
down  = '( x[1]==0 )'
walls  = left+'||'+right+'||'+down
driven = up
corner = '( x[0]==0 && x[1]==0 )'
bc = [
      DirichletBC(U.sub(p_pp), Constant(0), corner, method='pointwise'),
      DirichletBC(U.sub(p_si), Constant(0), corner, method='pointwise'),
      ]

def plotxy_velocity(ans, foldername):
   ux,uy,p,ome,psi = ans.split()
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
   ux,uy,p,ome,psi = ans.split()
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

def save_streamVorticy_txt(ans, foldername, re='1000'):
   print('saving')
   streamVorticity_pos = {
      '1000' : [  (0.5300, 0.5650), (0.8633, 0.1117), (0.0833, 0.0783), (0.9917, 0.0067), (0.0050, 0.0050)  ],
      '2500' : [  (0.5200, 0.5433), (0.8350, 0.0917), (0.0850, 0.1100), (0.9900, 0.0100), (0.0067, 0.0067)  ],
      '5000' : [  (0.5150, 0.5350), (0.8050, 0.0733), (0.0733, 0.1367), (0.9783, 0.0183), (0.0083, 0.0083)  ],
      '7500' : [  (0.5133, 0.5317), (0.7900, 0.0650), (0.0650, 0.1517), (0.9517, 0.0417), (0.0117, 0.0117)  ],
      '10000': [  (0.5117, 0.5300), (0.7767, 0.0600), (0.0583, 0.1633), (0.9350, 0.0667), (0.0167, 0.0200)  ], 
   } # end: streamVorticity
   ux,uy,p,ome,psi = ans.split()
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

def solve_re(cons_re, foldername):
   last_re = np.zeros(1)
   RE.eval(last_re, np.zeros(3))
   for re in range(int(last_re), cons_re+1, 25):
      print('Solving for Re = {}'.format(re))
      RE.assign(re);
      solve(FF==0, ans, bc)
   plotxy_velocity         (ans, foldername+'_Re'+str(cons_re))
   save_velocity_txt       (ans, foldername+'_Re'+str(cons_re))
   save_streamVorticy_txt  (ans, foldername+'_Re'+str(cons_re))

RE.assign(100)
for cons_re in [1000, 2500, 5000, 7500, 10000]:
   print('Solving for Re = {}'.format(cons_re))
   solve_re(cons_re, 'LSvort_R'+str(res))

# print assemble(MM)
# plot(u, title='Velocity')
# plot(p, title='Pressure')
# plot(w, title='Vorticity')
# plot(psi, title='Streamfunction')
# interactive()
