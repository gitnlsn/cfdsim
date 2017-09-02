from dolfin import *
from mshr   import *
import numpy as np
import matplotlib.pyplot as plt

res = 30
domain = Rectangle( Point(0,0), Point(1,1) )
mesh   = generate_mesh(domain, res)

FE_1 = FiniteElement('Lagrange', 'triangle', 2)
elem = MixedElement ([FE_1, FE_1, FE_1, FE_1, FE_1, FE_1])
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
ome,psi,omx,omy,psx,psy = split(ans)

RE = Constant(100)
x,y = 0,1

grad_ome = as_vector([  omx   ,   omy])
grad_psi = as_vector([  psx   ,   psy])
rot_psi  = as_vector([  psy   ,  -psx])
v_driven = as_vector([Constant(1), Constant(0)])
v_walls  = as_vector([Constant(0), Constant(0)])

LL = inner( grad_psi -grad(psi), 
            grad_psi -grad(psi)                       ) *dx \
   + inner( grad_ome -grad(ome), 
            grad_ome -grad(ome)                       ) *dx \
   + inner( ome +div(grad_psi),
            ome +div(grad_psi)                        ) *dx \
   + inner( div(grad_ome) -inner(rot_psi,grad_ome), 
            div(grad_ome) -inner(rot_psi,grad_ome)    ) *dx \
   + inner(rot_psi -v_driven, rot_psi -v_driven       ) *ds(ds_driven) \
   + inner(rot_psi -v_walls,  rot_psi -v_walls        ) *ds(ds_walls)

   # + inner( psi -Constant(0) ,
   #          psi -Constant(0)                          ) *ds(ds_walls) \
FF = derivative(LL, ans, TestFunction(U))

solve(FF==0, ans, [])

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
   psi,ome = ans.split()
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
   psi,ome = ans.split(); x,y = 0,1
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
   psi,ome = ans.split();
   ux,uy = Dx(psi,y), -Dx(psi,x)
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

def solve_re(cons_re, foldername):
   last_re = np.zeros(1)
   RE.eval(last_re, np.zeros(3))
   for re in range(int(last_re), cons_re+1, 100):
      print('Solving for Re = {}'.format(re))
      RE.assign(re);
      solve(FF==0, ans, bc)
   save_streamVorticy_txt(ans, foldername+'_Re'+str(cons_re))
   save_velocity_txt     (ans, foldername+'_Re'+str(cons_re))
   #plotxy_velocity       (ans, foldername+'_Re'+str(cons_re))

for cons_re in [1000]:
   print('Solving for Re = {}'.format(cons_re))
   solve_re(cons_re, 'WV_R'+str(res))

