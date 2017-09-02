'''
DESCRICAO:

DATA:    20.03.2017
AUTOR:   NELSON KENZO TAMASHIRO
'''
# ------ IMPORTACAO DE BIBLIOTECAS ------ #
from sympy import *
import numpy   as np
import control as ct
import matplotlib.pyplot as plt
#init_printing()

# ------ PARAMETROS FISICOS DO PROBLEMA ------ #
                     # SILICIO - 300K
R_dens = 2330.E+0    # densidade
c_capc = 712.0E+0    # capacidade termica
a_diff = 89.20E-6    # difusividade termica
hc_for =  200.E+0    # difusividade termica
TT_inf = 300.0E+0    # temperatura ambiente
phys_param = [R_dens, c_capc, a_diff, hc_for, TT_inf]
p_rho, p_cv, p_aa, p_hc, p_tinf= 0,1,2,3,4


# ------ DECLARACAO DE VARIAVEIS E PARAMETROS SIMBOLICOS ------ #
rho   = symbols('rho'      )
cv    = symbols('c'        )
aa    = symbols('alpha'    )
hc    = symbols('hc'       )
T_inf = symbols('T_inf'    )  # constantes fisicas #
phys_varbl = [rho, cv, aa, hc, T_inf]

# ------ TABELA DE NOS E DE ELEMENTOS ------ #
node = [             # TABELA DE NOS
         [0, 0, 0],
         [1, 1, 0],
         [2, 2, 0],
         [3, 0, 1],
         [4, 1, 1],
         [5, 2, 1],
                     ]

elem = [             # TABELA DE ELEMENTOS
         [0, 0, 1, 3, 4, 0],
         [1, 1, 2, 4, 5, 1],
                              ]

heat = [             # CALOR EXTERNO
         [0,    0,      0,     0,     0],
         [1, 5000,  10000,     0,  5000],
                                          ]

output_pos_list = [  # posicoes de temperatura medida
                     #[0.0, 0.0],
                     #[2.0, 1.0],
                     [0.3, 0.4],
                     [1.8, 0.2], ]

x,y = symbols('x y')
x0,x1,x2,x3  = symbols('x:4')
y0,y1,y2,y3  = symbols('y:4')

# ------ CONSTRUCAO DA MATRIZ LOCAL ------ #
def get_nodes_position(nodes_table, elem_table, elem_num):
   # 1) pega os nos na tabela de elementos
   elem_pos_n0, elem_pos_n1, elem_pos_n2, elem_pos_n3 = 1,2,3,4
   node0 = elem_table[elem_num][elem_pos_n0]
   node1 = elem_table[elem_num][elem_pos_n1]
   node2 = elem_table[elem_num][elem_pos_n2]
   node3 = elem_table[elem_num][elem_pos_n3]
   # 2) pega os x,y na tabela de nos
   px_node,py_node = 1,2
   x0,y0 = nodes_table[node0][px_node], nodes_table[node0][py_node]
   x1,y1 = nodes_table[node1][px_node], nodes_table[node1][py_node]
   x2,y2 = nodes_table[node2][px_node], nodes_table[node2][py_node]
   x3,y3 = nodes_table[node3][px_node], nodes_table[node3][py_node]   
   return (x0,x1,x2,x3), (y0,y1,y2,y3)

def get_local_B(x_var,y_var, xi, yi):
   x0,x1,x2,x3 = xi
   y0,y1,y2,y3 = yi
   x = x_var
   y = y_var
   B = Matrix([1,x,y,x*y]).T*Matrix([ 
                                       [1, x0, y0, x0*y0],
                                       [1, x1, y1, x1*y1],
                                       [1, x2, y2, x2*y2],
                                       [1, x3, y3, x3*y3], ]).inv()
   return simplify(B.T)

def get_local_nablaB(x_var,y_var, xi, yi):
   x0,x1,x2,x3 = xi
   y0,y1,y2,y3 = yi
   x = x_var
   y = y_var
   nB = Matrix([  [0,1,0,y],
                  [0,0,1,x], ])*Matrix([ 
                                       [1, x0, y0, x0*y0],
                                       [1, x1, y1, x1*y1],
                                       [1, x2, y2, x2*y2],
                                       [1, x3, y3, x3*y3], ]).inv()
   return simplify(nB.T)

def get_integration_interval(xi, yi):
   x_inf = min(xi)
   x_sup = max(xi)
   y_inf = min(yi)
   y_sup = max(yi)
   return x_inf,x_sup,y_inf,y_sup

def get_intBB (x_var, y_var, xi, yi, B):
   x_inf, x_sup, y_inf, y_sup = get_integration_interval(xi,yi)
   x,y = x_var, y_var
   M1 = integrate( 
        integrate(B*B.T,
                           (x, x_inf, x_sup)    ),
                           (y, y_inf, y_sup)    )
   return simplify(M1.reshape(4,4))

def get_intnBnB (x_var, y_var, xi, yi, nB):
   x_inf, x_sup, y_inf, y_sup = get_integration_interval(xi,yi)
   x,y = x_var, y_var
   M2 = integrate(
        integrate(nB*nB.T,
                           (x, x_inf, x_sup)    ),
                           (y, y_inf, y_sup)    )
   return simplify(M2.reshape(4,4))

def get_intB (x_var, y_var, xi, yi, B):
   x_inf, x_sup, y_inf, y_sup = get_integration_interval(xi,yi)
   x,y = x_var, y_var
   V1 = integrate(
        integrate(B,
                           (x, x_inf, x_sup)    ),
                           (y, y_inf, y_sup)    )
   return simplify(V1.reshape(4,1))

# ------ CONSTRUCAO DA MATRIZ GLOBAL ------ #
def get_local_global_map(elem_table, elem_num):
   elem_pos_n0, elem_pos_n1, elem_pos_n2, elem_pos_n3 = 1,2,3,4
   pos_n0 = elem_table[elem_num][elem_pos_n0]
   pos_n1 = elem_table[elem_num][elem_pos_n1]
   pos_n2 = elem_table[elem_num][elem_pos_n2]
   pos_n3 = elem_table[elem_num][elem_pos_n3]
   return [pos_n0, pos_n1, pos_n2, pos_n3]

def local_global_matrix_map(M_global, M_local, elem_table, elem_num):
   pos_local  = [0,1,2,3]
   pos_global = get_local_global_map(elem_table, elem_num)
   for i in range(4):
      for j in range(i,4):
         ni_local  = pos_local [i]
         nj_local  = pos_local [j]
         ni_global = pos_global[i]
         nj_global = pos_global[j]
         if i==j:
            M_global[ni_global, nj_global] += M_local[ni_local, nj_local]
         else:
            M_global[ni_global, nj_global] += M_local[ni_local, nj_local]
            M_global[nj_global, ni_global] += M_local[ni_local, nj_local]
   return M_global

def get_global_M1(nodes_table, elem_table):
   # 1) INICIA M1 COM ZEROS
   M1_size = len(nodes_table)
   M1 = zeros(M1_size, M1_size)
   x,y = symbols('x y')
   # 2) CONSTROI MATRIZES LOCAIS E MAPEIA NA MATRIZ GLOBAL
   elem_pos_num = 0
   for e in elem_table:
      elem_num = e[elem_pos_num]
      xi,yi    = get_nodes_position(nodes_table, elem_table, elem_num)
      B_local  = get_local_B  (x, y, xi, yi         )
      M1_local = get_intBB    (x, y, xi, yi, B_local)
      M1 = local_global_matrix_map(M1, M1_local, elem_table, elem_num)
   return M1

def local_global_vector_map(V_global, V_local, elem_table, elem_num):
   pos_local  = [0,1,2,3]
   pos_global = get_local_global_map(elem_table, elem_num)
   for i in range(4):
      n_local  = pos_local [i]
      n_global = pos_global[i]
      V_global[n_global] += V_local[n_local]
   return V_global

def get_global_V1(nodes_table, elem_table):
   # 1) INICIA V1 COM ZEROS
   V1_size = len(nodes_table)
   V1 = zeros(V1_size, 1)
   x,y = symbols('x y')
   # 2) CONSTROI MATRIZES LOCAIS E MAPEIA NA MATRIZ GLOBAL
   elem_pos_num = 0
   for e in elem_table:
      elem_num = e[elem_pos_num]
      xi,yi    = get_nodes_position(nodes_table, elem_table, elem_num)
      B_local  = get_local_B  (x, y, xi, yi         )
      V1_local = get_intB     (x, y, xi, yi, B_local)
      V1 = local_global_vector_map(V1, V1_local, elem_table, elem_num)
   return V1

def get_global_M2(nodes_table, elem_table):
   # 1) INICIA M2 COM ZEROS
   M2_size = len(nodes_table)
   M2 = zeros(M2_size, M2_size)
   x,y = symbols('x y')
   # 2) CONSTROI MATRIZES LOCAIS E MAPEIA NA MATRIZ GLOBAL
   elem_pos_num = 0
   for e in elem_table:
      elem_num = e[elem_pos_num]
      xi,yi    = get_nodes_position(nodes_table, elem_table, elem_num)
      nB_local = get_local_nablaB (x, y, xi, yi          )
      M2_local = get_intnBnB      (x, y, xi, yi, nB_local)
      M2 = local_global_matrix_map(M2, M2_local, elem_table, elem_num)
   return M2

def get_local_Heat_config(elem_table, heat_table, elem_num):
   elem_pos_heat = 5
   heat_num = elem_table[elem_num][elem_pos_heat]
   return Matrix(heat_table[heat_num][1:])

def get_global_V2(nodes_table, elem_table, heat_table):
   # 1) INICIA V2 COM ZEROS
   V2_size = len(nodes_table)
   V2 = zeros(V2_size, 1)
   x,y = symbols('x y')
   # 2) CONSTROI MATRIZES LOCAIS E MAPEIA NA MATRIZ GLOBAL
   elem_pos_num = 0
   for e in elem_table:
      elem_num = e[elem_pos_num]
      xi,yi    = get_nodes_position(nodes_table, elem_table, elem_num)
      B_local  = get_local_B  (x, y, xi, yi         )
      aux      = get_intBB    (x, y, xi, yi, B_local)
      heat_ff  = get_local_Heat_config(elem_table, heat_table, elem_num)
      V2 = local_global_vector_map(V2, aux*heat_ff, elem_table, elem_num)
   return V2

def control_matrix(M, V):
   Mc_size = M.shape[1]
   Mc = M*V
   aux = Matrix(Mc)
   for i in range(1,Mc_size):
      aux = M*aux
      Mc  = Mc.col_insert(i, aux)
   return Mc

def get_eigen(M):
   eig = M.eigenvals()
   open_loop_eig = []
   for e in eig:
      open_loop_eig.append(float(e))
   return open_loop_eig

class System:
   # CONSTRUCTOR
   def __init__ (self, A, B, C):
      self.A = A
      self.B = B
      self.C = C
   def eigenvalues(self, sym=True):
      if sym:
         A = self.A_sym
      else:
         A = self.A
      return get_eigen(A)
   def set_numeric_form(self, phys_varbl, phys_param):
      self.A_sym = Matrix(A)
      self.B_sym = Matrix(B)
      self.C_sym = Matrix(C)
      self.A = Subs(self.A, flatten(phys_varbl), flatten(phys_param)).doit()
      self.B = Subs(self.B, flatten(phys_varbl), flatten(phys_param)).doit()
      self.C = Subs(self.C, flatten(phys_varbl), flatten(phys_param)).doit()
      self.phys_varbl = phys_varbl
      self.phys_param = phys_param
   def check_observability(self, sym=True):
      if sym:
         A = self.A_sym
         C = self.C_sym
      else:
         A = self.A
         C = self.C
      Mo = C*A
      aux = Matrix(Mo).reshape(C.shape[0],C.shape[1])
      for i in range(1, A.shape[0]):
         aux = aux*self.A
         for j in range(C.shape[0]):
            Mo  = Matrix(Mo).row_insert(i*C.shape[0]+j, aux.row(j))
      self.Mo = Mo
      if Mo.rank() == A.shape[0]:
         return True
      else :
         return False
   def check_controlability(self, sym=True):
      if sym:
         A = self.A_sym
         B = self.B_sym
      else:
         A = self.A
         B = self.B
      Mc = A*B
      aux = Matrix(Mc).reshape(B.shape[0],B.shape[1])
      for i in range(1,A.shape[0]):
         aux = A*aux
         for j in range(B.shape[1]):
            Mc  = Mc.col_insert(i*B.shape[1]+j, aux.col(j))
      self.Mc = Mc
      if Mc.rank() == A.shape[0]:
         return True
      else :
         return False
   # RUNGE-KUTTA INTEGRATOR SETUP
   def set_RK (self, dt):
      X   = Matrix(symbols('X:'+str(self.A.shape[1])))
      U   = Matrix(symbols('U:'+str(self.B.shape[1])))
      dX1 = self.A*(X          ) +self.B*U
      dX2 = self.A*(X+dX1*dt/2.) +self.B*U
      dX3 = self.A*(X+dX2*dt/2.) +self.B*U
      dX4 = self.A*(X+dX3*dt   ) +self.B*U
      dX  = (dX1 +dX2*2. +dX3*2. +dX4)/6.
      self.output   = lambdify( (X,U), self.C*X, 'numpy')
      self.rk4_step = lambdify( (X,U), X+dX*dt , 'numpy')
   # RUNGE-KUTTA SYSTEM SIMULATION
   def simulate_system(self, x0, U, steps):
      hist = Matrix(   x0 )
      time = Matrix(  [0] )
      insp = Matrix( self.C*x0 )
      for i in range(steps):
         print('Progress = {}%%'.format(100.*i/steps))
         t_now  = time[0,i]
         x_now  = hist[:,i]
         x_next = Matrix( self.rk4_step(x_now, U) )
         y_next = Matrix( self.output  (x_now, U) )
         time = time.col_insert(i+1, Matrix([t_now+dt]))
         hist = hist.col_insert(i+1, x_next  )
         insp = insp.col_insert(i+1, y_next  )
      self.hist = hist
      self.time = time
      self.insp = insp
      return hist, insp, time
   def set_stateObsever(self, dt, speed=5):
      self.A_red, self.B_red, self.C_red, self.E_obs, self.P_obs \
          = generate_reduced_observer(self.A, self.B, self.C, speed)
      V   = Matrix(symbols('V:'+str(self.A_red.shape[1])))
      Y   = Matrix(symbols('Y:'+str(self.B_red.shape[1])))
      U   = Matrix(symbols('U:'+str(self.C_red.shape[1])))
      dV1 = self.A_red*(V          ) +self.B_red*Y +self.C_red*U
      dV2 = self.A_red*(V+dV1*dt/2.) +self.B_red*Y +self.C_red*U
      dV3 = self.A_red*(V+dV2*dt/2.) +self.B_red*Y +self.C_red*U
      dV4 = self.A_red*(V+dV3*dt   ) +self.B_red*Y +self.C_red*U
      dV  = (dV1 +dV2*2. +dV3*2. +dV4)/6.
      self.rk4_obsever_step = lambdify( (V,Y,U),
            V+dV*dt, 'numpy')
      self.state_recovery   = lambdify( (V,Y),
            self.P_obs.inv()*(V+self.E_obs*Y).col_join(Y), 'numpy')
   def observe_system(self, x0, U, steps):
      v_now     = get_reduced_state(x0, self.P_obs, self.E_obs)
      y_now     = self.insp[:,0]
      obs_state = Matrix( self.state_recovery   ( v_now, y_now   ) )
      for i in range(steps):
         print('Progress = {}%%'.format(100.*i/steps))
         y_now  = self.insp[:,i]
         v_now  = Matrix( self.rk4_obsever_step ( v_now, y_now, U)   )
         x_now  = Matrix( self.state_recovery   ( v_now, y_now   )   )
         obs_state = obs_state.col_insert(i+1, x_now )
      self.obs_state = obs_state
      return obs_state

def plot_history(hist):
   pos_time = hist.shape[0]-1
   leg_list = []
   for temp_i in range(pos_time):
      leg, = plt.plot( flatten( hist[pos_time,:]/(60.) ),
                       flatten( hist[temp_i  ,:]       ),
                       label ='Theta_'+str(temp_i)
                     )
      leg_list.append(leg)
   plt.title('Historico da Temperatura')
   plt.xlabel('Tempo (min)')
   plt.ylabel('Temperatura (K)')
   plt.grid(True)
   plt.gca().add_artist(plt.legend(handles=leg_list, loc=0))
   plt.show()

def find_elem(nodes_table, elem_table, pos_x, pos_y):
   elem_num = 0
   xi,yi = get_nodes_position(nodes_table, elem_table, elem_num)
   x_inf, x_sup = min(xi), max(xi)
   y_inf, y_sup = min(yi), max(yi)
   inside = (pos_x >= x_inf) and \
            (pos_x <= x_sup) and \
            (pos_y <= y_sup) and \
            (pos_y >= y_inf)
   while not inside:
      elem_num += 1
      xi,yi = get_nodes_position(nodes_table, elem_table, elem_num)
      x_inf, x_sup = min(xi), max(xi)
      y_inf, y_sup = min(yi), max(yi)
      inside = (pos_x > x_inf) and \
               (pos_x < x_sup) and \
               (pos_y < y_sup) and \
               (pos_y < y_sup)
   if inside:
      return elem_num
   else:
      return -1

def get_output_matrix(nodes_table, elem_table, position_list):
   C_aux = []
   for pos in position_list:
      pos_x, pos_y = pos[0], pos[1]
      x_var, y_var = symbols('x y')
      elem_num = find_elem(nodes_table, elem_table, pos_x, pos_y)
      if elem_num != -1:
         xi,yi = get_nodes_position(nodes_table, elem_table, elem_num)
         B     = get_local_B(x_var,y_var, xi, yi)
         B     = Subs(B, x_var, pos_x).doit()
         B     = Subs(B, y_var, pos_y).doit()
         B_aux = zeros(len(nodes_table),1)
         B_global = local_global_vector_map(B_aux, B, elem_table, elem_num)
         C_aux.append(B_global)
   C = C_aux[0]
   for i in range(1,len(C_aux)):
      C = C.col_insert(i, C_aux[i])
   return C.T

def choose_iter(elements, length):
   for i in xrange(len(elements)):
      if length == 1:
         yield (elements[i],)
      else:
         for next in choose_iter(elements[i+1:len(elements)], length-1):
            yield (elements[i],) + next

def choose(l, k):
   return list(choose_iter(l, k))

def complete_non_singular(C):
   lenght_square = C.shape[1]
   length_to_complete = C.shape[1] -C.shape[0]
   eye_m = eye(lenght_square)
   lines_index_comb = choose(range(lenght_square), length_to_complete)
   for index_tuple in lines_index_comb:
      possible_return = eye_m.row(index_tuple[0])
      for i in range(1,len(index_tuple)):
         possible_return = possible_return.row_insert(i, eye_m.row(index_tuple[i]))
      possible_return = possible_return.col_join(C)
      if possible_return.rank() == lenght_square:
         return possible_return

def generate_reduced_observer(A, B, C, speed=2):
   P      = complete_non_singular(C)
   A = sysfe.A
   B = sysfe.B
   C = sysfe.C
   fastest_pole = abs(min(get_eigen(A)))
   new_pole1 = (-0.6573 +0.8302j) *fastest_pole*2
   new_pole2 = (-0.6573 -0.8302j) *fastest_pole*2
   new_pole3 = (-0.9047 +0.2711j) *fastest_pole*2
   new_pole4 = (-0.9047 -0.2711j) *fastest_pole*2
   poles = [new_pole1, new_pole2, new_pole3, new_pole4]
   K_size = C.shape[1] -C.shape[0]
   A_aux  = P*A*P.inv()
   B_aux  = P*B
   A11    = Matrix(A_aux)[       :K_size,       :K_size ]
   A12    = Matrix(A_aux)[       :K_size, K_size:       ]
   A21    = Matrix(A_aux)[ K_size:      ,       :K_size ]
   A22    = Matrix(A_aux)[ K_size:      , K_size:       ]
   B11    = Matrix(B_aux)[       :K_size,       :       ]
   B21    = Matrix(B_aux)[ K_size:      ,       :       ]
   a = np.array(flatten(A11), dtype='float64').reshape(    K_size, K_size)
   b = np.array(flatten(A21), dtype='float64').reshape(C.shape[0], K_size)
   E = Matrix(ct.place(a.T, b.T, poles)).T
   A_red = A11 -E*A21
   B_red = A11*E -E*A21*E +A12 -E*A22
   C_red = B11 -E*B21
   return A_red, B_red, C_red, E, P

#sysfe.set_stateObsever(dt, speed=2)
#sysfe.E_obs; mg(sysfe.E_obs)

def mg(M):
   for i in range(4):
      print( ((   abs(M[i,0]) +abs(M[i,1]) ))/2. ,
             ((   abs(M[i,0]) -abs(M[i,1]) ))/2. ,
                  M[i,1])

def get_reduced_state(x0, P, E):
   reduced_order = E.shape[0]
   I = eye(reduced_order)
   I_E = I.row_join(-E)
   #pprint((I, E, I_E, P))
   return I_E*P*x0

#Al, Bl = Matrix(sysfe.A), Matrix(sysfe.B)
#A_red, B_red, C_red, E, P = generate_reduced_observer(Al,Bl, C)

# ------ MAIN ------ #
#if '__name__' == '__main__':
# 1) CRIACAO DE MATRIZES NOTAVEIS DO FEM
V1 = get_global_V1(node, elem)
V2 = get_global_V2(node, elem, heat)
M1 = get_global_M1(node, elem)
M2 = get_global_M2(node, elem)
M1_inv = M1.inv()

# 2) DEFINICAO DO SISTEMA EM ESPACO DE ESTADOS
A = -M1_inv*(M2*aa +M1*hc/(rho*cv))
B =  M1_inv*(V1*hc*T_inf/(rho*cv) + V2/(rho*cv))
C = get_output_matrix(node, elem, output_pos_list)

sysfe = System(A, B, C)
sysfe.set_numeric_form(phys_varbl, phys_param)
#print sysfe.check_controlability (sym=False)
#print sysfe.check_observability  (sym=False)

open_loop_eig = sysfe.eigenvalues(sym=False); pprint(open_loop_eig )
slowest_eig = max(open_loop_eig); fastest_eig = min(open_loop_eig)
#pprint((slowest_eig, fastest_eig ))

xrr   = Matrix([250, 250, 250, 250, 250, 250])
xoo   = Matrix([300, 300, 300, 300, 300, 300])
U     = Matrix([1])
dt    = 1/(20.*abs(fastest_eig))
steps = 400
sysfe.set_RK(dt)
sysfe.set_stateObsever(dt, speed=2)
#sysfe.E_obs; mg(sysfe.E_obs)

x,y,t = sysfe.simulate_system (xrr, U, steps)
o     = sysfe.observe_system  (xoo, U, steps)

d_hst = (x-o).row_insert(x.shape[0],t); plot_history(d_hst)
x_hst = x.row_insert(x.shape[0],t); plot_history(x_hst)
y_hst = y.row_insert(y.shape[0],t); plot_history(y_hst)
o_hst = o.row_insert(o.shape[0],t); plot_history(o_hst)

# ------ FIM DA MAIN ------ #

