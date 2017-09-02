'''
DESCRICAO:

DATA:    18.05.2017
AUTOR:   NELSON KENZO TAMASHIRO
'''
# ------ IMPORTACAO DE BIBLIOTECAS ------ #
from     sympy             import   *
import   numpy             as       np
import   control           as       ct
import   matplotlib.pyplot as       plt
init_printing()

# ------ PARAMETROS FISICOS DO PROBLEMA ------ #
                     # SILICIO - 300K
R_dens =  2330.E+0    # densidade
c_capc =  712.0E+0    # capacidade termica
a_diff =  89.20E-6    # difusividade termica
hc_for = 1500.0E+0    # difusividade termica
TT_inf =  300.0E+0    # temperatura ambiente
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
   V2  = zeros(V2_size, 1)
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
   def __init__ (self, A, B1, B2, C):
      self.A  = A
      self.B1 = B1
      self.B2 = B2
      self.C  = C
   def eigenvalues(self, sym=True):
      if sym:
         A = self.A_sym
      else:
         A = self.A
      return get_eigen(A)
   def set_numeric_form(self, phys_varbl, phys_param):
      self.A_sym  = Matrix(A )
      self.B1_sym = Matrix(B1)
      self.B2_sym = Matrix(B2)
      self.C_sym  = Matrix(C )
      self.A  = Subs(self.A , flatten(phys_varbl), flatten(phys_param)).doit()
      self.B1 = Subs(self.B1, flatten(phys_varbl), flatten(phys_param)).doit()
      self.B2 = Subs(self.B2, flatten(phys_varbl), flatten(phys_param)).doit()
      self.C  = Subs(self.C , flatten(phys_varbl), flatten(phys_param)).doit()
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
   # DETERMINISTIC INTEGRATOR SETUP
   def set_RK_DT (self, dt):
      X   = Matrix(symbols('X:'+str(self.A.shape[1]  )))
      U   = Matrix(symbols('U:'+str(self.B1.shape[1] )))
      dX1 = self.A*(X          ) +self.B1*U +self.B2
      dX2 = self.A*(X+dX1*dt/2.) +self.B1*U +self.B2
      dX3 = self.A*(X+dX2*dt/2.) +self.B1*U +self.B2
      dX4 = self.A*(X+dX3*dt   ) +self.B1*U +self.B2
      dX  = (dX1 +dX2*2. +dX3*2. +dX4)/6.
      self.rk4_sim_DT = lambdify( (X,U), X+dX*dt , 'numpy')
      self.out_sim_DT = lambdify( (X,U), self.C*X, 'numpy')
   # STOCHASTIC INTEGRATOR SETUP
   def set_RK_ST (self, dt):
      X   = Matrix(symbols('X:'+str(self.A.shape[1]  )))
      U   = Matrix(symbols('U:'+str(self.B1.shape[1] )))
      dBrown  = Matrix(symbols('dB:'+str(self.B1.shape[1] )))
      wNoise  = Matrix(symbols('Wh:'+str(self.C.shape[0]  )))
      dX1 = self.A*(X          ) +self.B1*(U+dBrown) +self.B2
      dX2 = self.A*(X+dX1*dt/2.) +self.B1*(U+dBrown) +self.B2
      dX3 = self.A*(X+dX2*dt/2.) +self.B1*(U+dBrown) +self.B2
      dX4 = self.A*(X+dX3*dt   ) +self.B1*(U+dBrown) +self.B2
      dX  = (dX1 +dX2*2. +dX3*2. +dX4)/6.
      self.rk4_sim_ST = lambdify( (X,U,dBrown), X +dX*dt, 'numpy')
      self.out_sim_ST = lambdify( (X,U,wNoise), self.C*X +wNoise, 'numpy')
   # DETERMINISTIC SYSTEM SIMULATION
   def sim_sys_DT(self, x0, U, steps):
      hist = Matrix(   x0 )
      time = Matrix(  [0] )
      insp = Matrix( self.C*x0 )
      for i in range(steps):
         print('Progress = {}%'.format(100.*i/steps))
         t_now  = time[0,i]
         x_now  = hist[:,i]
         U_now  = U[:,i]
         x_next = Matrix( self.rk4_sim_DT (x_now, U_now) )
         y_next = Matrix( self.out_sim_DT (x_now, U_now) )
         time = time.col_insert(i+1, Matrix([t_now+dt]))
         hist = hist.col_insert(i+1, x_next  )
         insp = insp.col_insert(i+1, y_next  )
      self.hist = hist
      self.time = time
      self.insp = insp
      return hist, insp, time
   # STOCHASTIC SYSTEM SIMULATION
   def sim_sys_ST(self, x0, U, steps, Q, variance=1E-3):
      dBrown  = Q*get_brownianStepVector(length=steps, size=self.B1.shape[0])
      wNoise  = get_whiteNoiseVector(mean_value=0.0, variance=variance, length=steps, size=self.C.shape[0])
      hist = Matrix(   x0 ) # estado inicial com probabilidade 1: wp1
      time = Matrix(  [0] )
      insp = Matrix( self.C*x0 )
      for i in range(steps):
         print('Progress = {}%'.format(100.*i/steps))
         t_now  = time  [0,i]
         x_now  = hist  [:,i]
         U_now  = U[:,i]
         dB_now = dBrown[:,i]
         wn_now = wNoise[:,i]
         x_next = Matrix( self.rk4_sim_ST (x_now, U_now, dB_now) )
         y_next = Matrix( self.out_sim_ST (x_now, U_now, wn_now) )
         time = time.col_insert(i+1, Matrix([t_now+dt]))
         hist = hist.col_insert(i+1, x_next  )
         insp = insp.col_insert(i+1, y_next  )
      self.hist = hist
      self.time = time
      self.insp = insp
      return hist, insp, time
   # LUENBERGER OBSERVER SETUP
   def set_observer_DT(self, dt, speed=2):
      self.A_red, self.B1_red, self.B2_red, self.C_red, self.E_obs, self.P_obs \
          = generate_reduced_observer(self.A, self.B1, self.B2, self.C, speed=speed)
      V   = Matrix(symbols('V:'+str(self.A_red.shape [1] )))
      Y   = Matrix(symbols('Y:'+str(self.B1_red.shape[1] )))
      U   = Matrix(symbols('U:'+str(self.B2_red.shape[1] )))
      dV1 = self.A_red*(V          ) +self.B1_red*Y +self.B2_red*U +self.C_red
      dV2 = self.A_red*(V+dV1*dt/2.) +self.B1_red*Y +self.B2_red*U +self.C_red
      dV3 = self.A_red*(V+dV2*dt/2.) +self.B1_red*Y +self.B2_red*U +self.C_red
      dV4 = self.A_red*(V+dV3*dt   ) +self.B1_red*Y +self.B2_red*U +self.C_red
      dV  = (dV1 +dV2*2. +dV3*2. +dV4)/6.
      self.rk4_obs_DT = lambdify( (V,Y,U), V+dV*dt, 'numpy')
      self.stt_obs_DT = lambdify(
            (V,Y),
            self.P_obs.inv()*(V+self.E_obs*Y).col_join(Y),
            'numpy')
   # KALMAN FILTER OBSERVER SETUP
   def set_observer_ST(self, dt, Q, R):
      self.set_RK_DT (dt=dt)
      self.R = R
      self.Q = Q
      X = Matrix(symbols('V:'+str(self.A.shape[0] )))
      Y = Matrix(symbols('Y:'+str(self.C.shape[0] )))
      U = Matrix(symbols('U:'+str(self.B1.shape[1] )))
      P = Matrix(symbols('P:'+str(self.A.shape[0] )   \
        +':'+str(self.A.shape[1]  ) )).reshape(*self.A.shape)
      K = Matrix(symbols('k:'+str(self.C.T.shape[0])  \
        +':'+str(self.C.T.shape[1]) )).reshape(*self.C.T.shape)
      G = self.B1
      M = self.C
      I = eye(P.shape[0])
      self.residuum = lambdify(
         flatten([ flatten(Y), flatten(X) ]),
         Y-M*X, 'numpy')
      self.rk4_obsST_Xminus  = lambdify(
         flatten([ flatten(X), flatten(U) ]),
         Matrix(self.rk4_sim_DT(X,U)), 'numpy')
      self.rk4_obsST_Xplus   = lambdify( 
         flatten([ flatten(X), flatten(Y),flatten(K)  ]),
         X +K*(Y-M*X),                 'numpy')
      dP1 = self.A*(P          ) +(P          )*self.A.T +G*Q*G.T
      dP2 = self.A*(P+dP1*dt/2.) +(P+dP1*dt/2.)*self.A.T +G*Q*G.T
      dP3 = self.A*(P+dP2*dt/2.) +(P+dP2*dt/2.)*self.A.T +G*Q*G.T
      dP4 = self.A*(P+dP3*dt   ) +(P+dP3*dt   )*self.A.T +G*Q*G.T
      dP  = (dP1 +dP2*2. +dP3*2. +dP4)/6.
      self.rk4_obsST_Pminus = lambdify( P, P+dP*dt, 'numpy' )
      # self.IKM = lambdify( K,   I-K*M, 'numpy')
      # self.KRK = lambdify( K, K*R*K.T, 'numpy')
      # IKM = Matrix(symbols('ikm:'+str(P.shape[0] )   \
      #    +':'+str(P.shape[1]  ) )).reshape(*P.shape)
      # KRK = Matrix(symbols('krk:'+str(P.shape[0] )   \
      #    +':'+str(P.shape[1]  ) )).reshape(*P.shape)
      # self.rk4_obsST_Pplus2 = lambdify(
      #    flatten([flatten(P), flatten(IKM), flatten(KRK)]),
      #    IKM*P*IKM.T +KRK, 'numpy' )
      self.rk4_obsST_Pplus  = lambdify( flatten([flatten(P),flatten(K)]),
            (I-K*M)*P*(I-K*M).T +K*R*K.T, 'numpy')
   # LUENBERGER OBSERVER SIMULATION
   def observe_system_DT(self, x0, U, steps, sample_freq=1):
      v_now     = get_reduced_state(x0, self.P_obs, self.E_obs)
      y_now     = self.insp[:,0]
      obs_state = Matrix( self.stt_obs_DT   ( v_now, y_now   ) )
      sample_count = 1
      for i in range(steps):
         print('Progress = {}%'.format(100.*i/steps))
         U_now  = U[:,i]
         if sample_count == sample_freq:
            y_now  = self.insp[:,i]
            sample_count = 1
         else:
            sample_count = sample_count+1
         v_now  = Matrix( self.rk4_obs_DT ( v_now, y_now, U_now ) )
         x_now  = Matrix( self.stt_obs_DT ( v_now, y_now        ) )
         obs_state = obs_state.col_insert(i+1, x_now )
      self.obs_state = obs_state
      return obs_state
   # KALMAN FILTER OBSERVER SIMULATION
   def observe_system_ST(self, x0, P0, U, steps, sample_freq=1):
      G = self.B1
      M = self.C
      sys_size  = x0.shape[0]
      y_now     = self.insp[:,0]
      X_plus    = x0
      P_plus    = P0
      obs_state_ST = x0
      obs_varic_ST = Matrix(P0).reshape(P0.shape[0]*P0.shape[1], 1) # covariancia de forma vetorial
      obs_resss_ST = Matrix(self.residuum(
                  *flatten([  flatten(y_now),
                              flatten(X_plus)    ])         )).reshape(*y_now.shape)
      sample_count = 1
      for i in range(steps):
         print('Progress = {}%'.format(100.*i/steps))
         y_now   = self.insp[:,i]
         U_now   = U[:,i]
         X_minus = Matrix(self.rk4_obsST_Xminus(*flatten([ flatten(X_plus), flatten(U_now) ])))
         P_minus = Matrix(self.rk4_obsST_Pminus(*flatten(P_plus))).reshape(*P0.shape)
         K_Kalm  = P_minus *M.T *(M*P_minus*M.T +R).inv()
         residum = Matrix(self.residuum(
                  *flatten([  flatten(y_now),
                              flatten(X_plus)    ])         )).reshape(*y_now.shape)
         # IKM = Matrix(self.IKM(*flatten(K_Kalm))).reshape(*P_minus.shape)
         # KRK = Matrix(self.KRK(*flatten(K_Kalm))).reshape(*P_minus.shape)
         # P_plus = Matrix(  self.rk4_obsST_Pplus2( 
         #       *flatten([  flatten(P_minus),
         #                   flatten(IKM),
         #                   flatten(KRK)         ])       )).reshape(P0.shape[0]*P0.shape[1], 1)
         if sample_count == sample_freq:
            P_plus  = Matrix(self.rk4_obsST_Pplus(
                  *flatten([  flatten(P_minus)  ,
                              flatten(K_Kalm)      ])       )).reshape(P0.shape[0]*P0.shape[1], 1)
            X_plus  = Matrix( self.rk4_obsST_Xplus(
                  *flatten([  flatten(X_minus)  ,
                              flatten(y_now)    ,
                              flatten(K_Kalm)      ])       )).reshape(P0.shape[0], 1)
            sample_count = 1
         else:
            P_plus = Matrix(P_minus).reshape(P0.shape[0]*P0.shape[1], 1)
            X_plus = Matrix(X_minus).reshape(P0.shape[0], 1)
            sample_count = sample_count+1
         obs_state_ST = obs_state_ST.col_insert(i+1, X_plus )
         obs_varic_ST = obs_varic_ST.col_insert(i+1, P_plus )
         obs_resss_ST = obs_resss_ST.col_insert(i+1, residum)
      self.obs_state_ST = obs_state_ST
      self.obs_varic_ST = obs_varic_ST
      self.obs_resss_ST = obs_resss_ST
      return obs_state_ST, obs_varic_ST, obs_resss_ST

def plot_history(hist, title_name, label_name, ylabel_name):
   pos_time = hist.shape[0]-1
   leg_list = []
   for temp_i in range(pos_time):
      leg, = plt.plot( flatten( hist[pos_time,:]/(60.) ),
                       flatten( hist[temp_i  ,:]       ),
                       label =label_name+'_'+str(temp_i)
                     )
      leg_list.append(leg)
   plt.title(title_name)
   plt.xlabel('Tempo (min)')
   plt.ylabel(ylabel_name)
   plt.grid(b=True, which='major', color='k', linestyle='-')
   plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
   plt.minorticks_on()
   plt.gca().add_artist(plt.legend(handles=leg_list, loc=0))
   plt.show()

def get_diagonal(hist):
   hist_size = int(hist.shape[0]**(1.0/2))
   hist_length = hist.shape[1]
   position = 0
   column   = 0
   variance = Matrix([hist[position,column]])
   for i in range(1, hist_size):
      position = i*(hist_size)+i
      variance = variance.row_insert(i,Matrix([hist[position,column]]))
   for j in range(1, hist_length):
      position = 0
      column   = j
      aux = Matrix([hist[position,column]])
      for i in range(1,hist_size):
         position = i*(hist_size)+i
         aux = aux.row_insert(i,Matrix([hist[position,column]]))
      variance = variance.col_insert(column, aux)
   return variance

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
   length_square = C.shape[1]
   length_to_complete = C.shape[1] -C.shape[0]
   eye_m = eye(length_square)
   lines_index_comb = choose(range(length_square), length_to_complete)
   for index_tuple in lines_index_comb:
      possible_return = eye_m.row(index_tuple[0])
      for i in range(1,len(index_tuple)):
         possible_return = possible_return.row_insert(i, eye_m.row(index_tuple[i]))
      possible_return = possible_return.col_join(C)
      if possible_return.rank() == length_square:
         return possible_return

def generate_reduced_observer(A, B1, B2, C, speed=2):
   P      = complete_non_singular(C)
   fastest_pole = abs(min(get_eigen(A)))
   new_pole1 = (-0.6573 +0.8302j) *fastest_pole*2
   new_pole2 = (-0.6573 -0.8302j) *fastest_pole*2
   new_pole3 = (-0.9047 +0.2711j) *fastest_pole*2
   new_pole4 = (-0.9047 -0.2711j) *fastest_pole*2
   poles = [new_pole1, new_pole2, new_pole3, new_pole4]
   K_size = C.shape[1] -C.shape[0]
   A_aux  = P*A*P.inv()
   B_aux  = P*B1
   C_aux  = P*B2
   A11    = Matrix(A_aux)[       :K_size,       :K_size ] # correspondente a A
   A12    = Matrix(A_aux)[       :K_size, K_size:       ]
   A21    = Matrix(A_aux)[ K_size:      ,       :K_size ]
   A22    = Matrix(A_aux)[ K_size:      , K_size:       ]
   B11    = Matrix(B_aux)[       :K_size,       :       ] # correspondente a B1
   B21    = Matrix(B_aux)[ K_size:      ,       :       ]
   C11    = Matrix(C_aux)[       :K_size,       :       ] # correspondente a B2
   C21    = Matrix(C_aux)[ K_size:      ,       :       ]
   a = np.array(flatten(A11), dtype='float64').reshape(    K_size, K_size)
   b = np.array(flatten(A21), dtype='float64').reshape(C.shape[0], K_size)
   E = Matrix(ct.place(a.T, b.T, poles)).T
   A_red  = A11 -E*A21
   B1_red = A11*E -E*A21*E +A12 -E*A22
   B2_red = B11 -E*B21
   C_red  = C11 -E*C21
   return A_red, B1_red, B2_red, C_red, E, P

#sysfe.set_stateObsever(dt, speed=2)
#sysfe.E_obs; mg(sysfe.E_obs)

def get_reduced_state(x0, P, E):
   reduced_order = E.shape[0]
   I = eye(reduced_order)
   I_E = I.row_join(-E)
   #pprint((I, E, I_E, P))
   return I_E*P*x0

def get_whiteNoiseVector(mean_value=0.0, variance=1E-6, length=100, size=1):
   whiteNoise_vector = Matrix(np.random.normal( mean_value, variance, length )).T
   for i in range(1,size):
      whiteNoise_i = Matrix(np.random.normal( mean_value, variance, length )).T
      whiteNoise_vector = whiteNoise_vector.col_join(whiteNoise_i)
   return whiteNoise_vector

def get_brownianStepVector(length=100, size=1):
   whiteNoise = get_whiteNoiseVector(length=length, size=size)
   brownianStepVector = Matrix(zeros(size,length))
   for i in range(size):
      for j in range(1,length):
         if whiteNoise[i,j] > 0:
            brownianStepVector[i,j] = +1
         else:
            brownianStepVector[i,j] = -1
   return brownianStepVector

def get_U_2steps(U, steps=100):
   U_zero   = zeros(*U.shape)
   U_degree = Matrix(U_zero)
   pos_0 = steps*0.0
   pos_1 = steps*0.3
   pos_2 = steps*0.7
   pos_3 = steps*1.0
   for i in range(steps):
      if (i > pos_0 and i < pos_1) or (i > pos_2 and i < pos_3):
         U_degree = U_degree.col_insert(i+1, U)
      else:
         U_degree = U_degree.col_insert(i+1, U_zero)
   return U_degree

#get_whiteNoiseVector(mean_value=0.0, variance=1E-6, length=4, size=2); plt.plot(w); plt.show();
#get_brownianStepVector(length=10, size=2); plt.plot(b); plt.show();

#Al, Bl = Matrix(sysfe.A), Matrix(sysfe.B)
#A_red, B_red, C_red, E, P = generate_reduced_observer(Al,Bl, C)

# ------ MAIN ------ #
#if '__name__' == '__main__':
# 1) CRIACAO DE MATRIZES NOTAVEIS DO FEM
V1 = get_global_V1(node, elem)      # int( B.T     )
M1 = get_global_M1(node, elem)      # int( B.T *  B)
M2 = get_global_M2(node, elem)      # int(dB.T * dB)
M1_inv = M1.inv()

# 2) DEFINICAO DO SISTEMA EM ESPACO DE ESTADOS
A  = -M1_inv * (M2*aa +M1*hc/(rho*cv))    # matrix de massa do sistema
B1 =  M1_inv * (M1 /(rho*cv))             # matrix de entrada
B2 =  M1_inv * (V1 *hc*T_inf/(rho*cv))    # matrix constante de conveccao
C = get_output_matrix(node, elem, output_pos_list) # matrix de observabilidade

sysfe = System(A, B1, B2, C)
sysfe.set_numeric_form(phys_varbl, phys_param)
#print sysfe.check_controlability (sym=False)
#print sysfe.check_observability  (sym=False)

open_loop_eig = sysfe.eigenvalues(sym=False); pprint(open_loop_eig)
slowest_eig = max(open_loop_eig); fastest_eig = min(open_loop_eig)
#pprint((slowest_eig, fastest_eig ))

steps = 400
variance = 1.0

xrr   = Matrix(   [300, 300, 300, 300, 300, 300]   )
xoo   = Matrix(   [298, 298, 298, 298, 298, 298]   )
Poo   = eye(6)*100  # variancia da estimativa inicial
Q     = eye(6)      # variancia da geracao de calor
R     = eye(2)*variance
U     = Matrix(   [  0,   0, 500,   0, 500,1000]   )
Ustep = get_U_2steps(U, steps=steps)

dt    = 1/(5.*abs(fastest_eig))    # Periodo de simulacao
pprint(dt)
sample_freq = 5

sysfe.set_RK_ST(dt)
sysfe.set_observer_DT (dt=dt, speed=2)
sysfe.set_observer_ST (dt=dt, Q=Q*10, R=R*variance/2 )

x,y,t          = sysfe.sim_sys_ST        (x0=xrr,           U=Ustep, steps=steps, Q=Q*0, variance=0)
o_DT           = sysfe.observe_system_DT (x0=xoo,           U=Ustep, steps=steps, sample_freq=1)
o_ST,s_ST,r_ST = sysfe.observe_system_ST (x0=xoo, P0 = Poo, U=Ustep, steps=steps, sample_freq=sample_freq)
s=get_diagonal(s_ST)

u_hst    = Ustep.row_insert(Ustep.shape[0],t);  plot_history(u_hst,
   title_name='Entrada do Sistema', label_name='U', ylabel_name='Potencia (W)' )  # entrada do sistema
x_hst    = x.row_insert(        x.shape[0],t);  plot_history(x_hst, 
   title_name='Estados Reais do Sistema', label_name='Theta', ylabel_name='Temperatura (K)' )  # historico dos estados
y_hst    = y.row_insert(        y.shape[0],t);  plot_history(y_hst,
   title_name='Medicoes de Temperatura', label_name='Theta', ylabel_name='Temperatura (K)' )  # output do sistema
o_hst_DT = o_DT.row_insert(  o_DT.shape[0],t);  plot_history(o_hst_DT,
   title_name='Luenberguer - Estados Observados', label_name='Theta', ylabel_name='Temperatura (K)' )  # observacao deterministica
o_hst_ST = o_ST.row_insert(  o_ST.shape[0],t);  plot_history(o_hst_ST,
   title_name='Kalmam-Buci - Estados Observados', label_name='Theta', ylabel_name='Temperatura (K)' )  # observacao stocastica
r_hst    = r_ST.row_insert(  r_ST.shape[0],t);  plot_history(r_hst,
   title_name='Kalmam-Buci - Residuo', label_name='Theta', ylabel_name='Temperatura (K)' )  # variancia stocastica
v_hst    = s.row_insert(        s.shape[0],t);  plot_history(v_hst,
   title_name='Kalmam-Buci - Variancia', label_name='Sigma2', ylabel_name='Variancia da Temperatura (K^2)' )  # variancia stocastica
d_hst_DT = (o_DT- x).row_insert(x.shape[0],t);  plot_history(d_hst_DT,
   title_name='Luenberguer - Erro', label_name='Theta', ylabel_name='Temperatura (K)' )  # erro deterministica
d_hst_ST = (o_ST- x).row_insert(x.shape[0],t);  plot_history(d_hst_ST,
   title_name='Kalmam-Buci - Erro', label_name='Theta', ylabel_name='Temperatura (K)' )  # erro stocastico


# ------ FIM DA MAIN ------ #




