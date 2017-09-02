from     math              import pi
import   numpy             as np
import   matplotlib.pyplot as plt

max_tanh = np.tanh(1)
k1 = 1
k2 = 2
k3 = 4
k4 = 8
xx  = np.arange(0,1,0.01)
fx1 = np.tanh((xx*2.0-1.0)*k1)/(np.tanh(k1)*2.0)+1/2.0
fx2 = np.tanh((xx*2.0-1.0)*k2)/(np.tanh(k2)*2.0)+1/2.0
fx3 = np.tanh((xx*2.0-1.0)*k3)/(np.tanh(k3)*2.0)+1/2.0
fx4 = np.tanh((xx*2.0-1.0)*k4)/(np.tanh(k4)*2.0)+1/2.0
plt.plot(xx,fx1)
plt.plot(xx,fx2)
plt.plot(xx,fx3)
plt.plot(xx,fx4)
plt.show()
