from     math              import pi
import   numpy             as np
import   matplotlib.pyplot as plt

xx  = np.arange(0,1,0.01)
fx1 = np.sin(xx *pi-pi/2.0)/2.0+1.0/2.0
fx2 = np.sin(fx1*pi-pi/2.0)/2.0+1.0/2.0
fx3 = np.sin(fx2*pi-pi/2.0)/2.0+1.0/2.0
fx4 = np.sin(fx3*pi-pi/2.0)/2.0+1.0/2.0
plt.plot(xx,fx1)
plt.plot(xx,fx2)
plt.plot(xx,fx3)
plt.plot(xx,fx4)
plt.show()
