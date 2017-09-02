from     math              import pi
import   numpy             as np
import   matplotlib.pyplot as plt


min_exp = 0.1
max_exp = min_exp
delta = 0.00001

xx  = np.arange( 0, 1+delta, delta )
fx1 = xx*(1.0-xx)/0.25
plt.plot(xx,fx1)

for exp in [min_exp]:
   fxn = fx1**exp
   plt.plot(xx,fxn)

plt.show()
