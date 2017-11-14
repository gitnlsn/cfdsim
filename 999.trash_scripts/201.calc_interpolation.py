'''
 
SAO PAULO, 30 JUNHO 2017

SCRIPT CRIADO PARA AUTOMATIZAR INTERPOLACAO DE FUNCAO CUBICA

'''

from     numpy             import *
import   matplotlib.pyplot as plt
from     sympy             import *

x,d,t = symbols('x delta tau')
N0, N1, N2 = S(0), S(1), S(2)

var = Matrix(  [ 1,       x,         x**2,         x**3] )

A = Matrix ([  [N1,      N0,           N0,           N0],
               [N1,      N1,           N1,           N1],
               [N1, N1/N2-d, (N1/N2-d)**2, (N1/N2-d)**3],
               [N1, N1/N2+d, (N1/N2+d)**2, (N1/N2+d)**3],   ])

B = Matrix([   [0],
               [N1],
               [t],
               [1-t],   ])

C = simplify(A.inv()*B)

def get_function( delta=0.01, tau=0.01 ):
   return lambdify( x, (var.T*C.subs(d,0.01).subs(t,0.01).doit())[0], 'numpy')

f  = get_function(0.01,0.01)

xx = np.arange(0,1,0.01)
fx = f(xx)

plt.plot(xx,fx)
plt.show()

