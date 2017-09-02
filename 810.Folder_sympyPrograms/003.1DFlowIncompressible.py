
from sympy import *
import numpy as np
import matplotlib.pyplot as plt

def area1Dfunction (xi, yi, var, degree=2):
   ''' Min Least Squares Aproximation
      xi = [x0, x1, ..., xn].t
      yi = [y0, y1, ..., yn].t
      
      M = [1, xi, ..., xi**n]

      M.t*M*C = M.t*y
      => C = (M.t*M)**-1 *M.t*y
      
      B = [1, x, ..., x**d]

      F = B*C
   '''
   p = ones(len(yi),1)
   q = ones(1,1)
   M = ones(len(yi),1)
   B = Matrix([1])
   for n1 in range(1,degree+1):
      for n2 in range(len(yi)):
         p[n2]=p[n2]*xi[n2]
      M = M.col_insert(n1,p)
      q = q*var
      B = B.col_insert(n1,q)
   C = (M.T*M).inv() *M.T*yi
   F = (B*C).det()
   return F

def baseSpace1D(tipe='P', degree=1):
   if tipe == 'P':
      if degree==1:
         x = symbols('x')
         p = x1,x2 = symbols('x:2')
         m1 = Matrix([1, x])
         m2 = Matrix([
            [ 1,  1],
            [x1, x2]])
         B = m2.inv()*m1
   return B,x,p

def basicElem(base, var, trialDegree=0, testDegree=0):
   trial = test = base
   for i in range(trialDegree):
      trial = trial.diff(var)
   for i in range(testDegree):
      test = test.diff(var)
   return test*trial.T

def globalSys(Me, xi, x0, xn, Ne):
   Mg = zeros(Ne+1, Ne+1)
   Me = lambdify(xi, Me)
   d = (xn-x0)/Ne
   for i in range(Ne):
      x1 = x0 +d*i
      x2 = x0 +d*(i+1)
      p = Matrix([x1,x2])
      M = Me(*flatten(p))
      Mg[i  ,i  ] = Mg[i  ,i  ]+M[0,0]
      Mg[i  ,i+1] = Mg[i  ,i+1]+M[0,1]
      Mg[i+1,i  ] = Mg[i+1,i  ]+M[1,0]
      Mg[i+1,i+1] = Mg[i+1,i+1]+M[1,1]
   return Mg

def solve(Mg, x0):
   Mg[0,0] = 1
   for i in range(1,Mg.shape[0]):
      Mg[0,i] = 0
   B = zeros(Mg.shape[0],1)
   B[0,0] = x0
   yi = Mg.inv()*B
   return yi

if __name__ == '__main__':
   init_printing()
   x1, x2 = 0.,1.
   qElems = 5
   qNodes = qElems+1
   B,x,xi = baseSpace1D()
   Ax = Matrix([0,1,2])/2
   Ay = Matrix([4,3,4])/4
   A = area1Dfunction(Ax,Ay,x)
   dA = A.diff(x)
   E = basicElem(B,x,trialDegree=0,testDegree=0)*dA \
      +basicElem(B,x,trialDegree=1,testDegree=0)*A
   Me = integrate(E,(x,xi[0],xi[1]))
   Mg = globalSys(Me, xi, x1, x2, qElems)
   pprint (Mg)
   ss = solve(Mg, 1)
   plt.plot(np.linspace(x1, x2, qNodes),flatten(ss))
   plt.show()

