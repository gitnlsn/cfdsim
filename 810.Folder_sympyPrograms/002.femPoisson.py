

from sympy import *



def Jacobian(F,var):
   J = F.diff(var[0])
   for i in range(1,len(var)):
      J = J.col_insert(i,F.diff(var[i]))
   return J

def solveNonlinear(F,J,x,p,tol=1E-12):
   Jinv = lambdify(x,J.inv())
   Func = lambdify(x,F)
   p1 = p
   p2 = p1 -Jinv(*flatten(p1))*Func(*flatten(p1))
   while ((p2-p1).T*(p2-p1)).det() > tol:
      p1 = p2
      p2 = p1 -Jinv(*flatten(p1))*Func(*flatten(p1))
   return p2

if __name__ == "__main__":
   print 'femPoisson: debug mode.'
   s = x,y = symbols('x:2')
   F = Matrix(
      [x*x+4*y*y-9, 
      -14*x*x+18*y+45])
   J = Jacobian(F,s)
   p1 = Matrix([1.,1.])
   p2 = Matrix([1.,-1.])
   p1 = solveNonlinear(F,J,s,p1)
   p2 = solveNonlinear(F,J,s,p2)
   init_printing()
   pprint (p1)
   pprint (p2)
