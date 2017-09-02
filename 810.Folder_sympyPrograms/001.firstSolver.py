print('FirstSolver: Importing Sympy...')
from sympy import *

print('FirstSolver: Defining nonlinear problem...')
init_printing()

X = symbols('x:2')
x,y = X[0],X[1]

f = x*x +4*y*y -9
g = 18*y -14*x*x +45

print('\nFirstSolver: Function f:')
pprint (f)
print('\nFirstSolver: Function g:')
pprint (g)

F = Matrix([f,g])

J = F.diff(x)
J = J.col_insert(1, F.diff(y))
Ji = J.inv()

print('\nFirstSolver: Jacobian:')
pprint(J)

p1 = p2 = Matrix([1.,2.])

print('\nInicial Value:')
pprint(p1)
print('\nFirstSolver: Iterations Begining...')
p1 = p2; p2 = p1-Subs(Ji,X,p1).doit()*Subs(F,X,p1).doit()

tol, r, n = 1E-10, 1, 1
while r > tol:
   p1 = p2; p2 = p1-Subs(Ji,X,p1).doit()*Subs(F,X,p1).doit()
   r = ((p2-p1).T*(p2-p1)).det()
   n+=1
   print('({:7.5e},{:7.5e}): {:7.5e}'.format(p2[0],p2[1],r))

