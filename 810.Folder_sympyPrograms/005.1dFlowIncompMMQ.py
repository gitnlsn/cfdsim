print ('Importing libraries.')
from sympy import *
from mpmath import inf,norm
import numpy as np
import matplotlib.pyplot as plt

init_printing()

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
   var = symbols(var)
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


def lineMesh(var_y, L=1, qElems=10):
   qNodes = qElems + 1
   var_y = symbols(var_y+':'+str(qNodes))
   # Nodes Table
   Nodes = zeros(qNodes, 2)
   for i in range(qNodes):
      pos_x = Float(L)/qElems*i
      Nodes[i,0] = pos_x
      Nodes[i,1] = var_y[i]
   # Elements Table
   Elems = zeros(qElems, 2)
   for i in range(qElems):
      pos_node1 = i
      pos_node2 = i+1
      Elems[i,0] = pos_node1
      Elems[i,1] = pos_node2
   return Nodes, Elems


def baseSpace1D(var_x, var_y, tipe='P', degree=1):
   if tipe == 'P':
      # Lagrange Polinomials
      if degree == 1:
         x  = symbols(var_x)
         px = x1,x2 = symbols(var_x+':2')
         py = symbols(var_y+var_y+':2')
         m1 = Matrix([1, x])
         m2 = Matrix([
            [ 1,  1],
            [x1, x2]])
         B = m2.inv()*m1
         px = Matrix(px)
         py = Matrix(py)
         return B,x,px,py
      # Bases de funcoes nao lineares
      x  = symbols(var_x)
      px = symbols(var_x+':'+str(degree+1))
      py = symbols(var_y+var_y+':'+str(degree+1))
      m1 = ones(degree+1, 1)
      m2 = ones(degree+1,degree+1)
      for i in range(1,m1.shape[0]):
         m1[i,0] = x**i
         for j in range(m2.shape[1]):
            m2[i,j] = p[j]**i
      B = simplify(m2.inv()*m1)
      px = Matrix(px)
      py = Matrix(py)
   return B,x,px,py

def globalF(Elems, Nodes, F, px, py):
   pos_node1 = 0
   pos_node2 = 1
   pos_x = 0
   pos_y = 1
   qElems = Elems.shape[0]
   qNodes = Nodes.shape[0]
   G = 0
   Fj = lambdify(Matrix([px,py]), F)
   for elem in range(qElems):
      print('Assembly percentage: %.2f'%(100*Float(elem)/qElems))
      node1 = Elems[elem, pos_node1]
      node2 = Elems[elem, pos_node2]
      px1 = Nodes[node1, pos_x]
      px2 = Nodes[node2, pos_x]
      py1 = Nodes[node1, pos_y]
      py2 = Nodes[node2, pos_y]
      pxy = Matrix([px1,px2,py1,py2])
      Fi = Fj(*flatten(pxy))
      #Fi = Subs(F ,py,[py1,py2]).doit()
      #Fi = Subs(Fi,px,[px1,px2]).doit()
      G += Fi
   return G

def newtonFormulation(p, F):
   F = Matrix([F])
   dW = F.diff(p[0])
   for i in range(1, p.shape[0]):
      dW = dW.row_insert(i,F.diff(p[i]))
   J = dW.diff(p[0])
   for i in range(1,p.shape[0]):
      J  =  J.col_insert(i,dW.diff(p[i]))
   return p - J.inv()*dW

def Dirichlet(F, var, value):
   return F + (var-value)**2

def Jacobian(F, vars_x):
   J = F.diff(vars_x[0])
   for i in range(1,len(vars_x)):
      J = J.col_insert(i,F.diff(vars_x[i]))
   return J

def solveNonlinear(Nodes, G, p1, tol=1E-10, nmax=25):
   pos_x = 0
   pos_y = 1
   var = Nodes[:,pos_y]
   print('Setting Jacobian.')
   J = simplify(Jacobian(G,var))
   Ji_F = lambdify(var,J.inv()*G)
   p2 = p1 -Ji_F(*flatten(p1))
   r1 = norm(p2-p1, inf)
   r2 = norm(p2-p1, 2)
   n = 1
   pprint((n, r1, r2))
   x = np.linspace(x1,x2,qNodes)
   plt.plot(np.linspace(x1,x2,qNodes), flatten(p2))
   
   while r1>tol and r2>tol and n<nmax:
      p1 = p2
      p2 = p1 -Ji_F(*flatten(p1))
      r1 = norm(p2-p1, inf)
      r2 = norm(p2-p1, 2)
      pprint((n, r1, r2))
      plt.plot(np.linspace(x1,x2,qNodes), flatten(p2))
      n  += 1
   plt.show()
   return p2

if __name__ == '__main__':
   print ('Main: Debug Mode simulation.')
   
   x1, x2 = 0, 1
   vin = 1
   qElems = 20
   qNodes = qElems+1
   Pdegree = 1

   var_x = 'x'
   var_y = 'v'
   print ('Main: Creating mesh.')
   Nodes, Elems = lineMesh(var_y, L=x2-x1, qElems=qElems)
   print ('Main: Setting Function Space Base.')
   B,x,px,py = baseSpace1D(var_x, var_y, degree=Pdegree)

   Ax = Matrix([0,1,2])/2
   Ay = Matrix([4,3,4])/4
   A = area1Dfunction(Ax,Ay,var_x)
   dA = A.diff(x)

   v = B.T*py
   dv = B.diff(x).T*py

   print ('Main: Setting elementary formulation.')
   W = integrate(simplify((A*dv+v*dA)[0]**2), (x,px[0],px[1]) )
   pprint(W)
   print ('Main: Setting global assembly.')
   var = Nodes[:,1]
   F = globalF(Elems, Nodes, W, px, py)
   F = Dirichlet(F, var[0], 5)
   Newt = newtonFormulation(var, F)
   R = lambdify(var, F)
   ite = lambdify(var, Newt)

   print ('Main: Solving problem.')
   p0 = ones(qNodes,1)
   p2 = Matrix(ite(*flatten(p0)))
   r1 = norm(p2-p0, inf)
   r2 = R(*flatten(p2))
   x = np.linspace(x1,x2,qNodes)
   tol, nmax, n = 1E-10, 25, 1
   pprint((n, r1, r2))
   while r1>tol and n<nmax:
      n +=1
      p1 = p2
      p2 = Matrix(ite(*flatten(p0)))
      r1 = norm(p2-p1, inf)
      r2 = R(*flatten(p2))
      pprint((n,r1,r2))
      plt.plot(x, flatten(p2))
   plt.show()

