
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


def lineMesh(var_y, xi=0, xf=1, qElems=10):
   qNodes = qElems + 1
   var = Matrix(symbols(var_y[0]+':'+str(qNodes)))
   for i in range(1,len(var_y)):
      var = var.col_insert(i,
         Matrix(symbols(var_y[1]+':'+str(qNodes))))
   # Nodes Table
   Nodes = zeros(qNodes, 1)
   Nodes = Nodes.col_insert(1,var)
   for i in range(qNodes):
      x = Float(xf-xi)/qElems*i +xi
      Nodes[i,0] = x
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
      # Bases de funcoes nao lineares
      x  = symbols(var_x)
      px = symbols(var_x+':'+str(degree+1))
      py = Matrix(symbols(var_y[0]+var_y[0]+':'+str(degree+1)))
      for i in range(1,len(var_y)):
         py = py.col_insert(i,
            Matrix(symbols(var_y[i]+var_y[i]+':'+str(degree+1))))
      m1 = ones(degree+1, 1)
      m2 = ones(degree+1,degree+1)
      for i in range(1,m1.shape[0]):
         m1[i,0] = x**i
         for j in range(m2.shape[1]):
            m2[i,j] = px[j]**i
      B = simplify(m2.inv()*m1)
      px = Matrix(px)
   return B,x,px,py

def globalF(Elems, Nodes, F, px, py):
   pos_node1 = 0
   pos_node2 = 1
   pos_x = 0
   pos_y = [1,Elems.shape[1]]
   qElems = Elems.shape[0]
   qNodes = Nodes.shape[0]
   pp = flatten([flatten(px),flatten(py.T)])
   G = 0
   Fj = lambdify(pp, F, 'numpy')
   for elem in range(qElems):
      print('Assembly percentage: %.2f'%(100*Float(elem)/qElems))
      node1 = Elems[elem, pos_node1]
      node2 = Elems[elem, pos_node2]
      pn1 = Nodes[node1, :]
      pn2 = Nodes[node2, :]
      pp = flatten(pn1.row_insert(1,pn2).T)
      Fi = Fj(*pp)
      G += Fi
   print('Assembly percentage: %.2f'%(100))
   return G

def newtonSolver(F, var, p, nmax=25, tol=1E-10):
   R = lambdify(var,F)
   F = Matrix([F])
   dW = F.diff(var[0])
   for i in range(1, p.shape[0]):
      dW = dW.row_insert(i,F.diff(var[i]))
   J = dW.diff(var[0])
   for i in range(1,var.shape[0]):
      J  =  J.col_insert(i,dW.diff(var[i]))

   dW = lambdify(var,dW)
   JJ = lambdify(var,J)

   # Solving System
   n = 1
   qnodes = p.shape[0]/2
   x = flatten(np.linspace(x1,x2,qnodes))
   Ji = SparseMatrix(JJ(*flatten(p)))
   Fi = Matrix(dW(*flatten(p)))
   #pprint((var,p,Ji,Fi))
   p1 = p
   p2 = p -Ji.inv()*Fi
   r1 = norm(p2-p1, inf)
   r2 = R(*flatten(p2))
   pprint((n,r1,r2))
   plt.figure(1)
   plt.subplot(211)
   plt.title('Velocity')
   plt.plot(x, flatten(p2[:qnodes,0]))
   plt.subplot(212)
   plt.title('Pressure')
   plt.plot(x, flatten(p2[qnodes:,0]))
   while r1>tol and n<nmax:
      n +=1
      p1 = p2
      Ji = SparseMatrix(JJ(*flatten(p1)))
      Fi = Matrix(dW(*flatten(p1)))
      p2 = p1 -Ji.inv()*Fi
      r1 = norm(p2-p1, inf)
      r2 = R(*flatten(p2))
      pprint((n,r1,r2))
      #pprint(p2)
      plt.subplot(211)
      plt.plot(x, flatten(p2[:qnodes,0]))
      plt.subplot(212)
      plt.plot(x, flatten(p2[qnodes:,0]))
   plt.show()

   return p2

def Dirichlet(F, var, value):
   pprint((var-value)**2)
   return F + (var-value)**2

def Jacobian(F, vars_x):
   J = F.diff(vars_x[0])
   for i in range(1,len(vars_x)):
      J = J.col_insert(i,F.diff(vars_x[i]))
   return J


if __name__ == '__main__':
   print ('\nMain: Debug Mode simulation.')

   x1, x2 = 0, 1
   vin = 2
   qElems = 7
   qNodes = qElems+1
   Pdegree = 1
   var_x = 'x'
   var_y = ['v','p']

   print ('\nMain: Creating mesh.')
   Nodes, Elems = lineMesh(var_y, xi=0, xf=1, qElems=qElems)
   pprint((Nodes,Elems))
   print ('\nMain: Setting Function Space Base.')
   B,x,px,py = baseSpace1D(var_x, var_y, degree=Pdegree)
   #pprint((B,x,px,py))
   vv,pp = py[:,0], py[:,1]

   Ax = Matrix([0,1,2])/2
   Ay = Matrix([4,3,4])/4
   A = area1Dfunction(Ax,Ay,var_x)
   dA = A.diff(x)
   #pprint(A)
   eq1 = (B.T*vv*dA +B.diff(x).T*vv*A)[0]
   eq2 = (B.diff(x).T*pp + (B.T*vv)*(B.diff(x).T*vv))[0]
   #pprint(eq1)
   #pprint(eq2)
   print ('Main: Setting elementary formulation.')
   W = integrate(eq1**2+eq2**2, (x,px[0],px[1]))
   print ('Main: Setting global assembly.')
   F = globalF(Elems, Nodes, W, px, py)
   var = Matrix(flatten(Nodes[:,1:].T))
   F = Dirichlet(F, var[0         ], 1)
   F = Dirichlet(F, var[qNodes*2-1], 0)
   print ('Main: setting newton formulation.')
   #pp = randMatrix(qNodes*2,1)/10.
   pp = ones(qNodes*2,1)
   pp = newtonSolver(F, var, pp, nmax=100)

   plt.figure(2); x = flatten(np.linspace(x1,x2,qnodes))
   plt.subplot(211); plt.title('Velocity'); plt.plot(x, flatten(p2[:qnodes,0]))
   plt.subplot(212); plt.title('Pressure'); plt.plot(x, flatten(1000*p2[qnodes:,0]))
   plt.show()
