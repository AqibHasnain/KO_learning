import numpy as np

# Author: John Burkardt

### TO-DO ###
# need to write this all in pytorch and keep track of all gradients so that I don't have to worry about that when training
# can be a in-file function for deep_KG_learning. 

# Input
# a: the matrix

a = np.random.uniform(-1,1,size=(10,10))
np.savetxt("testmat.csv", a, delimiter=",")
n = a.shape[0]

q = 6 # don't know what this is yet
a2 = a.copy() # I think 'a' is the matrix of interest
a_norm = np.linalg.norm(a2,np.inf) 
ee = (int)(np.log2(a_norm)) + 1 
s = max(0,ee + 1)
a2 = a2/(2.0**s)
x = a2.copy()
c = 0.5
e = np.eye(n, dtype = np.complex64) + c*a2
d = np.eye(n, dtype = np.complex64) - c*a2

p = True
for k in range (2,q + 1):

  c = c * float(q - k + 1)/float(k * (2 * q - k + 1))

  x = np.dot(a2, x)

  e = e + c * x

  if (p):
    d = d + c*x
  else:
    d = d - c*x

  p = not p
#
#  E -> inverse(D) * E
#
e = np.linalg.solve (d,e)
#
#  E -> E^(2*S)
#
for k in range (0,s):
  e = np.dot(e,e)

print(e)
