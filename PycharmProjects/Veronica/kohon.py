import numpy as np
X = np.random.normal(loc=1, scale=10, size=(1000, 50))
print(X)
print ("-----------------------------------------")
m = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = ((X - m)  / std)
print (X_norm)
print ("-----------------------------------------")
Z = np.array([[4, 5, 0],
             [1, 9, 3],
             [5, 1, 1],
             [3, 3, 3],
             [9, 9, 9],
             [4, 7, 1]])


r = np.sum(Z, axis=1)
print (np.nonzero(r > 10))
print ("-----------------------------------------")

A = np.eye(3)
B = np.eye(3)
print (A)
print ("-----------------------------------------")
print (B)
print ("-----------------------------------------")

AB = np.vstack((A, B))
print (AB)