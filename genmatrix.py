from numpy.linalg import qr
from numpy.random import rand
from random import randint
import random as rd
from numpy.linalg import inv
import numpy as np
from qr import solve_svd, solve_qr

def t_func(x):
    return 2+(x/5)


x_size = 11
y_size = 8
matrix = []
for i in range(11):
    row = []
    for j in range(8):
        row.append(t_func(i)**j)
    matrix.append(row)

A = np.array(matrix)

c = [1 for _ in range(8)]
c = np.array(c)
c_true = c.T

z = np.random.rand(11)
print(f"Random vector z: {z}")

factor = np.linalg.norm(c)/np.linalg.norm(z)
z *= factor
print(f"Z after scaling: {z}")

b = A@c_true

#print(np.linalg.svd(A))

solved_x = solve_svd(A, b)

print(solved_x)

solved_x_qr = solve_qr(A, b)
print(solved_x_qr)

A_t_A = A.T@A

U, S, V = np.linalg.svd(A, full_matrices=False)

k2 = A.max() / A.min()
k2 = S[0]/S[-1]
print(f"Condition number: {k2}")
print(f"Condition number squared: {k2**2}")
k2_AT_A = A_t_A.max() / A_t_A.min()
k2_AT_A = np.linalg.cond(A_t_A)
print(f"Condition number of A^T*A: {k2_AT_A}")

for i in range(3): 
    epi = rd.uniform(0.0001, 0.001)*i
    b_noise = b + epi*z 
    print(f"\n\n\nNoise level: {epi}")
    solved_x_noise = solve_svd(A, b_noise)
    print(f"Solution with noise using SVD: {solved_x_noise}")

