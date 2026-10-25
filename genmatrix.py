from numpy.linalg import qr
from numpy.random import rand
from random import randint
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

b = A@c_true

print(np.linalg.svd(A))

solved_x = solve_svd(A, b)
print(solved_x)

solved_x_qr = solve_qr(A, b)
print(solved_x_qr)

