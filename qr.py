from os import name

from numpy.linalg import qr
from numpy.random import rand
from random import randint
from numpy.linalg import inv
import numpy as np





if name == "__main__":
    print("Random Numbers 0-1")
    matrix = rand(4, 2)
    Q, R = qr(matrix)
    print("Original Matrix:\n", matrix)
    print("\nQ Matrix:\n", Q)
    print("\nR Matrix:\n", R)


    N = 3
    M = 2
    matrix = [[randint(1, 10) for _ in range(M)] for _ in range(N)]

    print("\n\n\nWhole Numbers")
    Q, R = qr(matrix)
    print("Original Matrix:\n", matrix)
    print("\nQ Matrix:\n", Q)
    print("\nR Matrix:\n", R)
    Q_transpose = Q.T
    print("\nQ Transpose Matrix:\n", Q_transpose)

    b = np.array([1, 4, 2])
    x_val = inv(R) @ Q_transpose @ b

    print(f"\n\nSolution x: {x_val}")



    U,S, V = np.linalg.svd(matrix,False)
    print("\n\n\nSingular Value Decomposition")
    print("Original Matrix:\n", matrix)
    print("\nU Matrix:\n", U)
    print("\nS Matrix:\n", S)
    print("\nV Matrix:\n", V)

    x_val_svd = V.T @ np.diag(1/S) @ U.T @ b

    print(f"\n\nSolution x using SVD: {x_val_svd}")

def solve_svd(A, b):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = np.diag(1/S)
    return Vt.T @ S_inv @ U.T @ b

def solve_qr(A, b):
    Q, R = np.linalg.qr(A)
    Q_transpose = Q.T
    x_val = inv(R) @ Q_transpose @ b
    return x_val



