from numpy.linalg import qr
from numpy.random import rand
from random import randint
from numpy.linalg import inv

print("Random Numbers 0-1")
matrix = rand(4, 2)
Q, R = qr(matrix)
print("Original Matrix:\n", matrix)
print("\nQ Matrix:\n", Q)
print("\nR Matrix:\n", R)


N=3
M = 2
matrix =[]
for i in range(N):
    row = []
    for j in range(M):
        row.append(randint(1, 10))
    matrix.append(row)

print("\n\n\nWhole Numbers")
Q, R = qr(matrix)
print("Original Matrix:\n", matrix)
print("\nQ Matrix:\n", Q)
print("\nR Matrix:\n", R)
Q_transpose = Q.transpose()

print("\nQ Transpose Matrix:\n", Q_transpose)

b = [1,4,2]
x_val = inv(R)*Q_transpose*b

print(f"\n\nSolution x: {x_val}")

