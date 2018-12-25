import numpy as np
import csv
from sympy import *
import datetime


filename = sys.argv[1] if len(sys.argv) > 1 else "ratings_task1.csv"
N = int(sys.argv[2]) if len(sys.argv) > 1 else 100
M = int(sys.argv[3]) if len(sys.argv) > 2 else 4382
f = int(sys.argv[4]) if len(sys.argv) > 3 else 40
k = int(sys.argv[5]) if len(sys.argv) > 4 else 10
# N = 2 #numebr of users of matrix
# M = 3 #number of movies of matrix
# f = 2 #number of dimensions
# k = 10 #number of iterations
starttime = datetime.datetime.now()

dict = {}
index = 0
matrix = np.zeros((N,M))
with open(filename, "rb") as csvfile:
    spamreader = csv.reader(csvfile, delimiter = " ", quotechar = "|")
    next(spamreader, None)
    list = []
    for row in spamreader:
        value = row[0].split(",")[:-1]

        if value[1] not in dict:
            dict[value[1]] = index
            index += 1
        matrix[int(value[0]) - 1][dict[value[1]]] = float(value[2])

U = np.ones((N,f))
V = np.ones((f,M))

UU = np.zeros((N,M))

for it in range(k):

    for r in range(N):
        for s in range(f):
            numerator = 0
            denominator = 0
            for j in range(M):
                if matrix[r][j] > 0:

                    tt = np.dot(U[r, :], V[:, j]) - U[r][s] * V[s][j]
                    numerator += V[s][j] * (matrix[r][j] - tt)
                    denominator += V[s][j] ** 2
            U[r][s] = float(numerator) / denominator
    for s in range(M):
        for r in range(f):
            numerator = 0
            denominator = 0
            for i in range(N):
                if matrix[i][s] > 0:

                    tt = np.dot(U[i, :], V[:, s]) - U[i][r] * V[r][s]
                    numerator += U[i][r] * (matrix[i][s] - tt)
                    denominator += U[i][r] * U[i][r]
            V[r][s] = float(numerator) / denominator

    diffmatrix = matrix - np.dot(U , V)
    count = 0
    sum = 0
    for row in range(N):
        for col in range(M):
            if matrix[row][col] != 0:
                count += 1
                sum += diffmatrix[row,col] ** 2
    print ("%.4f"% np.sqrt(sum / count))
