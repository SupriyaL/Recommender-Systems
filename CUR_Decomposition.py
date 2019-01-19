import numpy as np
import math
from numpy import linalg
import time
from Singular_Value_Decomposition import svd, frobenius, precision_top_k_svd, ninety_percent
from CF import input_handle, spearman

np.random.seed(550)

"""
	Input: matrix 'a'
	Output: Total length of all columns
"""


def total_length(a):
    total_length = 0
    for i in range(0, a.shape[0]):
        for j in range(0, a.shape[1]):
            total_length = total_length + a[i][j] ** 2
    return total_length


def cur(a):
    x = total_length(a)
    # C
    p_c = []
    for i in range(0, a.shape[1]):
        column_sum = 0.0
        for j in range(0, a.shape[0]):
            column_sum = column_sum + a[j][i] ** 2
        p_c.append(column_sum / x)

    C = np.zeros(shape=(a.shape[0], 942))
    columns_chosen = np.random.choice(a.shape[1], 942, replace=False, p=p_c)
    y = 0
    for i in columns_chosen:
        d = (942 * p_c[i]) ** 0.5
        column = a[:, i]
        C[:, y] = column / d
        y += 1

    p_r = []
    for i in range(0, a.shape[0]):
        row_sum = 0.0
        for j in range(0, a.shape[1]):
            row_sum = row_sum + a[i][j] ** 2
        p_r.append(row_sum / x)

    R = np.zeros(shape=(942, a.shape[1]))
    rows_chosen = np.random.choice(a.shape[0], 942, replace=False, p=p_r)
    y = 0
    for i in rows_chosen:
        d = (942 * p_r[i]) ** 0.5
        row = a[i, :]
        R[y, :] = row / d
        y += 1

    # U
    overlap = np.zeros(shape=(len(rows_chosen), len(columns_chosen)))
    for i in range(0, len(rows_chosen)):
        for j in range(0, len(columns_chosen)):
            overlap[i][j] = a[rows_chosen[i]][columns_chosen[j]]

    U, V, sigma, m, n = svd(overlap)

    Y = V.T
    X = U.T
    for i in range(0, sigma.shape[0]):
        if (sigma[i][i] != 0):
            sigma[i][i] = 1 / sigma[i][i]

    Z = sigma
    overlap = np.dot(Y, np.dot(Z, X))
    return C, R, overlap


# MAIN
print("CUR Results")
a = input_handle()
start_time = time.time()
C, R, U = cur(a)

"""
reconstructed_cur = np.dot(C,np.dot(U,R))
print("Prediction Time: ",(time.time() - start_time),"seconds")
frobenius(reconstructed_cur,a)
spearman(a,reconstructed_cur,a,np.shape(a)[0] * np.shape(a)[1])
precision_top_k_svd(a,reconstructed_cur,5)
"""

print("After no similar columns or rows: ")
C_ninety, R_ninety, U_ninety = ninety_percent(a, C, R, U)
ninety_reconstructed_cur = np.dot(C_ninety, (np.dot(U_ninety, R_ninety)))
print("Prediction Time: ", (time.time() - start_time), "seconds")
frobenius(ninety_reconstructed_cur, a)
spearman(a, ninety_reconstructed_cur, a, np.shape(a)[0] * np.shape(a)[1])
precision_top_k_svd(a, ninety_reconstructed_cur, 5)
