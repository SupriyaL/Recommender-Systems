import numpy as np
from copy import copy, deepcopy
from itertools import islice
import time
from CF import input_handle, cos_sim, calculate_rmse, precision_top_k, spearman

"""
    Input: matrix 'a', user id, and movie id
    Output: Predicted rating using user-user collborative filtering with baseline approach 
"""


def baseline(a, u_id, m_id):
    masked_a = deepcopy(a)
    masked_a = np.ma.masked_array(masked_a, mask=masked_a == 0)
    mean_of_matrix = masked_a.mean()
    users_mean = masked_a.mean(axis=1)
    movies_mean = masked_a.mean(axis=0)
    for i in range(masked_a.shape[0]):  # iterate over rows
        masked_a[i] = masked_a[i] - users_mean[i]
    similarity = []
    for row in masked_a:
        similarity.append(cos_sim(masked_a[u_id - 1, :], row))
    indices_of_top = np.argsort(similarity)[-6:][::-1]
    sum = 0.0
    denominator = 0.0
    answer = 0.0
    base = mean_of_matrix + (users_mean[u_id - 1] - mean_of_matrix) + (movies_mean[m_id - 1] - mean_of_matrix)
    for i in indices_of_top:
        if (i != u_id - 1):
            x = mean_of_matrix + (users_mean[i] - mean_of_matrix) + (movies_mean[m_id - 1] - mean_of_matrix)
            sum = sum + similarity[i] * (a[i][m_id - 1] - x)
            denominator = denominator + similarity[i]
            answer = base + (sum / denominator)
    return (np.around(answer, decimals=2))


"""
    Input: matrix 'a'
    Output: rmse and spearman for matrix 'a' using baseline approach
"""


def rmse_and_spearman_for_baseline(a):
    d = deepcopy(a)
    n = 0
    for i in range(472, 943):
        for j in range(841, 1682):
            if (a[i][j] != 0):
                n += 1
                d[i][j] = 0.0
    mratings_rmse = np.ma.masked_equal(d, 0)
    movie_mean_rmse = mratings_rmse.mean(axis=1)
    for i in range(mratings_rmse.shape[0]):
        mratings_rmse[i] = mratings_rmse[i] - movie_mean_rmse[i]
    dictionary = dict()
    for i in range(0, len(a)):
        for j in range(0, len(a[0])):
            if (a[i][j] != 0 and d[i][j] == 0):
                if (i not in dictionary):
                    l = dict()
                    s = 0
                    for row in mratings_rmse:
                        x = (cos_sim(mratings_rmse[i, :], row))
                        l[s] = x
                        s += 1
                    result = sorted(l.items(), key=lambda x: x[1], reverse=True)
                    dictionary[i] = (result[1:6])
                    prediction_matrix_baseline(d, dictionary[i], i, j)
                else:
                    prediction_matrix_baseline(d, dictionary[i], i, j)
    calculate_rmse(d[472:943, 841:1682], a[472:943, 841:1682], n)
    spearman(a, d[472:943, 841:1682], a[472:943, 841:1682], n)


"""
    Function to find the predicted rating
"""


def prediction_matrix_baseline(d, top_similarities, x, y):
    sum = 0.0
    denominator = 0.0
    mean_of_matrix = d.mean()
    users_mean = d.mean(axis=1)
    movies_mean = d.mean(axis=0)
    base = mean_of_matrix + (users_mean[x - 1] - mean_of_matrix) + (movies_mean[y - 1] - mean_of_matrix)
    for i in top_similarities:
        u = mean_of_matrix + (users_mean[i[0]] - mean_of_matrix) + (movies_mean[y - 1] - mean_of_matrix)
        sum = sum + i[1] * (d[i[0]][y] - u)
        denominator = denominator + i[1]
        answer = base + (sum / denominator)
    d[x][y] = answer


# MAIN
a = input_handle()
print("2. Baseline Approach Results")
u_id = int(input("Enter the user ID: "))
m_id = int(input("Enter the movie ID: "))
start_time = time.time()
print("Predicted value is: ", baseline(a, u_id, m_id))
print("Prediction Time: ", (time.time() - start_time), "seconds")
rmse_and_spearman_for_baseline(a)
precision_top_k(a, u_id, 5)
