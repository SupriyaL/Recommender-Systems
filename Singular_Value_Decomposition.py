import numpy as np
import math
from numpy import linalg
from CF import input_handle, spearman
import time

"""
	Input: matrix 'a'
	Output: U,V,sigma
"""


def svd(a):
	# U

	U = np.dot(a, a.T)
	eigen_values_1, eigen_vectors_1 = linalg.eig(U)
	eigen_pairs_1 = {}
	for i in range(0, len(eigen_values_1)):
		eigen_pairs_1[eigen_values_1[i]] = eigen_vectors_1[:, i]
	eigen_values_1 = sorted(eigen_values_1)[::-1]
	result_1 = {}
	for j in eigen_values_1:
		result_1[round(j.real, 2)] = eigen_pairs_1[j].real
	# for k, v in result_1.items():
	#    print (k, '-->', v)
	eigen_values_u = []
	count_u = 0
	for j in result_1:
		if j > 0:
			eigen_values_u.append(float("{0:.2f}".format(j)))
			count_u += 1
	final_U = np.zeros(shape=(len(result_1[eigen_values_u[1]]), len(eigen_values_u)))
	for i in (eigen_values_u):
		y = 0
		for j in result_1[i]:
			final_U[y][eigen_values_u.index(i)] = j
			y += 1

	# V
	V = np.dot(a.T, a)
	eigen_values_2, eigen_vectors_2 = linalg.eig(V)
	eigen_pairs_2 = {}
	for i in range(0, len(eigen_values_2)):
		eigen_pairs_2[eigen_values_2[i]] = eigen_vectors_2[:, i]
	eigen_values_2 = sorted(eigen_values_2)[::-1]
	result_2 = {}
	for j in eigen_values_2:
		result_2[round(j.real, 2)] = eigen_pairs_2[j].real
	# for k, v in result_2.items():
	#   print (k, '-->', v)
	eigen_values_v = []
	count_v = 0
	for j in result_2:
		if j > 0:
			eigen_values_v.append(float("{0:.2f}".format(j)))
			count_v += 1
	final_V = np.zeros(shape=(len(result_2[eigen_values_v[1]]), len(eigen_values_v)))
	for i in (eigen_values_v):
		y = 0
		for j in result_2[i]:
			final_V[y][eigen_values_v.index(i)] = j
			y += 1
	final_V = final_V.T

	# sigma
	sigma_elements = list(set(eigen_values_u) | set(eigen_values_v))
	sigma_elements = sorted(sigma_elements)[::-1]
	sigma = np.zeros(shape=(len(sigma_elements), len(sigma_elements)))
	j = 0
	for i in sigma_elements:
		sigma[j][j] = i ** 0.5
		j += 1

	for i in (eigen_values_v):
		flag = True
		temp = np.dot(a, result_2[i])
		from_U = result_1[i]
		for j in range(0, len(temp)):
			if ((temp[j] > 0 and from_U[j] < 0) or (temp[j] < 0 and from_U[j] > 0)):  # (temp[j]/from_U[j])<0.0
				flag = False
				break
		if (flag == False):
			y = eigen_values_v.index(i)
			for columns in range(len(eigen_values_v)):
				final_U[columns][y] = final_U[columns][y] * -1

	return final_U, final_V, sigma, count_u, count_v


"""
	Input: matrix 'a' and its reconstructed matix 
	Output: Frobenius norm
	
"""


def frobenius(predictions, targets):
	differences = np.subtract(predictions, targets)
	differences_squared = differences ** 2
	sum_of_differences_squared = differences_squared.sum()
	print("Frobenius is: ", np.sqrt(sum_of_differences_squared))


"""
	
"""


def ninety_percent(a, final_U, final_V, sigma):
	sum = 0
	total = 0
	for j in range(0, len(sigma[0])):
		sum = sum + sigma[j][j]
	for j in range(0, len(sigma[0])):
		if ((total / sum) * 100) >= 90:
			break
		else:
			total = total + sigma[j][j]

	new_final_sigma = sigma[:j, :j]
	new_final_U = final_U[:, :j]
	new_final_V = final_V[:j, :]
	return new_final_U, new_final_V, new_final_sigma


def precision_top_k_svd(a, predicted_a, k):
	row_id = int(input("Enter user id: "))
	list_predicted = dict()
	for j in range(0, len(a[0])):
		if (a[row_id - 1][j] != 0):
			list_predicted[j] = predicted_a[row_id - 1][j]
	list_predicted = sorted(list_predicted.items(), key=lambda x: x[1], reverse=True)
	list_actual = dict()
	for j in range(0, len(a[0])):
		if (a[row_id - 1][j] != 0):
			list_actual[j] = (a[row_id - 1][j])
	list_actual = sorted(list_actual.items(), key=lambda x: x[1], reverse=True)
	list_actual = list_actual[0:k]
	number_of_relevant = len(list_actual)
	relevant = []
	recommended = []
	for i in list_actual:
		relevant.append(i[0])
	for i in list_predicted[0:number_of_relevant]:
		recommended.append(i[0])
	intersection = list(set(relevant) & set(recommended))
	number_of_recommended_items_that_are_relevant = len(intersection)
	print("Precision is: ", (number_of_recommended_items_that_are_relevant / number_of_relevant) * 100)


# MAIN
def main():
	pass


if __name__ == "__main__":
	print("SVD Results")
	a = input_handle()
	start_time = time.time()
	U, V, sigma, x, y = svd(a)
	reconstructed_matrix = np.dot(U, (np.dot(sigma, V)))
	print("Prediction Time: ", (time.time() - start_time), "seconds")
	frobenius(reconstructed_matrix, a)
	spearman(a, reconstructed_matrix, a, np.shape(a)[0] * np.shape(a)[1])
	precision_top_k_svd(a, reconstructed_matrix, 5)
	print("After 90% energy retention: ")
	# start_time = time.time()
	U_ninety, V_ninety, sigma_ninety = ninety_percent(a, U, V, sigma)
	ninety_reconstructed_matrix = np.dot(U_ninety, (np.dot(sigma_ninety, V_ninety)))
	print("Prediction Time: ", (time.time() - start_time), "seconds")
	frobenius(ninety_reconstructed_matrix, a)
	spearman(a, ninety_reconstructed_matrix, a, np.shape(a)[0] * np.shape(a)[1])
	precision_top_k_svd(a, ninety_reconstructed_matrix, 5)
	main()
