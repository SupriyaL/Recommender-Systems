import numpy as np
from copy import copy, deepcopy
from itertools import islice
import time

""" 
	Input: data.txt file with each row as <user_id,movie_id,rating> 
	Output: matrix 'a' 
"""


def input_handle():
	a = np.zeros(shape=(943, 1682))
	fo = open("data.txt", "r")
	for line in fo.readlines():
		name1, name2, value = line.strip().split('\t')
		a[int(name1) - 1, int(name2) - 1] = int(value)
	return a


"""
	Input: Two matrices 'a' and 'b'
	Output: Cosine similarity between 'a' and 'b'
"""


def cos_sim(a, b):
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)


"""
	Input: matrix 'a', user id, and movie id
	Output: Predicted rating using user-user collborative filtering 
"""


def collaborative_filtering(a, row_id, column_id):
	mratings = deepcopy(a)
	mratings = np.ma.masked_array(mratings, mask=mratings == 0)
	movie_mean = mratings.mean(axis=1)
	for i in range(mratings.shape[0]):
		mratings[i] = mratings[i] - movie_mean[i]
	similarity = []
	for row in mratings:
		similarity.append(cos_sim(mratings[row_id - 1, :], row))
	indices_of_top = np.argsort(similarity)[-6:][::-1]
	sum = 0.0
	denominator = 0.0
	for i in indices_of_top:
		if (i != row_id - 1):
			sum = sum + similarity[i] * a[i][column_id - 1]
			denominator = denominator + similarity[i]
	return (np.around(sum / denominator, decimals=2))


"""
	Function to find the predicted rating
"""


def prediction_matrix(d, top_similarities, x, y):
	sum = 0.0
	denominator = 0.0
	for i in top_similarities:
		sum = sum + i[1] * d[i[0]][y]
		denominator = denominator + i[1]
	d[x][y] = (sum / denominator)
	return d


"""
	Input: matrix 'a'
	Output: rmse and spearman for matrix 'a'
"""


def rmse_and_spearman(a):
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
					prediction_matrix(d, dictionary[i], i, j)
				else:
					prediction_matrix(d, dictionary[i], i, j)
	calculate_rmse(d[472:943, 841:1682], a[472:943, 841:1682], n)
	spearman(a, d[472:943, 841:1682], a[472:943, 841:1682], n)


def calculate_rmse(predictions, targets, n):
	differences = np.subtract(predictions, targets)
	differences_squared = differences ** 2
	mean_of_differences_squared = differences_squared.sum() / n
	print("RMSE is: ", np.sqrt(mean_of_differences_squared))


"""
	Function to find the Spearman Rank Correlation
"""


def spearman(a, predictions, targets, n):
	diff = np.subtract(predictions, targets)
	diff_squared = diff ** 2
	x = np.sum(diff_squared)
	answer = 1 - (6 * (x) / (n * (n ** 2 - 1)))
	print("Spearmean is: ", answer)


"""
	Function to find precision on top K
"""


def precision_top_k(a, row_id, k):
	list_predicted = dict()
	for j in range(0, a.shape[1]):
		if (a[row_id - 1][j] != 0):
			list_predicted[j] = (collaborative_filtering(a, row_id, j + 1))
	list_predicted = sorted(list_predicted.items(), key=lambda x: x[1], reverse=True)

	list_actual = dict()
	for j in range(0, a.shape[1]):
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
	print("Precision for user", row_id, "for top", k, "movies is: ",
		  (number_of_recommended_items_that_are_relevant / number_of_relevant) * 100)


def main():
	pass


if __name__ == "__main__":
	a = input_handle()
	print("Collaborative Filtering Results")
	u_id = int(input("Enter the user ID: "))
	m_id = int(input("Enter the movie ID: "))
	start_time = time.time()
	print("Predicted value is:", collaborative_filtering(a, u_id, m_id))
	print("Prediction Time: ", (time.time() - start_time), "seconds")
	rmse_and_spearman(a)
	precision_top_k(a, u_id, 5)
	main()
