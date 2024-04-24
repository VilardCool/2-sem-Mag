
X = [[0, 1, 2, 1],
     [1, 0, 1, 1],
     [0, 1, 1, 0],
     [0, 0, 1, 1],
     [0, 0, 2, 1],
     [1, 1, 2, 0],
     [1, 0, 2, 1],
     [1, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 1, 1]]

Y = [1, 1, 1]

"""
X = [[0, 0, 1, 0, 0],
     [1, 0, 0, 1, 1],
     [2, 0, 1, 0, 1],
     [0, 1, 0, 0, 1],
     [0, 1, 1, 0, 1],
     [0, 1, 1, 1, 0],
     [1, 0, 0, 1, 0],
     [2, 0, 0, 0, 1],
     [2, 1, 1, 0, 1],
     [0, 1, 1, 1, 0]]

Y = [2, 1, 1, 1]
"""

from math import sqrt

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	for v in distances:
		print(v)
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

prediction = predict_classification(X, Y, 3)

print("Class of Y: ", prediction)