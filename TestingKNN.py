import numpy as np
import random as r
import KNearestNeighbors as KNN

def initialize_points(num, c1, c2, d1, d2, arr):
    for i in range(num):
        upper_domain = d1 + c1 ** 0.5
        lower_domain = d1 - c1 ** 0.5
        total_domain = upper_domain - lower_domain
        x = total_domain * r.random() + lower_domain

        upper_bound = (c2 * (1 - (1 / c1) * ((x - d1) ** 2))) ** 0.5 + d2
        lower_bound = 2 * d2 - upper_bound
        total_bound = upper_bound - lower_bound
        y = total_bound * r.random() + lower_bound

        arr.append([x, y])

class1 = []
class2 = []
class3 = []
class4 = []

n1 = int((r.random() * 100) + 50)

initialize_points(n1, 8, 2, 2, 1, class1)
initialize_points(n1, 4, 4, 3, 5, class2)
initialize_points(n1, 4, 8, 8, 3, class3)
initialize_points(n1, 32, 32, -6, 8, class4)

class1 = np.array(class1)
class2 = np.array(class2)
class3 = np.array(class3)
class4 = np.array(class4)

model = KNN.KNN([class1, class2, class3, class4], 2, 4, 10)
test_inputs = np.array([[2, 1], [3, 5], [8, 3], [5, 3], [-1, 4]])
predictions = model.predict(test_inputs)
print(predictions)