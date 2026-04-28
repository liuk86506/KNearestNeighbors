import numpy as np
import copy as c

class KNN:
    def __init__(self, all_data, n, c, k):
        self.n = n
        self.c = c
        self.k = k

        self.c_data = all_data

    def predict(self, inputs):
        predictions = []
        for i in range(len(inputs)):
            c_data = c.deepcopy(self.c_data)
            c_data = np.abs(c_data - inputs[i])
            c_data = np.sum(c_data, axis = 2)

            indexes = []
            for j in range(self.c):
                indexes.append(0)

            found = 0
            while found < self.k:
                found_min_index = []
                found_min = []

                for j in range(len(indexes)):
                    found_min_index.append(np.argmin(c_data[j]))
                    found_min.append(np.min(c_data[j]))

                min_index = np.argmin(found_min)
                indexes[min_index]+=1
                c_data[min_index][found_min_index[min_index]] = np.max(c_data[min_index])

                found+=1

            predictions.append(np.argmax(indexes) + 1)

        return predictions