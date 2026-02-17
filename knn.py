import numpy as np
from scipy.sparse import csr_matrix

class KNNClass:
    def __init__(self, k=5):
        self.k = k

    def fit(self, D, y):
        self.D_train = D
        self.y_train = y

    def distance(self, x1, x2):
        difference = x1-x2
        dist = np.sqrt(difference.multiply(difference).sum())
        return dist
    
    def predict_one(self, x):
        distances = []

        for i in range(self.D_train.shape[0]):
            d = self.distance(self.D_train[i], x)
            distances.append((d, self.y_train[i]))

        distances.sort(key=lambda z: z[0])
        nearest = distances[:self.k]
        numbers = [number for (_,number) in nearest]
        return 1 if sum(numbers) >= (self.k / 2) else 0
    
    def predict(self, D):
        predictions = []
        for i in range(D.shape[0]):
            predictions.append(self.predict_one(D[i]))
        return np.array(predictions)
    
def predict_proba_knn(knn, D):
        probs = []
        for i in range(D.shape[0]):
            distances = []
            for j in range(knn.D_train.shape[0]):
                difference = D[i] - knn.D_train[j]
                d = np.sqrt(difference.multiply(difference).sum())
                distances.append((d, knn.y_train[j]))
            distances.sort(key=lambda z: z[0])
            nearest = distances[:knn.k]
            numbers = [number for (_,number) in nearest]
            probs.append(sum(numbers)/knn.k)
        return np.array(probs)