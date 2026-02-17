import time
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from knn import KNNClass
from sklearn.metrics import confusion_matrix, roc_auc_score

def train_models(D_train, y_train):
    results={}

    knn=KNNClass(k=5)
    start=time.time()
    knn.fit(D_train, y_train)
    finish=time.time()
    results["KNN"] = (knn, finish-start)

    bayes=MultinomialNB()
    start=time.time()
    bayes.fit(D_train, y_train)
    finish=time.time()
    results["Bayes"]=(bayes, finish-start)

    svm=LinearSVC()
    start=time.time()
    svm.fit(D_train, y_train)
    finish=time.time()
    results["SVM"]=(svm, finish-start)

    return results

def evaluate(model, D, y):
    start=time.time()
    y_pred = model.predict(D)
    finish=time.time()

    average = (finish-start) / D.shape[0]
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y, y_pred).ravel()

    accuracy = (true_pos+true_neg) / (true_pos+true_neg+false_pos+false_neg)
    precision = true_pos / (true_pos+false_pos)
    recall = true_pos / (true_pos+false_neg)
    specificity = true_neg / (true_neg+false_pos)

    return {"accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "specificity":specificity,
            "average_prediction_time":average
            }

