# Inspired by: https://github.com/karolpiczak/paper-2015-esc-dataset

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from utils import show_scores_pred

class SVM():
    def __init__(self, C=1, kernel='rbf', gamma='scale'):
        self.svm = SVC(C=C, gamma=gamma, kernel=kernel, class_weight='balanced', cache_size=1000)

    def train(self, X, Y):
        
        print("Starting SVM training...")
        self.svm.fit(X, Y)

        print("Training completed")
        print("")
        print("====[ Training scores ]====")
        pred_train = self.svm.predict(X)
        
        show_scores_pred(Y, pred_train, "SVM-train-")
    
    def test(self, X, Y, csvfile='./plots/stats.csv'):
        
        pred_test = self.svm.predict(X)
        
        print("")
        print("====[ Test scores ]====")
        show_scores_pred(Y, pred_test, "SVM-test-", csvfile)
    
    def evaluate(self,X):
        return self.svm.predict(X)[:,-1]
        