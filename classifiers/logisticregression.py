from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score,cohen_kappa_score, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

from utils import show_scores

class LogisticRegression:
    def __init__(self):
        self.logisticregression = LogisticRegressionClassifier(class_weight="balanced")


    def train(self, X, Y):
        
        print("Starting LogisticRegression training...")
        self.logisticregression.fit(X, Y)

        print("LogisticRegression Training completed")
        print("")
        print("====[ LogisticRegression Training scores ]====")
        pred_train = self.logisticregression.predict_proba(X)[:,1]
        show_scores(Y, pred_train, "Log-Reg-train-")

    def test(self, X_test, Y_test, csvfile='./plots/stats.csv'):
        
        pred_test = self.logisticregression.predict_proba(X_test)[:,1]
        print("")
        print("====[ LogisticRegression Test scores ]====")
        show_scores(Y_test, pred_test, "Log-Reg-test-", csvfile)
    
    def evaluate(self,X):
        return self.logisticregression.predict_proba(X)[:,-1]