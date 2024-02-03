from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score,cohen_kappa_score, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from utils import show_scores

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=6):
        self.randomforest = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42, class_weight="balanced")

    def train(self, X, Y):
        
        print("Starting RandomForest training...")
        self.randomforest.fit(X, Y)

        print("RandomForest Training completed")
        print("")
        print("====[ RandomForest Training scores ]====")
        pred_train = self.randomforest.predict_proba(X)[:,1]
        show_scores(Y, pred_train, "Rand-Forest-train-")
    
    def test(self, X_test, Y_test, csvfile='./plots/stats.csv'):
        
        pred_test = self.randomforest.predict_proba(X_test)[:,1]
        print("")
        print("====[ RandomForest Test scores ]====")
        show_scores(Y_test, pred_test, "Rand-Forest-test-", csvfile)
    
    def evaluate(self,X):
        return self.randomforest.predict_proba(X)[:,-1]