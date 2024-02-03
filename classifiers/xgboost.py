import xgboost
from sklearn.metrics import confusion_matrix, classification_report, f1_score,cohen_kappa_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from utils import show_scores

class XGBoost:
    def __init__(self, max_depth = 6):
        self.max_depth = max_depth

    def train(self, X, Y):

        posValues = np.unique(Y, return_counts=True)[1][1]
        negValues = np.unique(Y, return_counts=True)[1][0]
        
        self.xgb = xgboost.XGBClassifier(objective="binary:logistic", random_state=42, scale_pos_weight=negValues/posValues, max_depth=self.max_depth)
        print("Starting XGBoost training...")
        self.xgb.fit(X, Y)

        print("XGBoost Training completed")
        print("")
        print("====[ XGBoost Training scores ]====")
        pred_train = self.xgb.predict_proba(X)[:,1]
        show_scores(Y, pred_train, "XGB-train-")

    def test(self, X_test, Y_test, csvfile='./plots/stats.csv'):

        pred_test = self.xgb.predict_proba(X_test)[:,1]
        print("")
        print("====[ XGBoost Test scores ]====")
        show_scores(Y_test, pred_test,"XGB-test-", csvfile)

    def evaluate(self,X):
        return self.xgb.predict_proba(X)[:,-1]