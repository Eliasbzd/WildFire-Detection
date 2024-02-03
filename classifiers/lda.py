from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils import show_scores

class LDA:
    def __init__(self):
        self.lda = LinearDiscriminantAnalysis()

    def train(self, X, Y):
        print("Starting LDA training...")
        self.lda.fit(X, Y)

        print("LDA Training completed")
        print("")
        print("====[ LDA Training scores ]====")
        pred_train = self.lda.predict_proba(X)[:,1]
        show_scores(Y, pred_train, "LDA-train-")
    
    def test(self, X_test, Y_test, csvfile='./plots/stats.csv'):

        Y_probs = self.lda.predict_proba(X_test)[:,1]
        print("")
        print("====[LDA Test scores ]====")
        show_scores(Y_test, Y_probs, "LDA-train-",csvfile)
    
    def evaluate(self,X):
        return self.lda.predict_proba(X)[:,-1]




