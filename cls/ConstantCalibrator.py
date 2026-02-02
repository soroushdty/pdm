class ConstantCalibrator:
    def __init__(self, prob): self.prob = prob
    def predict(self, X): return np.full(X.shape[0], self.prob)
    def predict_proba(self, X): return np.column_stack([1-self.prob, self.prob])
