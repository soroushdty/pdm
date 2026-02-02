class Preprocessor:
    def __init__(self, target_variance=0.90):
        self.target_variance = target_variance
        self.scaler = StandardScaler()
        self.pca = None
        self.n_components_ = None
    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        n_max = min(X_scaled.shape)
        temp_pca = PCA(n_components=min(n_max, 500))
        temp_pca.fit(X_scaled)
        cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= self.target_variance) + 1
        self.n_components_ = d
        self.pca = PCA(n_components=d)
        self.pca.fit(X_scaled)
        return self
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
