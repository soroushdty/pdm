import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Preprocessor:
    def __init__(self, target_variance=0.90):
        self.target_variance = target_variance
        self.scaler = StandardScaler()
        self.pca = None
        self.n_components_ = None

    def fit(self, X, y=None):
        # Scale first
        X_scaled = self.scaler.fit_transform(X)

        # >>> DEBUG: inspect data right before PCA <<<
        print("[DEBUG] Preprocessor.fit: X shape:", X.shape)
        print("[DEBUG] Preprocessor.fit: X_scaled shape:", X_scaled.shape)
        print("[DEBUG] Preprocessor.fit: overall variance of X:", np.var(X))
        print("[DEBUG] Preprocessor.fit: overall variance of X_scaled:", np.var(X_scaled))
        per_feature_var = np.var(X_scaled, axis=0)
        print("[DEBUG] Preprocessor.fit: first 10 feature variances (scaled):",
              per_feature_var[:10])

        n_max = min(X_scaled.shape)
        # PCA was failing here because it wasn't imported
        temp_pca = PCA(n_components=min(n_max, 500))
        temp_pca.fit(X_scaled)
        
        # np.cumsum would also fail without the numpy import
        cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= self.target_variance) + 1
        self.n_components_ = d
        self.pca = PCA(n_components=d)
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)
