import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
X = np.load("data/processed/X_features.npy")
y = np.load("data/processed/y.npy")

print("Class distribution:", dict(zip(*np.unique(y, return_counts=True))))

# -----------------------------------------------------------------------------
# 1)  Standardise features  ➜  much better PCA scatter
# -----------------------------------------------------------------------------
X_std = StandardScaler().fit_transform(X)

# --------------------  PCA  --------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=y, palette="Set1", s=45, alpha=0.8)
plt.title("PCA (after standardisation)")
plt.xlabel(f"PC1  ({pca.explained_variance_ratio_[0]:.0%} var)")
plt.ylabel(f"PC2  ({pca.explained_variance_ratio_[1]:.0%} var)")
plt.legend(title="Eye State")
Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/pca_projection.png")
plt.show()

# --------------------  t-SNE  --------------------
tsne = TSNE(n_components=2, perplexity=15, max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_std)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1],
                hue=y, palette="Set1", s=45, alpha=0.8)
plt.title("t-SNE (standardised features)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Eye State")
plt.savefig("reports/figures/tsne_projection.png")
plt.show()
