#%%
# ---- Try with AI 2.2: the centre of each group as an image ----
centres = pca.inverse_transform(km.cluster_centers_)     # back from 50 numbers to 784 pixels
fig, axes = plt.subplots(1, 10, figsize=(12, 1.8))
for k in range(10):
    axes[k].imshow(centres[k].reshape(28, 28), cmap="Greys")
    axes[k].set_title("guess " + str(guess[k]))
    axes[k].set_xticks([])
    axes[k].set_yticks([])
plt.show()
#%%
# ---- Try with AI 4.2 (first box): a dendrogram of 40 random images ----
from scipy.cluster.hierarchy import linkage, dendrogram
rows = np.random.default_rng(0).choice(len(Z_fit), 40, replace=False)   # 40 random fitting images
merges = linkage(Z_fit[rows], method="ward")
plt.figure(figsize=(12, 4))
dendrogram(merges, labels=y_digits_fit[rows])
plt.show()
#%%
# ---- Try with AI 4.2 (second box): hierarchical clustering on 500 random images ----
rows = np.random.default_rng(0).choice(5625, 500, replace=False)
part = AgglomerativeClustering(n_clusters=10, linkage="ward").fit_predict(Z_fit[rows])
t = np.zeros((10, 10), dtype=int)
for i in range(500):
    t[part[i], y_digits_fit[rows][i]] += 1
print("500 images, majority vote:", round(t.max(axis=1).sum() / 500, 3))
