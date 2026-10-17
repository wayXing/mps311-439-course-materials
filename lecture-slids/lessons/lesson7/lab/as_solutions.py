#%%
# ---- Advanced Study A ----
zeros = X_digits_fit[y_digits_fit == 0]                   # the 557 fitting images of digit 0
S = np.cov(zeros.T)                                       # 784 x 784 covariance matrix of the 0s
values, vectors = np.linalg.eigh(S)                       # eigenvalues (ascending) and eigenvectors (columns)
order = np.argsort(values)[::-1]                          # largest first
values = values[order]
vectors = vectors[:, order]
print("share of the first eigenvalue:", round(values[0] / values.sum(), 4), "second:", round(values[1] / values.sum(), 4))
pca_zero = PCA(random_state=0).fit(zeros)                 # sklearn's PCA on the same images
first = vectors[:, 0]
diff_same = np.abs(first - pca_zero.components_[0]).max() # if the signs agree
diff_flip = np.abs(first + pca_zero.components_[0]).max() # if one is the negative of the other
print("largest difference below 1e-6:", min(diff_same, diff_flip) < 1e-6)
print("eigenvalues below 1e-10:", (values < 1e-10).sum())
plt.imshow(first.reshape(28, 28), cmap="RdBu_r")          # the first direction of the 0s
plt.axis("off")
plt.show()
#%%
# ---- Advanced Study B ----
mean = X_digits_fit.mean(axis=0)                          # the mean of the fitting images
Xc = X_digits_fit - mean                                  # centred fitting images
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)         # rows of Vt are the components
var = s ** 2 / (len(Xc) - 1)                              # variance along each component
print("first variance:", round(var[0], 4), "sklearn:", round(pca.explained_variance_[0], 4))
k = 10
Z = (X_digits_val - mean) @ Vt[:k].T                      # scores of the validation images, fitting mean
back = mean + Z @ Vt[:k]                                  # rebuilt from 10 numbers
print("reconstruction error, k = 10:", round(((X_digits_val - back) ** 2).sum(axis=1).mean(), 2))
#%%
# ---- Advanced Study C ----
raw = PCA(n_components=2).fit(X_train)                    # the lumps, not standardised
top = np.argmax(np.abs(raw.components_[0]))               # the largest weight in the first component
print("largest weight:", data.feature_names[top], " share kept by component 1:", round(raw.explained_variance_ratio_[0], 3))
