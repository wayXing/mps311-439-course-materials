#%%
# ---- Try with AI 2.2: the cumulative share against k ----
ks = np.arange(1, 201)                                    # k = 1 to 200
plt.plot(ks, cum[:200])                                   # the first 200 values
plt.axhline(0.9, linestyle=":", color="grey")
plt.xlabel("number of components k")
plt.ylabel("cumulative explained variance")
plt.show()
print("k = 10:", round(cum[9], 2), " k = 50:", round(cum[49], 2), " first k with 90%:", int(np.argmax(cum >= 0.9)) + 1)
#%%
# ---- Try with AI 3.1: the first six components as images ----
fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
limit = np.abs(pca.components_[:6]).max()                 # the same colour scale for all six
for k in range(6):
    axes[k].imshow(pca.components_[k].reshape(28, 28), cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[k].set_title(str(round(pca.explained_variance_ratio_[k], 3)))
    axes[k].set_xticks([])
    axes[k].set_yticks([])
plt.show()
print("shares of the first six:", pca.explained_variance_ratio_[:6].round(3))
#%%
# ---- Try with AI 4.2 (first box): the fitting images on two components ----
Z2 = pca.transform(X_digits_fit)                          # all 784 scores; we plot the first two
plt.figure(figsize=(6, 5))
plt.scatter(Z2[:, 0], Z2[:, 1], c=y_digits_fit, cmap="tab10", s=8)
for d in range(10):
    plt.text(Z2[y_digits_fit == d, 0].mean(), Z2[y_digits_fit == d, 1].mean(), str(d), fontsize=16, weight="bold")
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.show()
#%%
# ---- Try with AI 4.2 (second box): the lumps, as given in the box ----
from sklearn.datasets import load_breast_cancer           # the 569 lumps
from sklearn.preprocessing import StandardScaler          # rescales columns
data = load_breast_cancer()
y = 1 - data.target                                       # 1 = malignant
X_train, X_test, y_train, y_test = train_test_split(
    data.data, y, test_size=0.25, random_state=0, stratify=y)   # the split of Labs 4 and 5
scaler = StandardScaler()                                 # the measurements have very different scales
X_train_s = scaler.fit_transform(X_train)                 # learn and scale on the training lumps
pca_lumps = PCA(n_components=2)                           # two components
Z_lumps = pca_lumps.fit_transform(X_train_s)              # learn and project the training lumps
print(pca_lumps.explained_variance_ratio_.round(3))      # share kept by each component
#%%
# the AI's part
plt.scatter(Z_lumps[y_train == 0, 0], Z_lumps[y_train == 0, 1], facecolors="none", edgecolors="grey", label="benign")
plt.scatter(Z_lumps[y_train == 1, 0], Z_lumps[y_train == 1, 1], color="tab:blue", s=10, label="malignant")
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.legend()
plt.show()
