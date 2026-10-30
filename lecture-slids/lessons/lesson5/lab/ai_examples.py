#%%
# ---- Try with AI 1.3: the three bell curves on top of the data (load the wines again first) ----
data = load_wine()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
bins = np.linspace(11, 15, 25)                                    # the same bins for the three varieties
grid = np.linspace(11, 15, 400)                                   # alcohol values at which to draw the curves
colours = ["tab:red", "tab:green", "tab:blue"]
for k in [0, 1, 2]:
    alcohol_k = X_train[y_train == k, 0]
    plt.hist(alcohol_k, bins=bins, density=True, alpha=0.3, color=colours[k], label="variety " + str(k))
    plt.plot(grid, norm.pdf(grid, alcohol_k.mean(), alcohol_k.std(ddof=1)), color=colours[k])
plt.axvline(13.32, linestyle="--", color="black")                 # the wine of 1.2
plt.xlabel("alcohol (%)")
plt.ylabel("density")
plt.legend()
plt.show()
heights = []
for k in [0, 1, 2]:
    alcohol_k = X_train[y_train == k, 0]
    heights.append(round(norm.pdf(13.32, alcohol_k.mean(), alcohol_k.std(ddof=1)), 3))
print("heights at 13.32:", heights[0], heights[1], heights[2])
#%%
# ---- load the lumps again for the next boxes (the same data block as 3.0) ----
data = load_breast_cancer()
X = data.data
y = 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
#%%
# ---- Try with AI 3.1: the line the lumps lie on ----
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 2], s=12, facecolor="none", edgecolor="grey", label="benign")
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 2], s=12, color="tab:blue", label="malignant")
plt.xlabel("mean radius")
plt.ylabel("mean perimeter")
plt.legend()
plt.show()
#%%
# ---- Try with AI 3.2: the cross-validation curve of reg_param ----
values = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1]   # the values tried in 3.2
scores = []
for r in values:
    qda = QuadraticDiscriminantAnalysis(reg_param=r)
    scores.append(cross_val_score(qda, X_train_s, y_train, cv=kf).mean())
plt.plot(values, scores, marker="o")
plt.axhline(0.958, linestyle="--", color="grey")                 # no shrinking
plt.xscale("log")
plt.xlabel("reg_param")
plt.ylabel("cross-validated accuracy")
plt.show()
print("best:", values[scores.index(max(scores))], round(max(scores), 3))
