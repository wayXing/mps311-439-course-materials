#%%
# ---- Try with AI 1.2: a bell curve for each class on top of the histograms ----
bins = np.linspace(6, 29, 35)                                     # the same bins for both classes
plt.hist(radius[y_train == 0], bins=bins, density=True, alpha=0.4, color="grey", label="benign")
plt.hist(radius[y_train == 1], bins=bins, density=True, alpha=0.4, color="tab:blue", label="malignant")
grid = np.linspace(5, 30, 400)                                    # radii at which to draw the curves
plt.plot(grid, norm.pdf(grid, mean_ben, sd_ben), color="grey")    # bell curve of the benign class
plt.plot(grid, norm.pdf(grid, mean_mal, sd_mal), color="tab:blue")  # bell curve of the malignant class
plt.axvline(12.8, linestyle="--", color="black")                  # the opening lump
plt.xlabel("mean radius")
plt.ylabel("density")
plt.legend()
plt.show()
print("heights at 12.8:", round(norm.pdf(12.8, mean_ben, sd_ben), 3), round(norm.pdf(12.8, mean_mal, sd_mal), 3))
#%%
# ---- Try with AI 3.3 (first box): the cross-validation curve of reg_param ----
values = [0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]                    # the values tried in 3.2
scores = []
for r in values:
    qda = QuadraticDiscriminantAnalysis(reg_param=r)
    scores.append(cross_val_score(qda, X_digits_train, y_digits_train, cv=kf).mean())
positions = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]           # 0 drawn at 0.00001 on the log axis
plt.plot(positions, scores, marker="o")
plt.xscale("log")
plt.xticks(positions, ["0", "0.0001", "0.001", "0.01", "0.1", "0.5", "1"])
plt.xlabel("reg_param")
plt.ylabel("cross-validated accuracy")
plt.show()
print("best accuracy:", round(max(scores), 3))
#%%
# ---- Try with AI 3.3 (second box): the centre of each class of digits ----
qda = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X_digits_train, y_digits_train)   # the model of 3.3
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for k in range(10):
    ax = axes[k // 5, k % 5]                                      # row and column of this small plot
    ax.imshow(qda.means_[k].reshape(28, 28), cmap="Greys")          # the average image of digit k
    ax.set_title("digit " + str(k))
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
#%%
# ---- Try with AI 3.3 (third box): the two-feature data of the lecture, as given in the box ----
rng = np.random.default_rng(0)                            # the same random numbers as the lecture
X1 = rng.multivariate_normal([0, 0], [[0.35, 0.15], [0.15, 0.35]], 120)      # class 1: tight
X0 = rng.multivariate_normal([1.0, 0.6], [[2.2, -0.6], [-0.6, 1.6]], 240)    # class 0: spread out
X_two = np.vstack([X1, X0])                               # 360 points, 2 features
y_two = np.concatenate([np.ones(120), np.zeros(240)])     # their labels
lda_two = LinearDiscriminantAnalysis().fit(X_two, y_two)  # one shared spread
qda_two = QuadraticDiscriminantAnalysis().fit(X_two, y_two)   # one spread per class
print(lda_two.score(X_two, y_two), qda_two.score(X_two, y_two))   # accuracy on these 360 points
#%%
# the AI's part: the two boundaries
gx, gy = np.meshgrid(np.linspace(-3.5, 5.5, 300), np.linspace(-3.5, 4.5, 300))   # a grid over the plane
grid_points = np.column_stack([gx.ravel(), gy.ravel()])                         # one row per grid point
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, model, title in [(axes[0], lda_two, "LDA"), (axes[1], qda_two, "QDA")]:
    p = model.predict_proba(grid_points)[:, 1].reshape(gx.shape)               # probability of class 1
    ax.scatter(X_two[:, 0], X_two[:, 1], c=y_two, cmap="coolwarm", s=10)
    ax.contour(gx, gy, p, levels=[0.5], colors="green")                         # the boundary p = 0.5
    ax.set_title(title)
plt.show()
