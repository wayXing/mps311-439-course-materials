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
