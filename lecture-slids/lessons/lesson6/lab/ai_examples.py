#%%
# ---- Try with AI 1.2: a small tree as a diagram ----
from sklearn.tree import plot_tree
small = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_fit_two, y_fit)   # two questions deep
plt.figure(figsize=(10, 5))
plot_tree(small, feature_names=["radius", "texture"], class_names=["benign", "malignant"], filled=True)
plt.show()
print("root:", small.tree_.feature[0], round(small.tree_.threshold[0], 3))
#%%
# ---- Try with AI 2.2: training and validation accuracy against depth ----
depths = list(range(1, 21))
train_list = []
val_list = []
for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    tree.fit(X_digits_fit, y_digits_fit)
    train_list.append(tree.score(X_digits_fit, y_digits_fit))     # training accuracy
    val_list.append(tree.score(X_digits_val, y_digits_val))       # validation accuracy
plt.plot(depths, train_list, marker="o", label="training")
plt.plot(depths, val_list, marker="s", label="validation")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.legend()
plt.show()
print("best validation:", round(max(val_list), 3), "at depth", depths[val_list.index(max(val_list))])
#%%
# ---- Try with AI 3.1 (first box): feature importance on all 30 features ----
tree30 = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
order = np.argsort(tree30.feature_importances_)[::-1][:5]          # the five largest
plt.barh(data.feature_names[order][::-1], tree30.feature_importances_[order][::-1])
plt.xlabel("feature importance")
plt.show()
print("largest:", data.feature_names[order[0]], round(tree30.feature_importances_[order[0]], 2))
#%%
# ---- Try with AI 3.1 (second box): rectangles of a shallow and a deep tree ----
gx, gy = np.meshgrid(np.linspace(6, 29, 200), np.linspace(9, 40, 200))   # a grid over the two features
grid_points = np.column_stack([gx.ravel(), gy.ravel()])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
shallow = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_fit_two, y_fit)
regions = shallow.predict(grid_points).reshape(gx.shape)             # 1 where the tree says malignant
axes[0].contourf(gx, gy, regions, alpha=0.3, cmap="coolwarm")
axes[0].scatter(X_fit_two[:, 0], X_fit_two[:, 1], c=y_fit, cmap="coolwarm", s=8)
axes[0].set_title("max_depth 2")
deep = DecisionTreeClassifier(max_depth=10, random_state=0).fit(X_fit_two, y_fit)
regions = deep.predict(grid_points).reshape(gx.shape)
axes[1].contourf(gx, gy, regions, alpha=0.3, cmap="coolwarm")
axes[1].scatter(X_fit_two[:, 0], X_fit_two[:, 1], c=y_fit, cmap="coolwarm", s=8)
axes[1].set_title("max_depth 10")
plt.show()
