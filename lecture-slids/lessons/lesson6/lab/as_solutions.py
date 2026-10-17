#%%
# ---- Advanced Study A ----
inside = X_fit[:, 0] <= 15.045                            # the 228 fitting lumps on the left of the root cut
y_in = y_fit[inside]                                      # their labels (35 malignant)

radius_in = X_fit[inside, 0]                              # mean radius of the 228 lumps
values = np.unique(radius_in)                             # distinct values, sorted
cuts = (values[:-1] + values[1:]) / 2                     # midpoints between neighbours
best_cut = None
best_score = 1.0
for c in cuts:
    left = y_in[radius_in <= c]
    right = y_in[radius_in > c]
    p_left = left.mean()
    p_right = right.mean()
    score = (len(left) * 2 * p_left * (1 - p_left) + len(right) * 2 * p_right * (1 - p_right)) / len(y_in)
    if score < best_score:
        best_score = score
        best_cut = c
print("left node, radius: best cut", round(best_cut, 3), "weighted Gini", round(best_score, 3))
#%%
# the same search, copied, with column 1 (mean texture)
texture_in = X_fit[inside, 1]
values = np.unique(texture_in)
cuts = (values[:-1] + values[1:]) / 2
best_cut = None
best_score = 1.0
for c in cuts:
    left = y_in[texture_in <= c]
    right = y_in[texture_in > c]
    p_left = left.mean()
    p_right = right.mean()
    score = (len(left) * 2 * p_left * (1 - p_left) + len(right) * 2 * p_right * (1 - p_right)) / len(y_in)
    if score < best_score:
        best_score = score
        best_cut = c
print("left node, texture: best cut", round(best_cut, 3), "weighted Gini", round(best_score, 3))
two = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_fit[:, :2], y_fit)   # the tree of 1.1, two levels
print("tree, left node:", two.tree_.feature[1], round(two.tree_.threshold[1], 3))   # node 1 is the left child of the root
#%%
# the same search with the error rate: no cut leaves fewer than 35 of the 228 wrong
values = np.unique(radius_in)
errors = []
for c in (values[:-1] + values[1:]) / 2:
    left = y_in[radius_in <= c]
    right = y_in[radius_in > c]
    errors.append(int(min(left.sum(), len(left) - left.sum()) + min(right.sum(), len(right) - right.sum())))
print("fewest errors of any cut:", min(errors), "of", len(y_in), "; without a cut:", int(y_in.sum()))
#%%
# ---- Advanced Study B ----
rng = np.random.default_rng(0)                            # fixed random numbers, so the result repeats
for k in range(10):
    idx = rng.integers(0, 426, 426)                       # 426 row numbers, drawn with replacement
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(X_train[idx], y_train[idx])
    print(k, data.feature_names[tree.tree_.feature[0]], round(tree.tree_.threshold[0], 3))
#%%
# ---- Advanced Study C ----
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=0)   # 100 trees that vote
forest.fit(X_digits_fit, y_digits_fit)
print("random forest, validation accuracy:", round(forest.score(X_digits_val, y_digits_val), 3))
