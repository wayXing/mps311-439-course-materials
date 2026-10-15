#%%
# ---- Advanced Study A ----
lams = np.logspace(-3, 3, 13)
Z = Z27_train
conds = [np.linalg.cond(Z.T @ Z + lam * np.eye(27)) for lam in lams]
R27 = H27_train                                    # the same 27 columns, not standardised
conds_raw = [np.linalg.cond(R27.T @ R27 + lam * np.eye(27)) for lam in lams]
plt.loglog(lams, conds, marker="o", label="standardised")
plt.loglog(lams, conds_raw, marker="s", label="not standardised")
plt.xlabel("lambda")
plt.ylabel("condition number of X'X + lambda I")
plt.legend()
plt.show()
print("lambda:", lams.round(3))
print("standardised:", np.round(conds))
print("not standardised:", np.array(conds_raw))
print(np.linalg.cond(Z.T @ Z))                        # lambda = 0
print(np.linalg.cond(Z.T @ Z + 1.0 * np.eye(27)))     # lambda = 1
#%%
# ---- Advanced Study B ----
def soft(z, t):
    if z > t:
        return z - t
    if z < -t:
        return z + t
    return 0.0

Z = Z27_train
n = len(Z)
yc = y_train.to_numpy() - y_train.mean()
w = np.zeros(Z.shape[1])
r = yc.copy()                                       # residual for w = 0
for sweep in range(1000):
    for j in range(Z.shape[1]):
        rho = Z[:, j] @ (r + Z[:, j] * w[j]) / n
        new = soft(rho, 0.05) / (Z[:, j] @ Z[:, j] / n)
        r = r - Z[:, j] * (new - w[j])
        w[j] = new

ref = Lasso(alpha=0.05, max_iter=100000).fit(Z27_train, y_train)
print(np.abs(w - ref.coef_).max() < 0.05, (w != 0).sum(), (ref.coef_ != 0).sum())
#%%
# ---- Advanced Study C ----
ridge_grid = np.logspace(-2, 4, 40)
lasso_grid = np.logspace(-4, 1, 40)
ridge_path = [Ridge(alpha=a).fit(Z27_train, y_train).coef_ for a in ridge_grid]
lasso_path = [Lasso(alpha=a, max_iter=100000).fit(Z27_train, y_train).coef_ for a in lasso_grid]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ridge_grid, ridge_path)
axes[0].set_xscale("log")
axes[0].set_title("Ridge")
axes[0].set_xlabel("alpha")
axes[0].set_ylabel("coefficient (standardised columns)")
axes[1].plot(lasso_grid, lasso_path)
axes[1].set_xscale("log")
axes[1].set_title("Lasso")
axes[1].set_xlabel("alpha")
plt.show()
#%%
alphas_lasso = np.logspace(-4, 1, 26)
cv_rmse_lasso = []
for alpha in alphas_lasso:
    model = Lasso(alpha=alpha, max_iter=100000)
    scores = cross_val_score(model, Z27_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    cv_rmse_lasso.append(-scores.mean())
position = np.argmin(cv_rmse_lasso)
print("Lasso best alpha:", alphas_lasso[position], " CV RMSE:", cv_rmse_lasso[position])
#%%
for d in range(1, 7):
    Hd = PolynomialFeatures(degree=d, include_bias=False).fit_transform(X_train[["horsepower"]])
    Hd = StandardScaler().fit_transform(Hd)
    s = cross_val_score(LinearRegression(), Hd, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    print(d, round(-s.mean(), 3), round(s.std() / np.sqrt(5), 3))
