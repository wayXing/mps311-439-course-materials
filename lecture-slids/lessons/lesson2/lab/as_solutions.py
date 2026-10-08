#%%
# ---- Advanced Study A ----
Xtr2 = X_train[features].to_numpy(); Xte2 = X_test[features].to_numpy()
Xb = np.c_[np.ones(len(Xtr2)), Xtr2]; Xbt = np.c_[np.ones(len(Xte2)), Xte2]
ytr = y_train.to_numpy()
w_solve = np.linalg.solve(Xb.T @ Xb, Xb.T @ ytr)
w_inv = np.linalg.inv(Xb.T @ Xb) @ (Xb.T @ ytr)
w_lstsq = np.linalg.lstsq(Xb, ytr, rcond=None)[0]
print(w_solve, w_inv, w_lstsq)
print("max prediction difference vs sklearn:",
      max(np.abs(Xbt @ w - model2.predict(X_test[features])).max() for w in (w_solve, w_inv, w_lstsq)))
print(np.allclose(w_solve, np.r_[model2.intercept_, model2.coef_]),
      np.allclose(w_inv, w_solve), np.allclose(w_lstsq, w_solve))
#%%
X6 = X_train[all_features].to_numpy()
X6b = np.c_[np.ones(len(X6)), X6]
X6s = ((X_train[all_features] - X_train[all_features].mean()) / X_train[all_features].std()).to_numpy()
X6sb = np.c_[np.ones(len(X6s)), X6s]
for name, M in [("raw", X6b), ("standardised", X6sb)]:
    print(name, "cond(X) = %.3g" % np.linalg.cond(M), " cond(X'X) = %.3g" % np.linalg.cond(M.T @ M))
#%%
A = np.array([[1.0, 1.0], [1.0, 1.0001]])
print(np.linalg.solve(A, [2.0, 2.0]), np.linalg.solve(A, [2.0, 2.0001]), np.linalg.cond(A))
#%%
# ---- Advanced Study B ----
def gradient_descent(grad, w0, eta, steps):
    w = np.array(w0, dtype=float)
    for _ in range(steps):
        w = w - eta * grad(w)
    return w

for eta in (0.1, 0.5, 1.1):
    print(eta, [float(gradient_descent(lambda w: 2 * (w - 3), 0.0, eta, s)) for s in (1, 2, 5, 20)])
print(gradient_descent(lambda w: 2 * (w - 3), 0.0, 0.5, 1))
#%%
ytr = y_train.to_numpy()
Xtr2 = X_train[features].to_numpy()
mu, sd = X_train[features].mean(), X_train[features].std()
Zb = np.c_[np.ones(len(X_train)), ((X_train[features] - mu) / sd).to_numpy()]
Zbt = np.c_[np.ones(len(X_test)), ((X_test[features] - mu) / sd).to_numpy()]
n = len(ytr)
grad = lambda w: 2 / n * Zb.T @ (Zb @ w - ytr)
w_gd = gradient_descent(grad, np.zeros(3), 0.1, 500)
w_ref = np.linalg.lstsq(Zb, ytr, rcond=None)[0]
print("gd:", w_gd, "lstsq:", w_ref)
print("train MSE:", np.mean((Zb @ w_gd - ytr) ** 2))
print("test RMSE gd:", np.mean((Zbt @ w_gd - y_test.to_numpy()) ** 2) ** 0.5, " sklearn model2:", rmse(model2, features, X_test, y_test))
#%%
with np.errstate(all="ignore"):
    Rb = np.c_[np.ones(n), Xtr2]
    gr = lambda w: 2 / n * Rb.T @ (Rb @ w - ytr)
    print("raw, eta=0.1:", gradient_descent(gr, np.zeros(3), 0.1, 50))
    print("raw, eta=1e-7:", gradient_descent(gr, np.zeros(3), 1e-7, 500))
for eta in (0.01, 0.6):
    with np.errstate(all="ignore"):
        print("standardised, eta =", eta, gradient_descent(grad, np.zeros(3), eta, 500))
#%%
# ---- Advanced Study C ----
rng = np.random.default_rng(0)
def boot_hp(cols, B=200):
    out = []
    for _ in range(B):
        idx = rng.integers(0, len(X_train), len(X_train))
        m = LinearRegression().fit(X_train[cols].iloc[idx], y_train.iloc[idx])
        out.append(m.coef_[cols.index("horsepower")])
    return np.array(out)

b2 = boot_hp(["horsepower", "weight"])
b6 = boot_hp(all_features)
for name, b in [("hp + weight", b2), ("six features", b6)]:
    print(name, np.percentile(b, [2.5, 97.5]).round(3))
fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharex=True)
ax[0].hist(b2, bins=20); ax[0].set_title("hp + weight")
ax[1].hist(b6, bins=20); ax[1].set_title("six features")
for a in ax: a.axvline(0, color="black")
plt.show()
