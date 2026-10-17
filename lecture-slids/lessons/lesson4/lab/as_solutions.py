#%%
# ---- Advanced Study A ----
n, d = X_train_s.shape                       # 426 lumps, 30 features
w = np.zeros(d)                              # start from all weights 0
b = 0.0
lr = 0.5                                     # learning rate
losses = []
for step in range(5000):
    p = 1 / (1 + np.exp(-(X_train_s @ w + b)))                    # probability of malignant
    p = np.clip(p, 1e-12, 1 - 1e-12)                              # keep log away from 0 and 1
    if step % 100 == 0:
        losses.append(-np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p)))   # training cross-entropy
    grad_w = X_train_s.T @ (p - y_train) / n + w / (0.5 * n)      # gradient plus the penalty term for C = 0.5
    grad_b = np.mean(p - y_train)                                 # gradient for the intercept (no penalty)
    w = w - lr * grad_w                                           # one step downhill
    b = b - lr * grad_b
plt.plot(np.arange(0, 5000, 100), losses)
plt.xlabel("step")
plt.ylabel("training cross-entropy")
plt.show()
#%%
# self-check from the worksheet
ref = LogisticRegression(C=0.5).fit(X_train_s, y_train)  # the same penalty as yours
print(np.abs(w - ref.coef_[0]).max() < 0.02)              # True if your weights match
print(round(np.abs(ref.coef_).max(), 3))                  # the largest weight with C = 0.5
print("largest difference:", round(np.abs(w - ref.coef_[0]).max(), 3))
#%%
# ---- Advanced Study B ----
almost_free = LogisticRegression(C=1e6, max_iter=10000).fit(X_train_s, y_train)   # larger C = weaker penalty: almost no penalty
print("largest weight, almost no penalty:", round(np.abs(almost_free.coef_).max()))
print("training accuracy, almost no penalty:", almost_free.score(X_train_s, y_train))
default_model = LogisticRegression().fit(X_train_s, y_train)   # default C = 1, for comparison
print("largest weight, default:", round(np.abs(default_model.coef_).max(), 2))
p_default = cross_val_predict(                                 # cross-validated probabilities, default model
    LogisticRegression(), X_train_s, y_train, cv=kf, method="predict_proba")[:, 1]
cm = confusion_matrix(y_train, (p_default >= 0.5).astype(int), labels=[1, 0])
print("default: missed", cm[0, 1], "false alarms", cm[1, 0])
p_free = cross_val_predict(                                    # the same, almost no penalty
    LogisticRegression(C=1e6, max_iter=10000), X_train_s, y_train, cv=kf, method="predict_proba")[:, 1]
cm = confusion_matrix(y_train, (p_free >= 0.5).astype(int), labels=[1, 0])
print("almost no penalty: missed", cm[0, 1], "false alarms", cm[1, 0])
#%%
# ---- Advanced Study C ----
balanced = LogisticRegression(class_weight="balanced")     # malignant lumps count more in the loss
p_bal = cross_val_predict(balanced, X_train_s, y_train, cv=kf, method="predict_proba")[:, 1]
cm = confusion_matrix(y_train, (p_bal >= 0.5).astype(int), labels=[1, 0])
print("class_weight balanced, threshold 0.5: missed", cm[0, 1], "false alarms", cm[1, 0])
from sklearn.calibration import calibration_curve
frac, mean_p = calibration_curve(y_train, prob, n_bins=10) # fraction really malignant in each probability bin
plt.plot(mean_p, frac, marker="o")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("mean predicted probability")
plt.ylabel("fraction really malignant")
plt.show()
