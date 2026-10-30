#%%
# ---- Advanced Study A ----
r = 0.5                                          # how far to pull each spread towards a circle
d = X_train_s.shape[1]                           # 30 features
X_ben = X_train_s[y_train == 0]                  # training lumps of class 0
X_mal = X_train_s[y_train == 1]                  # training lumps of class 1
S_ben = (1 - r) * np.cov(X_ben.T) + r * np.eye(d)    # shrunk spread of class 0
S_mal = (1 - r) * np.cov(X_mal.T) + r * np.eye(d)    # shrunk spread of class 1
diff_ben = X_test_s - X_ben.mean(axis=0)         # each test lump minus the centre of class 0
diff_mal = X_test_s - X_mal.mean(axis=0)         # each test lump minus the centre of class 1
dist_ben = np.sum(diff_ben @ np.linalg.inv(S_ben) * diff_ben, axis=1)   # squared distance by the class's own spread
dist_mal = np.sum(diff_mal @ np.linalg.inv(S_mal) * diff_mal, axis=1)
log_ben = -0.5 * dist_ben - 0.5 * np.linalg.slogdet(S_ben)[1] + np.log(len(X_ben) / len(X_train_s))   # log of P(Y) times height
log_mal = -0.5 * dist_mal - 0.5 * np.linalg.slogdet(S_mal)[1] + np.log(len(X_mal) / len(X_train_s))
mine = (log_mal > log_ben).astype(int)          # 1 when malignant wins
#%%
# self-check from the worksheet
ref = QuadraticDiscriminantAnalysis(reg_param=0.5).fit(X_train_s, y_train)   # sklearn, the same shrinkage
print((mine != ref.predict(X_test_s)).sum())              # number of test lumps where the two disagree
print("test accuracy, mine:", round((mine == y_test).mean(), 3), " sklearn:", round(ref.score(X_test_s, y_test), 3))
#%%
# the benign lumps with r = 0: the spread is almost flat in one direction
S_zero = np.cov(X_train_s[y_train == 0].T)                # 30 x 30, no shrinking
print("smallest spread:", round(np.linalg.eigvalsh(S_zero).min(), 5), " largest:", round(np.linalg.eigvalsh(S_zero).max(), 1))
print("slogdet:", np.linalg.slogdet(S_zero))              # sign 1, log-determinant about -98.7
# np.linalg.inv(S_zero) multiplies by about 1 / 0.00003 along the flattest direction, so distances swing wildly
#%%
# ---- Advanced Study B ----
X_mal = X_train_s[y_train == 1]
X_ben = X_train_s[y_train == 0]
n1 = len(X_mal)
n0 = len(X_ben)
S_shared = ((n1 - 1) * np.cov(X_mal.T) + (n0 - 1) * np.cov(X_ben.T)) / (n1 + n0 - 2)   # the shared spread
mu1 = X_mal.mean(axis=0)
mu0 = X_ben.mean(axis=0)
w = np.linalg.inv(S_shared) @ (mu1 - mu0)                # weights of the straight boundary
b = -0.5 * mu1 @ np.linalg.inv(S_shared) @ mu1 + 0.5 * mu0 @ np.linalg.inv(S_shared) @ mu0 + np.log(n1 / n0)
lda = LinearDiscriminantAnalysis().fit(X_train_s, y_train)
print("largest difference in weights:", np.abs(w - lda.coef_[0]).max())
print("largest weight:", round(np.abs(w).max(), 1), " intercept:", round(b, 2), round(lda.intercept_[0], 2))
#%%
# ---- Advanced Study C ----
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()                                          # only the diagonal of each spread
print("naive Bayes:", round(cross_val_score(nb, X_train_s, y_train, cv=kf).mean(), 3))
print("numbers estimated: 2 x 30 means + 2 x 30 variances + 1 class probability =", 2 * 30 + 2 * 30 + 1)
rare = QuadraticDiscriminantAnalysis(priors=[0.99, 0.01]).fit(X_train[:, [0]], y_train)   # malignant only 1%
print("opening lump, P(Y=0) = 0.99 and P(Y=1) = 0.01:", round(rare.predict_proba(X_test[[47]][:, [0]])[0, 1], 4))
