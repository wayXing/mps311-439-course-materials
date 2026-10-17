#%%
# ---- Advanced Study A ----
k = 12                                                    # 12 groups (the Note uses 10)
results = []
for seed in range(10):                                   # ten different starting points
    rng = np.random.default_rng(seed)
    centres = Z_fit[rng.choice(len(Z_fit), k, replace=False)]   # k random images as starting centres
    for step in range(100):
        dist = ((Z_fit[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)   # every image to every centre
        nearest = dist.argmin(axis=1)                    # step 1: nearest centre
        new = centres.copy()                             # step 2: move each centre to its group's mean
        for j in range(k):
            members = Z_fit[nearest == j]
            if len(members) > 0:                         # an empty group keeps its old centre
                new[j] = members.mean(axis=0)
        if np.allclose(new, centres):                    # nothing moves: stop
            break
        centres = new
    results.append(((Z_fit - centres[nearest]) ** 2).sum())
print("ten starts:", np.round(results, 1))
print("best", round(min(results), 1), "worst", round(max(results), 1))
print("scikit-learn, k = 12:", round(KMeans(n_clusters=12, n_init=10, random_state=0).fit(Z_fit).inertia_, 1))
#%%
# ---- Advanced Study B ----
from sklearn.metrics import silhouette_score
for k in range(2, 16):
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z_fit)
    print(k, round(silhouette_score(Z_fit, labels), 3))
#%%
# ---- Advanced Study C ----
from sklearn.cluster import DBSCAN
rng = np.random.default_rng(0)
a1 = rng.uniform(0, 2 * np.pi, 120)
a2 = rng.uniform(0, 2 * np.pi, 200)
r1 = rng.normal(1, 0.12, 120)
r2 = rng.normal(3, 0.15, 200)
X_rings = np.vstack([np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)]), np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])])
for eps in [0.2, 0.5, 1.0]:
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(X_rings)
    print("eps", eps, "groups", len(set(lab) - {-1}), "noise", int((lab == -1).sum()))
