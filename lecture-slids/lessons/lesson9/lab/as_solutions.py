#%%
# ---- Advanced Study A ----
rng = np.random.default_rng(0)
angle = rng.uniform(0, 2 * np.pi, 300)
radius = np.r_[rng.normal(1.0, 0.25, 150), rng.normal(2.4, 0.3, 150)]
X_ring = np.c_[radius * np.cos(angle), radius * np.sin(angle)]   # the ring of Lesson 4
y_ring = np.r_[np.ones(150, int), np.zeros(150, int)]
rng = np.random.default_rng(0)                            # starting weights
W1 = rng.normal(0, 1, (2, 4))
b1 = np.zeros(4)
w2 = rng.normal(0, 1, 4)
b2 = 0.0
losses = []
for step in range(5000):
    Zh = X_ring @ W1 + b1                                 # weighted sums of the hidden units
    H = np.maximum(0, Zh)                                 # ReLU hidden features (the Note uses sigmoid)
    p = 1 / (1 + np.exp(-(H @ w2 + b2)))                  # output probability
    losses.append(-np.mean(y_ring * np.log(p) + (1 - y_ring) * np.log(1 - p)))
    g = (p - y_ring) / len(y_ring)                        # step 1
    gw2, gb2 = H.T @ g, g.sum()
    gH = np.outer(g, w2) * (Zh > 0)                       # steps 2 and 3: the slope of ReLU is 1 or 0
    gW1, gb1 = X_ring.T @ gH, gH.sum(axis=0)              # step 4
    W1, b1, w2, b2 = W1 - 0.5 * gW1, b1 - 0.5 * gb1, w2 - 0.5 * gw2, b2 - 0.5 * gb2   # step size 0.5
print("loss", round(losses[0], 3), "->", round(losses[-1], 3), "accuracy", round(((p > 0.5) == y_ring).mean(), 3))
#%%
# ---- Advanced Study B ----
def loss_at(W1_try):
    Hh = np.maximum(0, X_ring @ W1_try + b1)
    pp = 1 / (1 + np.exp(-(Hh @ w2 + b2)))
    return -np.mean(y_ring * np.log(pp) + (1 - y_ring) * np.log(1 - pp))
Zh = X_ring @ W1 + b1                                     # gradient at the final weights
H = np.maximum(0, Zh)
p = 1 / (1 + np.exp(-(H @ w2 + b2)))
gH = np.outer((p - y_ring) / len(y_ring), w2) * (Zh > 0)
analytic = (X_ring.T @ gH)[0, 0]
W1_plus, W1_minus = W1.copy(), W1.copy()
W1_plus[0, 0] += 1e-5
W1_minus[0, 0] -= 1e-5
numeric = (loss_at(W1_plus) - loss_at(W1_minus)) / 2e-5    # change in the loss over a step of 2e-5
print("analytic", analytic, "numeric", numeric)
print("agree:", bool(abs(analytic - numeric) < 1e-3 * abs(analytic)))   # the first three significant figures
#%%
# ---- Advanced Study C ----
keras.utils.set_random_seed(0)
drop = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation="softmax"),
])
drop.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hd = drop.fit(X_fit, y_fit, epochs=8, validation_data=(X_val, y_val), verbose=0).history
print("lowest validation loss at epoch", int(np.argmin(hd["val_loss"])) + 1, round(min(hd["val_loss"]), 3))
print("final gap (val loss - train loss):", round(hd["val_loss"][-1] - hd["loss"][-1], 3), "without dropout:", round(h["val_loss"][-1] - h["loss"][-1], 3))
