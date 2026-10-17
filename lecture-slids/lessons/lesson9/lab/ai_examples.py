#%%
# ---- Try with AI 1.3: the weights of the first 10 hidden units as images ----
W = model.layers[0].get_weights()[0]                     # 784 x 128: one column per unit
fig, axes = plt.subplots(1, 10, figsize=(12, 1.8))
for k in range(10):
    w = W[:, k].reshape(28, 28)                          # the 784 weights of unit k as an image
    v = np.abs(w).max()                                  # centre the colours at 0
    axes[k].imshow(w, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[k].set_title("unit " + str(k))
    axes[k].set_xticks([])
    axes[k].set_yticks([])
plt.show()
#%%
# ---- Try with AI 3.1: stop by itself when the validation loss stops improving ----
keras.utils.set_random_seed(0)
early = keras.Sequential([
    keras.layers.Input((784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
early.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
hist_early = early.fit(X_fit, y_fit, epochs=20, validation_data=(X_val, y_val), callbacks=[stop], verbose=0)
print("early stop after", len(hist_early.history["loss"]), "epochs")
print("validation accuracy", round(early.evaluate(X_val, y_val, verbose=0)[1], 4))
#%%
# ---- Try with AI 4.1 (first box): confusion matrix on the validation set ----
from sklearn.metrics import confusion_matrix
predicted = model.predict(X_val, verbose=0).argmax(axis=1)   # the most probable digit for each image
cm = confusion_matrix(y_val, predicted)
print(cm)
print("diagonal", np.trace(cm))
off = cm.copy()
np.fill_diagonal(off, 0)                                  # keep only the mistakes
for idx in np.argsort(off.ravel())[::-1][:3]:
    print("true", idx // 10, "predicted", idx % 10, "count", off.ravel()[idx])
#%%
# ---- Try with AI 4.1 (second box): the network on the ring ----
rng = np.random.default_rng(0)
angle = rng.uniform(0, 2 * np.pi, 300)
radius = np.r_[rng.normal(1.0, 0.25, 150), rng.normal(2.4, 0.3, 150)]
X_ring = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
y_ring = np.r_[np.ones(150, int), np.zeros(150, int)]
keras.utils.set_random_seed(0)
ring = keras.Sequential([keras.layers.Input((2,)), keras.layers.Dense(3, activation="sigmoid"),
                         keras.layers.Dense(1, activation="sigmoid")])
ring.compile(optimizer=keras.optimizers.Adam(0.05), loss="binary_crossentropy", metrics=["accuracy"])
h_ring = ring.fit(X_ring, y_ring, epochs=300, batch_size=300, verbose=0)
print("ring accuracy", round(h_ring.history["accuracy"][-1], 2))
g = np.linspace(-3.6, 3.6, 200)
gx, gy = np.meshgrid(g, g)
prob = ring.predict(np.c_[gx.ravel(), gy.ravel()], verbose=0).reshape(gx.shape)
plt.contourf(gx, gy, prob > 0.5, levels=[0.5, 1.5], colors=["#cfe3f3"])
plt.scatter(X_ring[:, 0], X_ring[:, 1], c=y_ring, cmap="coolwarm_r", s=12)
plt.gca().set_aspect("equal")
plt.show()
