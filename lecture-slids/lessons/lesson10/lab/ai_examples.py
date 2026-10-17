#%%
# ---- Try with AI 2.3: the 16 first-layer filters ----
W = cnn.layers[0].get_weights()[0]                       # 3 x 3 x 1 x 16
fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
for k in range(16):
    w = W[:, :, 0, k]                                    # filter k
    v = np.abs(w).max()
    ax = axes[k // 8, k % 8]
    ax.imshow(w, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title("filter " + str(k))
    ax.axis("off")
plt.show()
#%%
# ---- Try with AI 3.1: the CNN on shuffled pixels ----
X_fit_mixed_img = X_fit_mixed.reshape(-1, 28, 28, 1)
X_val_mixed_img = X_val_mixed.reshape(-1, 28, 28, 1)
keras.utils.set_random_seed(0)
cnn_mixed = keras.Sequential([
    keras.layers.Input((28, 28, 1)),
    keras.layers.Conv2D(16, 3, activation="relu"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation="softmax"),
])
cnn_mixed.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h_mixed = cnn_mixed.fit(X_fit_mixed_img, y_fit, epochs=2, validation_data=(X_val_mixed_img, y_val), verbose=0)
print("CNN on shuffled pixels:", round(h_mixed.history["val_accuracy"][-1], 4))
#%%
# ---- Try with AI 4.2: parameters, accuracy and seconds, as in Table A ----
import time
rows = []
for name, layers, A, B in [
        ("fully connected", [keras.layers.Input((784,)), keras.layers.Dense(128, activation="relu"),
                             keras.layers.Dense(128, activation="relu"), keras.layers.Dense(10, activation="softmax")], X_fit_flat, X_val_flat),
        ("CNN", [keras.layers.Input((28, 28, 1)), keras.layers.Conv2D(16, 3, activation="relu"), keras.layers.MaxPooling2D(2),
                 keras.layers.Conv2D(32, 3, activation="relu"), keras.layers.MaxPooling2D(2), keras.layers.Flatten(),
                 keras.layers.Dense(10, activation="softmax")], X_fit, X_val)]:
    keras.utils.set_random_seed(0)
    m = keras.Sequential(layers)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start = time.time()
    h = m.fit(A, y_fit, epochs=2, validation_data=(B, y_val), verbose=0)
    rows.append((name, m.count_params(), round(h.history["val_accuracy"][-1], 4), round(time.time() - start, 1)))
for row in rows:
    print(*row)
print("time ratio", round(rows[1][3] / rows[0][3], 1))
