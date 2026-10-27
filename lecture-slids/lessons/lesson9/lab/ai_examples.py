#%%
# ---- Try with AI 2.3: the weights of the first 10 hidden units as images ----
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
# ---- Try with AI 4.1: stop by itself when the validation loss stops improving ----
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
# ---- Try with AI Part 5: confusion matrix on the validation set ----
from sklearn.metrics import confusion_matrix
predicted = model.predict(X_val, verbose=0).argmax(axis=1)   # the most probable digit for each image
cm = confusion_matrix(y_val, predicted)
print(cm)
print("diagonal", np.trace(cm))
off = cm.copy()
np.fill_diagonal(off, 0)                                  # keep only the mistakes
for idx in np.argsort(off.ravel())[::-1][:3]:
    print("true", idx // 10, "predicted", idx % 10, "count", off.ravel()[idx])
