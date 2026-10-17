#%%
# ---- Advanced Study A ----
import time
X_shift = np.zeros_like(X_val)
X_shift[:, :, 2:] = X_val[:, :, :-2]
warm = keras.Sequential([keras.layers.Input((28, 28, 1)), keras.layers.Conv2D(8, 3, activation="relu"), keras.layers.Flatten(),
                         keras.layers.Dense(10, activation="softmax")])
warm.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
warm.fit(X_fit[:2000], y_fit[:2000], epochs=1, verbose=0)   # warm-up: the first fit also pays start-up time
print("Timings vary from run to run and machine to machine; compare them only within one table.")
table = []
for f1, f2, pool in [(8, 16, True), (16, 32, True), (32, 64, True), (16, 32, False)]:
    keras.utils.set_random_seed(0)
    layers = [keras.layers.Input((28, 28, 1)), keras.layers.Conv2D(f1, 3, activation="relu")]
    if pool:
        layers.append(keras.layers.MaxPooling2D(2))
    layers.append(keras.layers.Conv2D(f2, 3, activation="relu"))
    if pool:
        layers.append(keras.layers.MaxPooling2D(2))
    layers += [keras.layers.Flatten(), keras.layers.Dense(10, activation="softmax")]
    m = keras.Sequential(layers)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start = time.time()
    h = m.fit(X_fit, y_fit, epochs=2, validation_data=(X_val, y_val), verbose=0)
    seconds = round(time.time() - start, 1)
    shifted = m.evaluate(X_shift, y_val, verbose=0)[1]
    table.append((f"{f1}-{f2}" + ("" if pool else " no pooling"), m.count_params(), round(h.history["val_accuracy"][-1], 4), seconds, round(shifted, 4)))
for row in table:
    print(*row)
#%%
# ---- Advanced Study B ----
# A written answer: see the worksheet. Assignment 2, Data, rule 2 forbids models trained by someone else.
#%%
# ---- Advanced Study C ----
X_shift_fit = np.zeros_like(X_fit)
X_shift_fit[:, :, 2:] = X_fit[:, :, :-2]                  # every fitting image moved 2 pixels
X_more = np.concatenate([X_fit_flat, X_shift_fit.reshape(-1, 784)])
y_more = np.concatenate([y_fit, y_fit])
keras.utils.set_random_seed(0)
aug = keras.Sequential([keras.layers.Input((784,)), keras.layers.Dense(128, activation="relu"),
                        keras.layers.Dense(128, activation="relu"), keras.layers.Dense(10, activation="softmax")])
aug.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
aug.fit(X_more, y_more, epochs=3, verbose=0)
print("moved 2: before", round(dense.evaluate(X_shift.reshape(-1, 784), y_val, verbose=0)[1], 4),
      "after", round(aug.evaluate(X_shift.reshape(-1, 784), y_val, verbose=0)[1], 4),
      "unmoved after", round(aug.evaluate(X_val_flat, y_val, verbose=0)[1], 4))
