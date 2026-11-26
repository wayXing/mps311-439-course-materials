"""Lesson 10 Note v2: the numbers that are new in v2 (the rest stay in ../note/numbers.json).
Run:  uv run --no-project --python 3.12 --with tensorflow python build_numbers_v2.py   -> numbers_v2.json
Same fixed settings as ../note/build_numbers.py (MNIST from keras.datasets, first 50,000 fit, next 10,000 validation,
set_random_seed(0) before every network, 5 epochs; shift = move right, fill with 0).
  loss       the lecture's CNN after 5 epochs: the 10 probabilities of two validation images, and -log p of the true digit
  extension  one-line variants of the lecture's CNN, 5 epochs each: parameters, validation accuracy, after moves of 2 and 4
  augment    Lab Advanced A: both networks, 2 epochs, with and without one moved copy (-3 to 3 pixels) of every fitting image
  multiply   Note Advanced Study 3: multiplications per image of each network
"""
import os, json, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import numpy as np
import keras

HERE = os.path.dirname(os.path.abspath(__file__))
r = lambda v, k=4: round(float(v), k)
N = {}
(x_train, y_train), _ = keras.datasets.mnist.load_data()
X = (x_train / 255.0)[..., None]
y = y_train
Xf, yf, Xv, yv = X[:50000], y[:50000], X[50000:], y[50000:]
L = keras.layers


def shift(A, k):
    out = np.zeros_like(A)
    out[:, :, k:] = A[:, :, :28 - k] if k else A
    return out


def base():
    return [L.Conv2D(16, 3, activation="relu"), L.MaxPooling2D(2), L.Conv2D(32, 3, activation="relu"), L.MaxPooling2D(2), L.Flatten()]


def cnn(make):
    keras.utils.set_random_seed(0)                                         # the seed first, then the layers: as in ../note/build_numbers.py
    m = keras.Sequential([L.Input((28, 28, 1))] + make() + [L.Dense(10, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def dense():
    keras.utils.set_random_seed(0)
    m = keras.Sequential([L.Input((784,)), L.Dense(128, activation="relu"), L.Dense(128, activation="relu"), L.Dense(10, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


# ---------- loss: two validation images under the lecture's CNN ----------
m = cnn(base)
h = m.fit(Xf, yf, epochs=5, validation_data=(Xv, yv), verbose=0).history
P = m.predict(Xv, verbose=0)
pred = P.argmax(1)
right = int(np.where(pred == yv)[0][0])                                   # the first image it gets right
wrong = int(np.where((yv == 5) & (pred == 8))[0][0])                      # the first 5 it reads as 8
N["loss"] = dict(val=r(h["val_accuracy"][-1]), val_loss=[r(v) for v in h["val_loss"]], train_loss=[r(v) for v in h["loss"]],
                 images=[dict(index=i, true=int(yv[i]), pred=int(pred[i]), p=[r(v) for v in P[i]], minus_log=r(-np.log(P[i, yv[i]])),
                              img=[[r(v, 2) for v in row] for row in Xv[i, :, :, 0]]) for i in (right, wrong)],
                 mean_minus_log=r(-np.mean(np.log(P[np.arange(len(yv)), yv]))))

# ---------- extension: one-line variants ----------
V = {
    "base": base,
    "padding_same": lambda: [L.Conv2D(16, 3, padding="same", activation="relu"), L.MaxPooling2D(2),
                             L.Conv2D(32, 3, padding="same", activation="relu"), L.MaxPooling2D(2), L.Flatten()],
    "strides_2": lambda: [L.Conv2D(16, 3, strides=2, activation="relu"), L.Conv2D(32, 3, strides=2, activation="relu"), L.Flatten()],
    "global_average": lambda: base()[:-1] + [L.GlobalAveragePooling2D()],
    "three_conv": lambda: base()[:-1] + [L.Conv2D(64, 3, activation="relu"), L.Flatten()],
    "dropout": lambda: base() + [L.Dropout(0.5)],
    "random_translation": lambda: [L.RandomTranslation(0, 0.1, fill_mode="constant")] + base(),
}
N["extension"] = {}
for name, make in V.items():
    m = cnn(make)
    h = m.fit(Xf, yf, epochs=5, validation_data=(Xv, yv), verbose=0).history
    N["extension"][name] = dict(params=int(m.count_params()), val=r(h["val_accuracy"][-1]),
                                moved2=r(m.evaluate(shift(Xv, 2), yv, verbose=0)[1]), moved4=r(m.evaluate(shift(Xv, 4), yv, verbose=0)[1]))
    print(name, N["extension"][name], flush=True)

# ---------- Lab Advanced A: train on moved copies ----------
moved = np.zeros_like(Xf)
amounts = np.random.default_rng(0).integers(-3, 4, size=50000)
for k in range(-3, 4):
    pick = amounts == k
    if k > 0:
        moved[pick, :, k:] = Xf[pick, :, :-k]
    elif k < 0:
        moved[pick, :, :k] = Xf[pick, :, -k:]
    else:
        moved[pick] = Xf[pick]
Xm, ym = np.concatenate([Xf, moved]), np.concatenate([yf, yf])
N["augment"] = {}
for name, make, A, B, flat in [("dense", dense, Xf, yf, True), ("dense_moved", dense, Xm, ym, True),
                               ("cnn", lambda: cnn(base), Xf, yf, False), ("cnn_moved", lambda: cnn(base), Xm, ym, False)]:
    m = make()
    t = time.time()
    m.fit(A.reshape(-1, 784) if flat else A, B, epochs=2, verbose=0)
    t = time.time() - t
    acc = [r(m.evaluate(shift(Xv, k).reshape(-1, 784) if flat else shift(Xv, k), yv, verbose=0)[1]) for k in (0, 2, 4)]
    N["augment"][name] = dict(seconds=round(t, 1), val=acc[0], moved2=acc[1], moved4=acc[2])
    print(name, N["augment"][name], flush=True)

# ---------- Advanced Study 3: multiplications per image ----------
N["multiply"] = dict(dense=784 * 128 + 128 * 128 + 128 * 10,
                     conv1=26 * 26 * 16 * 9, conv2=11 * 11 * 32 * 3 * 3 * 16, out=800 * 10)
N["multiply"]["cnn"] = N["multiply"]["conv1"] + N["multiply"]["conv2"] + N["multiply"]["out"]

json.dump(N, open(os.path.join(HERE, "numbers_v2.json"), "w"))
show = dict(N)
show["loss"] = {k: v for k, v in N["loss"].items() if k != "images"}
show["loss"]["images"] = [{k: v for k, v in im.items() if k != "img"} for im in N["loss"]["images"]]
print(json.dumps(show, indent=1))
