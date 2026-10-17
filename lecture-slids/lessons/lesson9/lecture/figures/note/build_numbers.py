"""Lesson 9 Note: every number in the note, the figures, the widgets and the lab checks.
Run with a Python that has TensorFlow (on this Mac: the arm64 venv, see COURSE_MATERIAL_WORKFLOW §5):
    python build_numbers.py   -> numbers.json (and prints what the note quotes)
Rules fixed in the design card before writing the note:
  rings   the Lesson 4 ring: default_rng(0), radius N(1, 0.25) inner (class 1) and N(2.4, 0.3) outer, 150 each
  ring nets  Dense(h, sigmoid) then Dense(1, sigmoid); Adam(0.05), 300 epochs, whole data per step; set_random_seed(0)
  MNIST   keras.datasets file (mnist.npz); first 50,000 training images fit, the next 10,000 are the validation set
          (the split of Assignment 2); pixels / 255
  MNIST net  784 -> Dense(128, relu) -> Dense(128, relu) -> Dense(10, softmax); adam; 20 epochs; set_random_seed(0)
"""
import os, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import numpy as np
import keras
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../../../.."))
r = lambda v, k=3: round(float(v), k)
N = {}

# ---------- the Lesson 4 ring ----------
rng = np.random.default_rng(0)
ang = rng.uniform(0, 2 * np.pi, 300)
rr = np.r_[rng.normal(1.0, 0.25, 150), rng.normal(2.4, 0.3, 150)]
X = np.c_[rr * np.cos(ang), rr * np.sin(ang)]
y = np.r_[np.ones(150, int), np.zeros(150, int)]
acc_line = LogisticRegression().fit(X, y).score(X, y)
dist = np.sqrt((X ** 2).sum(1))[:, None]
acc_dist = LogisticRegression().fit(dist, y).score(dist, y)


def ring_net(h, act):
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.Input((2,)), keras.layers.Dense(h, activation=act), keras.layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=keras.optimizers.Adam(0.05), loss="binary_crossentropy", metrics=["accuracy"])
    hist = m.fit(X, y, epochs=300, batch_size=300, verbose=0)
    (W1, b1), (W2, b2) = [l.get_weights() for l in m.layers]
    return dict(acc=r(hist.history["accuracy"][-1]), loss=[r(v, 4) for v in hist.history["loss"][::10]],
                W1=[[r(v, 4) for v in row] for row in W1], b1=[r(v, 4) for v in b1],
                W2=[r(v, 4) for v in W2[:, 0]], b2=r(b2[0], 4))


nets = {f"sigmoid{h}": ring_net(h, "sigmoid") for h in [1, 2, 3, 4]}
nets.update({f"linear{h}": ring_net(h, "linear") for h in [1, 2, 3, 4]})
nets.update({f"relu{h}": ring_net(h, "relu") for h in [1, 2, 3, 4]})
N["ring"] = dict(x=[r(v) for v in X[:, 0]], y=[r(v) for v in X[:, 1]], label=y.tolist(),
                 acc_line=r(acc_line, 2), acc_dist=r(acc_dist, 2), nets=nets)

# ---------- MNIST ----------
d = np.load(os.path.join(ROOT, "course-site/public/assignments/assignment-two/mnist.npz"))
Xm = d["x_train"].reshape(-1, 784) / 255.0
ym = d["y_train"]
Xtr, ytr, Xv, yv = Xm[:50000], ym[:50000], Xm[50000:], ym[50000:]
lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)


def mnist_net(seed=0, act="relu"):
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([keras.layers.Input((784,)), keras.layers.Dense(128, activation=act),
                          keras.layers.Dense(128, activation=act), keras.layers.Dense(10, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


m = mnist_net()
hist = m.fit(Xtr, ytr, epochs=20, validation_data=(Xv, yv), verbose=0)
H = {k: [r(v, 4) for v in vals] for k, vals in hist.history.items()}
W1 = m.layers[0].get_weights()[0]  # 784 x 128
pick = np.argsort(-np.abs(W1).sum(0))[:12]  # the 12 units with the largest weights
mlin = mnist_net(act="linear")
hlin = mlin.fit(Xtr, ytr, epochs=5, validation_data=(Xv, yv), verbose=0)
mfb = mnist_net()                       # the same network, but each step uses all 50,000 images
hfb = mfb.fit(Xtr, ytr, epochs=5, batch_size=50000, validation_data=(Xv, yv), verbose=0)
seeds = {}
for s in [1, 2]:
    hs = mnist_net(s).fit(Xtr, ytr, epochs=20, validation_data=(Xv, yv), verbose=0)
    seeds[str(s)] = r(hs.history["val_accuracy"][-1], 4)
seeds["0"] = H["val_accuracy"][-1]
h1_5 = mnist_net(1).fit(Xtr, ytr, epochs=5, validation_data=(Xv, yv), verbose=0).history   # Lab 4.1: seed 1, 5 epochs
keras.utils.set_random_seed(0)          # the transfer question: 512 units per layer, otherwise the same
m512 = keras.Sequential([keras.layers.Input((784,)), keras.layers.Dense(512, activation="relu"),
                         keras.layers.Dense(512, activation="relu"), keras.layers.Dense(10, activation="softmax")])
m512.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h512 = m512.fit(Xtr, ytr, epochs=20, validation_data=(Xv, yv), verbose=0).history
N["mnist"] = dict(n_fit=len(ytr), n_val=len(yv), lr_val=r(lr.score(Xv, yv), 4), lr_train=r(lr.score(Xtr, ytr), 4),
                  params=int(m.count_params()), params_layers=[int(l.count_params()) for l in m.layers],
                  steps_per_epoch=int(np.ceil(50000 / 32)), hist=H,
                  best_val_loss_epoch=int(np.argmin(H["val_loss"]) + 1), best_val_loss=min(H["val_loss"]),
                  linear_val5=[r(v, 4) for v in hlin.history["val_accuracy"]], fullbatch_val5=[r(v, 4) for v in hfb.history["val_accuracy"]], relu_val5=H["val_accuracy"][:5],
                  seeds=seeds, seed1_val5=r(h1_5["val_accuracy"][-1], 4), wide512_val20=r(h512["val_accuracy"][-1], 4), wide512_best_val_loss_epoch=int(np.argmin(h512["val_loss"]) + 1), weights=[[r(v, 3) for v in W1[:, k]] for k in pick],
                  lr_weights=[[r(v, 3) for v in lr.coef_[k]] for k in range(10)])

# ---------- Advanced: one hidden layer in numpy on the ring ----------
rng = np.random.default_rng(0)
A1, c1 = rng.normal(0, 1, (2, 4)), np.zeros(4)
A2, c2 = rng.normal(0, 1, 4), 0.0
sig = lambda z: 1 / (1 + np.exp(-z))
losses = []
for step in range(5000):
    Hh = sig(X @ A1 + c1)                   # hidden features
    p = sig(Hh @ A2 + c2)                   # output probability
    losses.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    g = (p - y) / len(y)                    # d loss / d output score
    gA2, gc2 = Hh.T @ g, g.sum()
    gH = np.outer(g, A2) * Hh * (1 - Hh)    # chain rule back through the hidden layer
    gA1, gc1 = X.T @ gH, gH.sum(0)
    A1, c1, A2, c2 = A1 - 1.0 * gA1, c1 - 1.0 * gc1, A2 - 1.0 * gA2, c2 - 1.0 * gc2
N["adv"] = dict(acc=r(((p > 0.5) == y).mean()), loss_start=r(losses[0]), loss_end=r(losses[-1]), steps=5000)

json.dump(N, open(os.path.join(HERE, "numbers.json"), "w"), indent=1)
show = {"ring": {k: v for k, v in N["ring"].items() if k.startswith("acc")},
        "nets": {k: v["acc"] for k, v in nets.items()},
        "mnist": {k: v for k, v in N["mnist"].items() if k not in ("weights", "lr_weights", "hist")},
        "val_acc": H["val_accuracy"], "val_loss": H["val_loss"], "loss": H["loss"], "acc": H["accuracy"], "adv": N["adv"]}
print(json.dumps(show, indent=1))
