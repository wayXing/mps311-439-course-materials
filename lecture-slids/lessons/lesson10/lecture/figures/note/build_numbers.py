"""Lesson 10 Note: every number in the note, the figures, the widgets and the lab checks.
Run with a Python that has TensorFlow (on this Mac: the arm64 venv, see COURSE_MATERIAL_WORKFLOW §5):
    python build_numbers.py   -> numbers.json (and prints what the note quotes)
Rules fixed in the design card before writing the note:
  MNIST   keras.datasets file; first 50,000 training images fit, the next 10,000 are the validation set (Assignment 2)
  dense   the Lesson 9 network 784-128-128-10, adam, set_random_seed(0)
  CNN     Conv2D(16, 3, relu) - MaxPooling2D(2) - Conv2D(32, 3, relu) - MaxPooling2D(2) - Flatten - Dense(10, softmax)
  epochs  5 for the note; the lab trains for fewer epochs (see LAB below)
  shift   move every validation image k pixels to the right, filling the left columns with 0
  shuffle one fixed permutation of the 784 pixels, default_rng(0), the same for every image
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
X = x_train / 255.0
y = y_train
Xf, yf, Xv, yv = X[:50000], y[:50000], X[50000:], y[50000:]
perm = np.random.default_rng(0).permutation(784)


def shift(A, k):
    out = np.zeros_like(A)
    out[:, :, k:] = A[:, :, :28 - k] if k else A
    return out


def dense():
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.layers.Input((784,)), keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dense(128, activation="relu"), keras.layers.Dense(10, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def cnn(f1=16, f2=32, pool=True):
    keras.utils.set_random_seed(0)
    P = keras.layers.AveragePooling2D if pool == "avg" else keras.layers.MaxPooling2D
    L = [keras.layers.Input((28, 28, 1)), keras.layers.Conv2D(f1, 3, activation="relu")]
    if pool:
        L.append(P(2))
    L.append(keras.layers.Conv2D(f2, 3, activation="relu"))
    if pool:
        L.append(P(2))
    L += [keras.layers.Flatten(), keras.layers.Dense(10, activation="softmax")]
    m = keras.Sequential(L)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def fit(m, A, B, epochs=5):
    t = time.time()
    h = m.fit(A, yf, epochs=epochs, validation_data=(B, yv), verbose=0).history
    return h, time.time() - t


# ---------- the surprise: shuffled pixels ----------
md = dense()
hd, td = fit(md, Xf.reshape(-1, 784), Xv.reshape(-1, 784))
ms = dense()
hs, _ = fit(ms, Xf.reshape(-1, 784)[:, perm], Xv.reshape(-1, 784)[:, perm])
shifts = list(range(5))
dense_shift = [r(md.evaluate(shift(Xv, k).reshape(-1, 784), yv, verbose=0)[1]) for k in shifts]
N["dense"] = dict(val=[r(v) for v in hd["val_accuracy"]], params=int(md.count_params()), seconds=round(td, 1),
                  shuffled_val=[r(v) for v in hs["val_accuracy"]], shift=dense_shift)

# ---------- the CNN ----------
mc = cnn()
hc, tc = fit(mc, Xf[..., None], Xv[..., None])
cnn_shift = [r(mc.evaluate(shift(Xv, k)[..., None], yv, verbose=0)[1]) for k in shifts]
mcs = cnn()  # the CNN on shuffled pixels: the neighbours are gone
Xfs = Xf.reshape(-1, 784)[:, perm].reshape(-1, 28, 28, 1)
Xvs = Xv.reshape(-1, 784)[:, perm].reshape(-1, 28, 28, 1)
hcs, _ = fit(mcs, Xfs, Xvs)
mnp = cnn(pool=False)  # without pooling
hnp, tnp = fit(mnp, Xf[..., None], Xv[..., None])
np_shift = [r(mnp.evaluate(shift(Xv, k)[..., None], yv, verbose=0)[1]) for k in shifts]
mav = cnn(pool="avg")  # the other candidate: the mean of each 2 x 2 block
hav, tav = fit(mav, Xf[..., None], Xv[..., None])
av_shift = [r(mav.evaluate(shift(Xv, k)[..., None], yv, verbose=0)[1]) for k in shifts]
layer_params = [int(l.count_params()) for l in mc.layers]
N["cnn"] = dict(val=[r(v) for v in hc["val_accuracy"]], params=int(mc.count_params()), layer_params=layer_params,
                seconds=round(tc, 1), shift=cnn_shift, shuffled_val=[r(v) for v in hcs["val_accuracy"]],
                nopool=dict(val=r(hnp["val_accuracy"][-1]), params=int(mnp.count_params()), seconds=round(tnp, 1), shift=np_shift),
                avgpool=dict(val=r(hav["val_accuracy"][-1]), params=int(mav.count_params()), seconds=round(tav, 1), shift=av_shift))
# a layer with its own 3x3 weights at every position (no sharing), 16 detectors: how many weights?
N["local_params"] = 26 * 26 * 16 * (9 + 1)
N["shared_params"] = 16 * (9 + 1)

# ---------- errors on the validation set ----------
pc = mc.predict(Xv[..., None], verbose=0).argmax(1)
pd = md.predict(Xv.reshape(-1, 784), verbose=0).argmax(1)
cm = np.zeros((10, 10), int)
for t, p in zip(yv, pc):
    cm[t, p] += 1
off = cm.copy()
np.fill_diagonal(off, 0)
top = np.argsort(off.ravel())[::-1][:5]
wrong = np.where(pc != yv)[0]
# the 10 images shown: two of each of the five most common mistakes, so the pictures match the table
shown = []
for t in top:
    ti, pi = int(t // 10), int(t % 10)
    shown += [int(i) for i in wrong if yv[i] == ti and pc[i] == pi][:2]
shown = shown[:10]
N["errors"] = dict(cm=cm.tolist(), wrong=int(len(wrong)), dense_wrong=int((pd != yv).sum()),
                   cnn_right_dense_wrong=int(((pc == yv) & (pd != yv)).sum()), dense_right_cnn_wrong=int(((pc != yv) & (pd == yv)).sum()),
                   top=[dict(true=int(i // 10), pred=int(i % 10), n=int(off.ravel()[i])) for i in top],
                   examples=[dict(img=[[r(v, 2) for v in row] for row in Xv[i]], true=int(yv[i]), pred=int(pc[i])) for i in shown])

# ---------- what the first layer looks for ----------
W = mc.layers[0].get_weights()[0][:, :, 0, :]  # 3 x 3 x 16
idx = int(np.where(yv == 7)[0][0])
fmap = keras.Model(mc.inputs, mc.layers[0].output).predict(Xv[idx:idx + 1, :, :, None], verbose=0)[0]  # 26 x 26 x 16
N["filters"] = dict(w=[[[r(W[i, j, k], 3) for j in range(3)] for i in range(3)] for k in range(16)],
                    image=[[r(v, 2) for v in row] for row in Xv[idx]],
                    maps=[[[r(fmap[i, j, k], 2) for j in range(26)] for i in range(26)] for k in range(16)])
# three images for the sliding-filter widget
pick = [int(np.where(yv == d)[0][0]) for d in (1, 4, 7)]
N["widget_images"] = [dict(digit=int(yv[i]), img=[[r(v, 2) for v in row] for row in Xv[i]]) for i in pick]

# ---------- Advanced: how many filters? ----------
adv = {}
for f1, f2 in [(8, 16), (16, 32), (32, 64)]:
    m = cnn(f1, f2)
    h, t = fit(m, Xf[..., None], Xv[..., None])
    adv[f"{f1}-{f2}"] = dict(val=r(h["val_accuracy"][-1]), params=int(m.count_params()), seconds=round(t, 1))
N["adv"] = adv

# ---------- LAB: shorter training for Colab ----------
ml = dense()
hl, _ = fit(ml, Xf.reshape(-1, 784), Xv.reshape(-1, 784), epochs=3)
mls = dense()
hls, _ = fit(mls, Xf.reshape(-1, 784)[:, perm], Xv.reshape(-1, 784)[:, perm], epochs=3)
mlc = cnn()
hlc, _ = fit(mlc, Xf[..., None], Xv[..., None], epochs=2)
N["lab"] = dict(dense3=r(hl["val_accuracy"][-1]), shuffled3=r(hls["val_accuracy"][-1]), cnn2=r(hlc["val_accuracy"][-1]),
                dense3_shift2=r(ml.evaluate(shift(Xv, 2).reshape(-1, 784), yv, verbose=0)[1]),
                cnn2_shift2=r(mlc.evaluate(shift(Xv, 2)[..., None], yv, verbose=0)[1]))

json.dump(N, open(os.path.join(HERE, "numbers.json"), "w"))
show = {k: v for k, v in N.items() if k not in ("errors", "filters", "widget_images")}
show["errors"] = {k: v for k, v in N["errors"].items() if k not in ("cm", "examples")}
print(json.dumps(show, indent=1))
