# Course data

`mnist10k.npz`: a fixed subset of 10,000 MNIST training images, used in Lessons 4–8 wherever an image has to be looked at.

- Source: the training part of the MNIST file used in Assignment 2 (`keras.datasets.mnist`, 60,000 images).
- Selection: `np.sort(np.random.default_rng(0).choice(60000, 10000, replace=False))`.
- Arrays: `X` (10000 × 784, integers 0 to 255, one row per image, 28 × 28 pixels row by row) and `y` (the digit, 0 to 9).
- In a lab: `urllib.request.urlretrieve(URL, "mnist10k.npz")`, then `data = np.load("mnist10k.npz")`.

MNIST: Y. LeCun, C. Cortes and C. J. C. Burges, http://yann.lecun.com/exdb/mnist/ (licence CC BY-SA 3.0, as distributed with Keras).
