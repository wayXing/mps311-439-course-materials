#%%
# ---- Try with AI 2.2: the cross-validated probabilities as two histograms ----
bins = np.linspace(0, 1, 21)                                      # 20 bins between 0 and 1
plt.hist(prob[y_train == 1], bins=bins, alpha=0.6, label="malignant")   # the 159 malignant lumps
plt.hist(prob[y_train == 0], bins=bins, alpha=0.6, label="benign")      # the 267 benign lumps
plt.xlabel("cross-validated probability of malignant")
plt.ylabel("number of lumps")
plt.legend()
plt.show()
print("lumps in the two histograms:", (y_train == 1).sum(), (y_train == 0).sum())
#%%
# ---- Try with AI 3.2 (first box): missed cancers and false alarms for many thresholds ----
thresholds = np.arange(0.05, 0.96, 0.05)                          # 0.05, 0.10, ..., 0.95
missed_list = []
alarm_list = []
for t in thresholds:
    missed_list.append(((prob < t) & (y_train == 1)).sum())       # malignant but below the threshold
    alarm_list.append(((prob >= t) & (y_train == 0)).sum())       # benign but at or above the threshold
plt.plot(thresholds, missed_list, marker="o", label="missed cancers")
plt.plot(thresholds, alarm_list, marker="s", label="false alarms")
plt.xlabel("threshold")
plt.ylabel("count")
plt.legend()
plt.show()
print("at threshold 0.5:", missed_list[9], alarm_list[9])
#%%
# ---- Try with AI 3.2 (second box): the digits, fitted with the code given in the box ----
import urllib.request                                     # to download a file
DATA_URL = "https://raw.githubusercontent.com/wayXing/mps311-439-course-materials/main/lecture-slids/lessons/data/mnist10k.npz"
urllib.request.urlretrieve(DATA_URL, "mnist10k.npz")      # 10,000 MNIST images, about 1.6 MB
mnist = np.load("mnist10k.npz")                           # the file holds two arrays, X and y
X_digits = mnist["X"] / 255                               # 784 pixels per image (28 x 28), scaled to 0 to 1
y_digits = mnist["y"]                                     # the digit, 0 to 9
X_digits_train, X_digits_test, y_digits_train, y_digits_test = train_test_split(
    X_digits, y_digits, test_size=0.25, random_state=0, stratify=y_digits)   # 7500 to train, 2500 to test
digit_model = LogisticRegression(max_iter=1000, C=0.1)    # more steps; C = 0.1 regularises (Lesson 3), chosen by cross-validation
digit_model.fit(X_digits_train, y_digits_train)           # softmax and cross-entropy happen inside
print(digit_model.score(X_digits_test, y_digits_test))    # accuracy on the test images
#%%
# the AI's part: the 10 rows of weights as 28 x 28 images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for k in range(10):
    ax = axes[k // 5, k % 5]                                      # row and column of this small plot
    weights = digit_model.coef_[k].reshape(28, 28)                # 784 weights as a 28 x 28 image
    limit = np.abs(weights).max()                                 # same colour scale on both sides of 0
    ax.imshow(weights, cmap="RdBu_r", vmin=-limit, vmax=limit)    # red positive, blue negative
    ax.set_title("digit " + str(k))                               # which digit this row belongs to
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
