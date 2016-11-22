from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from kde import TwoClassKDE
#from balloon import BalloonNominate
import balloon
import numpy as np
import matplotlib.pyplot as plt


def img_scale(img):
    """Normalizes array values to [0, 1]."""
    return (img - img.min()) / (img.max() - img.min())

def prob_scale(img):
    """Given scores in [0, 1], normalizes so that scores below 0.5 remain below 0.5, and vice versa."""
    flags = img >= 0.5
    xmin, xmax = img.min(), img.max()
    a1 = 1 / (1 - 2 * xmin)
    b1 = -a1 * xmin
    a2 = 1 / (2 * xmax - 1)
    b2 = 1 - a2 * xmax
    return ~flags * (a1 * img + b1) + flags * (a2 * img + b2)

def img_histeq(img, nquant = 50):
    """Given an array of scalars, scales to [0, 1] and equalizes the histogram."""
    img2 = img_scale(img)
    (hist, bins) = np.histogram(img2.ravel(), bins = np.linspace(0, 1, nquant + 1), normed = True)
    P = np.cumsum(hist / np.sum(hist))
    img2 = np.interp(img2, np.linspace(0, 1, nquant), P)
    return img_scale(img2)

def img_histeq_prob(img, nquant = 50):
    flags = img >= 0.5
    return ~flags * 0.5 * img_histeq(~flags * img, nquant) + flags * 0.5 * (1 + img_histeq(flags * img, nquant))

def histeq_scale(nquant = 50):
    return (lambda img : img_histeq(img, nquant))

def histeq_prob_scale(nquant = 50):
    return (lambda img : img_histeq_prob(img, nquant))

def get_color(x):
    return x * np.array([1., 0., 0.]) + (1 - x) * np.array([0., 0., 1.])

# prepare predictors
logreg = LogisticRegression()
gnb = GaussianNB()
randfor = RandomForestClassifier(n_estimators = 200, n_jobs = 4)
kde = TwoClassKDE()
balloon = balloon.BalloonNominate(lamb = 0.5, deflate = False)
predictors = [logreg, gnb, randfor, kde, balloon]
predictor_names = ['logreg', 'gnb', 'randfor', 'kde', 'balloon']
num_preds = len(predictors)

# sample data from Gaussian mixture model
c = 4.0
sigma = 1.0
N = 10
mu = np.array([[-c, c], [c, c], [c, -c], [-c, -c]])
#mu = np.sqrt(2) * np.array([[0, c], [c, 0], [0, -c], [-c, 0]])
samps = np.zeros((4 * N, 2), dtype = float)
#labels = np.array([1] * N + [0] * N + [1] * N + [0] * N)  # alternating
labels = np.array([0] * (2 * N) + [1] * (2 * N))  # same sides
colors = np.array([get_color(label) for label in labels])
for i in range(4):
    for j in range(N):
        samps[i * N + j] = mu[i] + sigma * np.random.randn(2)

# make heatmaps for each predictor
k = 101
bnd = 1.2 * np.abs(samps).max()
x = np.linspace(-bnd, bnd, k)
X, Y = np.meshgrid(x, x)
Z = np.array([X, Y]).swapaxes(0, 2).reshape((k ** 2, 2))

# plot points overlaid on heatmap
def plot_predictor(pred_name, scaler = prob_scale):
    plt.figure()
    pred = predictors[predictor_names.index(pred_name)]
    pred.fit(samps, labels)
    heatmap = np.zeros((k, k, 3), dtype = float)
    scores = pred.predict_proba(Z)[:, 1]
    scores = scores.reshape((k, k)).transpose()
    scores = scaler(scores)
    for i in range(k):
        for j in range(k):
            heatmap[i, j] = get_color(scores[i, j])
    plt.scatter(X.flatten(), Y.flatten(), c = heatmap.reshape((k ** 2, 3)), alpha = 0.3, linewidth = 0, marker = 's', s = 100)
    plt.scatter(samps[:, 0], samps[:, 1], c = colors, s = 100, linewidth = 2)
    plt.xlim((-bnd, bnd))
    plt.ylim((-bnd, bnd))
    plt.show(block = False)




