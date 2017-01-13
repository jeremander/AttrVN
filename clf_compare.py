from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from kde import TwoClassKDE
#from balloon import BalloonNominate
import balloon
import numpy as np
import matplotlib.pyplot as plt
import optparse
import pandas as pd
import os


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
    """Scales/equalizes the below-0.5 and above-0.5 histograms."""
    flags = img >= 0.5
    return ~flags * 0.5 * img_histeq(~flags * img, nquant) + flags * 0.5 * (1 + img_histeq(flags * img, nquant))

def histeq_scale(nquant = 50):
    return (lambda img : img_histeq(img, nquant))

def histeq_prob_scale(nquant = 50):
    return (lambda img : img_histeq_prob(img, nquant))

def rank_scale(img):
    """Scales in [0, 1] by rank."""
    vals = sorted(set(img.ravel()))
    ranks_by_val = {val : float(i) for (i, val) in enumerate(vals)}
    img2 = np.vectorize(lambda val : ranks_by_val[val])(img)
    img2 /= img2.max()
    return img2

def get_color(x):
    return x * np.array([1., 0., 0.]) + (1 - x) * np.array([0., 0., 1.])


#predictor_names = ['logreg', 'gnb', 'randfor', 'kde']
predictor_names = ['logreg', 'gnb', 'randfor', 'kde1.0', 'infl0.5', 'defl0.5']
#predictor_names = ['infl0.0', 'infl0.5', 'infl1.0', 'defl0.0', 'defl0.5', 'defl1.0']
#predictor_names = ['kde0.001', 'kde0.01', 'kde0.1', 'kde1.0', 'kde10.0', 'kde100.0']
num_preds = len(predictor_names)

def predictor_by_name(name, n_jobs = 4):
    if (name == 'logreg'):
        return LogisticRegression()
    elif (name == 'gnb'):
        return GaussianNB()
    elif (name.startswith('randfor')):
        ntrees = int(name[7:]) if (len(name) > 7) else 200
        return RandomForestClassifier(n_estimators = ntrees, n_jobs = n_jobs)
    elif (name.startswith('kde')):
        bandwidth = float(name[3:]) if (len(name) > 3) else None
        return TwoClassKDE() if (bandwidth is None) else TwoClassKDE(bandwidth = bandwidth)
    elif (name.startswith('infl')):
        lamb = float(name[4:]) if (len(name) > 4) else 0.5
        return balloon.BalloonNominate(lamb = lamb, deflate = False)
    elif (name.startswith('defl')):
        lamb = float(name[4:]) if (len(name) > 4) else 0.5
        return balloon.BalloonNominate(lamb = lamb, deflate = True)
    else:
        raise ValueError("Invalid predictor name: %s" % name)

#scaler = histeq_scale(500)
scaler = rank_scale

# sample data from Gaussian mixture model
# c = 4.0
# sigma = 1.0
# N = 10
# mu = np.array([[-c, c], [c, c], [c, -c], [-c, -c]])
# #mu = np.sqrt(2) * np.array([[0, c], [c, 0], [0, -c], [-c, 0]])
# samps = np.zeros((4 * N, 2), dtype = float)
# #labels = np.array([1] * N + [0] * N + [1] * N + [0] * N)  # alternating
# labels = np.random.rand(4 * N) >= 0.5  # random
# #labels = np.array([0] * (2 * N) + [1] * (2 * N))  # same sides
# colors = np.array([get_color(label) for label in labels])
# for i in range(4):
#     for j in range(N):
#         samps[i * N + j] = mu[i] + sigma * np.random.randn(2)


def main():
    p = optparse.OptionParser()
    p.add_option('--dataset', '-d', type = str, help = 'dataset')
    p.add_option('-n', type = int, default = 50, help = 'number of sample points')
    p.add_option('-g', type = float, default = 0.0, help = 'garble rate of labels')
    p.add_option('-k', type = int, default = 51, help = 'number of grid points per dimension')
    p.add_option('-j', type = int, default = 4, help = 'number of threads')
    p.add_option('--seed', type = int, default = None, help = 'RNG seed')
    opts, args = p.parse_args()

    seed = np.random.randint(0, 2 ** 32) if (opts.seed is None) else opts.seed
    np.random.seed(seed)

    df = pd.read_csv('benchmark/2class/%s.csv' % opts.dataset, header = None)
    n = opts.n
    inds = np.random.choice(len(df), n, replace = True)
    seeds = np.asarray(df[[0, 1]])[inds]
    seeds -= seeds.mean(axis = 0)
    labels = np.asarray(df[2])[inds] ^ (np.random.rand(n) <= opts.g)
    colors = np.array([get_color(label) for label in labels])

    k = opts.k
    bnd = 1.2 * np.abs(seeds).max()
    x = np.linspace(-bnd, bnd, k)
    X, Y = np.meshgrid(x, x)
    Z = np.array([X, Y]).swapaxes(0, 2).reshape((k ** 2, 2))

    ncols = 3
    nrows = int(np.ceil(num_preds / ncols))

    fig, axes = plt.subplots(nrows, ncols, sharex = 'col', sharey = 'row', figsize = (12, 8), facecolor = 'white')
    axes = axes.reshape(nrows * ncols)
    for (ax, name) in zip(axes, predictor_names):
        print(name)
        pred = predictor_by_name(name, n_jobs = opts.j)
        pred.fit(seeds, labels)
        heatmap = np.zeros((k, k, 3), dtype = float)
        scores = pred.predict_proba(Z)[:, 1]
        scores = scores.reshape((k, k)).transpose()
        scores = scaler(scores)
        for i in range(k):
            for j in range(k):
                heatmap[i, j] = get_color(scores[i, j])
        ax.scatter(X.flatten(), Y.flatten(), c = heatmap.reshape((k ** 2, 3)), alpha = 0.3, linewidth = 0, marker = 's', s = 100)
        ax.scatter(seeds[:, 0], seeds[:, 1], c = colors, s = 100, linewidth = 2)
        ax.set_xlim((-bnd, bnd))
        ax.set_ylim((-bnd, bnd))
        ax.set_title(name)
    plotnames = set(os.listdir('benchmark/plots'))
    ctr = 0
    path = '%s_compare_g%s_%d_%d.png' % (opts.dataset, str(opts.g), seed, ctr)
    while (path in plotnames):
        ctr += 1
        path = '%s_compare_g%s_%d_%d.png' % (opts.dataset, str(opts.g), seed, ctr)
    plt.savefig('benchmark/plots/%s' % path)
    #plt.show()

if __name__ == "__main__":
    main()




