import numpy as np
import pandas as pd
import balloon
import optparse
import matplotlib.pyplot as plt
import os
from rstyle import *

def rank_scale(img):
    """Scales in [0, 1] by rank."""
    vals = sorted(set(img.ravel()))
    ranks_by_val = {val : float(i) for (i, val) in enumerate(vals)}
    img2 = np.vectorize(lambda val : ranks_by_val[val])(img)
    img2 /= img2.max()
    return img2

def get_color(x):
    return x * np.array([1., 0., 0.]) + (1 - x) * np.array([0., 0., 1.])


def main():
    p = optparse.OptionParser()
    p.add_option('--dataset', '-d', type = str, help = 'dataset')
    p.add_option('-n', type = int, default = 50, help = 'number of sample points')
    p.add_option('-g', type = float, default = 0.0, help = 'garble rate of labels')
    p.add_option('-k', type = int, default = 51, help = 'number of grid points per dimension')
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
    bnd = 1.25 * np.abs(seeds).max()
    x = np.linspace(-bnd, bnd, k)
    X, Y = np.meshgrid(x, x)
    Z = np.array([X, Y]).swapaxes(0, 2).reshape((k ** 2, 2))

    bln = balloon.BalloonNominate(0.5)
    bln.fit(seeds, labels)

    ncols = 3
    alpha_factor = 4.0
    fig, axes = plt.subplots(1, ncols, figsize = (26, 8), facecolor = 'white')
    for (axnum, ax) in enumerate(axes):
        alpha = alpha_factor ** ((ncols // 2) - axnum)
        (best_dist, intersection_sizes, best_indices) = bln.purity_set(Z, alpha = alpha)
        best_set_size = len(best_indices)
        scores = np.linspace(1, 0.6, best_set_size)
        best_colors = [get_color(score) for score in scores]
        best_pts = Z[best_indices]
        ax.scatter(best_pts[:, 0], best_pts[:, 1], c = best_colors, alpha = 0.5, linewidth = 0, marker = 's', s = 100)
        ax.scatter(seeds[:, 0], seeds[:, 1], c = colors, s = 100, linewidth = 2)
        ax.set_xlim((-bnd, bnd))
        ax.set_ylim((-bnd, bnd))
        ax.set_title('alpha = %s\nd = %s\n# pure points = %d' % (str(alpha), str(np.sqrt(best_dist)), best_set_size))
        ax.set_aspect('equal')
    plt.suptitle(opts.dataset, fontsize = 16, fontweight = 'bold')
    plt.subplots_adjust(top = 0.85, bottom = 0.1, left = 0.05, right = 0.95)


    plotnames = set(os.listdir('benchmark/plots'))
    ctr = 0
    path = '%s_purity_g%s_%d_%d.png' % (opts.dataset, str(opts.g), seed, ctr)
    while (path in plotnames):
        ctr += 1
        path = '%s_purity_g%s_%d_%d.png' % (opts.dataset, str(opts.g), seed, ctr)
    plt.savefig('benchmark/plots/%s' % path)
    #plt.show()

if __name__ == "__main__":
    main()

