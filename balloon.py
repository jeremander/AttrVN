"""Balloon nomination algorithm: For each point x let f_r(x) = lambda \sum_{i=1}^s+ I[|x - x+_i| \leq r] - (1 - lambda) \sum_{i=1}^s- I[|x - x-_i| \leq r], where x+ and x- are positive and negative seeds (s+ and s- of each). Then we may score the x's in this way for varying r, moving either large to small (deflation) or small to large (inflation), breaking ties accordingly."""

import numpy as np
from collections import defaultdict
from functools import reduce
import unittest
import pandas as pd
import logging
from logging import debug
logging.basicConfig(level = logging.CRITICAL, format = '%(message)s')


def flatten1(blocks):
    """Flattens a list of lists by one level."""
    return reduce(lambda x, y : x + y, blocks, [])

def block_sort(items, key, reverse):
    """Given a list of items, sorts them, but groups them by tied items."""
    d = defaultdict(list)
    for item in items:
        d[key(item)].append(item)
    return [d[k] for k in sorted(d.keys(), reverse = reverse)]


class BalloonNominate():
    def __init__(self, lamb, deflate = False):
        """Initialize with positive sample weighting coefficient (in [0, 1])."""
        assert (0 <= lamb <= 1)
        self.lamb = lamb
        self.deflate = deflate
    def fit(self, seeds, labels):
        """Saves seeds and their pos/neg labels, along with relevant counts."""
        assert set(labels).issubset({True, False})
        self.m = seeds.shape[1]
        self.seeds = seeds
        self.labels = (labels == 1)
        self.pos_seeds = self.seeds[self.labels]
        self.neg_seeds = self.seeds[~self.labels]
        self.s_plus = len(self.pos_seeds)
        self.s_minus = len(self.neg_seeds)
        self.s = self.s_plus + self.s_minus
    def get_dists(self, X):
        """Given n points X, computes distances from all the s seed points. Returns (n x s) matrix of distances between points and seeds."""
        assert (X.shape[1] == self.m)
        n = len(X)
        dists = np.zeros((n, self.s), dtype = float)
        for i in range(n):
            for j in range(self.s_plus):
                dists[i, j] = np.square(X[i] - self.pos_seeds[j]).sum()
            for j in range(self.s_minus):
                dists[i, self.s_plus + j] = np.square(X[i] - self.neg_seeds[j]).sum()
        return dists
    def get_row_sorted_dists(self, X):
        """Given n points, computes distances from all the s seed points. Returns (n x s) matrix of distances, where each row is sorted in increasing order, as well as a dictionary mapping each unique distance to dictionary from point indices to vector [num_pos, num_neg] of seeds having that distance from the point."""
        assert (X.shape[1] == self.m)
        n = len(X)
        dist_dict = defaultdict(dict)
        sorted_dists_by_row = np.zeros((n, self.s), dtype = float)
        for i in range(n):
            for j in range(self.s_plus):
                dist = np.square(X[i] - self.pos_seeds[j]).sum()  # use squared dist to save computation time
                #dist = np.linalg.norm(X[i] - self.pos_seeds[j])
                sorted_dists_by_row[i, j] = dist
                if (i in dist_dict[dist]):
                    dist_dict[dist][i][0] += 1
                else:
                    dist_dict[dist][i] = [1, 0]
            for j in range(self.s_minus):
                dist = np.square(X[i] - self.neg_seeds[j]).sum()
                #dist = np.linalg.norm(X[i] - self.neg_seeds[j])
                sorted_dists_by_row[i, self.s_plus + j] = dist
                if (i in dist_dict[dist]):
                    dist_dict[dist][i][1] += 1
                else:
                    dist_dict[dist][i] = [0, 1]
            sorted_dists_by_row[i] = sorted(sorted_dists_by_row[i])
        sorted_dists = np.array(sorted(dist_dict.keys()))
        return (sorted_dists, sorted_dists_by_row, dist_dict)
    def get_closest_points_to_seeds(self, X, N):
        """Returns N closest points to the set of positive seeds and N closest points to the set of negative seeds, sorted in order of increasing distance."""
        assert (X.shape[1] == self.m)
        n = len(X)
        dist_dict = defaultdict(list)
        sorted_dists_by_col = np.zeros((n, self.s), dtype = float)
        for j in range(self.s_plus):
            for i in range(n):
                dist = np.square(X[i] - self.pos_seeds[j]).sum()  # use squared dist to save computation time
                sorted_dists_by_col[i, j] = dist
                dist_dict[dist].append((i, j))
            sorted_dists_by_col[:, j] = sorted(sorted_dists_by_col[:, j])
        for j in range(self.s_minus):
            for i in range(n):
                dist = np.square(X[i] - self.neg_seeds[j]).sum()
                sorted_dists_by_col[i, self.s_plus + j] = dist
                dist_dict[dist].append((i, self.s_plus + j))
            sorted_dists_by_col[:, self.s_plus + j] = sorted(sorted_dists_by_col[:, self.s_plus + j])
        pos_close, neg_close = [], []
        pos_close_set, neg_close_set = set(), set()
        num_pos_close, num_neg_close = 0, 0
        for dist in sorted(dist_dict.keys()):
            if (num_pos_close == num_neg_close == N):
                break
            pts = dist_dict[dist]
            for (i, j) in pts:
                if (j < self.s_plus):
                    if ((num_pos_close < N) and (i not in pos_close_set)):
                        pos_close.append(i)
                        pos_close_set.add(i)
                        num_pos_close += 1
                else:
                    if ((num_neg_close < N) and (i not in neg_close_set)):
                        neg_close.append(i)
                        neg_close_set.add(i)
                        num_neg_close += 1
        return (pos_close, neg_close)
    def purity_set(self, X, alpha = 1.0):
        """Returns set of points that are all within some distance d of at least one positive seed and not within d of any negative seed (we call these points d-pure). Determines the best distance d as the one maximizing the size of this set. The alpha parameter is a factor to multiply by d when considering distances from negative seeds (lower alpha -> more permissive about negative seeds -> bigger purity set)."""
        n = X.shape[0]
        (pos, neg) = self.get_closest_points_to_seeds(X, n)
        dists = self.get_dists(X)  # doing redundant work here
        sorted_dists = sorted(set(dists.reshape(self.s * n)))
        pos_set, neg_set = set(), set()
        intersection_sizes = []
        i, j = 0, 0
        for d in sorted_dists:
            while ((i < len(pos)) and (dists[pos[i], :self.s_plus].min() <= d)):
                if (pos[i] not in neg_set):
                    pos_set.add(pos[i])
                i += 1
            while ((j < len(neg)) and (dists[neg[j], self.s_plus:].min() <= d * alpha)):
                if (neg[j] in pos_set):
                    pos_set.remove(neg[j])
                else:
                    neg_set.add(neg[j])
                j += 1
            size = len(pos_set)
            intersection_sizes.append(size)
            if ((i == len(pos)) and (j == len(neg))):
                break
        best_set_size = max(intersection_sizes)
        best_dist = sorted_dists[intersection_sizes.index(best_set_size)]
        best_indices = []
        for p in pos:
            if (dists[p, self.s_plus:].min() > best_dist * alpha):
                best_indices.append(p)
            if (len(best_indices) >= best_set_size):
                break
        assert(best_set_size == len(best_indices))
        return (best_dist, intersection_sizes, best_indices)
    def predict_proba(self, X):
        """Given n x m matrix X, returns rank-derived scores for the n data points, using either the inflation or deflation method."""
        (sorted_dists, sorted_dists_by_row, dist_dict) = self.get_row_sorted_dists(X)  # get distances by node
        if self.deflate:  # use descending order of distances if doing deflation
            sorted_dists = np.flipud(sorted_dists)
            sorted_dists_by_row = np.fliplr(sorted_dists_by_row)
        n = len(X)
        debug(dist_dict)
        debug(sorted_dists)
        debug(sorted_dists_by_row)
        min_dist, max_dist = sorted_dists[0], sorted_dists[-1]
        if self.deflate:
            scores = (self.lamb * self.s_plus - (1 - self.lamb) * self.s_minus) * np.ones(n, dtype = float)
        else:
            scores = np.zeros(n, dtype = float)
        ranked_list = np.array(range(n))  # ranking of the nodes
        cluster_sizes = [n]  # cluster is a rank equivalence class
        cluster_starts = [0]  # index in ranked_list where each cluster begins
        nonsingletons = set(range(n))  # nodes not yet in a singleton cluster
        cluster_by_row = np.zeros(n, dtype = int)
        ind_by_row = np.zeros(n, dtype = int)   # index of current dist in sorted_dists_by_row
        next_dist_by_row = sorted_dists_by_row[:, 0]  # array of next distance for each node to change its score
        done_marker = -np.inf if self.deflate else np.inf
        while (len(cluster_sizes) < n):
            new_cluster_sizes = defaultdict(list)  # maps each old cluster index to list of new subcluster sizes
            next_dist = next_dist_by_row.max() if self.deflate else next_dist_by_row.min()
            if np.isinf(next_dist):  # +/-inf signifies no more nodes will change
                break
            rows_with_next_dist = set(np.nonzero(next_dist_by_row == next_dist)[0]).intersection(nonsingletons)
            debug("nonsingletons: %s" % str(nonsingletons))
            debug("cluster_by_row: %s" % str(cluster_by_row))
            debug("cluster sizes: %s" % str(cluster_sizes))
            debug("cluster starts: %s" % str(cluster_starts))
            debug("next_dist = %f" % next_dist)
            debug("rows: %s" % str(rows_with_next_dist))
            for i in rows_with_next_dist:  # update scores of nodes with the given distance
                diff = self.lamb * dist_dict[next_dist][i][0] - (1 - self.lamb) * dist_dict[next_dist][i][1]
                if self.deflate:
                    scores[i] -= diff
                else:
                    scores[i] += diff
                ind = ind_by_row[i] + 1
                while ((ind < self.s) and (sorted_dists_by_row[i, ind] == next_dist)):
                    ind += 1
                ind_by_row[i] = ind  # increment node's index in sorted dist array until it's a new distance
                next_dist_by_row[i] = sorted_dists_by_row[i, ind] if (ind < self.s) else done_marker
                debug("\ti = %d" % i)
                debug("\tscores: %s" % str(scores))
                debug("\tind_by_row: %s" % str(ind_by_row))
                debug("\tnext_dist_by_row: %s" % str(next_dist_by_row))
            clusters_to_split = {cluster_by_row[i] for i in rows_with_next_dist}
            # debug("clusters_to_split: %s" % str(clusters_to_split))
            for k in clusters_to_split:  # split clusters containing altered members
                start, end = cluster_starts[k], cluster_starts[k] + cluster_sizes[k]
                clust = ranked_list[start : end]
                # can do this more efficiently by only considering nodes that changed
                items = block_sort([(i, scores[i]) for i in clust], key = lambda pair : pair[1], reverse = True)
                debug("\tk = %d" % k)
                debug("\tclust: %s" % clust)
                debug("\titems: %s" % items)
                debug("\tranked list (before): %s" % ranked_list)
                for newclust in items:
                    new_cluster_sizes[k].append(len(newclust))
                    if (len(newclust) == 1):
                        i = newclust[0][0]
                        nonsingletons.remove(i)
                        next_dist_by_row[i] = done_marker  # this node is a singleton, so no longer considered
                ranked_list[start : end] = [pair[0] for pair in flatten1(items)]
                debug("\tranked list (after): %s" % ranked_list)
            debug("\tnew cluster sizes: %s" % str(new_cluster_sizes))
            for k in clusters_to_split:  # relabel the clusters
                ctr = 0
                for (j, size) in enumerate(new_cluster_sizes[k]):
                    debug("(j, size): %s" % str((j, size)))
                    if (j == 0):
                        cluster_sizes[k] = size 
                    else:
                        cluster_sizes.append(size)
                        cluster_starts.append(cluster_starts[k] + ctr)
                        for i in range(cluster_starts[k] + ctr, cluster_starts[k] + ctr + size):
                            cluster_by_row[ranked_list[i]] = len(cluster_sizes) - 1
                    ctr += size
        debug(cluster_sizes)
        debug(ranked_list)
        debug(cluster_starts)
        debug(cluster_by_row)
        num_clusters = len(cluster_sizes)
        scores = np.zeros((n, 2))
        cluster_ranks = [pair[0] for pair in sorted(enumerate(cluster_starts), key = lambda pair : pair[1])]
        ranks_by_cluster = {k : j for (j, k) in enumerate(cluster_ranks)}
        for i in range(n):
            scores[i, 0] = ranks_by_cluster[cluster_by_row[i]] / (num_clusters - 1.0)
        scores[:, 1] = 1.0 - scores[:, 0]
        return scores


class TestBalloon(unittest.TestCase):
    def get_test_data(self, testnum):
        np.random.seed(0)
        if (testnum == 1):  # small example
            seeds = np.random.randn(5, 50)
            labels = np.random.randint(0, 2, 5)
            X = np.random.randn(10, 50)
            true_ranked_list = [6, 0, 8, 3, 5, 4, 2, 7, 9, 1]
        elif (testnum == 2):  # bigger example
            seeds = np.random.randn(20, 50)
            labels = np.random.randint(0, 2, 20)
            X = np.random.randn(200, 50)     
            true_ranked_list = [92, 151, 94, 172, 15, 118, 82, 154, 52, 84, 137, 27, 196, 192, 26, 39, 116, 173, 189, 139, 185, 104, 11, 70, 66, 7, 132, 133, 6, 165, 9, 169, 58, 119, 55, 144, 63, 153, 179, 152, 65, 38, 73, 97, 64, 95, 180, 199, 156, 162, 174, 109, 21, 134, 43, 13, 44, 188, 120, 79, 25, 187, 2, 103, 18, 77, 69, 16, 121, 122, 195, 30, 35, 181, 68, 145, 45, 1, 186, 47, 170, 59, 57, 83, 88, 85, 76, 48, 62, 67, 175, 127, 125, 81, 53, 112, 0, 107, 75, 191, 71, 105, 113, 86, 12, 102, 111, 80, 160, 49, 22, 42, 50, 143, 147, 128, 166, 61, 108, 135, 193, 41, 184, 161, 89, 177, 74, 114, 115, 3, 99, 93, 194, 14, 163, 5, 198, 131, 130, 37, 159, 164, 124, 168, 8, 126, 110, 51, 176, 29, 10, 96, 171, 32, 54, 190, 33, 183, 19, 106, 148, 46, 149, 72, 155, 23, 60, 100, 178, 28, 182, 56, 20, 90, 40, 24, 150, 87, 138, 141, 98, 91, 197, 78, 101, 146, 142, 36, 167, 123, 129, 140, 136, 34, 4, 117, 17, 158, 31, 157]   
        elif (testnum == 3):  # some equidistant points
            seeds = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype = float)
            labels = np.array([1, 1, 0, 0])
            X = np.array([[0, 0], [-2, 0], [2, 0], [0, 2], [0, -2]], dtype = float) 
            true_ranked_list = [1, 2, 0, 3, 4]
        elif (testnum == 4):  # points equal seeds
            seeds = np.random.randn(5, 50)
            labels = np.random.randint(0, 2, 5)
            X = np.vstack([seeds[0], np.random.randn(9, 50)])
            true_ranked_list = [0, 7, 1, 9, 4, 6, 5, 3, 8, 2]
        elif (testnum == 5):  # points are on a small lattice
            seeds = np.random.randint(-4, 5, (5, 2))
            labels = np.random.randint(0, 2, 5)
            X = np.random.randint(-4, 5, (10, 2))
            true_ranked_list = [0, 5, 1, 2, 8, 3, 7, 4, 9, 6]
        return (seeds, labels, X, true_ranked_list)
    def test_correctness(self, testnum, lamb = 0.5, deflate = False):
        (seeds, labels, X, true_ranked_list) = self.get_test_data(testnum)
        bln = BalloonNominate(lamb, deflate = deflate)
        bln.fit(seeds, labels)
        scores = bln.predict_proba(X)[:, 1]
        ranked_list = [pair[0] for pair in sorted(enumerate(scores), key = lambda pair : pair[1], reverse = True)]
        print("\nComputed ranking:")
        print(ranked_list)
        print("\nTrue ranking:")
        print(true_ranked_list)
        print("\n" + str(ranked_list == true_ranked_list))
        self.assertTrue(ranked_list == true_ranked_list)
    def test_time(self):
        np.random.seed(0)
        seeds = np.random.randn(100, 50)
        labels = np.random.randint(0, 2, 100)
        X = np.random.randn(1000, 50)
        infl = BalloonNominate(0.5, deflate = False)
        infl.fit(seeds, labels)
        scores = infl.predict_proba(X)[:, 1]
        ranked_list = [pair[0] for pair in sorted(enumerate(scores), key = lambda pair : pair[1], reverse = True)]                



# np.random.seed(0)
# seeds = np.random.randn(4, 2)
# labels = np.array([0, 0, 1, 1])
# X = np.random.randn(6, 2)
# infl = BalloonNominate(0.5, deflate = False)
# infl.fit(seeds, labels)
# defl = BalloonNominate(0.5, deflate = True)
# defl.fit(seeds, labels)
# import matplotlib.pyplot as plt 
# plt.scatter(seeds[:, 0], seeds[:, 1], s = 100, c = labels)
# plt.scatter(X[:, 0], X[:, 1], s = 100, c = 'black')

path = 'benchmark/2class/'
def get_data(name, s):
    df = pd.read_csv(path + '%s.csv' % name, header = None)
    inds = np.random.choice(len(df), s, replace = True)
    seeds = np.asarray(df[[0, 1]])[inds]
    seeds -= seeds.mean(axis = 0)
    labels = np.asarray(df[2])[inds]
    return (seeds, labels)

def get_mesh(data, k):
    bnd = 1.2 * np.abs(data).max()
    x = np.linspace(-bnd, bnd, k)
    X, Y = np.meshgrid(x, x)
    Z = np.array([X, Y]).swapaxes(0, 2).reshape((k ** 2, 2))
    return Z

import os
names = [name.split('.')[0] for name in os.listdir(path)]
name = 'square1'
s = 50
k = 51
(seeds, labels) = get_data(name, s)
Z = get_mesh(seeds, k)
bln = BalloonNominate(0.5)
bln.fit(seeds, labels)



