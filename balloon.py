"""Balloon nomination algorithm: For each point x let f_r(x) = lambda \sum_{i=1}^s+ I[|x - x+_i| \leq r] - (1 - lambda) \sum_{i=1}^s- I[|x - x-_i| \leq r], where x+ and x- are positive and negative seeds (s+ and s- of each). Then we may score the x's in this way for varying r, moving either large to small or small to large, breaking ties accordingly."""

import numpy as np
from collections import defaultdict
from functools import reduce

def flatten1(blocks):
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
        assert set(labels).issubset({True, False})
        self.m = seeds.shape[1]
        self.seeds = seeds
        self.labels = (labels == 1)
        self.pos_seeds = self.seeds[self.labels]
        self.neg_seeds = self.seeds[~self.labels]
        self.s_plus = len(self.pos_seeds)
        self.s_minus = len(self.neg_seeds)
        self.s = self.s_plus + self.s_minus
    def predict_proba(self, X):
        assert (X.shape[1] == self.m)
        n = len(X)
        dist_dict = defaultdict(dict)
        #dists_plus = np.zeros((n, self.s_plus), dtype = float)
        #dists_minus = np.zeros((n, self.s_minus), dtype = float)
        for i in range(n):
            for j in range(self.s_plus):
                dist = np.linalg.norm(X[i] - self.pos_seeds[j])
                #dists_plus[i, j] = dist
                if (i in dist_dict[dist]):
                    dist_dict[dist][i][0] += 1
                else:
                    dist_dict[dist][i] = [1, 0]
            for j in range(self.s_minus):
                dist = np.linalg.norm(X[i] - self.neg_seeds[j])
                #dists_minus[i, j] = dist
                if (i in dist_dict[dist]):
                    dist_dict[dist][i][1] += 1
                else:
                    dist_dict[dist][i] = [0, 1]
        #print(dist_dict)
        sorted_dists = sorted(dist_dict.keys())
        #print(sorted_dists)
        min_dist, max_dist = sorted_dists[0], sorted_dists[-1]
        if (not self.deflate):
            scores = np.zeros(n, dtype = float)
            ranked_list = np.array(range(n))
            cluster_sizes = [n]
            cluster_size_cumsum = np.concatenate([[0], np.cumsum(cluster_sizes)])
            for r in sorted_dists:  # could improve efficiency still
                # print("\nr = %f" % r)
                # print("cluster sizes: %s" % str(cluster_sizes))
                # print("cumsums:       %s" % str(cluster_size_cumsum))
                new_cluster_sizes = [[size] for size in cluster_sizes]
                for k in range(len(cluster_sizes)):
                    start, end = cluster_size_cumsum[k], cluster_size_cumsum[k + 1]
                    clust = ranked_list[start : end]
                    # print("\n\tk = %d" % k)
                    # print("\tstart = %d, end = %d" % (start, end))
                    # print("\tclust = %s" % clust)
                    if (len(clust) > 1):
                        for i in clust:
                            if (i in dist_dict[r]):
                                scores[i] += self.lamb * dist_dict[r][i][0] - (1 - self.lamb) * dist_dict[r][i][1]
                                #print("\tscores: %s" % scores)
                        items = block_sort([(i, scores[i]) for i in clust], key = lambda pair : pair[1], reverse = True)
                        new_cluster_sizes[k] = [len(newclust) for newclust in items]
                        ranked_list[start : end] = [pair[0] for pair in flatten1(items)]
                        #print("\tranked list: %s" % ranked_list)
                cluster_sizes = flatten1(new_cluster_sizes)
                cluster_size_cumsum = np.concatenate([[0], np.cumsum(cluster_sizes)])
                if (len(cluster_sizes) == n):
                    break
        num_clusters = len(cluster_sizes)
        scores = np.zeros((n, 2))
        for k in range(num_clusters):
            for j in range(cluster_size_cumsum[k], cluster_size_cumsum[k + 1]):
                scores[ranked_list[j], 0] = k / (num_clusters - 1.0)
        scores[:, 1] = 1.0 - scores[:, 0]
        return scores




