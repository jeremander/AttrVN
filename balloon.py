"""Balloon nomination algorithm: For each point x let f_r(x) = lambda \sum_{i=1}^s+ I[|x - x+_i| \leq r] - (1 - lambda) \sum_{i=1}^s- I[|x - x-_i| \leq r], where x+ and x- are positive and negative seeds (s+ and s- of each). Then we may score the x's in this way for varying r, moving either large to small (deflation) or small to large (inflation), breaking ties accordingly."""

import numpy as np
from collections import defaultdict
from functools import reduce, cmp_to_key
import unittest
import pandas as pd
import logging
import time
import random
from heap import *
from ctypes import *
from logging import debug
from copy import deepcopy
logging.basicConfig(level = logging.CRITICAL, format = '%(message)s')

libbln_filename = "balloon/cballoon/balloon.so"
def arr2D_to_ptr(arr):
    return (arr.__array_interface__['data'][0]  + np.arange(arr.shape[0]) * arr.strides[0]).astype(np.uintp)

def compute_squared_distances(points, seeds):
    libbln = CDLL(libbln_filename)
    _doublepp = np.ctypeslib.ndpointer(dtype = np.uintp, ndim = 1, flags = 'C') 
    libbln.py_compute_squared_distances.argtypes = [c_long, c_long, c_long, _doublepp, _doublepp, _doublepp]
    (n, dim) = points.shape
    (s, dim2) = seeds.shape
    assert (dim == dim2)
    points = np.asarray(points, dtype = float)
    seeds = np.asarray(seeds, dtype = float)
    D = np.zeros((n, s), dtype = float)
    libbln.py_compute_squared_distances(n, s, dim, arr2D_to_ptr(points), arr2D_to_ptr(seeds), arr2D_to_ptr(D))
    return D

def rank_from_distances(D, labels, deflate = False):
    libbln = CDLL(libbln_filename)
    _longp = np.ctypeslib.ndpointer(dtype = int, ndim = 1)
    _doublep = np.ctypeslib.ndpointer(dtype = float, ndim = 1)
    _doublepp = np.ctypeslib.ndpointer(dtype = np.uintp, ndim = 1, flags = 'C') 
    libbln.py_balloon_rank_from_distances.argtypes = [c_long, c_long, _doublepp, _doublep, _longp, c_bool]
    (n, s) = D.shape
    assert (labels.shape == (s,))
    D = np.asarray(D, dtype = float)
    ranks = np.zeros(n, dtype = int)
    D_copy = deepcopy(D)
    libbln.py_balloon_rank_from_distances(n, s, arr2D_to_ptr(D_copy), labels, ranks, deflate)
    return ranks   

def balloon_rank(points, seeds, labels, deflate = False):
    libbln = CDLL(libbln_filename)
    _longp = np.ctypeslib.ndpointer(dtype = int, ndim = 1)
    _doublep = np.ctypeslib.ndpointer(dtype = float, ndim = 1)
    _doublepp = np.ctypeslib.ndpointer(dtype = np.uintp, ndim = 1, flags = 'C') 
    libbln.py_balloon_rank.argtypes = [c_long, c_long, c_long, _doublepp, _doublepp, _doublep, _longp, c_bool]
    (n, dim) = points.shape
    (s, dim2) = seeds.shape
    assert (dim == dim2)
    points = np.asarray(points, dtype = float)
    seeds = np.asarray(seeds, dtype = float)
    labels = np.asarray(labels, dtype = float)
    ranks = np.zeros(n, dtype = int)
    libbln.py_balloon_rank(n, s, dim, arr2D_to_ptr(points), arr2D_to_ptr(seeds), labels, ranks, deflate)
    return ranks   

def block_sort(items, key, reverse):
    """Given a list of items, sorts them, but groups them by tied items."""
    d = defaultdict(list)
    for item in items:
        d[key(item)].append(item)
    return [d[k] for k in sorted(d.keys(), reverse = reverse)]

def arrange_by_index(M, I):
    M2 = M.copy()
    for i in range(M.shape[0]):
        M2[i] = M[i, I[i]]
    return M2

class RankHierarchicalClustering():
    """Class representing a set of indices associated with scores, clustered by equal score. When scores change, clusters are broken hierarchically (i.e. an index can change ranking with respect to its other cluster members, but not with respect to nonmembers. Breaking clusters ranks the new subclusters by decreasing score."""
    def __init__(self, n, score0):
        """Initialize all n scores to a constant (one cluster)."""
        self.n = n
        self.scores = score0 * np.ones(self.n, dtype = float)
        self.ranked_indices = np.array(range(self.n))
        self.ranks_by_index = np.array(range(self.n))
        self.num_clusters = 1
        self.cluster_sizes = [n]
        self.cluster_starts = [0]  # starting points of clusters in ranked index list
        self.cluster_scores = [score0]
        self.clusters_by_index = np.zeros(self.n, dtype = int)
        self.is_single = np.zeros(self.n, dtype = bool) if (self.n > 1) else np.array([True])
    def change_score(self, i, new_score):
        """Changes the score of index i, and readjusts the clustering accordingly. Returns list of indices that are new singletons."""
        new_singletons = []
        k = self.clusters_by_index[i]
        if (self.is_single[i]):
            self.cluster_scores[k] = new_score
        else:
            if (new_score != self.scores[i]):
                is_less = new_score < self.scores[i]
                rank = self.ranks_by_index[i]
                self.is_single[i] = True
                new_singletons.append(i)
                size, start = self.cluster_sizes[k], self.cluster_starts[k]
                new_rank = start + size - 1 if is_less else start
                j = self.ranked_indices[new_rank]
                self.ranked_indices[new_rank] = i
                self.ranked_indices[rank] = j
                self.ranks_by_index[i] = new_rank
                self.ranks_by_index[j] = rank
                self.cluster_sizes.append(1)
                self.cluster_sizes[k] -= 1
                self.cluster_starts.append(new_rank)
                self.cluster_scores.append(new_score)
                self.clusters_by_index[i] = self.num_clusters 
                self.num_clusters += 1
                if (self.cluster_starts[k] == new_rank):
                    self.cluster_starts[k] += 1
                if (self.cluster_sizes[k] == 1):
                    i2 = self.ranked_indices[self.cluster_starts[k]]
                    self.is_single[i2] = True
                    new_singletons.append(i2)
        self.scores[i] = new_score
        return new_singletons
    def change_scores(self, indices, new_scores):
        """Changes the scores corresponding to the given indices, and readjusts the clustering accordingly. Returns list of indices that are new singletons."""
        if (len(indices) == 1):
            return self.change_score(indices[0], new_scores[0])
        else:
            new_singletons = []
            index_score_pairs_by_cluster = defaultdict(list)
            for (i, new_score) in zip(indices, new_scores):
                index_score_pairs_by_cluster[self.clusters_by_index[i]].append((i, new_score))
            for (k, pairs) in index_score_pairs_by_cluster.items():
                cluster_score = self.cluster_scores[k]
                if (len(pairs) == 1):
                    (i, new_score) = pairs[0]
                    new_singletons += self.change_score(i, new_score)
                else:
                    new_cluster_score_pairs = block_sort(pairs, key = lambda pair : pair[1], reverse = False)
                    for subpairs in new_cluster_score_pairs:
                        new_score = subpairs[0][1]
                        if (new_score != cluster_score):
                            num_to_move = len(subpairs)
                            indices_to_move = set([pair[0] for pair in subpairs])
                            is_less = new_score < cluster_score
                            size, start = self.cluster_sizes[k], self.cluster_starts[k]
                            new_start = start + size - num_to_move if is_less else start
                            for ii in range(new_start, new_start + num_to_move):
                                i = self.ranked_indices[ii]
                                if (i in indices_to_move):
                                    indices_to_move.remove(i)
                                    j = i
                                else:
                                    j = indices_to_move.pop()
                                    jj = self.ranks_by_index[j]
                                    self.ranked_indices[ii] = j
                                    self.ranked_indices[jj] = i
                                    self.ranks_by_index[i] = jj
                                    self.ranks_by_index[j] = ii
                                self.clusters_by_index[j] = self.num_clusters
                                self.scores[j] = new_score
                            if (num_to_move == 1):
                                self.is_single[j] = True
                                new_singletons.append(j)
                            self.cluster_sizes.append(num_to_move)
                            self.cluster_sizes[k] -= num_to_move
                            self.cluster_starts.append(new_start)
                            self.cluster_scores.append(new_score)
                            self.num_clusters += 1
                            if (self.cluster_starts[k] == new_start):
                                self.cluster_starts[k] += num_to_move
                            if (self.cluster_sizes[k] == 1):
                                i2 = self.ranked_indices[self.cluster_starts[k]]
                                self.is_single[i2] = True
                                new_singletons.append(i2)
        return new_singletons
    def __repr__(self):
        df = pd.DataFrame()
        df['Score'] = self.scores
        df['Cluster'] = self.clusters_by_index
        df = df.loc[self.ranked_indices]
        return str(df)


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
        self.pos_seed_mags = np.square(np.linalg.norm(self.pos_seeds, axis = 1))
        self.neg_seed_mags = np.square(np.linalg.norm(self.neg_seeds, axis = 1))
        self.s_plus = len(self.pos_seeds)
        self.s_minus = len(self.neg_seeds)
        self.s = self.s_plus + self.s_minus
    def get_dists(self, X):
        assert (X.shape[1] == self.m)
        dists = np.zeros((X.shape[0], self.s), dtype = float)
        point_mags = np.square(np.linalg.norm(X, axis = 1))
        dists[:, :self.s_plus] = -2 * np.dot(X, self.pos_seeds.T) + self.pos_seed_mags[None, :]
        dists[:, self.s_plus:] = -2 * np.dot(X, self.neg_seeds.T) + self.neg_seed_mags[None, :]
        dists += point_mags[:, None]
        return dists
    def get_row_sorted_dists(self, X):
        dists = self.get_dists(X)
        sorted_dists_by_row = np.sort(dists, axis = 1)
        dist_dict = defaultdict(dict)
        n = X.shape[0]
        for i in range(n):
            row = dists[i]
            for j in range(self.s_plus):
                dist = row[j]
                if (i in dist_dict[dist]):
                    dist_dict[dist][i][0] += 1
                else:
                    dist_dict[dist][i] = [1, 0]
            for j in range(self.s_minus):
                dist = row[self.s_plus + j]
                if (i in dist_dict[dist]):
                    dist_dict[dist][i][1] += 1
                else:
                    dist_dict[dist][i] = [0, 1]    
        sorted_dists = np.sort(list(dist_dict.keys()))
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
        score0 = (self.lamb * self.s_plus - (1 - self.lamb) * self.s_minus) if self.deflate else 0.0
        rhc = RankHierarchicalClustering(n, score0)
        ind_by_row = np.zeros(n, dtype = int)   # index of current dist in sorted_dists_by_row
        next_dist_heap = BinaryHeap.build([KeyValuePair(*pair) for pair in enumerate(sorted_dists_by_row[:, 0])], increasing = (not self.deflate))
        while (next_dist_heap.current_size > 0):
            debug("next_dist_heap: %s" % str(next_dist_heap))
            next_dist = next_dist_heap.min()[1]
            indices = []
            while ((next_dist_heap.current_size > 0) and (next_dist_heap.min()[1] == next_dist)):
                i = next_dist_heap.delete_min()[0]
                if (not rhc.is_single[i]):
                    indices.append(i)
            debug("\nnonsingletons: %s" % str([i for i in range(n) if ~rhc.is_single[i]]))
            debug("cluster_by_row: %s" % str(rhc.clusters_by_index))
            debug("cluster sizes: %s" % str(rhc.cluster_sizes))
            debug("cluster starts: %s" % str(rhc.cluster_starts))
            debug("ind_by_row: %s" % str(ind_by_row))
            debug("next_dist = %f" % next_dist)
            debug("indices = %s" % str(indices))
            debug("next_dist_heap: %s" % str(next_dist_heap))
            new_scores = []
            for i in indices:
                diff = self.lamb * dist_dict[next_dist][i][0] - (1 - self.lamb) * dist_dict[next_dist][i][1]
                new_score = (rhc.scores[i] - diff) if self.deflate else (rhc.scores[i] + diff)
                new_scores.append(new_score)
            new_singletons = rhc.change_scores(indices, new_scores)
            debug(rhc)
            debug("is_single: %s" % str(rhc.is_single))
            debug("new singletons: %s" % str(new_singletons))
            for i in indices:
                if (not rhc.is_single[i]):
                    ind = ind_by_row[i] + 1
                    while ((ind < self.s) and (sorted_dists_by_row[i, ind] == next_dist)):
                        ind += 1
                    ind_by_row[i] = ind  # increment node's index in sorted dist array until it's a new distance
                    if (ind < self.s):
                        next_dist_heap.insert(KeyValuePair(i, sorted_dists_by_row[i, ind]))
        scores = np.zeros((n, 2))
        cluster_ranks = [pair[0] for pair in sorted(enumerate(rhc.cluster_starts), key = lambda pair : pair[1])]
        ranks_by_cluster = {k : j for (j, k) in enumerate(cluster_ranks)}
        for i in range(n):
            scores[i, 0] = ranks_by_cluster[rhc.clusters_by_index[i]] / (rhc.num_clusters - 1.0)
        scores[:, 1] = 1.0 - scores[:, 0]
        return scores
    def predict_proba2(self, X):
        """Given n x m matrix X, returns rank-derived scores for the n data points, using either the inflation or deflation method."""
        (sorted_dists, sorted_dists_by_row, dist_dict) = self.get_row_sorted_dists(X)  # get distances by node
        if self.deflate:  # use descending order of distances if doing deflation
            sorted_dists = np.flipud(sorted_dists)
            sorted_dists_by_row = np.fliplr(sorted_dists_by_row)
        n = len(X)
        #debug(dist_dict)
        #debug(sorted_dists)
        #debug(sorted_dists_by_row)
        min_dist, max_dist = sorted_dists[0], sorted_dists[-1]
        score0 = (self.lamb * self.s_plus - (1 - self.lamb) * self.s_minus) if self.deflate else 0.0
        rhc = RankHierarchicalClustering(n, score0)
        ind_by_row = np.zeros(n, dtype = int)   # index of current dist in sorted_dists_by_row
        next_dist_by_row = sorted_dists_by_row[:, 0]  # array of next distance for each node to change its score
        done_marker = -np.inf if self.deflate else np.inf
        while (not all(rhc.is_single)):
            next_dist = next_dist_by_row.max() if self.deflate else next_dist_by_row.min()
            if np.isinf(next_dist):  # +/-inf signifies no more nodes will change
                break
            rows_with_next_dist = set(np.nonzero((next_dist_by_row == next_dist) & (~rhc.is_single))[0])
            #debug("\nnonsingletons: %s" % str([i for i in range(n) if ~rhc.is_single[i]]))
            #debug("cluster_by_row: %s" % str(rhc.clusters_by_index))
            #debug("cluster sizes: %s" % str(rhc.cluster_sizes))
            #debug("cluster starts: %s" % str(rhc.cluster_starts))
            #debug("ind_by_row: %s" % str(ind_by_row))
            #debug("next_dist_by_row: %s" % str(next_dist_by_row))
            #debug("next_dist = %f" % next_dist)
            #debug("rows: %s" % str(rows_with_next_dist))
            indices = list(rows_with_next_dist)
            new_scores = []
            for i in indices:
                diff = self.lamb * dist_dict[next_dist][i][0] - (1 - self.lamb) * dist_dict[next_dist][i][1]
                new_score = (rhc.scores[i] - diff) if self.deflate else (rhc.scores[i] + diff)
                new_scores.append(new_score)
            new_singletons = rhc.change_scores(indices, new_scores)
            #debug(rhc)
            #debug("is_single: %s" % str(rhc.is_single))
            #debug("new singletons: %s" % str(new_singletons))
            for i in new_singletons:
                next_dist_by_row[i] = done_marker
            for i in indices:
                if (not rhc.is_single[i]):
                    ind = ind_by_row[i] + 1
                    while ((ind < self.s) and (sorted_dists_by_row[i, ind] == next_dist)):
                        ind += 1
                    ind_by_row[i] = ind  # increment node's index in sorted dist array until it's a new distance
                    next_dist_by_row[i] = sorted_dists_by_row[i, ind] if (ind < self.s) else done_marker
        scores = np.zeros((n, 2))
        cluster_ranks = [pair[0] for pair in sorted(enumerate(rhc.cluster_starts), key = lambda pair : pair[1])]
        ranks_by_cluster = {k : j for (j, k) in enumerate(cluster_ranks)}
        for i in range(n):
            scores[i, 0] = ranks_by_cluster[rhc.clusters_by_index[i]] / (rhc.num_clusters - 1.0)
        scores[:, 1] = 1.0 - scores[:, 0]
        return scores
    def predict_proba3(self, X):
        D = self.get_dists(X)
        n = X.shape[0]
        sorted_dists = np.concatenate([-1 * np.ones(1, dtype = float), np.array(sorted(set(D.flatten())))])
        M = np.searchsorted(sorted_dists, D)
        I_tilde = np.argsort(M, axis = 1, kind = 'mergesort')
        M_tilde = arrange_by_index(M, I_tilde)
        G_tilde = (np.diff(M_tilde, axis = 1) == 0)
        labels = np.concatenate([np.ones(self.s_plus, dtype = int), -1 * np.ones(self.s_minus, dtype = int)])
        # print(D)
        # print(M)
        # print(I_tilde)
        # print(M_tilde)
        # print(G_tilde)
        # print(self.s)
        def cmp_func(i1, i2):
            def get_nbr_influence(row, j):
                nbr_influence = labels[I_tilde[row, j]]
                if (j == self.s - 1):
                    return nbr_influence
                while (G_tilde[row, j]):
                    nbr_influence += labels[I_tilde[row, j + 1]]
                    if (j == self.s - 2):
                        break
                    else:
                        j += 1
                return nbr_influence
            for j in range(self.s):
                mij1, mij2 = M_tilde[i1, j], M_tilde[i2, j]
                if (mij2 < mij1):
                    nbr_influence = get_nbr_influence(i2, j)
                    if (nbr_influence != 0):
                        return (1 if (nbr_influence < 0) else -1)
                elif (mij2 > mij1):
                    nbr_influence = get_nbr_influence(i1, j)
                    if (nbr_influence != 0):
                        return (1 if (nbr_influence > 0) else -1)
                elif (mij1 == mij2):
                    nbr_influence1 = get_nbr_influence(i1, j)
                    nbr_influence2 = get_nbr_influence(i2, j)
                    if (nbr_influence1 != nbr_influence2):
                        return (1 if (nbr_influence1 > nbr_influence2) else -1)
            return 0
        ranks = np.array(sorted(reversed(range(n)), key = cmp_to_key(cmp_func)))
        print(ranks)
        scores = np.zeros((n, 2))
        for (ctr, i) in enumerate(ranks):
            scores[i, 1] = ctr / (n - 1.0)
        scores[:, 0] = 1.0 - scores[:, 1]
        #print(scores)
        return scores
    def predict_proba4(self, X):
        D = self.get_dists(X)
        n = X.shape[0]
        sorted_dists = np.concatenate([-1 * np.ones(1, dtype = float), np.array(sorted(set(D.flatten())))])
        M = np.searchsorted(sorted_dists, D)
        I_tilde = np.argsort(M, axis = 1, kind = 'mergesort')
        M_tilde = arrange_by_index(M, I_tilde)
        M_tilde_max = M_tilde.max(axis = 1)
        M_tilde_min = M_tilde.min(axis = 1)
        G_tilde = (np.diff(M_tilde, axis = 1) == 0)
        labels = np.concatenate([np.ones(self.s_plus, dtype = int), -1 * np.ones(self.s_minus, dtype = int)])
        # print(D)
        # print(M)
        # print(I_tilde)
        # print(M_tilde)
        # print(G_tilde)
        # print(self.s)
        def cmp_func(i1, i2):
            def get_nbr_influence(row, j):
                cursor = j 
                while ((j > 0) and G_tilde[row, j - 1]):
                    j -= 1
                    cursor = j
                nbr_influence = labels[I_tilde[row, cursor]]
                if (cursor == self.s - 1):
                    return nbr_influence
                while (G_tilde[row, cursor]):
                    nbr_influence += labels[I_tilde[row, cursor + 1]]
                    if (cursor == self.s - 2):
                        break
                    else:
                        cursor += 1
                return nbr_influence
            for j in range(self.s):
                mij1, mij2 = M_tilde[i1, j], M_tilde[i2, j]
                if (mij2 < mij1):
                    nbr_influence = get_nbr_influence(i2, j)
                    if (nbr_influence != 0):
                        return (1 if (nbr_influence < 0) else -1)
                elif (mij2 > mij1):
                    nbr_influence = get_nbr_influence(i1, j)
                    if (nbr_influence != 0):
                        return (1 if (nbr_influence > 0) else -1)
                elif (mij1 == mij2):
                    nbr_influence1 = get_nbr_influence(i1, j)
                    nbr_influence2 = get_nbr_influence(i2, j)
                    if (nbr_influence1 != nbr_influence2):
                        return (1 if (nbr_influence1 > nbr_influence2) else -1)
            # Handle the special case, that the radius of i1/i2 from all the
            # seeds is lesser than the radius of i2/i1
            if ((M_tilde_max[i1] < M_tilde_min[i2]) or (M_tilde_min[i1] > M_tilde_max[i2])):
                nbr_influence1 = get_nbr_influence(i1, 0)
                nbr_influence2 = get_nbr_influence(i2, 0)
                if (nbr_influence1 != nbr_influence2):
                    return (1 if (nbr_influence1 > nbr_influence2) else -1)
            return 0
        ranks = np.array(sorted(reversed(range(n)), key = cmp_to_key(cmp_func)))
        scores = np.zeros((n, 2))
        for (ctr, i) in enumerate(ranks):
            scores[i, 1] = ctr / (n - 1.0)
        scores[:, 0] = 1.0 - scores[:, 1]
        # print(ranks)
        # print(scores)
        return scores
    def predict_proba5(self, X):
        D = self.get_dists(X)
        n = X.shape[0]
        sorted_dists = np.concatenate([-1 * np.ones(1, dtype = float), np.array(sorted(set(D.flatten())))])
        M = np.searchsorted(sorted_dists, D)
        I_tilde = np.argsort(M, axis = 1, kind = 'mergesort')
        M_tilde = arrange_by_index(M, I_tilde)
        M_tilde_max = M_tilde.max(axis = 1)
        M_tilde_min = M_tilde.min(axis = 1)
        G_tilde = (np.diff(M_tilde, axis = 1) == 0)
        labels = np.concatenate([np.ones(self.s_plus, dtype = int), -1 * np.ones(self.s_minus, dtype = int)])
        # print(D)
        # print(M)
        # print(I_tilde)
        # print(M_tilde)
        # print(G_tilde)
        # print(self.s)
        def cmp_func(i1, i2):
            def get_nbr_influence(row, j):
                backtrack_cursor = j 
                while ((j > 0) and G_tilde[row, j - 1]):
                    j -= 1
                    backtrack_cursor = j
                nbr_influence = labels[I_tilde[row, backtrack_cursor]]
                if (backtrack_cursor == self.s - 1):
                    return nbr_influence
                while (G_tilde[row, backtrack_cursor]):
                    nbr_influence += labels[I_tilde[row, backtrack_cursor + 1]]
                    if (backtrack_cursor == self.s - 2):
                        break
                    else:
                        backtrack_cursor += 1
                return nbr_influence
            cursor1, cursor2 = 0, 0
            while (min(cursor1, cursor2) < self.s):
                new_cursor1, new_cursor2 = cursor1, cursor2
                j1, j2 = cursor1, cursor2
                if (cursor1 >= self.s):
                    j1 = self.s - 1
                    new_cursor2 = cursor2 + 1
                elif (cursor2 >= self.s):
                    j2 = self.s - 1
                    new_cursor1 = cursor1 + 1
                elif (M_tilde[i2, cursor2] >= M_tilde[i1, cursor1]):
                    new_cursor1 = cursor1 + 1
                elif (M_tilde[i1, cursor1] >= M_tilde[i2, cursor2]):
                    new_cursor2 = cursor2 + 1
                else:
                    raise Exception("Illegal state.")
                mij1, mij2 = M_tilde[i1, j1], M_tilde[i2, j2]
                if (mij2 < mij1):
                    nbr_influence = get_nbr_influence(i2, j2)
                    if (nbr_influence != 0):
                        return (1 if (nbr_influence < 0) else -1)
                elif (mij2 > mij1):
                    nbr_influence = get_nbr_influence(i1, j1)
                    if (nbr_influence != 0):
                        return (1 if (nbr_influence > 0) else -1)
                elif (mij1 == mij2):
                    nbr_influence1 = get_nbr_influence(i1, j1)
                    nbr_influence2 = get_nbr_influence(i2, j2)
                    if (nbr_influence1 != nbr_influence2):
                        return (1 if (nbr_influence1 > nbr_influence2) else -1)
                cursor1, cursor2 = new_cursor1, new_cursor2
                pass
            # Handle the special case, that the radius of i1/i2 from all the
            # seeds is lesser than the radius of i2/i1
            if ((M_tilde_max[i1] < M_tilde_min[i2]) or (M_tilde_min[i1] > M_tilde_max[i2])):
                nbr_influence1 = get_nbr_influence(i1, 0)
                nbr_influence2 = get_nbr_influence(i2, 0)
                if (nbr_influence1 != nbr_influence2):
                    return (1 if (nbr_influence1 > nbr_influence2) else -1)
            return 0
        ranks = np.array(sorted(reversed(range(n)), key = cmp_to_key(cmp_func)))
        scores = np.zeros((n, 2))
        for (ctr, i) in enumerate(ranks):
            scores[i, 1] = ctr / (n - 1.0)
        scores[:, 0] = 1.0 - scores[:, 1]
        # print(np.array([[cmp_func(i1, i2) for i2 in range(n)] for i1 in range(n)]))
        # print(ranks)
        # print(scores)
        return scores
    def predict_proba6(self, X):
        D = self.get_dists(X)
        n = X.shape[0]
        sorted_dists = np.concatenate([-1 * np.ones(1, dtype = float), np.array(sorted(set(D.flatten())))])
        M = np.searchsorted(sorted_dists, D)
        I_tilde = np.argsort(M, axis = 1, kind = 'quicksort')
        M_tilde = arrange_by_index(M, I_tilde)
        #M_tilde_max = M_tilde.max(axis = 1)
        #M_tilde_min = M_tilde.min(axis = 1)
        G_tilde = (np.diff(M_tilde, axis = 1) == 0)
        #G_tilde = np.hstack([np.ones(n, dtype = bool).reshape((n, 1)), G_tilde])
        labels = np.concatenate([np.ones(self.s_plus, dtype = int), -1 * np.ones(self.s_minus, dtype = int)])
        # print(D)
        # print(M)
        # print(I_tilde)
        # print(M_tilde)
        # print(G_tilde)
        # print(self.s)
        def cmp_func(i1, i2):
            #print((i1,i2))
            #j1, j2 = 0, 0
            #cumsum1, cumsum2 = labels[I_tilde[i1, j1]], labels[I_tilde[i2, j2]]
            j1, j2 = -1, -1
            cumsum1, cumsum2 = 0, 0
            #mij1, mij2 = M_tilde[i1, j1], M_tilde[i2, j2]
            while (min(j1, j2) < self.s - 1):
                next_j1, next_j2 = min(self.s - 1, j1 + 1), min(self.s - 1, j2 + 1)
                mij1, mij2 = M_tilde[i1, next_j1], M_tilde[i2, next_j2]
                #print("j1 = %d, j2 = %d" % (j1, j2))
                #print("mij1 = %d, mij2 = %d" % (mij1, mij2))
                if (mij1 <= mij2):
                    #while ((j1 < self.s - 1) and (M_tilde[i1, j1 + 1] == mij1)):
                    while ((next_j1 > j1) or ((j1 < self.s - 1) and G_tilde[i1, j1])):
                        j1 += 1
                        cumsum1 += labels[I_tilde[i1, j1]]
                if (mij1 >= mij2):
                    #while ((j2 < self.s - 1) and (M_tilde[i2, j2 + 1] == mij2)):
                    while ((next_j2 > j2) or ((j2 < self.s - 1) and G_tilde[i2, j2])):
                        j2 += 1
                        cumsum2 += labels[I_tilde[i2, j2]]
                if (cumsum1 != cumsum2):
                    return (-1 if (cumsum1 > cumsum2) else 1)
            return 0
        ranks = np.array(sorted(range(n), key = cmp_to_key(cmp_func)))
        scores = np.zeros((n, 2))
        for (ctr, i) in enumerate(ranks):
            scores[i, 0] = ctr / (n - 1.0)
        scores[:, 1] = 1.0 - scores[:, 0]
        # print(np.array([[cmp_func(i1, i2) for i2 in range(n)] for i1 in range(n)]))
        # print(ranks)
        # print(scores)
        return scores
    def predict_proba7(self, X):
        D = self.get_dists(X)
        n = X.shape[0]
        sorted_dists_by_row = np.sort(D, axis = 1)
        I_tilde = np.argsort(D, axis = 1, kind = 'mergesort')
        D_tilde = arrange_by_index(D, I_tilde)
        G_tilde = (np.diff(D_tilde, axis = 1) == 0.0)
        labels = np.concatenate([np.ones(self.s_plus, dtype = int), -1 * np.ones(self.s_minus, dtype = int)])
        s_minus_one = self.s - 1
        # print(D)
        # print(D_tilde)
        # print(I_tilde)
        # print(G_tilde)
        # print(self.s)
        def cmp_func(i1, i2):
            #print((i1,i2))
            j1, j2 = -1, -1
            cumsum1, cumsum2 = 0, 0
            while (min(j1, j2) < s_minus_one):
                next_j1, next_j2 = min(s_minus_one, j1 + 1), min(s_minus_one, j2 + 1)
                dij1, dij2 = D_tilde[i1, next_j1], D_tilde[i2, next_j2]
                if (dij1 <= dij2):
                    while ((next_j1 > j1) or ((j1 < s_minus_one) and G_tilde[i1, j1])):
                        j1 += 1
                        cumsum1 += labels[I_tilde[i1, j1]]
                if (dij1 >= dij2):
                    while ((next_j2 > j2) or ((j2 < s_minus_one) and G_tilde[i2, j2])):
                        j2 += 1
                        cumsum2 += labels[I_tilde[i2, j2]]
                if (cumsum1 != cumsum2):
                    return (-1 if (cumsum1 > cumsum2) else 1)
            return 0
        ranks = np.array(sorted(range(n), key = cmp_to_key(cmp_func)))
        scores = np.zeros((n, 2))
        for (ctr, i) in enumerate(ranks):
            scores[i, 0] = ctr / (n - 1.0)
        scores[:, 1] = 1.0 - scores[:, 0]
        # print(np.array([[cmp_func(i1, i2) for i2 in range(n)] for i1 in range(n)]))
        # print(ranks)
        # print(scores)
        return scores
    def predict_proba8(self, X):
        labels = np.asarray(self.labels, dtype = float)
        if (self.lamb <= 0.5):
            labels = (1 / (1 - self.lamb)) * labels - 1
        else:
            labels = (1 / self.lamb) * labels + ((self.lamb - 1) / self.lamb)
        ranks = balloon_rank(X, self.seeds, labels, self.deflate)
        n = X.shape[0]
        scores = np.zeros((n, 2))
        for (ctr, i) in enumerate(ranks):
            scores[i, 0] = ctr / (n - 1.0)
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
        elif (testnum == 6):  # another small lattice
            seeds = np.array([[-3, 2], [3, 2], [3, -2]])
            labels = np.array([0, 1, 1])
            X = np.array([[-1, -3], [0, 1]])
            true_ranked_list = [0, 1]
        return (seeds, labels, X, true_ranked_list)
    def test_correctness(self, testnum, lamb = 0.5, deflate = False):
        (seeds, labels, X, true_ranked_list) = self.get_test_data(testnum)
        bln = BalloonNominate(lamb, deflate = deflate)
        bln.fit(seeds, labels)
        scores = bln.predict_proba8(X)[:, 1]
        ranked_list = [pair[0] for pair in sorted(enumerate(scores), key = lambda pair : pair[1], reverse = True)]
        print("\nComputed ranking:")
        print(ranked_list)
        print("\nTrue ranking:")
        print(true_ranked_list)
        print("\n" + str(ranked_list == true_ranked_list))
        self.assertTrue(ranked_list == true_ranked_list)
    def test_time(self, nseeds = 100, npoints = 1000, dim = 50):
        np.random.seed(0)
        seeds = np.random.randn(nseeds, dim)
        labels = np.random.randint(0, 2, nseeds)
        X = np.random.randn(npoints, dim)
        start = time.time()
        infl = BalloonNominate(0.5, deflate = False)
        infl.fit(seeds, labels)
        scores = infl.predict_proba(X)[:, 1]
        end = time.time()
        ranked_list = [pair[0] for pair in sorted(enumerate(scores), key = lambda pair : pair[1], reverse = True)]
        return end - start     
    def test_time5(self, nseeds = 100, npoints = 1000, dim = 50):
        np.random.seed(0)
        seeds = np.random.randn(nseeds, dim)
        labels = np.random.randint(0, 2, nseeds)
        X = np.random.randn(npoints, dim)
        start = time.time()
        infl = BalloonNominate(0.5, deflate = False)
        infl.fit(seeds, labels)
        scores = infl.predict_proba5(X)[:, 1]
        end = time.time()
        ranked_list = [pair[0] for pair in sorted(enumerate(scores), key = lambda pair : pair[1], reverse = True)]
        return end - start  
    def test_time6(self, nseeds = 100, npoints = 1000, dim = 50):
        np.random.seed(0)
        seeds = np.random.randn(nseeds, dim)
        labels = np.random.randint(0, 2, nseeds)
        X = np.random.randn(npoints, dim)
        start = time.time()
        infl = BalloonNominate(0.5, deflate = False)
        infl.fit(seeds, labels)
        scores = infl.predict_proba6(X)[:, 1]
        end = time.time()
        ranked_list = [pair[0] for pair in sorted(enumerate(scores), key = lambda pair : pair[1], reverse = True)]
        return end - start  
    def test_time7(self, nseeds = 100, npoints = 1000, dim = 50):
        np.random.seed(0)
        seeds = np.random.randn(nseeds, dim)
        labels = np.random.randint(0, 2, nseeds)
        X = np.random.randn(npoints, dim)
        start = time.time()
        infl = BalloonNominate(0.5, deflate = False)
        infl.fit(seeds, labels)
        scores = infl.predict_proba7(X)[:, 1]
        end = time.time()
        ranked_list = [pair[0] for pair in sorted(enumerate(scores), key = lambda pair : pair[1], reverse = True)]
        return end - start  
    def test_time8(self, nseeds = 100, npoints = 1000, dim = 50):
        np.random.seed(0)
        seeds = np.random.randn(nseeds, dim)
        labels = np.random.randint(0, 2, nseeds)
        X = np.random.randn(npoints, dim)
        start = time.time()
        infl = BalloonNominate(0.5, deflate = False)
        infl.fit(seeds, labels)
        scores = infl.predict_proba8(X)[:, 1]
        end = time.time()
        ranked_list = [pair[0] for pair in sorted(enumerate(scores), key = lambda pair : pair[1], reverse = True)]
        return end - start  
    def test_lattice(self, nseeds = 5, npoints = 10, dim = 2, bnd = 4, lamb = 0.5, deflate = False, seed = 0):
        np.random.seed(seed)
        seeds = np.random.randint(-bnd, bnd + 1, (nseeds, dim))
        labels = np.random.randint(0, 2, nseeds)
        X = np.random.randint(-bnd, bnd + 1, (npoints, 2))
        bln = BalloonNominate(lamb, deflate = deflate)
        bln.fit(seeds, labels)
        scores = bln.predict_proba(X)[:, 1]
        ranked_list = [pair[0] for pair in sorted(enumerate(scores), key = lambda pair : pair[1], reverse = True)]
        return ranked_list
    def test_lattice5(self, nseeds = 5, npoints = 10, dim = 2, bnd = 4, lamb = 0.5, deflate = False, seed = 0):
        np.random.seed(seed)
        seeds = np.random.randint(-bnd, bnd + 1, (nseeds, dim))
        labels = np.random.randint(0, 2, nseeds)
        X = np.random.randint(-bnd, bnd + 1, (npoints, 2))
        bln = BalloonNominate(lamb, deflate = deflate)
        bln.fit(seeds, labels)
        scores = bln.predict_proba5(X)[:, 1]
        ranked_list = [pair[0] for pair in sorted(enumerate(scores), key = lambda pair : pair[1], reverse = True)]
        return ranked_list
    def test_lattice7(self, nseeds = 5, npoints = 10, dim = 2, bnd = 4, lamb = 0.5, deflate = False, seed = 0):
        np.random.seed(seed)
        seeds = np.random.randint(-bnd, bnd + 1, (nseeds, dim))
        labels = np.random.randint(0, 2, nseeds)
        X = np.random.randint(-bnd, bnd + 1, (npoints, 2))
        bln = BalloonNominate(lamb, deflate = deflate)
        bln.fit(seeds, labels)
        scores = bln.predict_proba7(X)[:, 1]
        ranked_list = [pair[0] for pair in sorted(enumerate(scores), key = lambda pair : pair[1], reverse = True)]
        return ranked_list



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
#s = 50
#k = 51
s = 5
k = 7
(seeds, labels) = get_data(name, s)
Z = get_mesh(seeds, k)
bln = BalloonNominate(0.5)
bln.fit(seeds, labels)



