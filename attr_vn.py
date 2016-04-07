import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from importlib import reload
from collections import defaultdict
from scipy.sparse.linalg import eigsh
from scipy.sparse import *
from linop import *
from utils import *

def safe_divide(num, den):
    """Floating point division, with the convention that 0 / 0 = 0, (+/-)c / 0 = (+/-)inf for c > 0."""
    if (num == 0.0):
        return 0.0
    if (den == 0.0):
        return np.sign(num) * float('inf')
    return (num / den)

def symmetrize_sparse_matrix(M):
    """Given a sparse matrix M, returns M + M^T - diag(M) in csr format. The resulting matrix is symmetric."""
    M = M.tocsr()
    return M + M.transpose().tocsr() - diags(M.diagonal(), offsets = 0).tocsr()

def edgelist_to_sparse_adjacency_operator(filename, verbose = True):
    """Takes a filename of an undirected edge list (space-separated pairs of vertices indexed from 0 to n - 1)."""
    if verbose:
        print("\nLoading edges from '%s'..." % filename)
    edges_dict = defaultdict(list)
    rows, cols = [], []
    num_edges = 0
    n = -1
    for (i, j) in (map(int, line.split()) for line in open(filename, 'r')):
        num_edges += 1
        n = max(n, max(i, j))
        rows.append(i)
        cols.append(j)
    n += 1
    data = np.ones(num_edges, dtype = float)
    ij = np.array([rows, cols])
    A = csr_matrix((data, ij), shape = (n, n))
    return SymmetricSparseLinearOperator(symmetrize_sparse_matrix(A))

def embed_symmetric_operator(A, embedding = 'adj', k = 50, tol = None, verbose = True):
    """Given a SymmetricSparseLinearOperator, returns the k-dimensional spectral embedding of this operator or one of its variants, as well as the corresponding eigenvalues."""
    assert (embedding in ['adj', 'adj+diag', 'normlap', 'regnormlap']), "Invalid embedding."
    n = A.shape[0]
    k = min(k, n - 1)
    tol = 1.0 / n if (tol is None) else tol
    if verbose:
        matrix_type = 'adjacency' if (embedding == 'adj') else ('diagonal-added adjacency' if (embedding == 'diag+adj') else ('normalized Laplacian' if (embedding == 'normlap') else 'regularized normalized Laplacian'))
        print("\nComputing k = %d eigenvectors of %d x %d %s matrix..." % (k, n, n, matrix_type))
    if (embedding == 'adj'):
        (eigvals, features) = eigsh(A, k = k, tol = tol)
        features = np.sqrt(np.abs(eigvals)) * features  # scale the feature columns by the sqrt of the eigenvalues
    elif (embedding == 'adj+diag'):
        adj_diag = SparseDiagonalAddedAdjacencyOperator(A)
        (eigvals, features) = eigsh(adj_diag, k = k, tol = tol)
    elif (embedding == 'normlap'):
        normlap = SparseNormalizedLaplacian(A)
        (eigvals, features) = eigsh(normlap, k = k, tol = tol)
    elif (embedding == 'regnormlap'):
        regnormlap = SparseRegularizedNormalizedLaplacian(A)
        (eigvals, features) = eigsh(regnormlap, k = k, tol = tol)
    return (eigvals, features)

def get_elbows(x, n = 1, thresh = 0.0):
    """Given array of decreasing abs(eigenvalues), computes the elbows of the scree plot. From Zhu and Ghodsi, "Automatic dimensionality selection from the scree plot via the use of profile likelihood," 2006."""
    m = len(x)
    assert (m > 2)
    x = x[x >= thresh]
    ll = -1000 * np.ones(m - 1)
    for k in range(1, m):
        mu1 = np.mean(x[:k])
        mu2 = np.mean(x[k:])
        sig2 = (k * np.sum((x[:k] - mu1) ** 2) + (len(x) - k) * np.sum((x[k:] - mu2) ** 2)) / (m - 2)
        ll[k - 1] = -1 * (np.sum((x[:k] - mu1) ** 2) + np.sum((x[k:] - mu2) ** 2)) / (2 * sig2) - 0.5 * m * np.log(2 * np.pi * sig2)
    elbows = list([ll.argmax()])
    if ((n > 1) and ((elbows[0] + 1) < m)):
        other_elbows = get_elbows(x[(elbows[0] + 1):], n = n - 1) + elbows[0] + 1
        elbows += list(other_elbows)
    return elbows

# def scree_plot(eigvals, elbow = None, show = True, filename = None):
#     """Makes a scree plot of the absolute value of a series of eigenvalues using matplotlib. If elbow is not None, draws a vertical line at this index. If show = True, displays the plot. If filename is not None, saves the plot to this filename."""
#     plt.figure()
#     abs_eigvals = np.abs(eigvals)
#     ranked_abs_eigvals = np.array(sorted(abs_eigvals, reverse = True))
#     plt.plot(ranked_abs_eigvals, linewidth = 3)
#     if (elbow is not None):
#         plt.axvline(x = elbow, linewidth = 2, color = 'red', linestyle = 'dashed')
#     plt.title('Scree plot of eigenvalues')
#     plt.xlabel('rank', labelpad = 10)
#     plt.ylabel('abs(eigenvalue)', labelpad = 15)
#     plt.ylim(ymin = 0)
#     if filename:
#         plt.savefig(filename)
#     if show:
#         plt.show(block = False)


class PairwiseFreqAnalyzer(object):
    """Manages statistics related to unordered pairwise frequencies, such as pointwise mutual information. Represents the pairwise frequencies as a sparse matrix of counts of each pair using the DOK (dictionary of keys) format, then converts it to CSR (compressed sparse row) format."""
    def __init__(self, vocab):
        """Constructs PairwiseFreqAnalyzer object with a list of vocab. Creates a mapping from vocab to indices for the sparse matrix representation.""" 
        self.vocab = vocab  # the canonical vocab list (in order of matrix indices)
        self.vocab_indices = dict((v, i) for (i, v) in enumerate(self.vocab))  # maps vocab items to canonical indices
        self.vocab_freqs = dict((v, 0) for v in self.vocab)  # counts number of edges seen with each vocab word in it
        self.num_vocab = len(self.vocab)
        self.num_possible_pairs = (self.num_vocab * (self.num_vocab + 1)) // 2
        self.freq_mat = dok_matrix((self.num_vocab, self.num_vocab), dtype = np.int64)
    def add_pair(self, pair):
        """Given a pair of objects, adds one to the count of the object pair. If an item in the pair is not in the vocab list, raises a KeyError."""
        v1, v2 = pair
        [i, j] = sorted([self.vocab_indices[v1], self.vocab_indices[v2]])
        self.freq_mat[i, j] += 1
    def finalize_construction(self):
        """Once all pairs are added, performs some computations and converts the sparse matrix from dok to csr format."""
        self.total_edges = sum(self.freq_mat.values())
        self.freq_mat = self.freq_mat.tocsr()
        self.total_counts = self.freq_mat.sum()
        sym_freq_mat = self.freq_mat + self.freq_mat.transpose().tocsr() - diags(self.freq_mat.diagonal(), offsets = 0).tocsr()  # symmetrize the matrix
        for (i, v) in enumerate(self.vocab):
            self.vocab_freqs[v] = sym_freq_mat[i,:].data.sum()
    def empirical_freq(self, *items, delta = 0):
        """Returns frequency count of item. If two arguments are given, returns the observed count of the pair, disregarding order. If one argument is given, returns the observed count of the singleton. If delta > 0, adds delta to the edge counts."""
        assert (len(items) in [1, 2])
        if any([x not in self.vocab_freqs for x in items]):
            raise ValueError("Entries must be in the vocabulary.")
        if (len(items) == 1):
            freq = self.vocab_freqs[items[0]] + delta * self.num_vocab
            return freq
        i, j = sorted([self.vocab_indices[items[0]], self.vocab_indices[items[1]]])
        freq = self.freq_mat[i, j]
        return (freq + delta)
    def empirical_prob(self, *items, delta = 0):
        """Returns empirical probability of item. If two arguments are given, this is the smoothed number of occurrences of the pair divided by the smoothed number of edges, under add-delta smoothing. If one argument is given, this is the smoothed ratio of occurrences of the item in any pair ."""
        assert (len(items) in [1, 2])
        if (len(items) == 1):
            denominator = self.total_counts + delta * self.num_vocab ** 2
        else:
            denominator = self.total_edges + delta * self.num_possible_pairs
        return self.empirical_freq(*items, delta = delta) / denominator
    def conditional_prob(self, item1, item2, delta = 0):
        """Empirical conditional probability of item1 given item2."""
        return safe_divide(self.empirical_prob(item1, item2, delta = delta), self.empirical_prob(item2, delta = delta))
    def PMIs(self, item1, item2, delta = 0):
        """Pointwise mutual information of two items. Ranges from -inf to -log p(x,y) at most, with 0 for independence"""
        return np.log(self.empirical_prob(item1, item2, delta = delta)) - np.log(self.empirical_prob(item1, delta = delta)) - np.log(self.empirical_prob(item2, delta = delta))
    def PMId(self, item1, item2, delta = 0):
        """Negative pointwise mutual information of two items. Ranges from log p(x,y) to inf, with 0 for independence."""
        return -self.PMIs(item1, item2, delta = delta)
    def NPMI1s(self, item1, item2, delta = 0):
        """PMI normalized so that it is a similarity score ranging from 0 to 1, with 1/2 for independence."""
        return safe_divide(np.log(self.empirical_prob(item1, delta = delta)) + np.log(self.empirical_prob(item2, delta = delta)), 2.0 * np.log(self.empirical_prob(item1, item2, delta = delta)))
    def NPMI1d(self, item1, item2, delta = 0):
        """PMI normalized so that it is a dissimilarity score ranging from 0 to 1, with 1/2 for independence."""
        return 1.0 - self.NPMI1s(item1, item2, delta = delta)
    def NPMI2s(self, item1, item2, delta = 0):
        """PMI transformed so that it is a similarity score ranging from 0 to inf, with 1 for independence."""
        return -np.log(1.0 - self.NPMI1s(item1, item2, delta = delta)) / np.log(2.0)
    def NPMI2d(self, item1, item2, delta = 0):
        """PMI transformed so that it is a dissimilarity score ranging from 0 to inf, with 1 for independence."""
        return -np.log(self.NPMI1s(item1, item2, delta = delta)) / np.log(2.0)
    def to_sparse_sim_matrix(self, sim = 'NPMI1s'):
        """Returns a sparse similarity/dissimilarity matrix based on co-occurrence of the vocabulary items. Options are 'PMIs', 'NPMI1s', 'NPMI2s', and 'conditional_prob', which have different ranges. No smoothing is done yet, since that would ruin the sparsity."""
        assert (sim in ['PMIs', 'NPMI1s', 'NPMI2s', 'conditional_prob']), "Invalid similarity option."
        n = len(self.freq_mat.data)
        sim_func = self.__class__.__dict__[sim]
        log_single_freqs = np.log(np.array([self.empirical_freq(self.vocab[i]) for i in range(self.num_vocab)]))
        log_total_edges = np.log(self.total_edges)
        log_total_counts = np.log(self.total_counts)
        shift = -log_total_edges + 2 * log_total_counts
        data = np.zeros(n, dtype = float)
        coo = self.freq_mat.tocoo()  # convert to coo_matrix
        coo.data = np.log(coo.data)  # store the log-frequencies
        # efficiently compute the score
        for (k, (i, j, log_freq)) in enumerate(zip(coo.row, coo.col, coo.data)):
            if (sim == 'PMIs'):
                data[k] = log_freq - log_single_freqs[i] - log_single_freqs[j] + shift
            elif (sim == 'NPMI1s'):
                data[k] = (log_single_freqs[i] + log_single_freqs[j] - 2 * log_total_counts) / (2 * (log_freq - log_total_edges))
            else:
                data[k] = sim_func(self, self.vocab[i], self.vocab[j])
        mat = coo_matrix((data, (coo.row, coo.col)), shape = (self.num_vocab, self.num_vocab)).tocsr()
        mat = symmetrize_sparse_matrix(mat)
        return mat
    def to_sparse_sim_operator(self, sim = 'NPMI1s', delta = 0):
        """Returns a SparseLinearOperator object encoding the sparse + low-rank representation of the PMI similarity matrix. This can be used in place of an actual matrix in various computations. If sim != 'PMIs', can use an alternative formulation of PMI, but only if delta = 0."""
        assert ((delta > 0) if (sim == 'PMIs') else (delta == 0))
        if (sim != 'PMIs'):  # just use the sparse similarity matrix with no smoothing
            csr_mat = self.to_sparse_sim_matrix(sim)
            return SymmetricSparseLinearOperator(csr_mat)
        log_delta = np.log(delta)
        coo = self.freq_mat.tocoo()
        data = np.log(coo.data + delta) - log_delta
        F = coo_matrix((data, (coo.row, coo.col)), shape = (self.num_vocab, self.num_vocab)).tocsr()
        F = symmetrize_sparse_matrix(F)
        u = np.log(np.array([self.empirical_freq(self.vocab[i], delta = delta) for i in range(self.num_vocab)]))
        Delta = log_delta + np.log(self.total_edges + delta * self.num_possible_pairs)
        return PMILinearOperator(F, Delta, u)


class AttributeAnalyzer(object):
    """Class for analyzing node attribute frequencies for various text attributes."""
    def __init__(self, attr_filename, num_nodes, attr_types = None):
        """Creates an AttributeAnalyzer object from a csv file (;-separated) of node attributes. If attr_types is a specified list, only includes these attribute types."""
        self.num_nodes = num_nodes
        self.attr_df = pd.read_csv(attr_filename, sep = ';')
        self.attr_df['attributeVal'] = self.attr_df['attributeVal'].astype(str)
        self.attr_types = list(sorted(set(self.attr_df['attributeType'])))
        if (attr_types is not None):
            assert set(attr_types).issubset(self.attr_types)
            self.attr_types = list(sorted(attr_types))
        self.num_attr_types = len(self.attr_types)
        self.attr_df = self.attr_df[np.vectorize(lambda t : t in set(self.attr_types))(self.attr_df['attributeType'])]
        self.attributed_nodes = sorted(list(set(self.attr_df['node'])))
        self.attr_freqs_by_type = dict((t, defaultdict(int)) for t in self.attr_types)
        for (t, val) in zip(self.attr_df['attributeType'], self.attr_df['attributeVal']):
            self.attr_freqs_by_type[t][val] += 1
        self.num_unique_attrs_by_type = dict((t, len(self.attr_freqs_by_type[t])) for t in self.attr_types)
        self.num_attr_instances_by_type = dict((t, sum(self.attr_freqs_by_type[t].values())) for t in self.attr_types)
        self.sorted_attr_freqs_by_type = dict((t, sorted(self.attr_freqs_by_type[t].items(), key = lambda pair : pair[1], reverse = True)) for t in self.attr_types)
        self.attrs_by_node_by_type = dict((attr_type, defaultdict(set)) for attr_type in self.attr_types)
        for (i, node, attr_type, attr_val) in self.attr_df.itertuples():
                self.attrs_by_node_by_type[attr_type][node].add(attr_val)
    def attr_freq_df(self, rank_thresh = 100):
        afdf = pd.DataFrame(columns = ['rank', 'attrVal', 'freq', 'cumulative %', 'type'])
        for t in self.attr_types:
            df = pd.DataFrame(columns = afdf.columns)
            df['rank'] = list(range(rank_thresh))
            saf = self.sorted_attr_freqs_by_type[t]
            df['attrVal'] = [pair[0] for pair in saf[:rank_thresh]]
            df['freq'] = [pair[1] for pair in saf[:rank_thresh]]
            df['cumulative %'] = 100 * np.cumsum(df['freq']) / self.num_attr_instances_by_type[t]
            df['type'] = t
            afdf = afdf.append(df)
        return afdf
    def rank_plot(self, rank_thresh = 100, show = True, filename = None):
        pass
    #     """Returns plot of the frequencies of the attributes, sorted by rank."""
    #     plt.figure()
    #     afdf = self.attr_freq_df(rank_thresh)
    #     cmap = plt.cm.gist_ncar
    #     colors = {i : cmap(int((i + 1) * cmap.N / (self.num_attr_types + 1.0))) for i in range(self.num_attr_types)}
    #     fig, (ax1, ax2) = plt.subplots(2, 1, sharex = False, sharey = False, facecolor = 'white')
    #     plots_for_legend = []
    #     for (i, t) in enumerate(self.attr_types):
    #         afdf_for_type = afdf[afdf['type'] == t]
    #         plots_for_legend.append(ax1.plot(afdf_for_type['rank'], np.log10(afdf_for_type['freq']), color = colors[i], linewidth = 2)[0])
    #         ax2.plot(afdf_for_type['rank'], afdf_for_type['cumulative %'], color = colors[i], linewidth = 2)
    #     ax1.set_title('Attribute frequencies by type', fontweight = 'bold')
    #     ax2.set_xlabel('rank')
    #     ax1.set_ylabel('log10(freq)')
    #     ax2.set_ylabel('cumulative %')
    #     ax2.set_ylim((0, 100))
    #     ax1.grid(True, 'major', color = 'w', linestyle = '-')
    #     ax2.grid(True, 'major', color = 'w', linestyle = '-')
    #     ax1.set_axisbelow(True)
    #     ax2.set_axisbelow(True)
    #     ax1.patch.set_facecolor('0.89')
    #     ax2.patch.set_facecolor('0.89')
    #     plt.figlegend(plots_for_legend, self.attr_types, 'right', fontsize = 10)
    #     if filename:
    #         plt.savefig(filename)
    #     if show:
    #         plt.show(block = False)
    # def loglog_rank_plot(self, show = True, filename = None):
    #     """Returns log-log plot of the frequencies of the attributes, sorted by rank."""
    #     plt.figure()
    #     cmap = plt.cm.gist_ncar
    #     colors = {i : cmap(int((i + 1) * cmap.N / (self.num_attr_types + 1.0))) for i in range(self.num_attr_types)}
    #     fig, ax = plt.subplots(1, 1, facecolor = 'white')
    #     plots_for_legend = []
    #     for (i, t) in enumerate(self.attr_types):
    #         log10_freqs = np.log10(np.array([pair[1] for pair in self.sorted_attr_freqs_by_type[t]]))
    #         log10_ranks = np.log10(np.arange(1, len(log10_freqs) + 1))  # use 1-up rank indexing, since it doesn't matter
    #         plots_for_legend.append(ax.plot(log10_ranks, log10_freqs, color = colors[i], linewidth = 2)[0])
    #     ax.set_title('Attribute frequencies by type', fontweight = 'bold')
    #     ax.set_xlabel('log10(rank)')
    #     ax.set_ylabel('log10(freq)')
    #     ax.grid(True, 'major', color = 'w', linestyle = '-')
    #     ax.patch.set_facecolor('0.89')
    #     ax.set_axisbelow(True)
    #     plt.figlegend(plots_for_legend, self.attr_types, 'right', fontsize = 10)
    #     if filename:
    #         plt.savefig(filename)
    #     if show:
    #         plt.show(block = False)
    def attr_report(self, rank_thresh = 100):
        """Text string containing frequency info of each attr_type, and the list of top-ranked attribute values with their frequencies."""
        pd.set_option('display.max_rows', rank_thresh)
        afdf = self.attr_freq_df(rank_thresh)
        s = ''
        for t in self.attr_types:
            typelen = len(t)
            s += '#' * (typelen + 4) + '\n' + '# ' + t + ' #\n' + '#' * (typelen + 4) + '\n\n'
            s += 'number of overall occurrences = %d\n' % self.num_attr_instances_by_type[t]
            s += 'number of unique occurrences  = %d\n\n' % self.num_unique_attrs_by_type[t]
            s += afdf[afdf['type'] == t][['rank', 'attrVal', 'freq', 'cumulative %']].to_string(index = False)
            s += '\n\n'
        return s
    def make_pairwise_freq_analyzer(self, attr_type, edges_filename, verbose = True):
        """Makes PairwiseFreqAnalyzer object for a given attribute type using edges from a given edge list filename. The PairwiseFreqAnalyzer object can be used to perform statistics on pairwise attribute counts and to compute pairwise similarity matrices between attributes."""
        if verbose:
            print("\nMaking PairwiseFreqAnalyzer for attribute type '%s' using edges from file '%s'..." % (attr_type, edges_filename))
        attrs_by_node = self.attrs_by_node_by_type[attr_type]
        vocab = set()
        for i in range(self.num_nodes):
            if (i in attrs_by_node):
                vocab.update(attrs_by_node[i])
            else:  # include unique unknown token for each unattributed node
                vocab.add('*???*_%d' % i)
        vocab = sorted(list(vocab)) # sort alphabetically
        pfa = PairwiseFreqAnalyzer(vocab)
        with open(edges_filename, 'r') as f:
            for (i, line) in enumerate(f):
                v1, v2 = [int(token) for token in line.split()[:2]]
                if (v1 in attrs_by_node):
                    if (v2 in attrs_by_node):
                        for val1 in attrs_by_node[v1]:
                            for val2 in attrs_by_node[v2]:
                                pfa.add_pair((val1, val2))
                    else:
                        for val1 in attrs_by_node[v1]:
                            pfa.add_pair((val1, ('*???*_%d' % v2)))
                else:
                    if (v2 in attrs_by_node):
                        for val2 in attrs_by_node[v2]:
                            pfa.add_pair((('*???*_%d' % v1), val2))
                    else:
                        pfa.add_pair((('*???*_%d' % v1), ('*???*_%d' % v2)))
        pfa.finalize_construction()
        return pfa
    def make_uncollapsed_operator(self, pfa, attr_type, sim = 'NPMI1s', delta = 0.0, verbose = True):
        """Given a PairwiseFreqAnalyzer for an attribute type, creates the uncollapsed SparseLinearOperator representing the attribute similarities. In this operator, node similarities are replicates of their corresponding attribute similarities (or the average of these, if multiple attributes occur). sim can be 'PMIs', 'NPMI1s', or 'conditional_prob'."""
        attrs_by_node = self.attrs_by_node_by_type[attr_type]
        if verbose:
            print("\nMaking uncollapsed %s operator..." % sim)
        attr_block = pfa.to_sparse_sim_operator(sim, delta)
        mapping = []
        for i in range(self.num_nodes):
            if i in attrs_by_node:
                mapping.append({pfa.vocab_indices[v] for v in attrs_by_node[i]})
            else:
                mapping.append({pfa.vocab_indices['*???*_%d' % i]})
        collapser = CollapseOperator(np.array(mapping), pfa.num_vocab)
        return SymmetricSparseLinearOperator(collapser.transpose() * (attr_block * collapser))
    def get_attribute_indicator(self, attr, attr_type):
        """Given an attribute and its type, returns a Series of indicators for the graph vertices. 1 indicates the presence of the attribute, 0 indicates the presence of other attributes but not the given attribute, and nan indicates the lack of any attributes in that type."""
        attrs_by_node = self.attrs_by_node_by_type[attr_type]
        ind = np.zeros(self.num_nodes, dtype = float)
        for i in range(self.num_nodes):
            if (i in attrs_by_node):
                attrs = attrs_by_node[i]
                if (len(attrs) == 0):
                    ind[i] = np.nan
                elif (attr in attrs):
                    ind[i] = 1.0
            else:
                ind[i] = np.nan
        return pd.Series(ind)

