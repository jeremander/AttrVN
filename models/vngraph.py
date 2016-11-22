import numpy as np
import networkx as nx
import os, sys
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse import dok_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse.linalg import cg
prev_path = os.path.abspath('..')
sys.path.append(prev_path)
from linop import *

def normalize(vec):
    """Normalizes a vector to have unit norm."""
    return vec / np.linalg.norm(vec)

def normalize_mat_rows(mat):
    """Normalizes the rows of a matrix, in-place."""
    for i in range(mat.shape[0]):
        mat[i] = normalize(mat[i])

def get_cumulative_precisions(true_labels, scores):
    """Given a vector of true labels (binary) and a vector of scores, returns vector of cumulative precisions, where precision at k is the proportion of labels, ranked according to decreasing scores, that have label True."""
    n = len(true_labels)
    assert (len(scores) == n)
    pairs = sorted(zip(true_labels, scores), key = lambda pair : pair[1], reverse = True)
    sorted_labels = np.array([pair[0] for pair in pairs])
    return sorted_labels.cumsum() / (np.arange(n) + 1.0)

def scree_plot(eigvals, show = True, filename = None):
    """Makes a scree plot of the absolute value of a series of eigenvalues using matplotlib. If show = True, displays the plot. If filename is not None, saves the plot to this filename."""
    abs_eigvals = np.abs(eigvals)
    ranked_abs_eigvals = np.array(sorted(abs_eigvals, reverse = True))
    plt.plot(ranked_abs_eigvals, linewidth = 3)
    ax = plt.axes()
    ax.set_title('Scree plot of eigenvalues')
    ax.set_xlabel('rank', labelpad = 10)
    ax.set_ylabel('abs(eigenvalue)', labelpad = 15)
    ax.set_ylim(ymin = 0)
    if filename:
        plt.savefig(filename)
    if show:
        plt.show(block = False)

class UndirectedGraph(nx.Graph):
    def degrees(self):
        """Returns array of node degrees."""
        return np.array([pair[1] for pair in self.degree_iter()])
    def component_sizes(self):
        """Returns array of connected component sizes."""
        return np.array(list((len(comp) for comp in nx.connected_components(self))))
    def copy_edges_from_graph(self, graph):
        """While keeping all nodes the same as before, copies all edges from another graph to this one. First removes all edges currently present."""
        self.remove_edges_from(self.edges_iter())
        self.add_edges_from(graph.edges_iter())
    def make_sparse_adjacency_operator(self):
        """Reads the graph from edge list, and returns it as a SymmetricSparseLinearOperator object."""
        if (not hasattr(self, 'sparse_adjacency_operator')):
            n = self.number_of_nodes()
            A = dok_matrix((n, n))
            for (i, j) in self.edges_iter():
                A[i, j] = 1.0
            A = A.tocsr()
            A = A + A.transpose().tocsr()
            self.sparse_adjacency_operator = SymmetricSparseLinearOperator(A)
    def make_graph_embedding_matrix(self, embedding = 'adj', k = 50, tol = None, plot = False, verbose = True):
        """Makes graph embedding matrix, keeping features of only the nodes possessing at least one attribute. embedding options are adjacency matrix (adj), diagonal-added adjacency matrix (adj+diag), normalized Laplacian (normlap), and regularized normalized Laplacian (regnormlap)."""
        if (not (hasattr(self, 'graph_embedding') and (self.graph_embedding_type == embedding) and (self.graph_embedding_k >= k))):
            assert (embedding in ['adj', 'adj+diag', 'normlap', 'regnormlap'])
            self.make_sparse_adjacency_operator()
            A = self.sparse_adjacency_operator
            n = A.shape[0]
            k = min(k, n - 1)
            tol = 1.0 / n if (tol is None) else tol
            matrix_type = 'adjacency' if (embedding == 'adj') else ('diagional-added adjacency' if (embedding == 'diag+adj') else ('normalized Laplacian' if (embedding == 'normlap') else 'regularized normalized Laplacian'))
            if verbose:
                print("\t\tComputing k = %d eigenvectors of %d x %d %s matrix..." % (k, n, n, matrix_type))
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
            self.graph_embedding_matrix = features
            self.graph_embedding_type = embedding
            self.graph_embedding_k = k
            if plot:
                scree_plot(eigvals, show = True)
                
class VNGraph(UndirectedGraph):
    """Graph class for doing vertex nomination. Has methods for embedding the graph as well. To do vertex nomination, the member 'blocks_by_node' must be a binary vector of block labels, and 'observed_flags' must be a binary vector indicating which nodes are observed."""
    def vn_supervised(self, k, embedding = 'adj', classifier = 'logreg', normalize = True, num_trees = 100, verbose = True):
        """Performs vertex nomination on a partially occluded graph by embedding the adjacency/Laplacian matrix and doing soft classification. k is number of features; embedding is 'adj' (adjacency A), 'adj+diag' (A + D/n), 'normlap' (normalized Laplacian D^(-1/2) * A * D^(-1/2)), or 'regnormlap' (regularized normalized Laplacian (D + tau * I)^(-1/2) * A * (D + tau * I)^(-1/2), for tau = mean(D)); classifier is 'logreg' (logistic regression), 'gnb' (Gaussian Naive Bayes), 'rfc' (random forest), or 'boost' (AdaBoost); normalize = True iff feature vectors are to be projected to sphere; num_trees is the number of trees to use if random forest or AdaBoost is used. Returns the vector of class probabilities corresponding to the unobserved nodes."""
        self.make_sparse_adjacency_operator()
        self.make_graph_embedding_matrix(embedding = embedding, k = k, tol = None, plot = False, verbose = verbose)
        if normalize:
            normalize_mat_rows(self.graph_embedding_matrix)
        if (classifier == 'logreg'):
            clf = LogisticRegression()
        elif (classifier == 'gnb'):
            clf = GaussianNB()
        elif (classifier == 'rfc'):
            clf = RandomForestClassifier(n_estimators = num_trees)
        else:  # AdaBoost
            clf = AdaBoostClassifier(n_estimators = num_trees)
        observed_nodes, unobserved_nodes = self.observed_flags.nonzero()[0], (~(self.observed_flags)).nonzero()[0]
        train_in, test_in = self.graph_embedding_matrix[observed_nodes, :k], self.graph_embedding_matrix[unobserved_nodes, :k]
        train_out = self.blocks_by_node[observed_nodes]
        clf.fit(train_in, train_out)
        test_probs = clf.predict_proba(test_in)[:, 1]
        return test_probs
    def vn_randomwalk(self, steps = 1, stoch = True):
        """Performs vertex nomination on a partially occluded graph by stepping out a Markov chain using the stochasticized adjacency matrix some number of times (where the initial distribution is the superposition of uniform distributions over observations, positive if block 1, negative if block 0) and taking the mean of the distribution vectors for each step."""
        assert ((~self.observed_flags).sum() > 0), "Must have at least one unobserved node to do vertex nomination."
        self.make_sparse_adjacency_operator()
        A = self.sparse_adjacency_operator
        if stoch:
            A = A.to_column_stochastic()
        n = A.shape[0]
        observed_nodes, unobserved_nodes = self.observed_flags.nonzero()[0], (~self.observed_flags).nonzero()[0]
        num_ones = (self.blocks_by_node[observed_nodes] == 1).sum()
        num_zeros = len(observed_nodes) - num_ones
        x_plus, x_minus = np.zeros((n, steps + 1), dtype = float), np.zeros((n, steps + 1), dtype = float)
        if stoch:
            p_plus = (1.0 / num_ones) if (num_ones >= 1) else 0.0
            p_minus = (1.0 / num_zeros) if (num_zeros >= 1) else 0.0
        else:
            p_plus = p_minus = 1.0
        for i in observed_nodes:  # initialize the Markov chains
            if (self.blocks_by_node[i] == 1):
                x_plus[i, 0] = p_plus
            else:
                x_minus[i, 0] = p_minus
        for t in range(steps):
            x_plus[:, t + 1] = A * x_plus[:, t]
            x_minus[:, t + 1] = A * x_minus[:, t]
        scores = x_plus - x_minus  # scores for each step
        mean_scores = np.mean(scores, axis = 1)
        return mean_scores[unobserved_nodes]
    def vn_diffusion(self, proportional = False, tol = 1e-8):
        """Performs vertex nomination on a partially occluded graph by solving the steady-state heat equation on the graph, where seed temperatures (high for positive examples, low for negative examples) are the boundary conditions. If proportional = False, default seed temps are [-1, 1], otherwise default seed_temps are [-1 / (# observed 0), 1 / (# observed 1)], scaled so that the larger absolute value is 1."""
        assert ((~self.observed_flags).sum() > 0), "Must have at least one unobserved node to do vertex nomination."
        self.make_sparse_adjacency_operator()
        n = len(self.observed_flags)
        observed_nodes, unobserved_nodes = self.observed_flags.nonzero()[0], (~self.observed_flags).nonzero()[0]
        pos_seeds = (self.observed_flags & (self.blocks_by_node == 1)).nonzero()[0]
        neg_seeds = (self.observed_flags & (self.blocks_by_node == 0)).nonzero()[0]
        num_pos_seeds = (self.blocks_by_node[observed_nodes] == 1).sum()
        num_neg_seeds = len(observed_nodes) - num_pos_seeds
        num_seeds = num_pos_seeds + num_neg_seeds
        if proportional:
            seed_temps = np.array([-1.0 / (self.n - num_seeds), 1.0 / num_seeds])
            seed_temps /= np.abs(self.seed_temps).max()
        else:
            seed_temps = np.array([-1.0, 1.0])
        T0 = np.zeros(n, dtype = float)
        for i in neg_seeds:
            T0[i] = seed_temps[0]
        for i in pos_seeds:
            T0[i] = seed_temps[1]
        P = PermutationOperator(np.concatenate([observed_nodes, unobserved_nodes]))
        T0 = (P * T0)[:num_seeds]
        Lp = P * (SparseLaplacian(self.sparse_adjacency_operator) * P.inv())
        F_seed = RowSelectorOperator(num_seeds, n, range(num_seeds))
        F_nonseed = RowSelectorOperator(n - num_seeds, n, range(num_seeds, n))
        b = -(F_nonseed * (Lp * (F_seed.transpose() * T0)))
        A = F_nonseed * (Lp * F_nonseed.transpose())
        scores = cg(A, b, tol = tol)[0]
        return scores

