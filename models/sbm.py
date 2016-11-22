import fractions
import warnings
import scipy.stats
import itertools
from collections import defaultdict
from .vngraph import *
from importlib import reload
from copy import deepcopy
#from pgmpy.models import MarkovModel
#from pgmpy.factors import Factor


def plot_beta_pdf(rv):
    """Plot a beta distribution pdf. Input is an instance of scipy.stats.beta."""
    xs = np.linspace(0.0, 1.0, 10000)
    plt.plot(xs, rv.pdf(xs), linewidth = 2)
    plt.title("Beta pdf\nalpha = %.3f, beta = %.3f" % (rv.args[0], rv.args[1]))
    plt.show(block = False)

def beta_mode(rv):
    """Given a beta random variable, computes its mode. If alpha + beta = 2, returns 1/2. If alpha or beta < 1, issues a warning."""
    (alpha, beta) = rv.args
    if (min(alpha, beta) < 1.0):
        warnings.warn("beta distribution with parameter < 1")
    if (alpha + beta == 2.0):
        return 0.5
    return (alpha - 1.0) / (alpha + beta - 2.0)

def dirichlet_mode(rv):
    """Given a Dirichlet random variable, computes its mode. If any theta_i < 1, issues a warning."""
    if (rv.alpha.min() < 1.0):
        warnings.warn("Dirichlet distribution with parameter < 1")
    return (rv.alpha - 1.0) / (rv.alpha.sum() - len(rv.alpha))

def logp_sum_logp(logps):
    """Given array of log-probabilities, adds together the corresponding probabilities (avoiding underflow), then returns the log-probability of the sum."""
    max_logp = logps.max()
    scaled_probs = np.exp(logps - max_logp)
    return max_logp + np.log(scaled_probs.sum())

def shifted_logprobs_to_normalized_probs(logps):
    """Given array of (shifted) log-probabilities, returns array of normalized probabilities."""
    max_logp = logps.max()
    scaled_probs = np.exp(logps - max_logp)
    return scaled_probs / scaled_probs.sum()

def log_odds_to_prob(log_odds):
    """Given log(p / (1 - p)), returns p."""
    z = np.exp(log_odds)
    return z / (1 + z)

class Pair(object):
    def __init__(self, first, second):
        self.items = (first, second)
    def __getitem__(self, i):
        return self.items[i]
    def __repr__(self):
        return str(self.items)

class Fraction(fractions.Fraction):
    def __repr__(self):
        return str(self)

def safe_div(x, y, dtype = float):
    if (dtype == Pair):
        return Pair(x, y)
    elif (dtype == Fraction):
        return Fraction(x, y)
    else:
        return (float(x) / y) if (y != 0) else np.nan

def empirical_block_probs(blocks_by_node, observed_flags, K, dtype = float):
    """Computes empirical block membership probabilities (known only)."""
    block_counts = np.bincount(blocks_by_node[observed_flags], minlength = K)    
    return np.vectorize(lambda x, y : safe_div(x, y, dtype))(block_counts, block_counts.sum())

def empirical_edge_probs(blocks_by_node, observed_flags, K, edges_iter, dtype = float):
    """Computes empirical edge probabilities (between pairs of known edges only). edges_iter is an iterator of edges."""
    block_counts = np.bincount(blocks_by_node[observed_flags], minlength = K)    
    nums = np.zeros((K, K), dtype = int)
    denoms = np.zeros((K, K), dtype = int)
    for i in range(K):
        for j in range(i, K):
            if (i == j):
                denoms[i, i] = block_counts[i] * (block_counts[i] - 1) // 2
            else:
                denoms[i, j] = denoms[j, i] = block_counts[i] * block_counts[j]
    for (v1, v2) in edges_iter:
        (i, j) = (blocks_by_node[v1], blocks_by_node[v2])
        if (observed_flags[v1] and observed_flags[v2]):
            nums[i, j] += 1
            if (i != j):
                nums[j, i] += 1
    return np.vectorize(lambda x, y : safe_div(x, y, dtype))(nums, denoms)

class SBM(object):
    """Class for Stochastic Block Models."""
    def __init__(self, edge_probs, block_probs = None):
        """edge_probs is symmetric matrix of communication probabilities; block_probs is vector of block probabilities."""
        assert (edge_probs.shape[0] == edge_probs.shape[1])
        assert np.allclose(edge_probs.transpose(), edge_probs)
        assert (0.0 <= edge_probs.min() <= edge_probs.max() <= 1.0)
        self.edge_probs = edge_probs
        self.K = edge_probs.shape[0]
        if (block_probs is None):
            self.block_probs = np.ones(self.K, dtype = float) / self.K  # uniform distribution
        else:
            assert (len(block_probs) == self.K)
            assert (0.0 <= block_probs.min() <= block_probs.max() <= 1.0)
            assert (sum(block_probs) == 1.0)
            self.block_probs = block_probs
        # compute log-probs as one-time work
        self.log_block_probs = np.log(self.block_probs)  
        self.log_edge_probs = np.log(self.edge_probs)
        self.log_one_minus_edge_probs = np.log(1.0 - self.edge_probs)
    def _sample_block_memberships(self, n = None, block_memberships = None):
        """Samples block memberships in an SBM if block_memberships is None; otherwise, these are the block memberships. Returns SBMGraph with the given memberships."""
        if (block_memberships is None):
            if (n is None):
                blocks_by_node = [i for i in range(self.K) for j in range(10)]  # 10 nodes per block by default
                n = 10 * self.K
            else:
                blocks_by_node = list(np.random.choice(self.K, n, p = self.block_probs))
        else:
            if (n is not None):
                assert (n == sum(block_memberships))
            blocks_by_node = [i for i in range(self.K) for j in range(block_memberships[i])]
        if (self.K == 2):
            return TwoBlockSBMGraph(np.array(blocks_by_node))
        return SBMGraph(self.K, np.array(blocks_by_node))
    def sample(self, n = None, block_memberships = None):
        """Samples an SBM using the inefficient (O(N^2), edge-by-edge) method. Optional block_memberships is a vector of memberships by block."""
        g = self._sample_block_memberships(n, block_memberships)
        for i in range(g.n):
            for j in range(i + 1, g.n):
                if (np.random.rand() <= self.edge_probs[g.blocks_by_node[i], g.blocks_by_node[j]]):
                    g.add_edge(i, j)
        return g
    def sample_sparse(self, n = None, block_memberships = None):
        """Samples an SBM using the efficient (sparse binomial sampling) method. Optional block_memberships is a vector of memberships by block."""
        g = self._sample_block_memberships(n, block_memberships)
        for i in range(self.K):
            nodes_i = np.arange(g.n)[g.blocks_by_node == i]
            for j in range(i, self.K):
                if (i == j):
                    block_count = g.block_counts[i]
                    edge_set_size = block_count * (block_count - 1) // 2
                    num_edges = np.random.binomial(edge_set_size, self.edge_probs[i, i])
                    edge_indices = np.random.choice(range(edge_set_size), num_edges, replace = False)
                    for k in edge_indices:
                        ii = int(((2 * block_count - 1) - np.sqrt((1 - 2 * block_count) ** 2 - 8 * k)) // 2)
                        jj = ii + 1 + k - int((ii * (2 * block_count - 1) - (ii ** 2)) // 2)
                        g.add_edge(nodes_i[ii], nodes_i[jj])
                        edge = (nodes_i[ii], nodes_i[jj])
                else:
                    nodes_j = np.arange(g.n)[g.blocks_by_node == j]
                    block_count_i, block_count_j = g.block_counts[i], g.block_counts[j]
                    edge_set_size = block_count_i * block_count_j
                    num_edges = np.random.binomial(edge_set_size, self.edge_probs[i, j])
                    edge_indices = np.random.choice(range(edge_set_size), num_edges, replace = False)
                    for k in edge_indices:
                        ii = int(k // block_count_j)
                        jj = k % block_count_j
                        g.add_edge(nodes_i[ii], nodes_j[jj])
        return g

class TwoBlockSBM(SBM):
    """Special case of SBM where there are only two blocks."""
    def __init__(self, p00, p01, p11, p1 = None):
        edge_probs = np.array([[p00, p01], [p01, p11]])
        block_probs = None if (p1 is None) else np.array([1.0 - p1, p1])
        super().__init__(edge_probs, block_probs)

class SBMGraph(VNGraph):
    """An instantiation of a Stochastic Block Model."""
    def __init__(self, K, blocks_by_node, observed_flags = None):
        super().__init__()
        self.K = K
        self.n = len(blocks_by_node)
        if (self.n > 0):
            assert (0 <= blocks_by_node.min() <= blocks_by_node.max() < K)
        self.blocks_by_node = blocks_by_node
        self.observed_flags = np.ones(self.n, dtype = bool) if (observed_flags is None) else observed_flags
        self.observed_nodes, self.unobserved_nodes = self.observed_flags.nonzero()[0], (~self.observed_flags).nonzero()[0]
        # only include known blocks
        self.block_counts = np.bincount(self.blocks_by_node[self.observed_flags], minlength = self.K)
        if (min(self.block_counts) < 2):
            warnings.warn("Block exists with low membership (less than 2).")
        for i in range(self.n):
            self.add_node(i, block = self.blocks_by_node[i])
    def mcar_occlude(self, num_to_occlude = None, occlusion_prob = 0.75):
        """Occludes nodes in an i.i.d. Bernoulli fashion (MCAR = "missing completely at random"). Either the probability of occlusion is supplied or the number to occlude is supplied."""
        if (num_to_occlude is None):
            occlusion_flags = (np.random.rand(self.n) <= occlusion_prob)
        else:
            assert (num_to_occlude <= self.n)
            occlusion_flags = np.random.permutation(([True] * num_to_occlude) + ([False] * (self.n - num_to_occlude)))
        g = TwoBlockSBMGraph(self.blocks_by_node, ~occlusion_flags) if (self.K == 2) else SBMGraph(self.K, self.blocks_by_node, ~occlusion_flags)
        g.copy_edges_from_graph(self)
        return g
    def to_two_block_sbm(self, selected_block):
        """Converts the SBMGraph into an identical graph with only two blocks, where the selected block becomes block 1, and the rest of the blocks are combined to form block 0."""
        g = TwoBlockSBMGraph(np.array([1 if (block == selected_block) else 0 for block in self.blocks_by_node]), self.observed_flags)
        g.copy_edges_from_graph(self)
        return g
    def draw(self, with_labels = False):
        """Assumes the nodes have been assigned blocks and colors them appropriately. Unobserved nodes are colored black."""
        plt.figure()
        cmap = plt.cm.gist_ncar
        cdict = {i : cmap(int((i + 1) * cmap.N / (self.K + 1.0))) for i in range(self.K)}
        black_color = (0.0, 0.0, 0.0, 1.0)
        nx.draw_networkx(self, node_color = [(cdict[self.blocks_by_node[i]] if self.observed_flags[i] else black_color) for i in range(self.n)], with_labels = with_labels, node_size = 100) 
        plt.axes().get_xaxis().set_ticks([])
        plt.axes().get_yaxis().set_ticks([])
        plt.title("Stochastic Block Model\n%d blocks, %d nodes" % (self.K, self.n))
        plt.show(block = False)
    def empirical_block_probs(self, dtype = float):
        return empirical_block_probs(self.blocks_by_node, self.observed_flags, self.K, dtype = dtype)
    def empirical_edge_probs(self, dtype = float):
        return empirical_edge_probs(self.blocks_by_node, self.observed_flags, self.K, self.edges_iter(), dtype = dtype)

class TwoBlockSBMGraph(SBMGraph):
    def __init__(self, blocks_by_node, observed_flags = None):
        """Initialize from binary block indicators and observed indicators."""
        super().__init__(2, blocks_by_node, observed_flags)
    def vn_exact_marginals(self, sbm = None):
        """Creates SBM_MRF model using the given parameters (or if sbm = None, the MAP estimate of these parameters using a uniform prior), then computes the exact marginal probabilities of the unknown nodes. Can only do this if there are < 25 unknown nodes in order not to exceed computational constraints."""
        mrf = SBM_MRF(self)
        if (sbm is None):
            mrf.init_sbm(style = 'MAP')
        else:
            mrf.set_sbm(sbm)
        mrf.compute_exact_logp_marginals()
        return mrf.exact_logp_marginals[self.unobserved_nodes, 1]
    def vn_induced_subgraph(self, sbm = None):
        """Performs vertex nomination on a partially occluded graph by computing (up to additive constant) log-odds of full conditional probabilities of each unknown node w.r.t. the observed subgraph."""
        if (sbm is None):
            mrf = SBM_MRF(self)
            mrf.init_sbm(style = 'MAP', observed = True)
            sbm = mrf.sbm
        self.make_sparse_adjacency_operator()
        A = self.sparse_adjacency_operator
        n = self.n
        x = np.zeros(n, dtype = float)
        plus_val = np.log((sbm.edge_probs[1, 1] * (1 - sbm.edge_probs[0, 1])) / (sbm.edge_probs[0, 1] * (1 - sbm.edge_probs[1, 1])))
        minus_val = np.log((sbm.edge_probs[1, 0] * (1 - sbm.edge_probs[0, 0])) / (sbm.edge_probs[0, 0] * (1 - sbm.edge_probs[1, 0])))
        for i in self.observed_nodes:
            if (self.blocks_by_node[i] == 1):
                x[i] = plus_val
            else:
                x[i] = minus_val
        scores = A * x
        return scores[self.unobserved_nodes]
    def vn_mean_field_naive(self, sbm = None):
        """Performs vertex nomination on a partially occluded graph by computing (up to additive constant) log-odds of conditional probabilities of each unknown node w.r.t. the observed subgraph, along with naive mean-field approximation over unknown nodes."""
        if (sbm is None):
            mrf = SBM_MRF(self)
            mrf.init_sbm(style = 'MAP', observed = True)
            sbm = mrf.sbm
        self.make_sparse_adjacency_operator()
        A = self.sparse_adjacency_operator
        n = self.n
        x = np.zeros(n, dtype = float)
        Lambda0 = (sbm.edge_probs[0, 0] + sbm.edge_probs[0, 1]) / 2.0
        Lambda1 = (sbm.edge_probs[1, 0] + sbm.edge_probs[1, 1]) / 2.0
        plus_val = np.log((sbm.edge_probs[1, 1] * (1 - sbm.edge_probs[0, 1])) / (sbm.edge_probs[0, 1] * (1 - sbm.edge_probs[1, 1])))
        minus_val = np.log((sbm.edge_probs[1, 0] * (1 - sbm.edge_probs[0, 0])) / (sbm.edge_probs[0, 0] * (1 - sbm.edge_probs[1, 0])))
        neutral_val = np.log((Lambda1 * (1 - Lambda0)) / (Lambda0 * (1 - Lambda1)))
        # geom mean instead of arithmetic? seems to be a bit worse
        #neutral_val = 0.5 * (sbm.log_edge_probs[1, 0] + sbm.log_edge_probs[1, 1] + sbm.log_one_minus_edge_probs[0, 0] + sbm.log_one_minus_edge_probs[0, 1] - sbm.log_edge_probs[0, 0] - sbm.log_edge_probs[0, 1] - sbm.log_one_minus_edge_probs[1, 0] - sbm.log_one_minus_edge_probs[1, 1])
        for i in range(n):
            if (i in self.observed_nodes):
                if (self.blocks_by_node[i] == 1):
                    x[i] = plus_val
                else:
                    x[i] = minus_val
            else:
                x[i] = neutral_val
        scores = A * x
        return scores[self.unobserved_nodes]
    def vn_mean_field_opt(self, sbm = None, tol = 1e-12, max_iters = 100, verbose = False):
        """Performs vertex nomination on a partially occluded graph using the mean-field approximation algorithm (see Koller & Friedman, Ch. 11.5)."""
        if (sbm is None):
            mrf = SBM_MRF(self)
            mrf.init_sbm(style = 'MAP', observed = True)
            sbm = mrf.sbm
        self.make_sparse_adjacency_operator()
        n = self.n
        A_mat = self.sparse_adjacency_operator.F
        observed_vec = np.zeros(n, dtype = float)
        observed_1_val = sbm.log_edge_probs[1, 1] + sbm.log_one_minus_edge_probs[0, 1] - sbm.log_edge_probs[0, 1] - sbm.log_one_minus_edge_probs[1, 1]
        observed_0_val = sbm.log_edge_probs[1, 0] + sbm.log_one_minus_edge_probs[0, 0] - sbm.log_edge_probs[0, 0] - sbm.log_one_minus_edge_probs[1, 0]
        unobserved_nbr0_val = sbm.log_edge_probs[1, 0] - sbm.log_edge_probs[0, 0]
        unobserved_nbr1_val = sbm.log_edge_probs[1, 1] - sbm.log_edge_probs[0, 1]
        unobserved_nonnbr0_val = sbm.log_one_minus_edge_probs[1, 0] - sbm.log_one_minus_edge_probs[0, 0]
        unobserved_nonnbr1_val = sbm.log_one_minus_edge_probs[1, 1] - sbm.log_one_minus_edge_probs[0, 1]
        unobserved_probs = 0.5 * np.ones(n, dtype = float)  # initialize unobserved marginals uniformly
        for i in self.observed_nodes:
            if (self.blocks_by_node[i] == 1):
                observed_vec[i] = observed_1_val
            else:
                observed_vec[i] = observed_0_val
            unobserved_probs[i] = 0.0
        observed_indicator = np.asarray(self.observed_flags, dtype = float)
        observed_scores = (sbm.log_block_probs[1] - sbm.log_block_probs[0]) + A_mat * observed_vec
        iteration = 0 
        delta = np.inf
        while ((delta > tol) and (iteration < max_iters)):  # stop when all score changes are beneath the tolerance, or when we've reached the max number of iterations
            if verbose:
                print("\t\tMean field iteration #%d" % iteration)
            delta = 0.0
            for i in self.unobserved_nodes:
                unobserved_vec_nbr = (1 - unobserved_probs - observed_indicator) * unobserved_nbr0_val + unobserved_probs * unobserved_nbr1_val 
                unobserved_vec_nonnbr = (1 - unobserved_probs - observed_indicator) * unobserved_nonnbr0_val + unobserved_probs * unobserved_nonnbr1_val
                log_odds = observed_scores[i] + (A_mat[i] * unobserved_vec_nbr)[0] + unobserved_vec_nonnbr.sum() - unobserved_vec_nonnbr[i] - (A_mat[i] * unobserved_vec_nonnbr)[0]
                prob = log_odds_to_prob(log_odds)
                delta = max(delta, np.abs(prob - unobserved_probs[i]))
                unobserved_probs[i] = prob
            if verbose:
                print("\t\t\tMax delta(probs) = %g" % delta)
            iteration += 1
        return np.log(unobserved_probs[self.unobserved_nodes])  # return log-probs of unobserved nodes
    def vn_gibbs(self, iters = 100, burn = 0, thin = 1, sbm = None, verbose = False):
        """Performs vertex nomination using Gibbs sampling, returning the estimated marginal probabilities for each unobserved node. If sbm is None, uses MAP estimate of the SBM parameters to initialize, then includes these parameters in the sampling, using uniform priors. Otherwise, SBM parameters are fixed to the specified values."""
        mrf = SBM_MRF(self)
        if (sbm is None):
            mrf.init_sbm(style = 'MAP', observed = True)
        else:
            mrf.set_sbm(sbm)
        mrf.gibbs_sample(iters = iters, burn = burn, thin = thin, update_params = (sbm is None), verbose = verbose)
        return mrf.traces['unobserved_blocks'].mean(axis = 1)
    def vn_sparse_approx_bp(self, sbm = None, tol = 1e-8, max_iters = 100, verbose = False):
        """Performs vertex nomination by constructing MRF assuming sparsity assumption (non-edge potentials are negligible), then performs loopy BP to estimate the posterior marginals. If sbm is None, uses MAP estimate of the SBM parameters when doing BP."""
        mrf = SBM_MRF(self)
        if (sbm is None):
            mrf.init_sbm(style = 'MAP', observed = True)
        else:
            mrf.set_sbm(sbm)
        approx_mrf = SparseApproxSBM_MRF.from_mrf(mrf)
        approx_mrf.belief_propagate(tol = tol, max_iters = max_iters, verbose = verbose)
        return approx_mrf.bp_logp_marginals[self.unobserved_nodes, 1]
    @classmethod
    def from_vngraph(cls, graph):
        """Convenience constructor for initalizing 2-block SBM from a VNGraph containing attributes 'blocks_by_node' and 'observed_flags'."""
        g = cls(graph.blocks_by_node, graph.observed_flags)
        g.copy_edges_from_graph(graph)
        return g


class MCMC(object):
    def plot(self, var_name, indices = (), true_dist = None):
        """Presents a trace plot, autocorrelation plot, and sample frequency histogram (normalized) for a given variable name to aid in visual inspection of MCMC convergence. indices are the indices of the array, if applicable. If true_dist is supplied, plots this pdf on top of the histogram."""
        samps = self.traces[var_name] if (len(indices) == 0) else self.traces[var_name][indices]
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.plot(samps)
        ax1.set_title('Trace plot of %s' % var_name)
        ax1.set_xlabel('sample #')
        ax2 = fig.add_subplot(312)
        ax2.acorr(samps - samps.mean(), maxlags = 100)
        ax2.set_title('Autocorrelations')
        ax2.set_xlabel('lag')
        ax2.set_ylabel('corr')
        ax2.set_xlim((0, 100))
        ax3 = fig.add_subplot(313)
        ax3.hist(samps, bins = 100, normed = True, color = 'blue')
        if (true_dist is not None):
            xs = np.linspace(samps.min(), samps.max(), 1000)
            ax3.plot(xs, true_dist.pdf(xs), color = 'red', linewidth = 2)
        ax3.set_title('Empirical PDF')
        plt.tight_layout()
        plt.show(block = False)

class PairwiseMRF(object):
    """Pairwise Markov Random Field containing a graph and a method for computing log-probability (up to proportionality). Each node has a distribution on integers [0, ..., K - 1]. Should contain properties K, n, blocks_by_node, observed_flags, unobserved_nodes, etc.."""
    def logp(self):
        """Returns log probability of the joint distribution using the current SBM parameters. Work is on the order of the number of edges incident to unknown edges."""
        block_counts = np.array([pair[0] for pair in empirical_block_probs(self.blocks_by_node, np.ones(self.n, dtype = bool), self.K, dtype = Pair)])
        unobserved_edge_probs = empirical_edge_probs(self.blocks_by_node, np.ones(self.n, dtype = bool), self.K, self.unobserved_edges, dtype = Pair)
        total_logp = np.dot(block_counts, np.log(self.sbm.block_probs))
        for i in range(self.K):
            for j in range(i, self.K):
                # assume we're in an SBM or SparseApproxSBM setting
                total_logp += (unobserved_edge_probs[i, j][0] + self.observed_edge_probs[i, j][0]) * np.log(self.sbm.edge_probs[i, j]) 
                if isinstance(self, SBM_MRF):  # also include non-edges
                    total_logp += (unobserved_edge_probs[i, j][1] - (unobserved_edge_probs[i, j][0] + self.observed_edge_probs[i, j][0])) * np.log(1.0 - self.sbm.edge_probs[i, j])
        return total_logp
    def compute_exact_logp_marginals(self):
        """Computes exact marginal probabilities for each node. Does this by computing vector of log-probabilities for all possible states of the unobserved nodes, then summing appropriately. Requires the number of possible states be less than 2**25 to prevent excess memory consumption. Work & memory are O(K^num_unobserved)."""
        unobserved_flags = ~self.graph.observed_flags
        num_unobserved = int(unobserved_flags.sum())
        assert (self.graph.K ** num_unobserved < (1 << 25)), "Too many unobserved nodes."
        init_blocks_by_node = deepcopy(self.blocks_by_node)
        vec = np.zeros(self.graph.K ** num_unobserved, dtype = float)
        for (i, v) in enumerate(itertools.product(*[range(self.graph.K) for i in range(num_unobserved)])):
            self.blocks_by_node[unobserved_flags] = np.array(v, dtype = int)
            vec[i] = self.logp()
        self.blocks_by_node = init_blocks_by_node  # reset to initial state
        self.exact_logp_marginals = np.zeros((self.n, self.K), dtype = float)
        for (i, u) in enumerate(self.unobserved_nodes):
            size = self.K ** (num_unobserved - i - 1)
            logps = np.zeros(self.K, dtype = float)
            for j in range(self.K):
                inds = list(itertools.chain(*(range(start, start + size) for start in range(j * size, self.graph.K ** num_unobserved, self.graph.K * size))))
                logps[j] = logp_sum_logp(vec[inds])
            self.exact_logp_marginals[u, :] = logps - logp_sum_logp(logps)
        unobserved_nodes = set(self.unobserved_nodes)
        for i in range(self.n):
            if (i not in unobserved_nodes):
                for j in range(self.K):
                    if (self.blocks_by_node[i] != j):
                        self.exact_logp_marginals[i, j] = -np.inf

class SBM_MRF(MCMC, PairwiseMRF):
    """Markov Random Field model of an SBM where parameters are unknown, but where all edges and possibly some node memberships are observed. The parameters are given prior distributions (Dirichlet/beta for block probabilities, i.i.d. beta for edge probabilities). This model is capable of performing Gibbs sampling to estimate the posterior distributions of parameters and unknown node memberships."""
    def __init__(self, graph, theta = None, alphas = None, betas = None):
        """Initialize with SBMGraph (where -1's represent unknown nodes). If vector theta is supplied, this is hyperparameter vector (dimension m) for Dirichlet prior of block probabilities. If alpha and betas are supplied, they are each m x m arrays giving the beta distribution hyperparameters for the edge probabilities. Default is uniform alpha = 1, beta = 1."""
        self.graph = graph
        self.observed_nodes, self.unobserved_nodes = self.graph.observed_nodes, self.graph.unobserved_nodes
        self.unobserved_edges = [(i, j) for (i, j) in self.graph.edges_iter() if not (graph.observed_flags[i] and graph.observed_flags[j])]
        self.observed_edge_probs = empirical_edge_probs(self.graph.blocks_by_node, self.graph.observed_flags, self.graph.K, self.graph.edges_iter(), dtype = Pair)
        self.n, self.K = self.graph.n, self.graph.K
        if (theta is None):
            self.theta = np.ones(self.K)
        else:
            assert (len(theta) == self.K)
            self.theta = theta
        if (alphas is None):
            self.alphas = np.ones((self.K, self.K), dtype = float)
        else:
            assert ((alphas.shape == (self.K, self.K)) and (alphas.transpose() == alphas).all())
            self.alphas = alphas
        if (betas is None):
            self.betas = np.ones((self.K, self.K), dtype = float)
        else:
            assert ((betas.shape == (self.K, self.K)) and (betas.transpose() == betas).all())
            self.betas = betas
        # prior Dirichlet prior distribution (beta if m = 2)
        self.block_prob_prior = scipy.stats.dirichlet(self.theta) if (self.K > 2) else scipy.stats.beta(self.theta[1], self.theta[0])
        # array of prior beta prior distributions
        self.edge_prob_priors = np.vectorize(lambda alpha, beta : scipy.stats.beta(alpha, beta))(self.alphas, self.betas)
        # a copy of the block memberships that will change state
        self.blocks_by_node = np.array(self.graph.blocks_by_node)  
        self.traces = dict()
    def set_posteriors(self, observed = True):
        """Sets the posterior distributions (Dirichlet & betas) based on the current block memberships. If observed = True, only includes observed nodes in forming the posteriors."""
        observed_flags = self.graph.observed_flags if observed else np.ones(self.n, dtype = bool)
        block_probs = empirical_block_probs(self.blocks_by_node, observed_flags, self.K, dtype = Pair)
        edge_probs = empirical_edge_probs(self.blocks_by_node, observed_flags, self.K, self.graph.edges_iter(), dtype = Pair)
        # Dirichlet conjugate prior -> Dirichlet posterior
        posterior_theta = self.theta + np.array([pair[0] for pair in block_probs], dtype = float)
        self.block_prob_posterior = scipy.stats.dirichlet(posterior_theta) if (self.K > 2) else scipy.stats.beta(posterior_theta[1], posterior_theta[0])
        # beta conjugate prior -> beta posterior
        self.edge_prob_posteriors = np.empty((self.K, self.K), dtype = object)
        for i in range(self.K):
            for j in range(self.K):
                self.edge_prob_posteriors[i, j] = scipy.stats.beta(self.alphas[i, j] + edge_probs[i, j][0], self.betas[i, j] + edge_probs[i, j][1])
    def set_sbm(self, sbm):
        """Sets the current state of the SBM parameters with an SBM object. Also resets observed logp accordingly."""
        assert (isinstance(sbm, SBM) and (sbm.K == self.K))
        self.sbm = sbm
    def init_sbm(self, style = 'MAP', observed = True):
        """Initializes the SBM parameters for MCMC. If style is 'MLE', uses the empirical counts to initialize these parameters with the MLE estimate; if style is 'MAP', uses these along with the prior to obtain the MAP estimate; if style is 'prior', sample from the prior distributions; if style is 'posterior', sample from the posterior distributions. If observed = True, observations correspond to the observed flags; otherwise all are observed."""
        if (style == 'MLE'):
            observed_flags = self.graph.observed_flags if observed else np.ones(self.graph.n, dtype = int)
            edge_probs = empirical_edge_probs(self.graph.blocks_by_node, observed_flags, self.K, self.graph.edges_iter(), dtype = float)
            block_probs = empirical_block_probs(self.graph.blocks_by_node, observed_flags, self.K, dtype = float)
        elif (style == 'MAP'):
            self.set_posteriors(observed = observed)
            edge_probs = np.zeros((self.K, self.K), dtype = float)
            for i in range(self.K):
                for j in range(self.K):
                    edge_probs[i, j] = edge_probs[j, i] = beta_mode(self.edge_prob_posteriors[i, j])
            if (self.K > 2):
                block_probs = dirichlet_mode(self.block_prob_posterior)
            else:
                mode = beta_mode(self.block_prob_posterior)  # hopefully alpha and beta >= 1
                block_probs = np.array([1 - mode, mode])
        elif (style == 'prior'):
            edge_probs = np.zeros((self.K, self.K), dtype = float)
            for i in range(self.K):
                for j in range(i, self.K):
                    edge_probs[i, j] = edge_probs[j, i] = self.edge_prob_priors[i, j].rvs()
            if (self.K > 2):
                block_probs = self.block_prob_prior.rvs()[0]
            else:
                prob = self.block_prob_prior.rvs()
                block_probs = np.array([1 - prob, prob])
        else:  # posterior
            self.set_posteriors(observed = observed)
            edge_probs = np.zeros((self.K, self.K), dtype = float)
            for i in range(self.K):
                for j in range(i, self.K):
                    edge_probs[i, j] = edge_probs[j, i] = self.edge_prob_posteriors[i, j].rvs()
            if (self.K > 2):
                block_probs = self.block_prob_posterior.rvs()[0] 
            else:
                prob = self.block_prob_posterior.rvs()
                block_probs = np.array([1 - prob, prob])
        self.set_sbm(SBM(edge_probs, block_probs))
    def init_unobserved_nodes(self):
        """After the SBM parameters have been set, initialize the unobserved node memberships as i.i.d. samples from the SBM block probability distribution (note: this ignores conditioning on the edges). The purpose of this is just to give a random starting point for the Gibbs sampling."""
        cdf = self.sbm.block_probs.cumsum()
        for i in range(self.n):
            if (not self.graph.observed_flags[i]):  # sample from categorical distribution
                self.blocks_by_node[i] = cdf.searchsorted(np.random.rand())
    def init_block_counts(self):
        """Tallies global block counts as well as block counts for the neighbor sets of each node, which will be updated sequentially during the Gibbs sampling."""
        self.block_counts = np.array([pair[0] for pair in empirical_block_probs(self.blocks_by_node, np.ones(self.n, dtype = bool), self.K, dtype = Pair)])  
        #self.neighbor_counts = np.array([pair[1] for pair in self.graph.degree_iter()])
        self.neighbor_block_counts_by_node = np.zeros((self.n, self.K), dtype = int)
        for i in range(self.n):
            block = self.blocks_by_node[i]
            for nbr in self.graph.neighbors_iter(i):
                self.neighbor_block_counts_by_node[nbr][block] += 1
    def gibbs_step(self, update_params = True):
        """Using the current SBM parameters, goes through a full round of Gibbs sampling of the unknown nodes, using the full conditional distribution w.r.t. the preceding sample. If update_params = True, computes the posterior distributions for the SBM parameters and re-samples the parameters from these distributions."""
        # print(self.sbm.log_block_probs)
        # print(self.sbm.log_edge_probs)
        # print(self.sbm.log_one_minus_edge_probs)
        # print()
        #total = self.neighbor_block_counts_by_node.sum()
        for i in self.unobserved_nodes:
            # assert (self.block_counts.sum() == self.graph.n)
            # assert (self.neighbor_block_counts_by_node.sum() == total)
            old_block = self.blocks_by_node[i]
            neighbor_block_counts = self.neighbor_block_counts_by_node[i]
            nonneighbor_block_counts = self.block_counts - neighbor_block_counts
            nonneighbor_block_counts[old_block] -= 1  # don't count the node itself among its non-neighbors
            logps = self.sbm.log_block_probs + np.dot(self.sbm.log_edge_probs, neighbor_block_counts) + np.dot(self.sbm.log_one_minus_edge_probs, nonneighbor_block_counts)
            # test
            # logps2 = np.zeros(self.K, dtype = float)
            # for k in range(self.K):
            #     self.blocks_by_node[i] = k
            #     logps2[k] = self.logp()
            # assert np.abs((logps[0]-logps2[0])-(logps[1]-logps2[1])) < 1e-12
            # self.blocks_by_node[i] = old_block
            probs = shifted_logprobs_to_normalized_probs(logps)
            cdf = probs.cumsum()
            new_block = cdf.searchsorted(np.random.rand())
            # print("i = %d" % i)
            # print(self.block_counts)
            # print(neighbor_block_counts)
            # print(nonneighbor_block_counts)
            # print(logps)
            # print(probs)
            # print("new block: %d" % new_block)
            self.blocks_by_node[i] = new_block
            if (new_block != old_block):
                #print('changed')
                self.block_counts[old_block] -= 1
                self.block_counts[new_block] += 1
                nbrs = self.graph.neighbors(i)
                self.neighbor_block_counts_by_node[nbrs, old_block] -= 1
                self.neighbor_block_counts_by_node[nbrs, new_block] += 1
        if update_params:
            self.init_sbm(style = 'posterior', observed = False)  # sample parameters from full posterior distribution
        #return (self.sbm, self.blocks_by_node)
    def gibbs_sample(self, iters = 1000, burn = 0, thin = 1, update_params = True, verbose = False):
        """Performs iters iterations of Gibbs sampling on the posterior distribution of the unknown block labels (if update_params = True, also samples the SBM parameters from the posterior distribution in each step). burn is the number of initial samples to discard; thin is the interval at which to keep samples in order to reduce autocorrelation."""
        if update_params:  
            # initialize with MAP parameters, but could also sample from (limited) posterior?
            self.init_sbm(style = 'MAP', observed = True)
            self.traces['block_probs'] = np.zeros((self.K, iters), dtype = float)
            self.traces['block_probs'][:, 0] = self.sbm.block_probs
            self.traces['edge_probs'] = np.zeros((self.K, self.K, iters), dtype = float)
            self.traces['edge_probs'][:, :, 0] = self.sbm.edge_probs
        self.init_unobserved_nodes()
        self.init_block_counts()
        self.traces['unobserved_blocks'] = np.zeros((len(self.unobserved_nodes), iters), dtype = int)
        self.traces['unobserved_blocks'][:, 0] = self.blocks_by_node[self.unobserved_nodes]
        for t in range(1, iters):
            if verbose:
                print("\t\tGibbs iteration #%d" % t)
            self.gibbs_step(update_params = update_params)
            if update_params:
                self.traces['block_probs'][:, t] = self.sbm.block_probs
                self.traces['edge_probs'][:, :, t] = self.sbm.edge_probs
            self.traces['unobserved_blocks'][:, t] = self.blocks_by_node[self.unobserved_nodes]
        self.traces['unobserved_blocks'] = self.traces['unobserved_blocks'][:, burn:iters:thin]
        if update_params:
            self.traces['block_probs'] = self.traces['block_probs'][:, burn:iters:thin]
            self.traces['edge_probs'] = self.traces['edge_probs'][:, :, burn:iters:thin]


class SparseApproxSBM_MRF(PairwiseMRF):
    """Sparse approximation to an SBM in which the MRF dependency structure is the same as that of the graph (separated nodes are conditionally independent)."""
    def bp_init(self):
        """Set all initial log-messages between nodes to 0, and set all beliefs for unobserved nodes to the uniform distribution."""
        for (i, j) in self.unobserved_subgraph.edges_iter():
            self.unobserved_subgraph.edge[i][j]['log_msg'] = np.zeros(self.K, dtype = float)
        self.bp_logp_marginals = -np.log(self.K) * np.ones((self.n, self.K), dtype = float)
        for i in self.observed_nodes:
            for j in range(self.K):
                if (self.blocks_by_node[i] == j):
                    self.bp_logp_marginals[i, j] = 0.0
                else:
                    self.bp_logp_marginals[i, j] = -np.inf
    def bp_message_pass(self):
        """Send messages from each node to all of its neighbors."""
        for i in self.bp_ordering:
            nbrs = list(self.unobserved_subgraph[i].keys())
            # print("i = %d" % i)
            # print(nbrs)
            num_nbrs = len(nbrs)
            if (num_nbrs > 0):
                log_msgs = np.ma.array(np.vstack([self.unobserved_subgraph.edge[nbr][i]['log_msg'] for nbr in nbrs]), mask = np.zeros((num_nbrs, self.K), dtype = bool))
                log_mat = self.sbm.log_edge_probs + self.unobserved_subgraph.node[i]['log_factor']
                # print("log_msgs")
                # print(log_msgs)
                # print("log_mat")
                # print(log_mat)
                for (index, nbr) in enumerate(nbrs):
                    log_msgs.mask[index] = True
                    log_prod_incoming_msgs = log_msgs.sum(axis = 0).data
                    log_msgs.mask[index] = False
                    log_msg_mat = log_mat + log_prod_incoming_msgs
                    log_outgoing_msg = np.apply_along_axis(logp_sum_logp, 1, log_msg_mat)
                    log_outgoing_msg -= log_outgoing_msg.max()  # keep the messages on a reasonable scale
                    self.unobserved_subgraph.edge[i][nbr]['log_msg'] = log_outgoing_msg
                    # print((index, nbr))
                    # print("incoming")
                    # print(log_prod_incoming_msgs)
                    # print("log_msg_mat")
                    # print(log_msg_mat)
                    # print("%d -> %d message" % (i, nbr))
                    # print(self.unobserved_subgraph.edge[i][nbr]['log_msg'])
        #return max_diff
    def bp_belief_compute(self):
        """Compute beliefs for each node."""
        for i in self.bp_ordering:
            nbrs = self.unobserved_subgraph[i].keys()
            logprobs = np.array(self.unobserved_subgraph.node[i]['log_factor'])
            if (len(nbrs) > 0):
                logprobs += np.vstack([self.unobserved_subgraph.edge[nbr][i]['log_msg'] for nbr in self.unobserved_subgraph[i]]).sum(axis = 0)
            self.unobserved_subgraph.node[i]['beliefs'] = shifted_logprobs_to_normalized_probs(logprobs) 
    def belief_propagate(self, tol = 1e-8, max_iters = 100, verbose = False):
        """Performs belief propagation on the unobserved subgraph to acquire the posterior marginals (beliefs) of each node. Stops when either the max change in any belief is less than tol, or when max_iters is reached."""
        self.bp_init()
        max_diff = np.inf
        iteration = 0
        while ((max_diff > tol) and (iteration < max_iters)):
            if verbose:
                print("\t\tBP iteration #%d" % iteration)
            self.bp_message_pass()
            self.bp_belief_compute()
            max_diff = 0.0
            for i in self.unobserved_nodes:
                max_diff = max(max_diff, np.abs(np.exp(self.bp_logp_marginals[i]) - self.unobserved_subgraph.node[i]['beliefs']).max())
                self.bp_logp_marginals[i] = np.log(self.unobserved_subgraph.node[i]['beliefs'])
            if verbose:
                print("\t\t\tmax change in belief component = %g" % max_diff)
            iteration += 1
    @classmethod
    def from_mrf(cls, sbm_mrf):
        """Initializes a SparseApproxSBM_MRF from an SBM_MRF, in which the factors corresponding to non-edges have been removed. This brings about the local dependency structure."""
        mrf = cls()
        mrf.graph = sbm_mrf.graph
        mrf.sbm = sbm_mrf.sbm
        mrf.K, mrf.n, mrf.blocks_by_node, mrf.observed_flags, mrf.observed_nodes, mrf.unobserved_nodes, mrf.unobserved_edges, mrf.observed_edge_probs = mrf.graph.K, mrf.graph.n, mrf.graph.blocks_by_node, mrf.graph.observed_flags, sbm_mrf.observed_nodes, sbm_mrf.unobserved_nodes, sbm_mrf.unobserved_edges, sbm_mrf.observed_edge_probs
        usg = nx.DiGraph()  # set up observed subgraph with potentials, to be used in BP
        for i in range(mrf.n):
            if (not mrf.observed_flags[i]):
                usg.add_node(i)
                usg.node[i]['log_factor'] = np.array(mrf.sbm.log_block_probs)
        for (i, j) in mrf.graph.edges_iter():
            if mrf.observed_flags[i]:
                if (not mrf.observed_flags[j]):
                    usg.node[j]['log_factor'] += mrf.sbm.log_edge_probs[mrf.blocks_by_node[i]]
            else:
                if mrf.observed_flags[j]:
                    usg.node[i]['log_factor'] += mrf.sbm.log_edge_probs[mrf.blocks_by_node[j]]
                else:  # edges should go in both ways for sending messages
                    usg.add_edge(i, j)
                    usg.add_edge(j, i)
                    # usg[i][j]['log_factor'] = mrf.sbm.log_edge_probs  # always the same edge potential, so don't need it explicitly
        mrf.unobserved_subgraph = usg
        mrf.bp_ordering = []
        nodeset = set(usg.nodes())
        g = deepcopy(usg)
        while (len(nodeset) > 0):  # rank nodes by increasing degree (relative to subgraph of already-visited nodes)
            node_degree_pairs = sorted(g.degree_iter(), key = lambda pair : pair[1])
            min_degree = node_degree_pairs[0][1]
            for (node, deg) in node_degree_pairs:
                if (deg > min_degree):
                    break
                else:
                    mrf.bp_ordering.append(node)
                    nodeset.remove(node)
            g = g.subgraph(nodeset)
        return mrf



s1 = TwoBlockSBM(0.1, 0.05, 0.2, p1 = 0.25)
g1 = s1.sample(100)
mrf1 = SBM_MRF(g1)
s2 = TwoBlockSBM(0.1, 0.05, 0.1, p1 = 0.25)
g2 = s2.sample(100)
mrf2 = SBM_MRF(g2)
s3 = TwoBlockSBM(0.1, 0.05, 0.2, p1 = 0.5)
g3 = s3.sample(100)
mrf3 = SBM_MRF(g3)
s4 = TwoBlockSBM(0.1, 0.05, 0.1, p1 = 0.5)  # non-identifiable
g4 = s4.sample(100)
mrf4 = SBM_MRF(g4)

