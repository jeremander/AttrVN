from sbm import *
from diffusion import *
from expsuite import PyExperimentSuite, ListWithNoSpaces
import numpy as np
import pandas as pd
import time

def time_format(seconds):
    """Formats a time into a convenient string."""
    s = ''
    if (seconds >= 3600):
        s += "%dh," % (seconds // 3600)
    if (seconds >= 60):
        s += "%dm," % ((seconds % 3600) // 60)
    s += "%.3fs" % (seconds % 60)
    return s

def print_params(params):
    longest_param_len = max([len(param) for param in params])
    for param in sorted(params):
        print("%s%s : %s" % (' ' * (longest_param_len - len(param)), str(param), str(params[param])))
    print()

class SBMExperimentSuite(PyExperimentSuite):
    def reset(self, params, rep):
        """Initializes SBM."""
        if (params['verbosity'] >= 1):
            print("\nStarting experiment with params...")
            print_params(params)
        if (params['K'] == 2):
            if ('lambda' in params):  # scale the minimum communication probability to lambda / n
                min_p = np.min([params['p00'], params['p01'], params['p11']])
                scale = (params['lambda'] / params['n']) / min_p
                for param in ['p00', 'p01', 'p11']:
                    params[param] *= scale
            self.sbm = TwoBlockSBM(params['p00'], params['p01'], params['p11'], params['p1'])
        else:
            raise ValueError("Only two-block SBM supported for now.")
        if ('num_observed' not in params):
            params['num_observed'] = int(params['frac_observed'] * params['n'])
        self.prec_df = pd.DataFrame(columns = range(params['iterations']))
        self.times = np.zeros(params['iterations'], dtype = float)
    def iterate(self, params, rep, n):
        """Draws a random graph from an SBM and occludes some number of vertices. Randomly occludes some number of nodes, then does vertex nomination on all the unobserved nodes. Reuses the same graph iters_per_graph times before resampling."""
        if (params['verbosity'] >= 2):
            print("\tIteration #%d" % n)
        np.random.seed(hash((rep, n)) % (1 << 32))  # use both the repetition and the iteration
        if (n % params['iters_per_graph'] == 0):  # sample a new graph
            self.graph = self.sbm.sample(params['n'])
        observed_graph = self.graph.mcar_occlude(params['n'] - params['num_observed'])
        true_memberships = observed_graph.blocks_by_node[~(observed_graph.observed_flags)]
        start_time = time.time()
        if (params['vn_method'] == 'randomwalk'):
            scores = observed_graph.vn_randomwalk(steps = params['steps'], stoch = params['stoch'])
        elif (params['vn_method'] == 'induced_subgraph'):
            scores = observed_graph.vn_induced_subgraph(sbm = None if params['estimate'] else self.sbm)
        elif (params['vn_method'] == 'mean_field_naive'):
            scores = observed_graph.vn_mean_field_naive(sbm = None if params['estimate'] else self.sbm)
        elif (params['vn_method'] == 'mean_field_opt'):
            scores = observed_graph.vn_mean_field_opt(sbm = None if params['estimate'] else self.sbm, tol = params['tol'], max_iters = params['mean_field_iters'], verbose = (params['verbosity'] >= 3))
        elif (params['vn_method'] == 'supervised'):
            try:
                scores = observed_graph.vn_supervised(k = params['k'], embedding = params['embedding'], classifier = params['classifier'], normalize = params['normalize'], num_trees = params['num_trees'], verbose = (params['verbosity'] >= 3))
            except IndexError:
                return dict()
        elif (params['vn_method'] == 'gibbs'):
            # if estimate = True, use MAP estimate to initialize SBM parameters, then include these parameters in the sampling chain
            scores = observed_graph.vn_gibbs(iters = params['mcmc_iters'], burn = params['burn'], thin = params['thin'], sbm = None if params['estimate'] else self.sbm, verbose = (params['verbosity'] >= 3))
        elif (params['vn_method'] == 'sparse_approx_bp'):
            scores = observed_graph.vn_sparse_approx_bp(sbm = None if params['estimate'] else self.sbm, tol = params['tol'], max_iters = params['bp_iters'], verbose = (params['verbosity'] >= 3))
        elif (params['vn_method'] == 'exact'):
            # if estimate = True, use MAP estimate instead of exact SBM parameters
            try:
                scores = observed_graph.vn_exact_marginals(sbm = None if params['estimate'] else self.sbm)
            except AssertionError:
                return dict()
        elif (params['vn_method'] == 'diffusion'):
            dg = DiffusionGraph.from_vngraph(observed_graph, proportional = params['proportional'])
            dg.diffusion(params['rate'], max_iters = params['diffusion_iters'], tol = params['tol'], verbosity = params['verbosity'] - 1)
            scores = dg.temps[~dg.observed_flags]
        else:
            raise ValueError("Invalid vertex nomination method: %s" % params['vn_method'])
        self.times[n] = time.time() - start_time
        if (params['verbosity'] >= 2):
            print("\t%s" % time_format(self.times[n]))
        self.prec_df[n] = get_cumulative_precisions(true_memberships, scores)
        ret = dict()
        if (n == params['iterations'] - 1):
            #ret['prec_df'] = self.prec_df  # uncomment if we want to save all the results
            ret['mean_prec'] = ListWithNoSpaces(self.prec_df.mean(axis = 1))
            ret['stderr_prec'] = ListWithNoSpaces(self.prec_df.std(axis = 1) / np.sqrt(params['iterations']))
            avg_prec = self.prec_df.mean(axis = 0)
            ret['mean_avg_prec'] = avg_prec.mean()
            ret['stderr_avg_prec'] = avg_prec.std() / np.sqrt(params['iterations'])
            ret['mean_time'] = self.times.mean()
            ret['stderr_time'] = self.times.std() / np.sqrt(params['iterations'])
            if (params['verbosity'] >= 1):
                print("mean time per iteration : %s" % time_format(ret['mean_time']))
        return ret


if __name__ == "__main__":
    sbm_suite = SBMExperimentSuite()
    sbm_suite.start()
