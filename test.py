import sys
import onetime
import imp
import itertools
import time
import os
from pyexperiment.expsuite import PyExperimentSuite, ListWithNoSpaces
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from kde import TwoClassKDE
from scipy.sparse.linalg import minres
from attr_vn import *
from copy import deepcopy

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

class TestParams():
    """Dummy class to bundle parameters."""
    pass

class AttrVNExperimentSuite(PyExperimentSuite):
    def __init__(self):
        super().__init__()
        param_module = imp.load_source('params', 'params.py')
        self.pm = TestParams()
        pm = self.pm
        for var in dir(param_module):  # builtin vars pollute the namespace; filter them out
            if (not var.startswith('__')):
                pm.__dict__[var] = param_module.__dict__[var]
        pm.verbosity = 1 if pm.verbose else 0

        # partition attribute types into text/discrete (str dtype) or numeric
        self.text_attr_types, self.num_attr_types = [], []
        for (attr_type, dtype) in pm.predictor_attr_types.items(): 
            if (dtype is str):
                self.text_attr_types.append(attr_type)
            else:
                self.num_attr_types.append(attr_type)
        self.attr_types = self.text_attr_types + self.num_attr_types  # all attribute types

        # get data frame of numeric features
        if (pm.verbosity >= 1):
            print("Gathering numeric features...")
        start_time = time.time()
        self.num_df = pd.read_csv(pm.attr_filename, sep = ';')
        self.num_df = self.num_df[np.vectorize(lambda t : t in set(self.num_attr_types))(self.num_df['attributeType'])]
        self.num_df = self.num_df.pivot(index = 'node', columns = 'attributeType', values = 'attributeVal')
        self.num_df = self.num_df.convert_objects(convert_numeric = True)
        if (pm.verbosity >= 1):
            print(time_format(time.time() - start_time))

        # load the one-time work (for both embedding & randomwalk)
        vn_method = pm.vn_method
        pm.vn_method = 'embedding'
        (self.context_features, self.text_attr_features_by_type) = onetime.get_onetime_work('.', pm)
        pm.vn_method = 'randomwalk'
        (self.A, self.text_attr_pfas_by_type) = onetime.get_onetime_work('.', pm)
        pm.vn_method = vn_method

        self.n = self.context_features.shape[0]  # the number of nodes

        # create AttributeAnalyzer
        if (pm.verbosity >= 1):
           print("\nCreating AttributeAnalyzer...")
        self.aa = timeit(AttributeAnalyzer, pm.verbosity >= 1)(pm.attr_filename, self.n, self.text_attr_types)

        # make the appropriate sparse linear operators
        if (pm.verbosity >= 1):
            print("\nMaking SparseLinearOperators...")
        text_attr_operators_by_type = {at : self.aa.make_uncollapsed_operator(pfa, at, sim = pm.sim, delta = pm.delta, verbose = (pm.verbosity >= 1)) for (at, pfa) in self.text_attr_pfas_by_type.items()}
        self.context_rw = self.A.to_column_stochastic()
        self.content_rws = [text_attr_operators_by_type[at].to_column_stochastic() for at in self.text_attr_types]
        self.context_laplacian = SparseLaplacian(self.A)
        self.content_laplacians = [SparseLaplacian(text_attr_operators_by_type[at]) for at in self.text_attr_types]

        # load the attribute file
        self.nom_attr_df = pd.read_csv(pm.nomination_path)

    def reset(self, params, rep):
        self.pm.__dict__.update(params)  # overwrite default params with test parameters
        pm = self.pm
        if (pm.verbosity >= 1):
            print("\n\n#################################################\nStarting experiment %d of %d with params...\n" % (params['***EXP_NUM***'], self.num_experiments_))
            print_params(params)

        # read nomination attributes & their types from a file
        nomination_attr_type = self.nom_attr_df['attributeType'][pm.nom_ind]
        nomination_attr_val = self.nom_attr_df['attribute'][pm.nom_ind]

        # distinguish text/numeric predictor types from nomination type
        self.predictor_attr_types = [at for at in self.attr_types if (at != nomination_attr_type)] # all predictor attribute types
        self.text_predictor_attr_types = [at for at in self.predictor_attr_types if (at in self.text_attr_types)]

        # identify seeds
        self.invalid_option = False  # flag on whether to skip experiment
        self.ind = self.aa.get_attribute_indicator(nomination_attr_val, nomination_attr_type)
        self.true_seeds, self.false_seeds = self.ind[self.ind == 1].index, self.ind[self.ind == 0].index
        self.all_seeds = set(self.true_seeds).union(set(self.false_seeds))
        self.num_true_seeds, self.num_false_seeds = len(self.true_seeds), len(self.false_seeds)
        self.num_seeds = self.num_true_seeds + self.num_false_seeds
        if (pm.verbosity >= 1):
            print("\n%d known instances of %s (%d +%s, %d -%s)" % (self.num_seeds, nomination_attr_type, self.num_true_seeds, nomination_attr_val, self.num_false_seeds, nomination_attr_val))
        if ((pm.num_true_samps > self.num_true_seeds) or (pm.num_false_samps > self.num_false_seeds)):
            print("\nWarning: not enough seed instances. Skipping experiment.")
            self.invalid_option = True
        if (pm.verbosity >= 1):
            print("Sampling %d positive seeds, %d negative seeds" % (pm.num_true_samps, pm.num_false_samps))
        self.num_pos_in_test = self.num_true_seeds - pm.num_true_samps
        self.num_train = pm.num_true_samps + pm.num_false_samps
        self.num_test = self.num_true_seeds + self.num_false_seeds - pm.num_true_samps - pm.num_false_samps
        self.guess_rate = self.num_pos_in_test / self.num_test

        # prepare the algorithm
        if (pm.vn_method == 'embedding'):

            # construct classifier
            if (pm.classifier == 'logreg'):
                self.clf = LogisticRegression()
            elif (pm.classifier == 'naive_bayes'):
                self.clf = GaussianNB()
            elif (pm.classifier == 'randfor'):
                self.clf = RandomForestClassifier(n_estimators = pm.num_trees, n_jobs = 1)
            elif (pm.classifier == 'boost'):
                self.clf = AdaBoostClassifier(n_estimators = pm.num_trees)
            elif (pm.classifier == 'kde'):
                self.clf = TwoClassKDE()
                #train_in = mat[training]
                #if pm.verbose:
                #    print("\nCross-validating to optimize KDE bandwidth...")
                #timeit(clf.fit_with_optimal_bandwidth)(train_in, train_out, gridsize = pm.kde_cv_gridsize, dynamic_range = pm.kde_cv_dynamic_range, cv = pm.kde_cv_folds, verbose = int(pm.verbose), n_jobs = pm.n_jobs)
            else:
                raise ValueError("Invalid classifier '%s'." % pm.classifier)

            # get embedding features
            if (pm.verbosity >= 1):
                print("\nStacking feature vectors...")
            mats = []
            if (pm.info != 'content'):
                if pm.sphere_context:
                    normalize_mat_rows(self.context_features)
                mats.append(self.context_features)
            if (pm.info != 'context'):
                for at in self.text_predictor_attr_types:
                    if pm.sphere_content:
                        normalize_mat_rows(self.text_attr_features_by_type[at])
                    mats.append(self.text_attr_features_by_type[at])
            if (len(self.num_attr_types)):
                # impute missing numeric data (using naive mean or median of known values)
                imputer = Imputer(strategy = pm.imputation)
                mats.append(imputer.fit_transform(self.num_df))
            self.mat = StandardScaler().fit_transform(np.hstack(mats))
            del(mats)

            # perform PCA on features, if desired (don't do this with multithreading!)
            if pm.use_pca:
                ncomps = self.mat.shape[1] if (pm.max_eig_pca is None) else min(pm.max_eig_pca, self.mat.shape[1])
                pca = PCA(n_components = ncomps, whiten = pm.whiten)
                if (pm.verbosity >= 1):
                    print("\nPerforming PCA on feature matrix...")
                self.mat = pca.fit_transform(self.mat)
                sq_sing_vals = pca.explained_variance_
                if (pm.which_elbow > 0):
                    try:
                        elbows = get_elbows(sq_sing_vals, n = pm.which_elbow, thresh = 0.0)
                        k = elbows[min(len(elbows), pm.which_elbow) - 1]
                    except AssertionError:  # can't compute the elbow
                        k = len(sq_sing_vals)
                else:
                    k = len(sq_sing_vals)
                self.mat = self.mat[:, :k]  # truncate the PCA'ed feature matrix at k columns

        else:
            if (len(self.num_attr_types) > 0): 
                print("Warning: Graph-only method not compatible with numeric features. Skipping experiment.")
                self.invalid_option = True

            self.sparse_ops = []
            if (pm.info != 'content'):
                context_op = self.context_rw if (pm.vn_method == 'randomwalk') else self.context_laplacian
                self.sparse_ops.append(context_op)
            if (pm.info != 'context'):
                content_ops = self.content_rws if (pm.vn_method == 'randomwalk') else self.content_laplacians
                self.sparse_ops += [op for (i, op) in enumerate(content_ops) if (self.text_attr_types[i] != nomination_attr_type)]
            if (pm.combination_style == 'mean'):  # average all the matrices
                self.sparse_ops = [(1.0 / len(self.sparse_ops)) * reduce(lambda x, y : x + y, self.sparse_ops)]

        # set up data arrays to save
        self.prec_df = pd.DataFrame(columns = range(params['iterations']))
        self.times = np.zeros(params['iterations'], dtype = float)
    def iterate(self, params, rep, trial):
        pm = self.pm
        if (pm.verbosity >= 2):
            print("\tTrial #%d" % trial)
        if self.invalid_option:  # skip iteration
            return dict()
        np.random.seed(hash((rep, trial)) % (1 << 32))  # use both the repetition and the iteration

        # sample the seeds
        ts = self.true_seeds[np.random.choice(range(self.num_true_seeds), pm.num_true_samps, replace = False)]
        fs = self.false_seeds[np.random.choice(range(self.num_false_seeds), pm.num_false_samps, replace = False)]
        training = list(ts) + list(fs)
        test = list(self.all_seeds.difference(set(training)))
        train_out, test_out = self.ind[training], self.ind[test]
        df = pd.DataFrame(index = test)
        df['node'] = test_out

        # perform the vertex nomination
        start_time = time.time()
        if (pm.vn_method == 'embedding'):
            train_in = self.mat[training]
            test_in = self.mat[test]
            if (self.clf == 'kde'):  # do cross-validation for bandwidth fitting
                self.clf.fit_with_optimal_bandwidth(train_in, train_out, gridsize = pm.kde_cv_gridsize, dynamic_range = pm.kde_cv_dynamic_range, cv = pm.kde_cv_folds)
            else:
                self.clf.fit(train_in, train_out)
            df['score'] = self.clf.predict_proba(test_in)[:, 1]  # output probs of the classifier

        else:
            scores = np.zeros((self.n, len(self.sparse_ops)))
            if (pm.vn_method == 'randomwalk'):
                for (j, rw) in enumerate(self.sparse_ops):
                    x_plus, x_minus = np.zeros(self.n), np.zeros(self.n)
                    p = (1.0 / self.num_true_seeds) if (self.num_true_seeds > 0) else (1.0 / self.n)
                    seeds_to_set = ts if (self.num_true_seeds > 0) else range(self.n)
                    for i in seeds_to_set:
                        x_plus[i] = p 
                    p = (1.0 / self.num_false_seeds) if (self.num_false_seeds > 0) else (1.0 / self.n)
                    seeds_to_set = fs if (self.num_false_seeds > 0) else range(self.n)
                    for i in seeds_to_set:
                        x_minus[i] = p
                    for t in range(pm.randomwalk_steps):
                        x_plus = rw * x_plus
                        x_minus = rw * x_minus
                    if (pm.randomwalk_score_style == 'arith'):
                        scores[:, j] = x_plus - x_minus
                    else:  # 'geom'
                        scores[:, j] = np.log(x_plus) - np.log(x_minus)
            else:  # diffusion
                if pm.diffusion_bias:
                    seed_temps = np.array([-1.0 / self.num_false_seeds, 1.0 / self.num_true_seeds])
                    seed_temps /= np.abs(seed_temps).max()
                else:
                    seed_temps = np.array([-1.0, 1.0])
                unobserved_flags = np.ones(self.n, dtype = bool)
                unobserved_flags[training] = False
                unobserved_nodes = unobserved_flags.nonzero()[0]
                T0 = np.zeros(self.n, dtype = float)
                for i in fs:
                    T0[i] = seed_temps[0]
                for i in ts:
                    T0[i] = seed_temps[1]
                P = PermutationOperator(np.concatenate([training, unobserved_nodes]))
                T0 = (P * T0)[:self.num_train]
                F_seed = RowSelectorOperator(self.num_train, self.n, range(self.num_train))
                F_nonseed = RowSelectorOperator(self.n - self.num_train, self.n, range(self.num_train, self.n))
                for (j, L) in enumerate(self.sparse_ops):
                    Lp = P * L * P.inv()
                    b = -(F_nonseed * (Lp * (F_seed.transpose() * T0)))
                    A = F_nonseed * (Lp * F_nonseed.transpose())
                    Tsol = minres(A, b, tol = pm.diffusion_tol)[0]
                    scores[:, j] = P.inv() * np.concatenate([T0, Tsol])
            if (pm.score_fusion_style == 'mean'):
                df['score'] = scores.mean(axis = 1)[test]
            else:  # 'max'
                df['score'] = scores.max(axis = 1)[test]

        self.times[trial] = time.time() - start_time
        df = df.sort_values(by = 'score', ascending = False)
        prec = np.cumsum(np.asarray(df['node'])[:pm.topN_save]) / np.arange(1.0, pm.topN_save + 1.0)
        self.prec_df[trial] = prec

        # get the data to save off
        ret = dict()
        if (trial == params['iterations'] - 1):  
            # ret['prec_df'] = self.prec_df  # uncomment if we want to save all the results
            ret['mean_prec'] = ListWithNoSpaces(self.prec_df.mean(axis = 1))
            ret['stderr_prec'] = ListWithNoSpaces(self.prec_df.std(axis = 1) / np.sqrt(pm.iterations))
            avg_prec = self.prec_df.mean(axis = 0)
            ret['mean_avg_prec'] = avg_prec.mean()
            ret['stderr_avg_prec'] = avg_prec.std() / np.sqrt(pm.iterations)
            ret['mean_time'] = self.times.mean()
            ret['stderr_time'] = self.times.std() / np.sqrt(pm.iterations)
            ret['num_pos_in_test'] = self.num_pos_in_test
            ret['num_test'] = self.num_test
            ret['guess_rate'] = self.guess_rate

            if (pm.verbosity >= 1):
                print('mean time per trial : %s' % time_format(ret['mean_time']))

        return ret


if __name__ == "__main__":
    print('cd %s\n' % sys.argv[1])
    this_path = os.path.abspath(__file__)
    os.chdir(sys.argv[1])
    test_suite = AttrVNExperimentSuite()
    test_suite.module_path_ = this_path
    test_suite.start()