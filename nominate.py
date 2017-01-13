"""After obtaining all the one-time work, performs vertex nomination on nodes whose nomination_attr_type value is unknown.

    EMBEDDING:
    Stacks all the embedding matrices and applies supervised binary classification. Optionally uses leave-one-out cross-validation to nominate the known nodes as well.

    RANDOMWALK:
    Performs a random walk on each stochasticized weighted adjacency matrix using both positive and negative seeds, combining either the scores or the random walks themselves.

    DIFFUSION:
    Applies heat equation repeatedly until convergence, with seeds as boundary conditions (+ seeds hot, - seeds cold)

    Usage: python3 nominate.py [path]

    The directory [path] must include a file params.py containing all necessary parameters."""

import sys
import onetime
import imp
import itertools
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from balloon import BalloonNominate
from kde import TwoClassKDE
from attr_vn import *

def main():

    path = sys.argv[1].strip('/')
    pm = imp.load_source('params', path + '/params.py')
    attr_filename = path + '/' + pm.attr_filename

    if (pm.rng_seed is not None):
        np.random.seed(pm.rng_seed)

    # partition attribute types into text/discrete (str dtype) or numeric
    text_attr_types, num_attr_types = [], []
    for (attr_type, dtype) in pm.predictor_attr_types.items():
        if (attr_type != pm.nomination_attr_type):
            if (dtype is str):
                text_attr_types.append(attr_type)
            else:
                num_attr_types.append(attr_type)
    attr_types = text_attr_types + num_attr_types  # all predictor attribute types

    # get data frame of numeric features
    if pm.verbose:
        print("Gathering numeric features...")
    start_time = time.time()
    num_df = pd.read_csv(attr_filename, sep = ';')
    num_df = num_df[np.vectorize(lambda t : t in set(num_attr_types))(num_df['attributeType'])]
    num_df = num_df.pivot(index = 'node', columns = 'attributeType', values = 'attributeVal')
    num_df = num_df.convert_objects(convert_numeric = True)
    if pm.verbose:
        print(time_format(time.time() - start_time))

    # load the one-time work
    onetime_work = onetime.main()
    n = onetime_work[0].shape[0]  # the number of nodes

    # identify seeds
    if pm.verbose:
        print("\nCreating AttributeAnalyzer...")
    a = timeit(AttributeAnalyzer, pm.verbose)(attr_filename, n, text_attr_types + [pm.nomination_attr_type])
    ind = a.get_attribute_indicator(pm.nomination_attr_val, pm.nomination_attr_type)
    true_seeds, false_seeds = ind[ind == 1].index, ind[ind == 0].index
    num_true_seeds, num_false_seeds = len(true_seeds), len(false_seeds)
    training = list(ind[ind >= 0].index)
    test = list(ind[~(ind >= 0)].index)  # complement of seed set
    assert ((num_true_seeds > 1) and (num_false_seeds > 1))  # can't handle this otherwise, yet
    if pm.verbose:
        print("\n%d total seeds (%d positive, %d negative)" % (num_true_seeds + num_false_seeds, num_true_seeds, num_false_seeds))
    train_out = ind[training]
    print(pm.vn_method)
    if (pm.vn_method == 'embedding'):

        # stack feature vectors, projecting to sphere if desired
        if pm.verbose:
            print("\nStacking feature vectors...")
        start_time = time.time()
        mats = []
        # get embedding features
        (context_features, text_attr_features_by_type) = onetime_work
        embedding_mats = []
        if pm.use_context:
            if pm.sphere_context:
                normalize_mat_rows(context_features)
            embedding_mats.append(context_features)
        for attr_type in text_attr_types:
            if pm.sphere_content:
                normalize_mat_rows(text_attr_features_by_type[attr_type])
            embedding_mats.append(text_attr_features_by_type[attr_type])
        if (len(text_attr_types) > 0):
            mats += embedding_mats
        if (len(num_attr_types) > 0):
            # impute missing numeric data (using naive mean or median of the known values)
            imputer = Imputer(strategy = pm.imputation)
            mats.append(imputer.fit_transform(num_df))
        mat = np.hstack(mats)
        if pm.verbose:
            print(time_format(time.time() - start_time))

        # standard-scale the columns
        mat = StandardScaler().fit_transform(mat)

        # perform PCA on features, if desired
        if pm.use_pca:
            ncomps = mat.shape[1] if (pm.max_eig_pca is None) else min(pm.max_eig_pca, mat.shape[1])
            pca = PCA(n_components = ncomps, whiten = pm.whiten)
            if pm.verbose:
                print("\nPerforming PCA on feature matrix...")
            mat = timeit(pca.fit_transform)(mat)
            sq_sing_vals = pca.explained_variance_
            if (pm.which_elbow > 0):
                elbows = get_elbows(sq_sing_vals, n = pm.which_elbow, thresh = 0.0)
                k = elbows[min(len(elbows), pm.which_elbow) - 1]
            else:
                k = len(sq_sing_vals)
            mat = mat[:, :k]

        # construct classifier
        if (pm.classifier == 'logreg'):
            clf = LogisticRegression()
        elif (pm.classifier == 'naive_bayes'):
            clf = GaussianNB()
        elif (pm.classifier == 'randfor'):
            clf = RandomForestClassifier(n_estimators = pm.num_trees, n_jobs = pm.n_jobs)
        elif (pm.classifier == 'boost'):
            clf = AdaBoostClassifier(n_estimators = pm.num_trees)
        elif (pm.classifier == 'kde'):
            clf = TwoClassKDE()
            train_in = mat[training]
            if pm.verbose:
                print("\nCross-validating to optimize KDE bandwidth...")
            timeit(clf.fit_with_optimal_bandwidth)(train_in, train_out, gridsize = pm.kde_cv_gridsize, dynamic_range = pm.kde_cv_dynamic_range, cv = pm.kde_cv_folds, verbose = int(pm.verbose), n_jobs = pm.n_jobs)
        elif (pm.classifier == 'inflate'):
            clf = BalloonNominate(pm.lamb, deflate = False)
        elif (pm.classifier == 'deflate'):
            clf = BalloonNominate(pm.lamb, deflate = True)
        else:
            raise ValueError("Invalid classifier '%s'." % pm.classifier)

        # cross-validate
        if (pm.cv_max > 0):
            true_seeds_for_cv = list(true_seeds[np.random.permutation(range(num_true_seeds))])
            false_seeds_for_cv = list(false_seeds[np.random.permutation(range(num_false_seeds))])
            # include equal proportion of positive & negative examples in cross-validation, if possible
            cv_seeds = list(itertools.islice(filter(None, sum(itertools.zip_longest(true_seeds_for_cv, false_seeds_for_cv), ())), pm.cv_max))
            num_cv_seeds = len(cv_seeds)
            start_time = time.time()
            cv_probs = np.zeros(num_cv_seeds, dtype = float)
            num_true = ind[cv_seeds].sum()
            guess_rate = num_true / num_cv_seeds
            training_set = set(training)
            if pm.verbose:
                print("\nCross-validating %d seeds (%d positive, %d negative) with %s = %s..." % (num_cv_seeds, num_true, num_cv_seeds - num_true, pm.nomination_attr_type, pm.nomination_attr_val))
            for (i, seed) in enumerate(cv_seeds):
                training_set.remove(seed)  # remove sample
                cv_train = list(training_set)
                cv_train_in = mat[cv_train]
                cv_train_out = ind[cv_train]
                clf.fit(cv_train_in, cv_train_out)
                cv_in = mat[[seed]]
                cv_probs[i] = clf.predict_proba(cv_in)[0, 1]
                training_set.add(seed)     # add back sample
            cv_df = pd.DataFrame(columns = ['node', 'prob'] + [pm.nomination_attr_type] + attr_types)
            cv_df['node'] = cv_seeds
            cv_df['prob'] = cv_probs
            for attr_type in [pm.nomination_attr_type] + text_attr_types:
                attrs_by_node = a.attrs_by_node_by_type[attr_type]
                cv_df[attr_type] = [str(attrs_by_node[node]) if (len(attrs_by_node[node]) > 0) else '{}' for node in cv_seeds]
            for attr_type in num_attr_types:
                vals = num_df[attr_type]
                cv_df[attr_type] = ['' if np.isnan(vals[node]) else str(vals[node]) for node in cv_seeds]
            cv_df = cv_df.sort_values(by = 'prob', ascending = False)
            cumulative_prec = np.cumsum(np.asarray(ind[cv_df['node']])) / np.arange(1.0, num_cv_seeds + 1.0)
            AP = np.mean(cumulative_prec)  # average precision
            if pm.verbose:
                print(time_format(time.time() - start_time))
                print("\nguess rate = %5f" % guess_rate)
                print("average precision = %5f" % AP)
                print("cumulative precisions:")
                print(cumulative_prec)
            if pm.save_info:
                cv_df.to_csv(path + '/%s_%s_cv_nomination.txt' % (pm.nomination_attr_type, pm.nomination_attr_val), index = False, sep = '\t')
                plt.figure()
                plt.plot(cumulative_prec, color = 'blue', linewidth = 2)
                plt.axhline(y = guess_rate, color = 'black', linewidth = 2, linestyle = 'dashed')
                plt.axvline(x = num_true, color = 'black', linewidth = 2, linestyle = 'dashed')
                plt.xlabel('rank')
                plt.ylabel('prec')
                plt.ylim((0, min(1.0, 1.1 * cumulative_prec.max())))
                plt.title('Cumulative precision of cross-validated seeds\nAP = %5f' % AP, fontweight = 'bold')
                plt.savefig(path + '/%s_%s_cv_prec.png' % (pm.nomination_attr_type, pm.nomination_attr_val))

        # nominate the unknown nodes
        start_time = time.time()
        if pm.verbose:
            print("\nNominating unknown nodes...")
        train_in = mat[training]
        clf.fit(train_in, train_out)
        test_in = mat[test]
        test_scores = clf.predict_proba(test_in)[:, 1]
        if pm.verbose:
            print(time_format(time.time() - start_time)) 
    
    else:
        if (len(num_attr_types) > 0):
            print("\nWARNING: VN method '%s' not compatible with numeric attributes -- ignoring these...")
        # get all sparse matrices
        if pm.verbose:
            print("\nObtaining sparse matrices...")
        (A, text_attr_pfas_by_type) = onetime_work
        text_attr_operators_by_type = {attr_type : a.make_uncollapsed_operator(pfa, attr_type, sim = pm.sim, delta = pm.delta, verbose = pm.verbose) for (attr_type, pfa) in text_attr_pfas_by_type.items()}
        sparse_ops = ([A] if pm.use_context else []) + list(text_attr_operators_by_type.values())
        if (pm.vn_method == 'randomwalk'):
            sparse_ops = [op.to_column_stochastic() for op in sparse_ops]
        else:  # diffusion
            sparse_ops = [SparseLaplacian(op) for op in sparse_ops]
        if (pm.combination_style == 'mean'):  # average all the matrices
            sparse_ops = [(1.0 / len(sparse_ops)) * reduce(lambda x, y : x + y, sparse_ops)]
        scores = np.zeros((n, len(sparse_ops)))
        start_time = time.time()
        if (pm.vn_method == 'randomwalk'):
            if pm.verbose:
                print("\nStepping random walk(s)...")
            for (j, rw) in enumerate(sparse_ops):
                x_plus, x_minus = np.zeros(n), np.zeros(n)
                p = (1.0 / num_true_seeds) if (num_true_seeds > 0) else (1.0 / n)
                seeds_to_set = true_seeds if (num_true_seeds > 0) else range(n)
                for i in seeds_to_set:
                    x_plus[i] = p 
                p = (1.0 / num_false_seeds) if (num_false_seeds > 0) else (1.0 / n)
                seeds_to_set = false_seeds if (num_false_seeds > 0) else range(n)
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
            if pm.verbose:
                print("\nPerforming diffusion...")
            if pm.diffusion_bias:
                seed_temps = np.array([-1.0 / num_false_seeds, 1.0 / num_true_seeds])
                seed_temps /= np.abs(seed_temps).max()
            else:
                seed_temps = np.array([-1.0, 1.0])
            mean_temp = (num_false_seeds * seed_temps[0] + num_true_seeds * seed_temps[1]) / n
            unobserved_flags = ~(ind >= 0)
            for (j, L) in enumerate(sparse_ops):
                temps = mean_temp * np.ones(n)
                for i in false_seeds:
                    temps[i] = seed_temps[0]
                for i in true_seeds:
                    temps[i] = seed_temps[1]
                counter = itertools.count() if (pm.diffusion_max_iters is None) else range(pm.diffusion_max_iters)
                for t in counter:
                    dt_temp = -L * temps * unobserved_flags
                    temps += pm.diffusion_rate * dt_temp
                    error = pm.diffusion_rate * np.linalg.norm(dt_temp)
                    if (error < pm.diffusion_tol):
                        break
                scores[:, j] = temps
        if (pm.score_fusion_style == 'mean'):
            test_scores = scores.mean(axis = 1)[test]
        else:  # 'max'
            test_scores = scores.max(axis = 1)[test]
        if pm.verbose:
            print(time_format(time.time() - start_time)) 


    nom_df = pd.DataFrame(columns = ['node', 'score'] + attr_types)
    nom_df['node'] = test
    nom_df['score'] = test_scores
    for attr_type in text_attr_types:
        attrs_by_node = a.attrs_by_node_by_type[attr_type]
        nom_df[attr_type] = [str(attrs_by_node[node]) if (len(attrs_by_node[node]) > 0) else '{}' for node in test]
    for attr_type in num_attr_types:
        vals = num_df[attr_type]
        nom_df[attr_type] = ['' if np.isnan(vals[node]) else str(vals[node]) for node in test]
    nom_df = nom_df.sort_values(by = 'score', ascending = False)
    nom_df[:pm.nominate_max].to_csv(path + '/%s_%s_nomination.out' % (pm.nomination_attr_type, pm.nomination_attr_val), index = False, sep = '\t')
    if pm.verbose:
        print("\nSaved results to %s/%s_%s_nomination.out" % (path, pm.nomination_attr_type, pm.nomination_attr_val))


if __name__ == "__main__":
    main()
