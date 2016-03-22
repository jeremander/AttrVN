"""After obtaining all the desired embeddings, stacks them and applies supervised learning to nominate nodes whose nomination_attr_type value is unknown. Optionally uses leave-one-out cross-validation to nominate the known nodes as well.

    Usage: python3 nominate.py [path]

    The directory [path] must include a file params.py containing all necessary parameters."""

import os
import sys
import embed
import imp
import itertools
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from attr_vn import *

def main():

    path = sys.argv[1].strip('/')
    pm = imp.load_source('params', path + '/params.py')
    attr_filename = path + '/' + pm.attr_filename

    if (pm.rng_seed is not None):
        np.random.seed(pm.rng_seed)

    # stack feature vectors, projecting to sphere if desired
    start_time = time.time()
    (context_features, attr_features_by_type) = embed.main()
    attr_types = list(attr_features_by_type.keys())
    embedding_mats = []
    if pm.use_context:
        if pm.sphere:
            normalize_mat_rows(context_features)
        embedding_mats.append(context_features)
    for attr_type in attr_types:
        if pm.sphere:
            normalize_mat_rows(attr_features_by_type[attr_type])
        embedding_mats.append(attr_features_by_type[attr_type])
    mat = np.hstack(embedding_mats)
    if pm.verbose:
        print(time_format(time.time() - start_time))

    # identify seeds
    n = mat.shape[0]
    if pm.verbose:
        print("\nCreating AttributeAnalyzer...")
    a = timeit(AttributeAnalyzer, pm.verbose)(attr_filename, n, attr_types + [pm.nomination_attr_type])
    ind = a.get_attribute_indicator(pm.nomination_attr_val, pm.nomination_attr_type)
    true_seeds, false_seeds = ind[ind == 1].index, ind[ind == 0].index
    num_true_seeds, num_false_seeds = len(true_seeds), len(false_seeds)
    training = list(ind[ind >= 0].index)
    assert ((num_true_seeds > 1) and (num_false_seeds > 1))  # can't handle this otherwise, yet
    if pm.verbose:
        print("\n%d total seeds (%d positive, %d negative)" % (num_true_seeds + num_false_seeds, num_true_seeds, num_false_seeds))

    if (pm.classifier == 'logreg'):
        clf = LogisticRegression()
    elif (pm.classifier == 'naive_bayes'):
        clf = GaussianNB()
    elif (pm.classifier == 'randfor'):
        clf = RandomForestClassifier(n_estimators = pm.num_trees)
    elif (pm.classifier == 'boost'):
        clf = AdaBoostClassifier(n_estimators = pm.num_trees)
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
            train_in = mat[cv_train]
            train_out = ind[cv_train]
            clf.fit(train_in, train_out)
            cv_in = mat[[seed]]
            cv_probs[i] = clf.predict_proba(cv_in)[0, 1]
            training_set.add(seed)     # add back sample
        cv_df = pd.DataFrame(columns = ['node', 'prob'] + [pm.nomination_attr_type] + attr_types)
        cv_df['node'] = cv_seeds
        cv_df['prob'] = cv_probs
        for attr_type in [pm.nomination_attr_type] + attr_types:
            attrs_by_node = a.attrs_by_node_by_type[attr_type]
            cv_df[attr_type] = [str(attrs_by_node[node]) if (len(attrs_by_node[node]) > 0) else '{}' for node in cv_seeds]
        cv_df = cv_df.sort_values(by = 'prob', ascending = False)
        cumulative_prec = np.cumsum(np.asarray(ind[cv_df['node']])) / np.arange(1.0, num_cv_seeds + 1.0)
        MAP = np.mean(cumulative_prec)  # mean average precision
        if pm.verbose:
            print(time_format(time.time() - start_time))
            print("\nguess rate = %5f" % guess_rate)
            print("mean average precision = %5f" % MAP)
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
            plt.title('Cumulative precision of cross-validated seeds\nMAP = %5f' % MAP, fontweight = 'bold')
            plt.savefig(path + '/%s_%s_cv_prec.png' % (pm.nomination_attr_type, pm.nomination_attr_val))

    # nominate the unknown nodes
    start_time = time.time()
    if pm.verbose:
        print("\nNominating unknown nodes...")
    train_in = mat[training]
    train_out = ind[training]
    clf.fit(train_in, train_out)
    test = list(ind[~(ind >= 0)].index)  # complement of seed set
    test_in = mat[test]
    test_probs = clf.predict_proba(test_in)[:, 1]
    if pm.verbose:
        print(time_format(time.time() - start_time))    
    nom_df = pd.DataFrame(columns = ['node', 'prob'] + attr_types)
    nom_df['node'] = test
    nom_df['prob'] = test_probs
    for attr_type in attr_types:
        attrs_by_node = a.attrs_by_node_by_type[attr_type]
        nom_df[attr_type] = [str(attrs_by_node[node]) if (len(attrs_by_node[node]) > 0) else '{}' for node in test]
    nom_df = nom_df.sort_values(by = 'prob', ascending = False)
    nom_df[:pm.nominate_max].to_csv(path + '/%s_%s_nomination.out' % (pm.nomination_attr_type, pm.nomination_attr_val), index = False, sep = '\t')
    if pm.verbose:
        print("\nSaved results to %s/%s_%s_nomination.out" % (path, pm.nomination_attr_type, pm.nomination_attr_val))


if __name__ == "__main__":
    main()
