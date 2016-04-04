"""This script reads in graph and attribute data, constructs similarity matrices for each text attribute of interest, then embeds each of these into Euclidean space. 

    Usage: python3 embed.py [path]

    The directory [path] must include a file params.py containing all necessary parameters."""

import os
import sys
import imp
from attr_vn import *


def main():

    path = sys.argv[1].strip('/')
    pm = imp.load_source('params', path + '/params.py')
    edges_filename = path + '/' + pm.edges_filename
    attr_filename = path + '/' + pm.attr_filename
    filenames = os.listdir(path)

    # load/perform context embedding
    if pm.use_context:
        embedding_filename_prefix = '*context*_embedding'
        valid_filenames = [filename for filename in filenames if filename.startswith(embedding_filename_prefix) and filename.endswith('.pickle')]
        if (pm.load_embeddings and (len(valid_filenames) > 0)):  # just use the first valid filename
            context_features = timeit(load_object, pm.verbose)(path, valid_filenames[0][:-7], 'pickle', verbose = pm.verbose)
        else:
            A = edgelist_to_sparse_adjacency_operator(edges_filename, verbose = pm.verbose)
            (eigvals, context_features) = timeit(embed_symmetric_operator, pm.verbose)(A, embedding = pm.embedding, k = pm.max_eig, tol = None, verbose = pm.verbose)
            abs_eigvals = np.array(sorted(np.abs(eigvals), reverse = True))
            if (pm.which_elbow > 0):
                elbows = get_elbows(abs_eigvals, n = pm.which_elbow, thresh = 0.0)
                k = elbows[min(len(elbows), pm.which_elbow) - 1]
            else:
                k = len(eigvals)
            if pm.verbose:
                print("\nKeeping first k = %d eigenvectors..." % k)
            context_features = context_features[:, :k] 
            obj_name = '*context*_embedding_%s_k=%d' % (pm.embedding, k)
            timeit(save_object, pm.verbose)(context_features, path, obj_name, 'pickle', verbose = pm.verbose)
            if pm.save_info:
                np.savetxt(path + '/*context*_eigvals.csv', eigvals, fmt = '%f')
                scree_plot(eigvals, k, show = False, filename = path + '/' + '*context*_scree.png')
    else:
        context_features = None

    # load/perform attribute embeddings
    text_attr_types = [attr_type for (attr_type, dtype) in pm.predictor_attr_types.items() if dtype is str]
    embedding_filename_prefixes = ['%s_embedding_sim=%s_delta=%s_%s' % (attr_type, pm.sim, str(pm.delta), pm.embedding) for attr_type in text_attr_types]
    text_attr_features_by_type = dict()
    if pm.load_embeddings:  # first see what can be loaded from files
        for (attr_type, embedding_filename_prefix) in zip(text_attr_types, embedding_filename_prefixes):  
            valid_filenames = [filename for filename in filenames if filename.startswith(embedding_filename_prefix) and filename.endswith('.pickle')]
            if (len(valid_filenames) > 0):  # just use the first valid filename
                text_attr_features_by_type[attr_type] = timeit(load_object, pm.verbose)(path, valid_filenames[0][:-7], 'pickle', verbose = pm.verbose)
    if (len(text_attr_features_by_type) < len(text_attr_types)):  # need to construct AttributeAnalyzer to get remaining attribute embeddings
        if pm.verbose:
            print("\nCreating AttributeAnalyzer...")
        a = timeit(AttributeAnalyzer, pm.verbose)(attr_filename, context_features.shape[0], text_attr_types)
        if pm.save_info:
            a.rank_plot(rank_thresh = pm.rank_thresh, show = False, filename = path + '/' + 'attr_rank_plot.png')
        with open(path + '/attr_report.txt', 'w') as f:
            f.write(a.attr_report(rank_thresh = pm.rank_thresh))
    attr_types_to_embed = [attr_type for attr_type in text_attr_types if (attr_type not in text_attr_features_by_type)]
    for attr_type in attr_types_to_embed:  # make attribute embedding for each text attribute type
        pfa = timeit(a.make_pairwise_freq_analyzer, pm.verbose)(attr_type, edges_filename, verbose = pm.verbose)
        sim_op = timeit(a.make_uncollapsed_operator, pm.verbose)(pfa, attr_type, sim = pm.sim, delta = pm.delta, verbose = pm.verbose)
        (eigvals, attr_features) = timeit(embed_symmetric_operator, pm.verbose)(sim_op, embedding = pm.embedding, k = pm.max_eig, tol = None, verbose = pm.verbose)
        abs_eigvals = np.array(sorted(np.abs(eigvals), reverse = True))
        if (pm.which_elbow > 0):
            elbows = get_elbows(abs_eigvals, n = pm.which_elbow, thresh = 0.0)
            k = elbows[min(len(elbows), pm.which_elbow) - 1]
        else:
            k = len(eigvals)
        if pm.verbose:
            print("\nKeeping first k = %d eigenvectors..." % k)
        attr_features = attr_features[:, :k] 
        text_attr_features_by_type[attr_type] = attr_features
        obj_name = '%s_embedding_sim=%s_delta=%s_%s_k=%d' % (attr_type, pm.sim, str(pm.delta), pm.embedding, k)
        timeit(save_object, pm.verbose)(attr_features, path, obj_name, 'pickle', verbose = pm.verbose)
        if pm.save_info:
            np.savetxt(path + '/%s_eigvals.csv' % attr_type, eigvals, fmt = '%f')
            scree_plot(eigvals, k, show = False, filename = path + '/%s_scree.png' % attr_type)

    return (context_features, text_attr_features_by_type)
 

if __name__ == "__main__":
    main()
