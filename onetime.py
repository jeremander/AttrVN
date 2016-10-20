"""This script reads in graph and attribute data, constructs similarity matrices for each text attribute of interest, then (if embedding is to be done) embeds each of these into Euclidean space. 

    Usage: python3 onetime.py [path]

    The directory [path] must include a file params.py containing all necessary parameters."""

import os
import sys
import imp
from attr_vn import *

def get_onetime_work(path, pm):
    """Given work path and an object bundling the VN parameters (usually extracted from a params.py file), return the onetime work as a tuple (context_work, content_work)."""
    edges_filename = path + '/' + pm.edges_filename
    attr_filename = path + '/' + pm.attr_filename
    filenames = os.listdir(path)

    # load/perform context embedding
    if (pm.use_context and (pm.vn_method == 'embedding')):
        embedding_filename_prefix = '*context*_embedding'
        valid_filenames = [filename for filename in filenames if filename.startswith(embedding_filename_prefix) and filename.endswith('.pickle')]
        if (pm.load_embeddings and (len(valid_filenames) > 0)):  # just use the first valid filename
            context_features = timeit(load_object, pm.verbose)(path, valid_filenames[0][:-7], 'pickle', verbose = pm.verbose)
        else:
            A = edgelist_to_sparse_adjacency_operator(edges_filename, verbose = pm.verbose)
            (eigvals, context_features) = timeit(embed_symmetric_operator, pm.verbose)(A, embedding = pm.embedding, k = pm.max_eig, tol = None, verbose = pm.verbose)
            pairs = sorted(enumerate(np.abs(eigvals)), key = lambda pair : pair[1], reverse = True)
            indices, abs_eigvals = map(np.array, zip(*pairs))
            if (pm.which_elbow > 0):
                elbows = get_elbows(abs_eigvals, n = pm.which_elbow, thresh = 0.0)
                k = elbows[min(len(elbows), pm.which_elbow) - 1]
            else:
                k = len(eigvals)
            if pm.verbose:
                print("\nKeeping first k = %d eigenvectors..." % k)
            context_features = context_features[:, indices[:k]] 
            obj_name = '*context*_embedding_%s_k=%d' % (pm.embedding, k)
            timeit(save_object, pm.verbose)(context_features, path, obj_name, 'pickle', verbose = pm.verbose)
            if pm.save_info:
                np.savetxt(path + '/*context*_eigvals.csv', eigvals, fmt = '%f')
                scree_plot(eigvals, k, show = False, filename = path + '/' + '*context*_scree.png')
        n = context_features.shape[0]
    else:
        context_features = None
        A = edgelist_to_sparse_adjacency_operator(edges_filename, verbose = pm.verbose)
        n = A.shape[0]

    # load/perform attribute similarity computation & embedding
    text_attr_types = [attr_type for (attr_type, dtype) in pm.predictor_attr_types.items() if dtype is str]
    pfa_filename_prefixes = ['%s_pfa' % attr_type for attr_type in text_attr_types]
    embedding_filename_prefixes = ['%s_embedding_sim=%s_delta=%s_%s' % (attr_type, pm.sim, str(pm.delta), pm.embedding) for attr_type in text_attr_types]
    text_attr_pfas_by_type = dict()
    text_attr_features_by_type = dict()
    if ((pm.vn_method == 'embedding') and pm.load_embeddings):  # see if embeddings can be loaded from files
        for (attr_type, embedding_filename_prefix) in zip(text_attr_types, embedding_filename_prefixes):  
            valid_filenames = [filename for filename in filenames if filename.startswith(embedding_filename_prefix) and filename.endswith('.pickle')]
            if (len(valid_filenames) > 0):  # just use the first valid filename
                text_attr_features_by_type[attr_type] = timeit(load_object, pm.verbose)(path, valid_filenames[0][:-7], 'pickle', verbose = pm.verbose)
    attr_types_to_process = [attr_type for attr_type in text_attr_types if (attr_type not in text_attr_features_by_type)]  # only process types for which we haven't loaded embeddings
    if ((pm.vn_method != 'embedding') or (len(attr_types_to_process) > 0)):  # need to get PairwiseFreqAnalyzers
        if pm.load_pfa:
            for (attr_type, pfa_filename_prefix) in zip(attr_types_to_process, pfa_filename_prefixes):
                valid_filenames = [filename for filename in filenames if filename.startswith(pfa_filename_prefix) and filename.endswith('.pickle')]   
                if (len(valid_filenames) > 0):  # just use the first valid filename
                    text_attr_pfas_by_type[attr_type] = timeit(load_object, pm.verbose)(path, valid_filenames[0][:-7], 'pickle', verbose = pm.verbose)
    attr_types_to_make_pfas = [attr_type for attr_type in attr_types_to_process if (attr_type not in text_attr_pfas_by_type)]
    if (len(attr_types_to_process) > 0):  # need to create AttributeAnalyzer
        if pm.verbose:
            print("\nCreating AttributeAnalyzer...")
        a = timeit(AttributeAnalyzer, pm.verbose)(attr_filename, n, attr_types_to_process) 
        if pm.save_info:
            a.rank_plot(rank_thresh = pm.rank_thresh, show = False, filename = path + '/' + 'attr_rank_plot.png')
        with open(path + '/attr_report.txt', 'w') as f:
            f.write(a.attr_report(rank_thresh = pm.rank_thresh))   
    for attr_type in attr_types_to_make_pfas:  # make PFA
        pfa = timeit(a.make_pairwise_freq_analyzer, pm.verbose)(attr_type, edges_filename, verbose = pm.verbose)
        text_attr_pfas_by_type[attr_type] = pfa
        if (pm.vn_method != 'embedding'):  # save PFA
            obj_name = '%s_pfa' % attr_type
            timeit(save_object, pm.verbose)(pfa, path, obj_name, 'pickle', verbose = pm.verbose)
    if (pm.vn_method == 'embedding'):
        for attr_type in attr_types_to_process:  # make attribute embedding for each text attribute type
            pfa = text_attr_pfas_by_type[attr_type]
            sim_op = timeit(a.make_uncollapsed_operator, pm.verbose)(pfa, attr_type, sim = pm.sim, delta = pm.delta, verbose = pm.verbose)
            (eigvals, attr_features) = timeit(embed_symmetric_operator, pm.verbose)(sim_op, embedding = pm.embedding, k = pm.max_eig, tol = None, verbose = pm.verbose)
            pairs = sorted(enumerate(np.abs(eigvals)), key = lambda pair : pair[1], reverse = True)
            indices, abs_eigvals = map(np.array, zip(*pairs))
            if (pm.which_elbow > 0):
                elbows = get_elbows(abs_eigvals, n = pm.which_elbow, thresh = 0.0)
                k = elbows[min(len(elbows), pm.which_elbow) - 1]
            else:
                k = len(eigvals)
            if pm.verbose:
                print("\nKeeping first k = %d eigenvectors..." % k)
            attr_features = attr_features[:, indices[:k]] 
            text_attr_features_by_type[attr_type] = attr_features
            obj_name = '%s_embedding_sim=%s_delta=%s_%s_k=%d' % (attr_type, pm.sim, str(pm.delta), pm.embedding, k)
            timeit(save_object, pm.verbose)(attr_features, path, obj_name, 'pickle', verbose = pm.verbose)
            if pm.save_info:
                np.savetxt(path + '/%s_eigvals.csv' % attr_type, eigvals, fmt = '%f')
                scree_plot(eigvals, k, show = False, filename = path + '/%s_scree.png' % attr_type)

    if (pm.vn_method == 'embedding'):
        return (context_features, text_attr_features_by_type)
    else:
        return (A, text_attr_pfas_by_type)


def main():
    path = sys.argv[1].strip('/')
    pm = imp.load_source('params', path + '/params.py')
    return get_onetime_work(path, pm)

if __name__ == "__main__":
    main()
