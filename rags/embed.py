#!/usr/bin/env python3
"""This script reads in graph and attribute data, constructs similarity matrices for each text attribute of interest, then embeds each of these into Euclidean space.

    Usage: python3 embed.py [path]

    The directory [path] must include a file params.py containing all necessary parameters."""

import importlib.util
import numpy as np
import os
from pathlib import Path
import sys

from rags.attr_vn import AttributeAnalyzer, edgelist_to_sparse_adjacency_operator, embed_symmetric_operator, get_elbows, scree_plot
from rags.utils import load_object, save_object, timeit


def main(path):

    os.chdir(path)
    spec = importlib.util.spec_from_file_location('params', 'params.py')
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    filenames = os.listdir()

    # load/perform context embedding
    if pm.use_context:
        embedding_filename_prefix = '*context*_embedding'
        valid_filenames = [filename for filename in filenames if filename.startswith(embedding_filename_prefix) and filename.endswith('.pickle')]
        if (pm.load_embeddings and (len(valid_filenames) > 0)):  # just use the first valid filename
            context_features = timeit(load_object, pm.verbose)(valid_filenames[0], verbose = pm.verbose)
        else:
            A = edgelist_to_sparse_adjacency_operator(pm.edges_filename, verbose = pm.verbose)
            (eigvals, context_features) = timeit(embed_symmetric_operator, pm.verbose)(A, embedding = pm.embedding, k = pm.max_eig, tol = None, verbose = pm.verbose)
            pairs = sorted(enumerate(np.abs(eigvals)), key = lambda pair : pair[1], reverse = True)
            indices, abs_eigvals = map(np.array, zip(*pairs))
            if (pm.which_elbow > 0):
                elbows = get_elbows(abs_eigvals, n = pm.which_elbow, thresh = 0.0)
                k = elbows[min(len(elbows), pm.which_elbow) - 1]
            else:
                k = len(eigvals)
            if pm.verbose:
                print(f'Keeping first k = {k} eigenvectors...')
            context_features = context_features[:, indices[:k]]
            obj_name = '*context*_embedding_%s_k=%d' % (pm.embedding, k)
            timeit(save_object, pm.verbose)(context_features, f'{obj_name}.pickle', verbose = pm.verbose)
            if pm.save_info:
                np.savetxt(path + '/*context*_eigvals.csv', eigvals, fmt = '%f')
                scree_plot(eigvals, k, show = False, filename = path + '/' + '*context*_scree.png')
        n = context_features.shape[0]
    else:
        context_features = None
        A = edgelist_to_sparse_adjacency_operator(pm.edges_filename, verbose = pm.verbose)
        n = A.shape[0]

    # load/perform attribute embeddings
    text_attr_types = [attr_type for (attr_type, dtype) in pm.predictor_attr_types.items() if dtype is str]
    embedding_filename_prefixes = ['%s_embedding_sim=%s_delta=%s_%s' % (attr_type, pm.sim, str(pm.delta), pm.embedding) for attr_type in text_attr_types]
    text_attr_features_by_type = dict()
    if pm.load_embeddings:  # first see what can be loaded from files
        for (attr_type, embedding_filename_prefix) in zip(text_attr_types, embedding_filename_prefixes):
            valid_filenames = [filename for filename in filenames if filename.startswith(embedding_filename_prefix) and filename.endswith('.pickle')]
            if (len(valid_filenames) > 0):  # just use the first valid filename
                text_attr_features_by_type[attr_type] = timeit(load_object, pm.verbose)(valid_filenames[0], verbose = pm.verbose)
    if (len(text_attr_features_by_type) < len(text_attr_types)):  # need to construct AttributeAnalyzer to get remaining attribute embeddings
        if pm.verbose:
            print("\nCreating AttributeAnalyzer...")
        a = timeit(AttributeAnalyzer, pm.verbose)(pm.attr_filename, n, text_attr_types)
        if pm.save_info:
            a.rank_plot(rank_thresh = pm.rank_thresh, show = False, filename = 'attr_rank_plot.png')
        with open('attr_report.txt', 'w') as f:
            f.write(a.attr_report(rank_thresh = pm.rank_thresh))
    attr_types_to_embed = [attr_type for attr_type in text_attr_types if (attr_type not in text_attr_features_by_type)]
    for attr_type in attr_types_to_embed:  # make attribute embedding for each text attribute type
        pfa = timeit(a.make_pairwise_freq_analyzer, pm.verbose)(attr_type, pm.edges_filename, verbose = pm.verbose)
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
            print(f'\nKeeping first k = {k} eigenvectors...')
        attr_features = attr_features[:, indices[:k]]
        text_attr_features_by_type[attr_type] = attr_features
        obj_name = f'{attr_type}_embedding_sim={pm.sim}_delta={pm.delta}_{pm.embedding}_k={k}'
        timeit(save_object, pm.verbose)(attr_features, f'{obj_name}.pickle', verbose = pm.verbose)
        if pm.save_info:
            np.savetxt(f'{attr_type}_eigvals.csv', eigvals, fmt = '%f')
            scree_plot(eigvals, k, show = False, filename = f'{attr_type}_scree.png')

    return (context_features, text_attr_features_by_type)


if __name__ == "__main__":

    path = Path(sys.argv[1]).resolve()
    main(path)
