"""Computes eigenvalues and eigenvectors of the PMI similarity matrices for a given attribute type. Saves the results of this along with kMeans clustering of the attributes, and the assignment of graph nodes to clusters."""

import pickle
import time
import numpy as np
import pandas as pd
import optparse
from scipy.sparse import coo_matrix, diags
from gplus import *



# save off:
# matrix or LinearOperator for similarity matrix
# eigenvalues and scree plot
# embedded vectors corresponding to attributes
def main():
    p = optparse.OptionParser()
    p.add_option('--attr_type', '-a', type = str, help = 'attribute type')
    p.add_option('-p', type = str, help = 'PMI type (PMIs, NPMI1s, or NPMI2s)')
    p.add_option('-e', type = str, help = 'embedding (adj, normlap, regnormlap)')
    p.add_option('-s', action = 'store_true', default = False, help = 'normalize in sphere')
    p.add_option('-d', type = float, help = 'smoothing parameter')
    p.add_option('-k', type = int, help = 'number of eigenvalues')
    p.add_option('-t', type = float, default = None, help = 'tolerance for eigsh')
    p.add_option('-v', action = 'store_true', default = False, help = 'save scree plot')
    opts, args = p.parse_args()

    attr_type = opts.attr_type
    sim = opts.p
    embedding = opts.e
    assert (embedding in ['adj', 'normlap', 'regnormlap'])
    sphere = opts.s
    delta = opts.d
    k = opts.k
    nclusts = opts.c
    tol = opts.t
    save_plot = opts.v
    topN = 50  # for the report
    assert (((sim == 'PMIs') or (delta == 0)) and (sim in ['PMIs', 'NPMI1s', 'NPMI2s']))

    data_folder = 'gplus0_lcc/data/PMI/'
    report_folder = 'gplus0_lcc/reports/PMI/'
    plot_folder = 'gplus0_lcc/plots/PMI/'
    file_prefix1 = ('%s_%s_%s_delta' % (attr_type, sim, embedding)) + str(delta) + ('_k%d' % k)
    file_prefix2 = ('%s_%s_%s_delta' % (attr_type, sim, embedding)) + str(delta) + ('_k%d%s_c%d' % (k, '_normalized' if sphere else '', nclusts))

    print_flush("\nLoading AttributeAnalyzer...")
    a = AttributeAnalyzer()
    a.load_pairwise_freq_analyzer(attr_type)
    a.make_attrs_by_node_by_type()
    attrs_by_node = a.attrs_by_node_by_type[attr_type]
    pfa = a.pairwise_freq_analyzers[attr_type]
    n = pfa.num_vocab
    tol = (1.0 / n) if (tol is None) else tol  # use 1/n instead of machine precision as default tolerance
    attr_indices, attr_vocab = get_attr_indices(pfa, a.attributed_nodes)

    try:  
        print_flush("\nLoading labels from '%s%s_labels.csv'..." % (data_folder, file_prefix2))
        labels = np.loadtxt('%s%s_labels.csv' % (data_folder, file_prefix2), dtype = int)
        print_flush("\nLoading cluster centers from '%s%s_cluster_centers.csv'..." % (data_folder, file_prefix2))
        cluster_centers = np.loadtxt('%s%s_cluster_centers.csv' % (data_folder, file_prefix2), delimiter = ',')
        print_flush("\nLoading eigenvalues from '%s%s_eigvals.csv'..." % (data_folder, file_prefix1))
        eigvals = np.loadtxt('%s%s_eigvals.csv' % (data_folder, file_prefix1), delimiter = ',')
        print_flush("\nLoading embedded features from '%s%s_features.pickle'..." % (data_folder, file_prefix1))
        features = pickle.load(open('%s%s_features.pickle' % (data_folder, file_prefix1), 'rb'))
        if sphere:
            for i in range(len(attr_indices)):  
                features[i] = normalize(features[i])
    except FileNotFoundError:
        print_flush("Failed to load.")
        try:
            print_flush("\nLoading eigenvalues from '%s%s_eigvals.csv'..." % (data_folder, file_prefix1))
            eigvals = np.loadtxt('%s%s_eigvals.csv' % (data_folder, file_prefix1), delimiter = ',')
            print_flush("\nLoading embedded features from '%s%s_features.pickle'..." % (data_folder, file_prefix1))
            features = pickle.load(open('%s%s_features.pickle' % (data_folder, file_prefix1), 'rb'))
        except FileNotFoundError:
            print_flush("Failed to load.")
            print_flush("\nComputing similarity matrix (%s)..." % sim)
            sim_op = pfa.to_sparse_PMI_operator(sim, delta)
            matrix_type = 'adjacency' if (embedding == 'adj') else ('normalized Laplacian' if (embedding == 'normlap') else 'regularized normalized Laplacian')
            print_flush("\nComputing eigenvectors of %s matrix (k = %d)..." % (matrix_type, k))
            if (embedding == 'adj'):
                (eigvals, features) = timeit(eigsh)(sim_op, k = k, tol = tol)
                features = np.sqrt(np.abs(eigvals)) * features  # scale the feature columns by the sqrt of the eigenvalues
            elif (embedding == 'normlap'):
                normlap = SparseNormalizedLaplacian(sim_op)
                (eigvals, features) = timeit(eigsh)(normlap, k = k, tol = tol)
            elif (embedding == 'regnormlap'):
                regnormlap = SparseRegularizedNormalizedLaplacian(sim_op)
                (eigvals, features) = timeit(eigsh)(regnormlap, k = k, tol = tol)
            features = features[attr_indices, :]  # free up memory by deleting embeddings of nodes with no attributes
            np.savetxt('%s%s_eigvals.csv' % (data_folder, file_prefix1), eigvals, delimiter = ',')
            pickle.dump(features, open('%s%s_features.pickle' % (data_folder, file_prefix1), 'wb'))
        if sphere:  # normalize the features to have unit norm (better for kMeans)
            for i in range(len(attr_indices)):  
                features[i] = normalize(features[i])

        print_flush("\nClustering attribute feature vectors into %d clusters using kMeans..." % nclusts)
        labels = timeit(km.fit_predict)(features)
        # save the cluster labels
        np.savetxt('%s%s_labels.csv' % (data_folder, file_prefix2), np.array(labels, dtype = int), delimiter = ',', fmt = '%d')
        # save the cluster centers
        cluster_centers = km.cluster_centers_
        np.savetxt('%s%s_cluster_centers.csv' % (data_folder, file_prefix2), cluster_centers, delimiter = ',')
        # save the attribute cluster report
        with open('%s%s_cluster_report.txt' % (report_folder, file_prefix2), 'w') as f:
            f.write(generate_cluster_report(a, attr_type, labels, topN))

    if save_plot:
        print_flush("\nSaving scree plot to '%s%s_screeplot.png'..." % (plot_folder, file_prefix1))
        scree_plot(eigvals, show = False, filename = '%s%s_screeplot.png' % (plot_folder, file_prefix1))

    print_flush("\nAssigning cluster labels to each node...")
    indices_by_vocab = dict((v, i) for (i, v) in enumerate(attr_vocab))
    centers = [normalize(center) for center in cluster_centers] if sphere else cluster_centers
    def assign_cluster(node):
        """Assigns -1 to a node with no attribute present. Otherwise, takes the cluster whose center is closest to the mean of the attribute vectors. Uses cosine distance if sphere = True, otherwise Euclidean distance."""
        if (node not in attrs_by_node):
            return -1
        else:
            attrs = list(attrs_by_node[node])
            if (len(attrs) == 1):
                return labels[indices_by_vocab[attrs[0]]]
            else:
                vec = np.zeros(k, dtype = float)
                for attr in attrs:
                    vec += features[indices_by_vocab[attr]]
                vec /= len(attrs)
                if sphere:
                    vec = normalize(vec)
                    sims = [np.dot(vec, center) for center in centers]
                else:
                    sims = [-np.linalg.norm(vec - center) for center in centers]
                max_index, max_sim = -1, -float('inf')
                for (i, sim) in enumerate(sims):
                    if (sim > max_sim):
                        max_index = i
                        max_sim = sim
                return max_index

    # save file with the list of cluster labels for each node
    clusters_by_node = [assign_cluster(i) for i in range(a.num_vertices)]
    np.savetxt('%s%s_node_labels.csv' % (data_folder, file_prefix2), np.array(clusters_by_node, dtype = int), delimiter = ',', fmt = '%d')
    print_flush("\nDone!")


if __name__ == "__main__":
    main()