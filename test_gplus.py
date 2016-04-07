done_import = False
while (not done_import):
    try:
        import sys
        import embed
        import imp
        import itertools
        import optparse
        from copy import deepcopy
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.linear_model import LogisticRegression
        from kde import TwoClassKDE
        from attr_vn import *
        from rstyle import *
        #import matplotlib
        #matplotlib.use('Agg')
        done_import = True
    except:
        pass
        

def legend_str(var, param, suppress_var):
    return str(param) if suppress_var else ('%s=%s' % (var, param))

def main():
    #classifier_vals = ['logreg', 'randfor', 'boost', 'kde']
    classifier_vals = ['kde']
    embedding_info_vals = ['context', 'NPMIs', 'both']
    #sphere_content_vals = [True, False]
    sphere_content_vals = [True]
    params = {'classifier' : classifier_vals, 'embedding_info' : embedding_info_vals, 'sphere_content' : sphere_content_vals}

    # free to permute these (but not remove them)
    vars_by_distinguisher = {'color' : 'classifier', 'xfacet' : 'embedding_info', 'yfacet' : 'sphere_content'}
    #vars_by_distinguisher = {'color' : 'embedding_info', 'xfacet' : 'classifier', 'yfacet' : 'sphere_content'}

    numvals_by_distinguisher = {dist : len(params[var]) for (dist, var) in vars_by_distinguisher.items()} 
    cmap = plt.cm.gist_ncar
    colors = {j : cmap(int((j + 1) * cmap.N / (numvals_by_distinguisher['color'] + 1.0))) for j in range(numvals_by_distinguisher['color'])} if ('color' in vars_by_distinguisher) else {0 : 'blue'}
    vars_to_suppress_in_legend = ['embedding_info', 'classifier']  # show values but not variable names

    gplus_attr_types = ['employer', 'major', 'places_lived', 'school']

    pd.options.display.max_rows = None
    pd.options.display.width = 1000

    topN_save = 1000    # number of precisions to save
    topN_plot = 500     # number of precisions to plot

    p = optparse.OptionParser()
    p.add_option('--attr', '-a', type = str, help = 'attribute')
    p.add_option('--attr_type', '-t', type = str, help = 'attribute type')
    p.add_option('--pos_seeds', '-p', type = int, default = 50, help = 'number of positive seeds')
    p.add_option('--neg_seeds', '-n', type = int, default = 50, help = 'number of negative seeds')
    p.add_option('--num_samples', '-S', type = int, default = 50, help = 'number of Monte Carlo samples')
    p.add_option('--save_plot', '-v', action = 'store_true', default = False, help = 'save plot')
    opts, args = p.parse_args()

    attr, attr_type, pos_seeds, neg_seeds, num_samples, save_plot = opts.attr, opts.attr_type, opts.pos_seeds, opts.neg_seeds, opts.num_samples, opts.save_plot
    sqrt_samples = np.sqrt(num_samples)

    path = 'gplus0_sub'
    pm = imp.load_source('params', path + '/params.py')
    attr_filename = path + '/' + pm.attr_filename

    csv_path = 'test_gplus/%s_%s_+%d_-%d.csv' % (attr_type, attr, pos_seeds, neg_seeds)

    try:
        prec_df = pd.read_csv(csv_path, index = False)
    except:
        # load all feature matrices, AttributeAnalyzer, identify seeds
        sys.argv = ['embed', path]
        (context_features, attr_features_by_type) = embed.main()  # use sim, delta, embedding, etc. from params.py file
        assert ((context_features is not None) and (len(attr_features_by_type) == 4))
        other_attr_types = [at for at in gplus_attr_types if (at != attr_type)]
        n = context_features.shape[0]
        print("\nCreating AttributeAnalyzer...")
        a = timeit(AttributeAnalyzer, True)(attr_filename, n, gplus_attr_types)
        ind = a.get_attribute_indicator(attr, attr_type)
        true_seeds, false_seeds = ind[ind == 1].index, ind[ind == 0].index
        num_true_seeds, num_false_seeds = len(true_seeds), len(false_seeds)
        all_seeds = set(true_seeds).union(set(false_seeds))
        assert ((num_true_seeds > 1) and (num_false_seeds > 1))  # can't handle this otherwise, yet
        print("\n%d known instances of %s (%d positive, %d negative)" % (num_true_seeds + num_false_seeds, attr_type, num_true_seeds, num_false_seeds))
        if (pos_seeds >= num_true_seeds):
            print("\tWarning: changing pos_seeds from %d to %d." % (pos_seeds, num_true_seeds - 1))
            pos_seeds = num_true_seeds - 1
        if (neg_seeds >= num_false_seeds):
            print("\tWarning: changing neg_seeds from %d to %d." % (neg_seeds, num_false_seeds - 1))
            neg_seeds = num_false_seeds - 1
        print("Sampling %d positive seeds, %d negative seeds" % (pos_seeds, neg_seeds))
        num_pos_in_test = num_true_seeds - pos_seeds
        num_test = num_true_seeds + num_false_seeds - pos_seeds - neg_seeds
        guess_rate = num_pos_in_test / num_test
        topN_save = min(topN_save, num_test)
        topN_plot = min(topN_plot, topN_save)

        # construct classifiers
        clf_dict = {'logreg' : LogisticRegression(), 'naive_bayes' : GaussianNB(), 'randfor' : RandomForestClassifier(n_estimators = pm.num_trees), 'boost' : AdaBoostClassifier(n_estimators = pm.num_trees), 'kde' : TwoClassKDE()}
        prec_df = pd.DataFrame()  # for storing mean & stdev topN_save precisions for each parameter combo

        # run nomination
        for embedding_info in embedding_info_vals:
            for sphere_content in sphere_content_vals:
                print("\nembedding_info = %s, sphere_content = %s" % (embedding_info, str(sphere_content)))
                # stack all desired feature matrices, with or without projecting to sphere
                embedding_mats = []
                if (embedding_info != 'NPMIs'):
                    context_mat = deepcopy(context_features)
                    if pm.sphere_context:
                        normalize_mat_rows(context_mat)
                    embedding_mats.append(context_mat)
                if (embedding_info != 'context'):
                    for at in other_attr_types:
                        attr_mat = deepcopy(attr_features_by_type[at])
                        if sphere_content:
                            normalize_mat_rows(attr_mat)
                        embedding_mats.append(attr_mat)
                mat = np.hstack(embedding_mats)
                mat = StandardScaler().fit_transform(mat)
                if pm.use_pca:  # perform PCA on features, if desired
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
                precs_by_classifier = {classifier : np.zeros((num_samples, topN_save)) for classifier in classifier_vals}  # top N cumulative precisions
                for s in range(num_samples):
                    print("\nSEED = %d" % s)
                    np.random.seed(s)
                    ts = true_seeds[np.random.choice(range(num_true_seeds), pos_seeds, replace = False)]
                    fs = false_seeds[np.random.choice(range(num_false_seeds), neg_seeds, replace = False)]
                    training = list(ts) + list(fs)
                    test = list(all_seeds.difference(set(training)))
                    train_in, train_out = mat[training], ind[training]
                    test_in, test_out = mat[test], ind[test]
                    for classifier in classifier_vals:
                        print("classifier = %s" % classifier)
                        clf = clf_dict[classifier]
                        if (clf == 'kde'):
                            clf.fit_with_optimal_bandwidth(train_in, train_out, ridsize = pm.kde_cv_gridsize, dynamic_range = pm.kde_cv_dynamic_range, cv = pm.kde_cv_folds)
                        else:
                            clf.fit(train_in, train_out)
                        df = pd.DataFrame(index = test)
                        df['ind'] = test_out
                        df['prob'] = clf.predict_proba(test_in)[:, 1]
                        df = df.sort_values(by = 'prob', ascending = False)
                        prec = np.cumsum(np.asarray(df['ind'])[:topN_save]) / np.arange(1.0, topN_save + 1.0)
                        precs_by_classifier[classifier][s] = prec
                for classifier in classifier_vals:
                    prec_df[(embedding_info, sphere_content, classifier, 'mean_prec')] = precs_by_classifier[classifier].mean(axis = 0)
                    prec_df[(embedding_info, sphere_content, classifier, 'stderr_prec')] = precs_by_classifier[classifier].std(axis = 0) / sqrt_samples
        prec_df.to_csv(csv_path, index = False)


    # mean_cols = [col for col in prec_df.columns if (col[-1] == 'mean_prec')]
    # y_max = min(1.0, 1.1 * prec_df[mean_cols].max().max())

    # fig, axis_grid = plt.subplots(numvals_by_distinguisher['yfacet'], numvals_by_distinguisher['xfacet'], sharex = 'col', sharey = 'row', figsize = (12, 8), facecolor = 'white')
    # axis_grid = np.array(axis_grid).reshape((numvals_by_distinguisher['yfacet'], numvals_by_distinguisher['xfacet']))
    # plots_for_legend = []
    # keys_for_legend = []
    # param_dict = dict()
    # for x in range(numvals_by_distinguisher['xfacet']):
    #     param_dict[vars_by_distinguisher['xfacet']] = params[vars_by_distinguisher['xfacet']][x]
    #     for y in range(numvals_by_distinguisher['yfacet']):
    #         param_dict[vars_by_distinguisher['yfacet']] = params[vars_by_distinguisher['yfacet']][y]
    #         ax = axis_grid[y, x]
    #         for i in range(numvals_by_distinguisher['color']):
    #             param_dict[vars_by_distinguisher['color']] = params[vars_by_distinguisher['color']][i]
    #             mean_prec = prec_df[(param_dict['embedding_info'], param_dict['sphere_content'], param_dict['classifier'], 'mean_prec')][:topN_plot]
    #             stderr_prec = prec_df[(param_dict['embedding_info'], param_dict['sphere_content'], param_dict['classifier'], 'stderr_prec')][:topN_plot]
    #             plot, = ax.plot(np.arange(topN_plot), mean_prec, color = colors[i], linewidth = 2)
    #             if ('color' in vars_by_distinguisher):
    #                 if ((x == 0) and (y == 0)):
    #                     plots_for_legend.append(plot)
    #                     key = legend_str(vars_by_distinguisher['color'], params[vars_by_distinguisher['color']][i], vars_by_distinguisher['color'] in vars_to_suppress_in_legend)
    #                     keys_for_legend.append(key)
    #             ax.fill_between(np.arange(topN_plot), mean_prec - 2 * stderr_prec, mean_prec + 2 * stderr_prec, color = colors[i], alpha = 0.1)
    #         plot, = ax.plot(np.arange(topN_plot), guess_rate * np.ones(topN_plot, dtype = float), color = 'black', linestyle = 'dashed', linewidth = 2)
    #         plot.set_dash_capstyle('projecting')
    #         if ((x == 0) and (y == 0)):
    #             plots_for_legend.append(plot)
    #             keys_for_legend.append('guess')
    #         ax.axvline(x = num_pos_in_test, color = 'black', linestyle = 'dashed', linewidth = 2)
    #         if ((numvals_by_distinguisher['yfacet'] > 1) and (x == 0)):
    #             ax.annotate(legend_str(vars_by_distinguisher['yfacet'], str(params[vars_by_distinguisher['yfacet']][y]), vars_by_distinguisher['yfacet'] in vars_to_suppress_in_legend), xy = (0, 0.5), xytext = (-ax.yaxis.labelpad, 0), xycoords = ax.yaxis.label, textcoords = 'offset points', ha = 'right', va = 'center', rotation = 'vertical')
    #         if ((numvals_by_distinguisher['xfacet'] > 1) and (y == 0)):
    #             ax.annotate(legend_str(vars_by_distinguisher['xfacet'], str(params[vars_by_distinguisher['xfacet']][x]), vars_by_distinguisher['xfacet'] in vars_to_suppress_in_legend), xy = (0.5, 1.01), xytext = (0, 0), xycoords = 'axes fraction', textcoords = 'offset points', ha = 'center', va = 'baseline')
    #         ax.set_xlim((0, topN_plot - 1))
    #         ax.set_ylim((0.0, y_max))
    #         rstyle(ax)
    #         ax.patch.set_facecolor('0.89')

    # this_plot_title = 'Cumulative precision plots\n%s = %s, %d +seeds, %d -seeds' % (attr_type, attr, pos_seeds, neg_seeds)
    # fig.text(0.5, 0.04, 'rank', ha = 'center', fontsize = 14)
    # fig.text(0.02, 0.5, 'precision', va = 'center', rotation = 'vertical', fontsize = 14)
    # plt.figlegend(plots_for_legend, keys_for_legend, 'right', fontsize = 10)
    # plt.suptitle(this_plot_title, fontsize = 16, fontweight = 'bold')
    # plt.subplots_adjust(left = 0.11, right = 0.9)

    # if save_plot:
    #     plot_path = 'test_gplus/%s_%s_+%d_-%d.png' % (attr_type, attr, pos_seeds, neg_seeds)
    #     plt.savefig(plot_path)
    # plt.show()






if __name__ == "__main__":
    main()


