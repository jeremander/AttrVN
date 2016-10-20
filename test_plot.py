import optparse
import itertools
import matplotlib.pyplot as plt
from test import *
from rstyle import *
from expsuite import convert_param_to_dirname
from collections import defaultdict


distinguishers = ['xfacet', 'yfacet', 'linestyle', 'color']
max_numvals_by_distinguisher = {'xfacet' : 5, 'yfacet' : 5, 'linestyle' : 4, 'color' : 16}
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
cmap = plt.cm.gist_ncar

def dict_union(*dicts):
    return dict(itertools.chain(*map(lambda dct: list(dct.items()), dicts)))

def legend_str(var, param, suppress_var):
    return str(param) if suppress_var else ('%s=%s' % (var, param))

def main():
    p = optparse.OptionParser()
    p.add_option('--experiment', '-e', type = str, help = 'experiment name')
    opts, args = p.parse_args()

    experiment = opts.experiment

    # Get & organize variables for experiment
    suite = PyExperimentSuite()
    exp_paths = suite.get_exp(experiment)
    if (len(exp_paths) == 0):
        raise RuntimeError("No results from experiment %s." % experiment)
    exp_path = exp_paths[0]

    exps = suite.get_exps()
    params = suite.get_params(exp_path)  # get param dictionary for experiment
    print(params)
    vars_by_type = defaultdict(set)
    for var in params:
        if isinstance(params[var], list):
            vars_by_type['iter'].add(var)
        else:
            vars_by_type['constant'].add(var)

    if (experiment == 'embedding'):
        vars_by_distinguisher = {'yfacet' : 'info', 'color' : 'classifier'}
        vars_for_title = ['info', 'classifier', 'use_pca', 'sphere_context', 'sphere_content']
    elif (experiment == 'randomwalk'):
        vars_by_distinguisher = {'xfacet' : 'info', 'yfacet' : 'randomwalk_score_style', 'color' : 'randomwalk_steps', 'linestyle' : 'score_fusion_style'}
        vars_for_title = ['info', 'combination_style', 'randomwalk_score_style', 'score_fusion_style', 'randomwalk_steps']
    elif (experiment == 'diffusion'):
        vars_by_distinguisher = {'xfacet' : 'info', 'yfacet' : 'diffusion_bias', 'linestyle' : 'score_fusion_style'}
        vars_for_title = ['info', 'combination_style', 'diffusion_bias', 'score_fusion_style']
    elif (experiment == 'compare'):
        vars_by_distinguisher = {'yfacet' : 'info', 'xfacet' : 'vn_method', 'color' : 'randomwalk_steps'}
        vars_for_title = ['classifier', 'score_fusion_style']

    vars_for_title = [var for var in vars_for_title if (var in params)]
    vars_to_suppress_in_legend = ['info', 'classifier']  # show values but not variable names

    # Determine which variables are constant and which actually vary
    dists_to_delete = []
    for var in vars_by_type['iter']:
        if (len(params[var]) == 1):
            vars_by_type['constant'].add(var)
            params[var] = params[var][0]
            for (dist, var1) in vars_by_distinguisher.items():
                if (var1 == var):
                    dists_to_delete.append(dist)
        elif (var not in vars_by_distinguisher.values()):
            vars_by_type['outer'].add(var)
    for dist in dists_to_delete:  # delete any distinguishing vars that have no iteration
        del(vars_by_distinguisher[dist])

    vars_by_type['inner'] = set(vars_by_distinguisher.values())  # vars for plotting
    assert (vars_by_type['inner'].issubset(set(params.keys())))
    assert ('nom_ind' not in vars_by_type['inner'])

    constant_iter_dict = {var : params[var] for var in vars_by_type['iter'].intersection(vars_by_type['constant'])}
    vars_for_title = [var for var in vars_for_title if (var not in vars_by_type['inner'])]

    # read the attribute types & values
    nom_attr_df = pd.read_csv(sys.argv[1] + '/' + params['nomination_path'])
    nomination_attr_type = nom_attr_df['attributeType'][params['nom_ind']]
    nomination_attr_val = nom_attr_df['attribute'][params['nom_ind']]

    outer_dict_iter = ({var : val for (var, val) in zip(vars_by_type['outer'], param_tuple)} for param_tuple in itertools.product(*[params[var1] for var1 in vars_by_type['outer']]))
    for outer_dict in outer_dict_iter:
        print(outer_dict)
        numvals_by_distinguisher = {dist : len(params[var]) if isinstance(params[var], list) else 1 for (dist, var) in vars_by_distinguisher.items()}
        for dist in vars_by_distinguisher.keys():
            if (numvals_by_distinguisher[dist] > max_numvals_by_distinguisher[dist]):
                raise ValueError("Maximum number of values for %s is %d." % (dist, max_numvals_by_distinguisher[dist]))
        for dist in distinguishers:
            if (dist not in numvals_by_distinguisher):
                numvals_by_distinguisher[dist] = 1

        suite.mkdir(exp_path + '/plots')
        colors = {j : cmap(int((j + 1) * cmap.N / (numvals_by_distinguisher['color'] + 1.0))) for j in range(numvals_by_distinguisher['color'])} if ('color' in vars_by_distinguisher) else {0 : 'blue'}

        nom_ind = outer_dict['nom_ind'] if ('nom_ind' in outer_dict) else constant_iter_dict['nom_ind']
        nom_attr_type, nom_attr_val = nom_attr_df['attributeType'][nom_ind], nom_attr_df['attribute'][nom_ind]

        plot_title = 'VN Precision\n%s: %s' % (nom_attr_type, nom_attr_val)
        pair = (params['num_true_samps'] if ('num_true_samps' in vars_by_type['constant']) else None, params['num_false_samps'] if ('num_false_samps' in vars_by_type['constant']) else None)
        if any(pair):
            plot_title += ' (%s%s%s)' % (('+' + str(pair[0])) if pair[0] else '', ', ' if all(pair) else '', ('-' + str(pair[1])) if pair[1] else '')
        plot_title += '\n'

        plt.close('all')
        plot_fail = False
        fig, axis_grid = plt.subplots(numvals_by_distinguisher['yfacet'], numvals_by_distinguisher['xfacet'], sharex = 'col', sharey = 'row', figsize = (12, 8), facecolor = 'white')
        axis_grid = np.array(axis_grid).reshape((numvals_by_distinguisher['yfacet'], numvals_by_distinguisher['xfacet']))
        plots_for_legend = []
        keys_for_legend = []

        inner_dict = dict()
        for x in range(numvals_by_distinguisher['xfacet']):
            if ('xfacet' in vars_by_distinguisher):
                inner_dict[vars_by_distinguisher['xfacet']] = params[vars_by_distinguisher['xfacet']][x]
            for y in range(numvals_by_distinguisher['yfacet']):
                if ('yfacet' in vars_by_distinguisher):
                    inner_dict[vars_by_distinguisher['yfacet']] = params[vars_by_distinguisher['yfacet']][y]
                ax = axis_grid[y, x]
                max_num_test = 0
                for i in range(numvals_by_distinguisher['color']):
                    if ('color' in vars_by_distinguisher):
                        inner_dict[vars_by_distinguisher['color']] = params[vars_by_distinguisher['color']][i]
                    for j in range(numvals_by_distinguisher['linestyle']):
                        if ('linestyle' in vars_by_distinguisher):
                            inner_dict[vars_by_distinguisher['linestyle']] = params[vars_by_distinguisher['linestyle']][j]
                        loopvar_dict = dict_union(outer_dict, inner_dict, constant_iter_dict)
                        try:
                            mean_prec = np.array(suite.get_values_fix_params(exp_path, 0, 'mean_prec', 'last', **loopvar_dict)[0][0]['mean_prec'])
                            stderr_prec = np.array(suite.get_values_fix_params(exp_path, 0, 'stderr_prec', 'last', **loopvar_dict)[0][0]['stderr_prec'])
                            num_test = len(mean_prec)
                            max_num_test = max(max_num_test, num_test)
                        except:
                            print("Warning: file error for params...")
                            print(loopvar_dict)
                            print()
                            plot_fail = True
                            break
                            #continue
                        plot, = ax.plot(np.arange(num_test), mean_prec, color = colors[i], linestyle = linestyles[j], linewidth = 2)
                        plot.set_dash_capstyle('projecting')
                        if (('linestyle' in vars_by_distinguisher) or ('color' in vars_by_distinguisher)):
                            if (((x == 0) and (y == 0)) and ((i == 0) or (j == 0))):
                                plots_for_legend.append(plot)
                                key = ', '.join([legend_str(vars_by_distinguisher[dist], params[vars_by_distinguisher[dist]][k], vars_by_distinguisher[dist] in vars_to_suppress_in_legend) for (dist, k) in [('color', i), ('linestyle', j)] if (dist in vars_by_distinguisher)])
                                keys_for_legend.append(key)
                        ax.fill_between(np.arange(num_test), mean_prec - 2 * stderr_prec, mean_prec + 2 * stderr_prec, color = colors[i], alpha = 0.1)
                    if plot_fail:
                        break
                if plot_fail:
                    break

                # plot horizontal dashed line for guess precision
                try:
                    guess_rate = suite.get_values_fix_params(exp_path, 0, 'guess_rate', 'last', **loopvar_dict)[0][0]['guess_rate']
                except TypeError:
                    continue
                plot, = ax.plot(np.arange(max_num_test), guess_rate * np.ones(max_num_test, dtype = float), color = 'black', linestyle = 'dashed', linewidth = 2)
                plot.set_dash_capstyle('projecting')
                if ((x == 0) and (y == 0)):
                    plots_for_legend.append(plot)
                    keys_for_legend.append('guess')

                # plot vertical dashed line for pos/neg test threshold
                num_pos_in_test = suite.get_values_fix_params(exp_path, 0, 'num_pos_in_test', 'last', **loopvar_dict)[0][0]['num_pos_in_test']
                ax.axvline(x = num_pos_in_test, color = 'black', linestyle = 'dashed', linewidth = 2)

                if ((numvals_by_distinguisher['yfacet'] > 1) and (x == 0)):
                    ax.annotate(legend_str(vars_by_distinguisher['yfacet'], str(params[vars_by_distinguisher['yfacet']][y]), vars_by_distinguisher['yfacet'] in vars_to_suppress_in_legend), xy = (0, 0.5), xytext = (-ax.yaxis.labelpad, 0), xycoords = ax.yaxis.label, textcoords = 'offset points', ha = 'right', va = 'center')
                if ((numvals_by_distinguisher['xfacet'] > 1) and (y == 0)):
                    ax.annotate(legend_str(vars_by_distinguisher['xfacet'], str(params[vars_by_distinguisher['xfacet']][x]), vars_by_distinguisher['xfacet'] in vars_to_suppress_in_legend), xy = (0.5, 1.01), xytext = (0, 0), xycoords = 'axes fraction', textcoords = 'offset points', ha = 'center', va = 'baseline')
                ax.set_xlim((0, max_num_test - 1))
                ax.set_ylim((0.0, 1.0))
                rstyle(ax)
                ax.patch.set_facecolor('0.89')
            if plot_fail:
                break

        if plot_fail:
            continue

        plot_params = dict_union({var : params[var] for var in vars_for_title}, outer_dict)
        this_plot_title = plot_title + ', '.join(['%s=%s' % (var, str(plot_params[var])) for var in vars_for_title])
        print(plot_params)
        #print(this_plot_title)

        fig.text(0.5, 0.04, 'rank', ha = 'center', fontsize = 14)
        fig.text(0.02, 0.5, 'precision', va = 'center', rotation = 'vertical', fontsize = 14)
        plt.figlegend(plots_for_legend, keys_for_legend, 'right', fontsize = 10)
        plt.suptitle(this_plot_title, fontsize = 16 - len(vars_for_title) // 5, fontweight = 'bold')
        plt.subplots_adjust(left = 0.11, right = 0.85)

        plot_path = exp_path + '/plots/' + ('%s=%s' % (nom_attr_type, nom_attr_val)) + ',' + ','.join(['='.join(map(str, pair)) for pair in sorted(plot_params.items())]) + '.png'
        print(plot_path)
        plt.savefig(plot_path)
        #plt.show()


if __name__ == "__main__":
    main()