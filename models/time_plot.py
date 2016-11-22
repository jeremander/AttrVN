import optparse
import itertools
import matplotlib.pyplot as plt
from sbm_suite import *
from rstyle import *
from expsuite import convert_param_to_dirname
from collections import defaultdict


suites_by_experiment = {'twoblock_time' : SBMExperimentSuite}
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

    suite = suites_by_experiment[experiment]()
    exp_paths = suite.get_exp(experiment)
    if (len(exp_paths) == 0):
        raise RuntimeError("No results from experiment %s." % experiment)
    exp_path = exp_paths[0]
    exps = suite.get_exps()
    params = suite.get_params(exp_path)  # get param dictionary for experiment
    vars_by_type = defaultdict(set)
    for var in params:
        if (var != 'n'):
            if isinstance(params[var], list):
                vars_by_type['iter'].add(var)
            else:
                vars_by_type['constant'].add(var)

    if (experiment == 'twoblock_time'):
        vars_by_distinguisher = {'color' : 'vn_method', 'linestyle' : 'lambda', 'xfacet' : 'p00', 'yfacet' : 'p11'}
        guess_var = 'p1'
        plot_title = 'Two-Block SBM: Timing comparison\n'
        #vars_for_title = ['p1', 'p00', 'p01', 'p11', 'lambda', 'frac_observed', 'steps', 'embedding', 'classifier', 'normalize', 'estimate', 'mcmc_iters', 'mean_field_iters', 'bp_iters', 'iterations']
        vars_for_title = ['p1', 'p00', 'p01', 'p11', 'frac_observed', 'mcmc_iters', 'mean_field_iters']
    vars_for_title = [var for var in vars_for_title if (var in params)]
    vars_to_suppress_in_legend = ['vn_method', 'embedding']  # show values but not variable names

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

    constant_iter_dict = {var : params[var] for var in vars_by_type['iter'].intersection(vars_by_type['constant'])}
    vars_for_title = [var for var in vars_for_title if (var not in vars_by_type['inner'])]

    outer_dict_iter = ({var : val for (var, val) in zip(vars_by_type['outer'], param_tuple)} for param_tuple in itertools.product(*[params[var1] for var1 in vars_by_type['outer']]))
    for outer_dict in outer_dict_iter:
        numvals_by_distinguisher = {dist : len(params[var]) if isinstance(params[var], list) else 1 for (dist, var) in vars_by_distinguisher.items()}
        for dist in vars_by_distinguisher.keys():
            if (numvals_by_distinguisher[dist] > max_numvals_by_distinguisher[dist]):
                raise ValueError("Maximum number of values for %s is %d." % (dist, max_numvals_by_distinguisher[dist]))
        for dist in distinguishers:
            if (dist not in numvals_by_distinguisher):
                numvals_by_distinguisher[dist] = 1

        suite.mkdir(exp_path + '/plots')
        colors = {j : cmap(int((j + 1) * cmap.N / (numvals_by_distinguisher['color'] + 1.0))) for j in range(numvals_by_distinguisher['color'])} if ('color' in vars_by_distinguisher) else {0 : 'blue'}

        fig, axis_grid = plt.subplots(numvals_by_distinguisher['yfacet'], numvals_by_distinguisher['xfacet'], sharex = True, sharey = True, figsize = (12, 8), facecolor = 'white')
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
                for i in range(numvals_by_distinguisher['color']):
                    if ('color' in vars_by_distinguisher):
                        inner_dict[vars_by_distinguisher['color']] = params[vars_by_distinguisher['color']][i]
                    for j in range(numvals_by_distinguisher['linestyle']):
                        if ('linestyle' in vars_by_distinguisher):
                            inner_dict[vars_by_distinguisher['linestyle']] = params[vars_by_distinguisher['linestyle']][j]
                        loopvar_dict = dict_union(outer_dict, inner_dict, constant_iter_dict)
                        nvals, mean_times, stderr_times = [], [], []
                        for nval in params['n']:
                            try:
                                mean_time = suite.get_values_fix_params(exp_path, 0, 'mean_time', 'last', n = nval, **loopvar_dict)[0][0]['mean_time']
                                stderr_time = suite.get_values_fix_params(exp_path, 0, 'stderr_time', 'last', n = nval, **loopvar_dict)[0][0]['stderr_time']
                                nvals.append(nval)
                                mean_times.append(mean_time)
                                stderr_times.append(stderr_time)
                            except TypeError:  # experiment doesn't exist
                                continue
                        log2_nvals = np.log2(np.array(nvals))
                        mean_times, stderr_times = np.array(mean_times), np.array(stderr_times)
                        log2_mean_times = np.log2(mean_times)
                        log2_lower = np.log2(mean_times - 2 * stderr_times)
                        log2_upper = np.log2(mean_times + 2 * stderr_times)
                        plot, = ax.plot(log2_nvals, log2_mean_times, color = colors[i], linestyle = linestyles[j], linewidth = 2, marker = 'o')
                        plot.set_dash_capstyle('projecting')
                        if (('linestyle' in vars_by_distinguisher) or ('color' in vars_by_distinguisher)):
                            if (((x == 0) and (y == 0)) and ((i == 0) or (j == 0))):
                                plots_for_legend.append(plot)
                                key = ', '.join([legend_str(vars_by_distinguisher[dist], params[vars_by_distinguisher[dist]][k], vars_by_distinguisher[dist] in vars_to_suppress_in_legend) for (dist, k) in [('color', i), ('linestyle', j)] if (dist in vars_by_distinguisher)])
                                keys_for_legend.append(key)
                        ax.fill_between(log2_nvals, log2_lower, log2_upper, color = colors[i], alpha = 0.1)
                if ((numvals_by_distinguisher['yfacet'] > 1) and (x == 0)):
                    ax.annotate(legend_str(vars_by_distinguisher['yfacet'], str(params[vars_by_distinguisher['yfacet']][y]), vars_by_distinguisher['yfacet'] in vars_to_suppress_in_legend), xy = (0, 0.5), xytext = (-ax.yaxis.labelpad, 0), xycoords = ax.yaxis.label, textcoords = 'offset points', ha = 'right', va = 'center')
                if ((numvals_by_distinguisher['xfacet'] > 1) and (y == 0)):
                    ax.annotate(legend_str(vars_by_distinguisher['xfacet'], str(params[vars_by_distinguisher['xfacet']][x]), vars_by_distinguisher['xfacet'] in vars_to_suppress_in_legend), xy = (0.5, 1.01), xytext = (0, 0), xycoords = 'axes fraction', textcoords = 'offset points', ha = 'center', va = 'baseline')
                ax.set_xlim((np.log2(min(params['n'])), np.log2(max(params['n']))))
                rstyle(ax)
                ax.patch.set_facecolor('0.89')

        plot_params = dict_union({var : params[var] for var in vars_for_title}, outer_dict)
        this_plot_title = plot_title + ', '.join(['%s=%s' % (var, str(plot_params[var])) for var in vars_for_title])

        fig.text(0.5, 0.04, 'log2(n)', ha = 'center', fontsize = 14)
        fig.text(0.02, 0.5, 'log2(secs)', va = 'center', rotation = 'vertical', fontsize = 14)
        plt.figlegend(plots_for_legend, keys_for_legend, 'right', fontsize = 10)
        plt.suptitle(this_plot_title, fontsize = 16 - len(vars_for_title) // 5, fontweight = 'bold')
        plt.subplots_adjust(left = 0.11, right = 0.85)

        plot_path = exp_path + '/plots/' + '_'.join(['='.join(map(str, pair)) for pair in sorted(plot_params.items())]) + '.png'
        plt.savefig(plot_path)
        plt.show()


if __name__ == "__main__":
    main()