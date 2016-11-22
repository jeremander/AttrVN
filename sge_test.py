"""This script is a wrapper for a pyexperiment test that parallelizes experiments across a Sun Grid Engine (SGE). User provides grid parameters, as well as the name of the test script and its parameters."""

import sys
import os
import optparse
import subprocess
import numpy as np
from pyexperiment.expsuite import PyExperimentSuite

class SGETestSuite(PyExperimentSuite):
    class SGETestOptions():
        pass
    def __init__(self, experiments):
        self.options = SGETestSuite.SGETestOptions()
        self.options.ncores = 1
        self.options.config = 'experiments.cfg'
        self.options.experiments = experiments
        self.parse_cfg()


def main():
    optparser = optparse.OptionParser()
    optparser.add_option('-j', '--jobs',
        action='store', dest='njobs', type='int', default=1,
        help="number of jobs")
    optparser.add_option('-G', '--gigs', 
        action='store', dest='gigs', type='int', default=32,
        help="gigabytes of memory")
    optparser.add_option('-H', '--hours',
        action='store', dest='hours', type='int', default=24,
        help="hours of grid time")
    optparser.add_option('-e', '--experiment',
        action='append', dest='experiments', type='string', default=None,
        help="run only selected experiments, by default run all experiments in config file.")

    opts, args = optparser.parse_args()
    assert (len(opts.experiments) == 1), "Can only do one experiment at a time."

    print('cd %s\n' % sys.argv[1])
    os.chdir(sys.argv[1])

    suite = SGETestSuite(opts.experiments)

    paramlist = []
    for exp in suite.cfgparser.sections():
        if not suite.options.experiments or exp in suite.options.experiments:
            params = suite.items_to_params(suite.cfgparser.items(exp))
            params['name'] = exp
            paramlist.append(params)
    iter_vars = [var for (var, val) in paramlist[0].items() if isinstance(val, list)]

    paramlist = suite.expand_param_list(paramlist)
    num_exps = len(paramlist)
    exps_per_job = num_exps // opts.njobs
    perm = np.random.permutation(num_exps)
    suite.mkdir('tmp')
    chunksize = int(np.ceil(len(paramlist) / opts.njobs))
    for i in range(opts.njobs):
        #expnames = []
        with open('tmp/experiments%d.cfg' % i, 'w') as f:
            for j in range(chunksize):
                ctr = i * chunksize + j
                if (ctr < len(paramlist)):
                    expname = opts.experiments[0] + '/' + ','.join(['%s=%s' % (var, val) for (var, val) in paramlist[ctr].items() if (var in iter_vars)])
                    f.write('[%s]\n' % expname)
                    for (var, val) in paramlist[ctr].items():
                        valstr = ("'%s'" % val) if isinstance(val, str) else str(val)
                        f.write('%s = %s\n' % (var, valstr))
                    f.write('\n')

    os.chdir('..')
    script = "#!/usr/bin/env python3\nimport os\ntask_id = int(os.environ['SGE_TASK_ID'])\ni = task_id - 1\nos.system('sleep %d' % (15 * i))\nos.system('python3 -u test.py " + sys.argv[1] + " -n 1 -c tmp/experiments%d.cfg' % i)\n"
    filename = "%s/tmp/script.py" % sys.argv[1]
    open(filename, 'w').write(script)
    os.chmod(filename, 0o770)

    cmd = "qsub -t 1-%d -q all.q -l num_proc=%d,mem_free=%dG,h_rt=%d:00:00 -b Y -V -cwd -j yes -o %s/tmp -N experiment '%s/tmp/script.py'" % (opts.njobs, 1, opts.gigs, opts.hours, sys.argv[1], sys.argv[1])
    print(cmd)
    subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)


if __name__ == "__main__":
    main()
