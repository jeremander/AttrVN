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

    print('cd %s\n' % sys.argv[1])
    os.chdir(sys.argv[1])

    suite = SGETestSuite(opts.experiments)

    paramlist = []
    for exp in suite.cfgparser.sections():
        if not suite.options.experiments or exp in suite.options.experiments:
            params = suite.items_to_params(suite.cfgparser.items(exp))
            params['name'] = exp
            paramlist.append(params)

    paramlist = suite.expand_param_list(paramlist)
    num_exps = len(paramlist)
    exps_per_job = num_exps // opts.njobs
    perm = np.random.permutation(num_exps)
    suite.mkdir('tmp')
    for i in range(opts.njobs):
        with open('tmp/experiment%d.cfg' % i, 'w') as f:
            f.write('[experiment%d]\n' % i)
            for (var, val) in paramlist[i].items():
                f.write('%s = %s\n' % (var, str(val)))

    os.chdir('..')
    script = """
#! /bin/bash
EXP_NUM=`expr $SGE_TASK_ID - 1`
python3 test.py %s -n 1 -c tmp/experiment$EXP_NUM.cfg -e experiment$EXP_NUM
""" % sys.argv[1]
    open("%s/tmp/script.sh" % sys.argv[1], 'w').write(script)

    cmd = "qsub -t 1-%d -q all.q -l num_proc=%d,mem_free=%dG,h_rt=%d:00:00 -b Y -V -cwd -j yes -o %s/tmp -N experiment '%s/tmp/script.sh'" % (opts.njobs, 1, opts.gigs, opts.hours, sys.argv[1], sys.argv[1])
    print(cmd)
    subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)


if __name__ == "__main__":
    main()
