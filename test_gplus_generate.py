import subprocess
import os.path
import pandas as pd 

selected_attrs = pd.read_csv('selected_attrs.csv')

pos_neg_vals = [(10, 5), (5, 20), (20, 40), (50, 50), (200, 400), (400, 400), (800, 1600)]
num_samples = 50
memory = 24  # number of GB

for (pos_seeds, neg_seeds) in pos_neg_vals:
    for (attr, attr_type) in zip(selected_attrs['attribute'], selected_attrs['attributeType']):
        if (selected_attrs[(selected_attrs['attribute'] == attr) & (selected_attrs['attributeType'] == attr_type)]['freq'].iloc[0] >= 2 * pos_seeds):
            safe_attr = '_'.join(attr.split())
            subcmd = "time python3 -u test_gplus.py -a '%s' -t '%s' -p %d -n %d -S %d -v" % (attr, attr_type, pos_seeds, neg_seeds, num_samples)
            cmd = 'qsub -q all.q -l num_proc=1,mem_free=%dG,h_rt=48:00:00 -b Y -V -cwd -j yes -o . -N gplus_%s_%s_p%d_n%d "%s"' % (memory, safe_attr, attr_type, pos_seeds, neg_seeds, subcmd)
            print(cmd)
            subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)
