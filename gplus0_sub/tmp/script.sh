
#! /bin/bash
EXP_NUM=`expr $SGE_TASK_ID - 1`
python3 test.py gplus0_sub -n 1 -c tmp/experiments.cfg -e experiment$EXP_NUM
