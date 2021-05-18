import numpy as np
import pandas as pd
import os, sys, glob
import refitt 
import json
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))
qname='physics'
threads=24
instrument='ZTF_public'#, ZTF_full
runs=''
sc_name='find_kNN'
for tst in range(52,refitt.window+1):
  runs+='time python '+sc_name+'.py '+str(tst)+'\n'

with open(dir_path+'/'+sc_name+'_list_2', 'w') as f:
   f.writelines(runs)

JOBSCRIPT=f"""#!/bin/sh -l
#SBATCH -A {qname}
#SBATCH -N 1
#SBATCH -n {threads}
#SBATCH -t 8-00:00:00

export PATH="~/.linuxbrew/bin:$PATH"
module load learning/conda-5.1.0-py36-cpu
module load ml-toolkit-cpu/keras/2.1.5
cat {dir_path}/{sc_name}_list_2 | parallel -j{threads-1} {{}}
"""

with open(dir_path+'/'+sc_name+'.sub', 'w') as f:
   f.write(JOBSCRIPT)
proc = subprocess.Popen(['sbatch', dir_path+'/'+sc_name+'.sub'])

