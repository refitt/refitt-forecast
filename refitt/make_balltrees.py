import numpy as np
import pandas as pd
import os, sys, glob
from refitt import defs
import json
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))
qname='refitt'
threads=24
runs=''
sc_name='make_balltrees'
for phase in range(defs.horizon,defs.window+1):
  runs+='python -c "from refitt import utils; utils.'+sc_name+'('+str(phase)+')"\n'

with open(dir_path+'/'+sc_name+'_list', 'w') as f:
   f.writelines(runs)

JOBSCRIPT=f"""#!/bin/sh -l
#SBATCH -A {qname}
#SBATCH -N 1
#SBATCH -n {threads}
#SBATCH -t 8-00:00:00

cd {dir_path}
module load anaconda/5.1.0-py36
source activate /depot/cassiopeia/data/ari/refitt/envs
cat {dir_path}/{sc_name}_list | parallel -j{threads-1} {{}}
"""

with open(dir_path+'/'+sc_name+'.sub', 'w') as f:
   f.write(JOBSCRIPT)
proc = subprocess.Popen(['sbatch', dir_path+'/'+sc_name+'.sub'])

