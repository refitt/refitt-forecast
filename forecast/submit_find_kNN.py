import os, sys, glob
import subprocess
from forecast import defs

dir_path = os.path.dirname(os.path.realpath(__file__))
qname='refitt'
threads=24
runs=''
sc_name='find_kNN'
for tst in range(defs.window+1,defs.horizon,-1):
  runs+='time python '+sc_name+'.py '+str(tst)+'\n'

with open(dir_path+'/'+sc_name+'_1_list', 'w') as f:
   f.writelines(runs)

JOBSCRIPT=f"""#!/bin/sh -l
#SBATCH -A {qname}
#SBATCH -N 1
#SBATCH -n {threads}
#SBATCH -t 8-00:00:00

module load anaconda/5.1.0-py36
source activate /depot/cassiopeia/data/ari/refitt-forecast/envs
cat {dir_path}/{sc_name}_1_list | parallel -j{threads-1} {{}}
"""

with open(dir_path+'/'+sc_name+'.sub', 'w') as f:
   f.write(JOBSCRIPT)
proc = subprocess.Popen(['sbatch', dir_path+'/'+sc_name+'.sub'])

