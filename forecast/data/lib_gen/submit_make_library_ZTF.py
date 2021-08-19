import os, sys, glob
import subprocess
from forecast import defs

dir_path = os.path.dirname(os.path.realpath(__file__))
qname='physics'
threads=len(defs.lib_classes.keys())+1
runs=''
sc_name='make_library_ZTF'
for cls in defs.lib_classes.keys():
  runs+='time python '+sc_name+'.py '+cls+'\n'

with open(dir_path+'/'+sc_name+'_list', 'w') as f:
   f.writelines(runs)

JOBSCRIPT=f"""#!/bin/sh -l
#SBATCH -A {qname}
#SBATCH -N 1
#SBATCH -n {threads}
#SBATCH -t 8-00:00:00

cd {dir_path}
module load anaconda/5.1.0-py36
source activate /depot/cassiopeia/data/ari/refitt-forecast/envs
cat {dir_path}/{sc_name}_list | parallel -j{threads-1} {{}}
"""

with open(dir_path+'/'+sc_name+'.sub', 'w') as f:
   f.write(JOBSCRIPT)
proc = subprocess.Popen(['sbatch', dir_path+'/'+sc_name+'.sub'])
