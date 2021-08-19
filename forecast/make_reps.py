import numpy as np
import pandas as pd
import os, sys, glob
import json
import subprocess
from forecast import defs, kernel

dir_path = os.path.dirname(os.path.realpath(__file__))
qname='refitt'
threads=24
runs=''
sc_name='make_reps'
survey='ZTF_public'
for phase in range(50,defs.window+1): #defs.horizon
  library_loc=kernel.select_library(survey,phase)
  for c in defs.lib_classes.keys():
     if not os.path.exists(defs.DATA_PATH+c+'/'+library_loc):
        os.makedirs(defs.DATA_PATH+c+'/'+library_loc)
     for event in glob.glob(defs.DATA_PATH+c+'/*.json'):
        name=(defs.DATA_PATH+c+'/'+library_loc+'/'+
                os.path.basename(event).split('.')[0]+'_Xception')
        runs+="python -c \"from forecast import kernel; import pandas as pd; import numpy as np; " 
        runs+="LC=pd.read_json('"+event+"',orient='index').sort_values(by=['mjd']); "
        runs+="obj=kernel.Transient('"+event+"',LC,'"+defs.DATA_PATH+c+"/"+library_loc+"',current_mjd=LC['mjd'].min()+"+str(phase)+"); "
        runs+="obj.create_AE_rep(); "
        runs+="np.save('"+name+"',obj.AE_rep)\"\n"

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

