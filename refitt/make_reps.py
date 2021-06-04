#this code makes several calls to refitt's create_AE_rep function in parallel

import numpy as np
import pandas as pd
import os, sys, glob
import refitt 
import json
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))
qname='physics'
threads=24
runs=''
sc_name='make_reps'
for phase in range(1,31):#refitt.horizon,refitt.window+1):
  library_loc=refitt.select_library('ZTF_public',phase)
  for c in refitt.lib_classes.keys():
     if not os.path.exists(refitt.DATA_PATH+c+'/'+library_loc):
        os.makedirs(refitt.DATA_PATH+c+'/'+library_loc)
     for event in glob.glob(refitt.DATA_PATH+c+'/train/*.json'):
        name=(refitt.DATA_PATH+c+'/'+library_loc+'/'+
                os.path.basename(event).split('.')[0]+'_Xception')
        runs+="python -c \"import refitt; import pandas as pd; import numpy as np; " 
        runs+="LC=pd.read_json('"+event+"',orient='index').sort_values(by=['mjd']); "
        runs+="obj=refitt.Transient('"+event+"',LC,current_mjd=LC['mjd'].min()+"+str(phase)+"); "
        runs+="obj.create_AE_rep(); "
        runs+="np.save('"+name+"',obj.AE_rep)\"\n"

with open(dir_path+'/'+sc_name+'_list', 'w') as f:
   f.writelines(runs)

JOBSCRIPT=f"""#!/bin/sh -l
#SBATCH -A {qname}
#SBATCH -N 1
#SBATCH -n {threads}
#SBATCH -t 8-00:00:00

export PATH="~/.linuxbrew/bin:$PATH"
module load learning/conda-5.1.0-py36-cpu
module load ml-toolkit-cpu/keras/2.1.5
cat {dir_path}/{sc_name}_list | parallel -j{threads-1} {{}}
"""
with open(dir_path+'/'+sc_name+'.sub', 'w') as f:
   f.write(JOBSCRIPT)
proc = subprocess.Popen(['sbatch', dir_path+'/'+sc_name+'.sub'])

