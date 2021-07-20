#!/depot/cassiopeia/apps/refitt/build/main/bin/python

import numpy as np
import pandas as pd
import os, sys, glob, shutil
import subprocess
from astropy.time import Time
from datetime import datetime, timedelta
from refitt import utils

qname='standby'
wt={'refitt':'2:00','standby':'2:00','debug':'30'}
threads=24

def run_refitt():
  dir_path = os.path.dirname(os.path.realpath(__file__))
  today=datetime.today()
  now=round(Time(today).mjd,3)
  dest=dir_path+'/ZTF/ZTF_'+today.strftime('%Y-%m-%dT%H%M')
  os.makedirs(dest)
  os.makedirs(dest+'/meta')
  os.makedirs(dest+'/priority')
  os.makedirs(dest+'/bandwise_forecasts')
  
  JOBSCRIPT = f"""#!/bin/bash -l
#SBATCH -A {qname}
#SBATCH -N 1
#SBATCH -n {threads}
#SBATCH -t {wt[qname]}:00

cd {dir_path}
module load anaconda/5.1.0-py36
source activate /depot/cassiopeia/data/ari/refitt/envs
python get_objects_antares.py {dest} {now}
ls {dest}/*.json | parallel -j{threads-1} 'python kernel.py {{}}'
ls {dest}/*_prediction.json | parallel -j{threads-1} 'python make_bandwise_jsons.py {{}}'
python baseline.py {dest} {now}
python prioritize_old.py {dest}
python prioritize_simple.py {dest}
  """
  with open(dir_path+'/run.sub', 'w') as f:
      f.write(JOBSCRIPT)
  proc = subprocess.Popen(['sbatch', dir_path+'/run.sub'])

if __name__ == '__main__':
  try:
    utils.log('Daily refitt run started')
    run_refitt()
  except:
    utils.email('Daily refitt run failed with error '+str(sys.exc_info()[0]))
    utils.log('Daily refitt run failed with error '+str(sys.exc_info()[0]))

