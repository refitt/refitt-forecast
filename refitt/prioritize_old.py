# Order by time to peak (grouped by 1 day) 
# with small uncertainty on plus side (grouped by 12 hours) 
# if minus uncertainity is in future make higher
# finally prioritize high moe to priortize SNe
# also, moe<1. and tpeak<14.

import numpy as np
import pandas as pd
import os, sys, shutil, glob
import json
import pdb
from refitt import defs
from refitt import kernel

event_folder=str(sys.argv[1]) #full path
cols=['object','band','moe','mag','mag_err','tpeak','size_uncer','min_tpeak']
df=pd.DataFrame(columns=cols)
for fname in glob.glob(event_folder+'/*_prediction.json'):#os.listdir(event_folder):
  with open(fname, 'r') as f:
    preds=json.load(f)
  instrument=preds['instrument']
  band_list=kernel.get_band_info(instrument)
  for i,b in enumerate(band_list):
    tpeak=preds['time_to_peak_'+defs.band_name_dict[b]]
    df=pd.concat([df,pd.DataFrame([[preds['ztf_id'],defs.band_name_dict[b],preds['moe'],
                  preds['next_mag_mean_'+defs.band_name_dict[b]],
                  preds['next_mag_sigma_'+defs.band_name_dict[b]],
                  abs(round(tpeak[0])),abs(round(2*(tpeak[1]-tpeak[2]))/2.),
                  -1.*np.sign(tpeak[2])]],columns=cols)])

df=df[(df['moe']<0.5) & (df['tpeak']<=7.) & (df['size_uncer']<=14.)]
df=df.groupby(['tpeak','size_uncer','min_tpeak'],sort=True).apply(
                                 lambda x: x.sort_values(['moe'])).reset_index(drop=True)
df.index+=1
#ordering is import to achieve:
# Prioritize things near peak with small uncertainity and minpeak still to occur with high confidence
with open(event_folder+'/priority/priority.csv','w') as f:
  df.to_csv(f,columns=['object','band','mag','mag_err'],header=False)

