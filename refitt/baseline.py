# scan list of events within folder supplied ZTF_YYYY-MM-DD*
# report in priority.csv style sorted by mag 

import os, sys, shutil, glob
import json
import pandas as pd

event_folder=str(sys.argv[1]) #full path
now=float(sys.argv[2])

event_dict={}
for fname in glob.glob(event_folder+'/meta/*'):#os.listdir(event_folder):
   with open(fname, 'r') as f:
     info=json.load(f)
   if not info['g_mag']:
     band='r'
     mag=info['r_mag']
     mag_err=info['r_mag_err']
   elif not info['r_mag']:
     band='g'
     mag=info['g_mag']
     mag_err=info['g_mag_err']
   elif info['r_mag']<info['g_mag']:
     band='r'
     mag=info['r_mag']
     mag_err=info['r_mag_err']
   else:
     band='g'
     mag=info['g_mag']
     mag_err=info['g_mag_err']
   obj=fname.split(event_folder+'/meta/')[1].split('_')[0]
   try:
     with open(event_folder+'/'+fname+'_prediction_'+str(now)+'.json', 'r') as f:
       preds=json.load(f)
     pred_mags=[preds['next_mag_mean_g'],preds['next_mag_mean_r']]
     pred_mag_errs=[preds['next_mag_sigma_g'],preds['next_mag_sigma_r']]
     mag=max(pred_mags)
     mag_err=pred_mag_errs[pred_mags.index(max(pred_mags))]
   except:
     pass
   event_dict[obj]={'band':band,'mag':mag,'mag_err':mag_err}

event_df=pd.DataFrame.from_dict(event_dict,orient='index')
event_df=event_df.sort_values(by=['mag']).reset_index()
event_df.index=event_df.index+1
with open(event_folder+'/priority/baseline.csv','w') as f:
   event_df.to_csv(f, header=False)

