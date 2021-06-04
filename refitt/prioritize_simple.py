import numpy as np
import pandas as pd
import os, sys, shutil, glob
import refitt
import json
import pdb
import refitt.utils as utils

event_folder=str(sys.argv[1]) #full path
cols=['object','band','moe','magg','magr','tpeak','plus_uncer','min_tpeak','class','sig',
     'mdmc','del_mag','bright']
df=pd.DataFrame(columns=cols)
for fname in glob.glob(event_folder+'/*_prediction.json'):#os.listdir(event_folder):
  with open(fname, 'r') as f:
    preds=json.load(f)
  band_list=refitt.get_band_info(preds['instrument'])
  df_LC=pd.read_json(event_folder+'/'+preds['ztf_id']+'.json',orient='index').sort_values(by=['mjd'])
  for i,b in enumerate(band_list):
    tpeak=preds['time_to_peak_'+refitt.band_name_dict[b]]
    df=pd.concat([df,pd.DataFrame([[
                  preds['ztf_id'],refitt.band_name_dict[b],preds['moe'],
                  preds['next_mag_mean_g'],preds['next_mag_mean_r'],
                  abs(round(tpeak[0])),abs(round(2*tpeak[1])/2.),
                  -1.*np.sign(tpeak[0]+tpeak[2]),preds['class'][0],preds['class'][1],
                  round(preds['mdmc'],1),round(max(
                  [(group['mag'].max()-group['mag'].min()) for pb,group in df_LC.groupby('passband')])
                  ,1),df_LC['mag'].min()]],columns=cols)])

# Prioritize things near peak (grouped by 1 day) with small uncertainity (grouped by 12 hours)
# and minpeak still to occur with small observed error
# also, moe<0.5 and tpeak<7.
df_peak=df[(df['moe']<0.5) & (df['tpeak']<=7.) & (df['plus_uncer']<=14.)] # meant to be conservative
df_peak=df_peak.groupby(['tpeak','plus_uncer','min_tpeak'],sort=True).apply(
                                 lambda x: x.sort_values(['moe'])).reset_index(drop=True)
df_peak.index+=1
with open(event_folder+'/priority/peak.csv','w') as f:
  df_peak.to_csv(f,columns=['object','band','tpeak','plus_uncer','min_tpeak','moe'],index_label='rank')

# Objects with high confidence classifiation and small observed error
df_class=df[(df['moe']<0.5)]
df_class=df_class.groupby(['sig'],sort=True).apply(lambda x: x.sort_values(['moe'],ascending=False))
df_class=df_class.iloc[::-1].drop_duplicates(subset='object').reset_index(drop=True)
df_class.index+=1
with open(event_folder+'/priority/class.csv','w') as f:
  df_class.to_csv(f,columns=['object','class','sig','moe'],index_label='rank')

# Objects with, large mdmc, and large observed error 
# but not brighter than 17 mag and at least ~1 mag change and observed error > 0.2
df_an=df.groupby(['mdmc'],sort=True).apply(lambda x: x.sort_values(['moe'])).reset_index(drop=True)
df_an=df_an[(df_an['bright']>17.) & (df_an['del_mag']>0.9) & (df_an['mdmc']>0.2) & (df_an['moe']>0.3)]
df_an=df_an.iloc[::-1].drop_duplicates(subset='object').reset_index(drop=True)
df_an.index+=1
with open(event_folder+'/priority/anomaly.csv','w') as f:
  df_an.to_csv(f,columns=['object','del_mag','mdmc','moe'],index_label='rank')

utils.log('Finished making priority lists. See {}'.format(event_folder+'/priority'))

