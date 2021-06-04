import json
import refitt
import os,sys
import pdb

fl=str(sys.argv[1])
path,fname = os.path.split(fl)
with open(fl, 'r') as f:
  preds=json.load(f)

instrument='ZTF_public'
obj_keys=['ztf_id','instrument','time_since_trigger','current_time','num_obs','class','time_arr','mdmc','moe']
band_keys=['phase','next_mag_mean','next_mag_sigma','time_to_peak','mag_mean','mag_sigma']

band_list=refitt.get_band_info(instrument)
for i,b in enumerate(band_list):
  d={}
  for key in obj_keys:
    d[key]=preds[key]
  band=refitt.band_name_dict[b]
  d['filter']=band+'-ztf'
  for key in band_keys:  
    d[key]=preds[key+'_'+band]
  with open(path+'/bandwise_forecasts/'+os.path.splitext(fname)[0]+'_'+band+'.json','w') as f:
    json.dump(d,f,indent=4)

