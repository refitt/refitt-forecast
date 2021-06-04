import numpy as np
import pandas as pd
import os,sys,glob,pdb
import random

import refitt

time_since_trigger=int(sys.argv[1])
instrument='ZTF_public'
classf=pd.DataFrame()

library_loc=refitt.select_library(time_since_trigger,instrument)
for c in refitt.lib_classes.keys():
  fname=np.array([])
  for event in glob.glob(refitt.refitt_loc+'/'+c+'/'+library_loc+'/*'):
    fname=np.append(fname,np.array(event)) if fname.size else np.array(event)
  for n,f in enumerate(np.random.choice(fname,size=round(20*len(fname)/100))):
    fl=f.split(instrument)
    fl=fl[0]+os.path.basename(fl[1]).split('_Xception')[0]+'.json'
    df_LC=pd.read_json(fl,orient='index')
    trigger_date=df_LC['mjd'].min()
    obs_LC=df_LC[df_LC['mjd']<=trigger_date+time_since_trigger]

    band_list=refitt.get_band_info(instrument)
    AE_rep=np.load(f)
    kNN=refitt.kNN_at_tst(time_since_trigger,instrument)
    c_list=refitt.find_NNs(AE_rep,kNN,time_since_trigger,instrument,start=1)
    c_rel,cguess=refitt.get_class(c_list,kNN,reset_index=False) 
    classf=pd.concat([classf,pd.DataFrame([[c,cguess]])])

classf.to_pickle('classf_'+str(time_since_trigger)+'.pkl')

