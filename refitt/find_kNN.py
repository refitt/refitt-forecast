# decision notes: Not using r because strongly variable r values and NNs
#                 Not able to envelope because class guess is made per k choice

import numpy as np
import pandas as pd
import json,pickle
import os,sys,glob,pdb
import refitt

phase=int(sys.argv[1])
kmax=50
k_arr=range(len(refitt.lib_classes.keys())+2,kmax+1,1) #min to ensure at least 3 neighbours selected
objective=np.zeros((len(k_arr),3))
for i,k in enumerate(k_arr):
  objective[i,0]=k

library_loc=refitt.select_library('ZTF_public',phase)
fnames=np.load(refitt.DATA_PATH+'balltree_AE_'+str(phase)+'_fnames.npy')
for n,fl in enumerate(fnames):#np.random.choice(fnames,size=round(10*len(fnames)/100))):
  df_LC=pd.read_json(fl,orient='index').sort_values(by=['mjd'])
  obj=refitt.Transient(fl,df_LC,current_mjd=df_LC['mjd'].min()+phase)
  time_predict=np.arange(obj.trigger,obj.trigger+refitt.window+refitt.resol,refitt.resol)
  AE_fl=fl.split('train')[0]+library_loc+'/'+os.path.basename(fl).split('.')[0]+'_Xception.npy'
  obj.AE_rep=np.load(AE_fl)
  obj.find_NNs(kmax,start=1).predict_kNN(time_predict,obj.ref_list)
  for i,k in enumerate(k_arr):
    #with full k_arr, find all rel classes for each k and predict on those rels
    c_rel,cguess=obj.get_class(obj.ref_list,k,reset_index=False)
    c_rel_ind=np.zeros(obj.ref_list.shape[0],dtype=bool) 
    c_rel_ind[c_rel.index.values]=True
    obj.summarize_kNN(c_rel_ind)
    if obj.status>90:
      error_metric_mtmc=0.
      for j, obs in df_LC.iterrows():
        color_i=int(obs['passband'])-1
        idx=int((obs['mjd']-obj.trigger)/refitt.resol)
        if not obs['mjd']>obj.trigger+refitt.window: #sometimes a photo may take to after window
          error_metric_mtmc+=abs(obj.mag_predict_mean[color_i,idx]-obs['mag'])
                    #mag_pred_sigma[color_i,idx]))
      objective[i,1]+=error_metric_mtmc/df_LC.shape[0]#alpha*mdmc-alpha*mdoe+mdoe
      objective[i,2]+=1

np.save(refitt.DATA_PATH+'obj_'+str(phase)+'.npy',objective)

