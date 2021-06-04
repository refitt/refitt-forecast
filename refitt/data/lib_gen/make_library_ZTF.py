# will overwrite all files
import json
import pandas as pd
import numpy as np
import math
import os,sys,glob,time
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from scipy import stats
from joblib import dump, load
from astropy.coordinates import SkyCoord
from astropy import units as u
import sncosmo
import sfdmap
dustmap = sfdmap.SFDMap('sfddata-master')

import refitt

events_per_bin=500
types=refitt.lib_classes
kde=load('revisit_kde.joblib')
noise=load('noise_kde.joblib')
df_samp=pd.read_pickle('sampling.pkl')
uncer_param_df=pd.read_pickle('uncer_params.pkl')

def fit_Ia_model(sn_name):
  data=sncosmo.read_lc('sncosmo/'+sn_name+'.json', format='json')
  with open('train_lcs/'+sn_name+'_meta.json','r') as f:
    sn_meta=json.load(f)
  c=SkyCoord(sn_meta['R.A.'],sn_meta['Declination'],
            unit=(u.hourangle, u.deg))
  dust=sncosmo.CCM89Dust() #r_v=A(V)/E(B−V); A(V) is total extinction in V
  model=sncosmo.Model(source='salt2',
                      effects=[dust],
                      effect_names=['mw'],
                      effect_frames=['obs'])
  ebv=dustmap.ebv(c) #ICRS frame
  model.set(z=data.meta['z'],mwebv=ebv)  # set the model's redshift.
  result,fitted_model=sncosmo.fit_lc(data,model,
                                      ['t0','x0','x1','c'])
  return result,fitted_model

def resimulate_Ia(sn_name,df_sim,z_new,ra_new,dec_new):
  band_name={1:'ztfg',2:'ztfr'}
  success=True
  try:
    result,fitted_model=fit_Ia_model(sn_name)
    if result.chisq>300.: success=False
  except RuntimeError:
    success=False
  if success:
    #put model in new conditions
    c=SkyCoord(ra_new,dec_new,unit='deg')
    ebv=dustmap.ebv(c) #ICRS frame
    fitted_model.set(z=z_new,mwebv=ebv)
    zp=df_sim['passband'].map(refitt.band_name_dict).map(refitt.ZTF_zp_dict)
    flux=fitted_model.bandflux(df_sim['passband'].map(band_name),
                              df_sim['mjd'],
                              zp=zp,
                              zpsys='ab')
    df_sim['mag']=zp-2.5*np.log10(flux+flux*noise.sample(
                                                n_samples=df_sim.shape[0])[:,0])
  return df_sim,success

def resimulate_GP(sn_name,df_sim,z_new,*args):
  success=True
  with open('train_lcs/'+sn_name+'_meta.json','r') as f:
    meta=json.load(f)
  z_org=float(meta['z'])
  df_LC=pd.read_json('train_lcs/'+sn_name+'.json',orient='index')
  df_LC=df_LC.sort_values(by=['mjd'],kind='mergesort')
  # sample GP fit at contracted timestamps and redshifted mean wvls
  obs_src_time=df_LC['mjd'].min()+(df_LC['mjd']-df_LC['mjd'].min())/(1.+z_org)
  sim_src_time=df_sim['mjd'].min()+(df_sim['mjd']-df_sim['mjd'].min())/(1.+z_new)
  df_sim['mag'],mag_pred_std=refitt.GP_predict_ts(
                                            obs_src_time.values,
                                            df_LC['mag'].values,
                                            df_LC['mag_err'].values,
                                            df_LC['passband'].values,
                                            sim_src_time.values,
                                            df_sim['passband'].values,
                                            k_corr_scale=(1.+z_org)/(1.+z_new)
                                            )
  #dim/brighten source
  df_sim['mag']=df_sim['mag']+5.*np.log10(refitt.cosmo.luminosity_distance(z_new)/
                              refitt.cosmo.luminosity_distance(z_org))
  return df_sim,success

for c in classes.keys():
  sn_list=[]
  for sn in glob.glob('train_lcs/*_meta.json'):
    df_LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
    if df_LC[df_LC['mjd']<df_LC['mjd'][df_LC['mag'].idxmin()]].shape[0]==0:
      continue #conservative elimination
    with open(sn,'r') as f:
      meta=json.load(f)
    if meta['Type']==c:
      sn_list.append(sn)
  if not os.path.exists(refitt.DATA_PATH+c):
    os.makedirs(refitt.DATA_PATH+c)
  inj_stats=pd.DataFrame()
  sim_stats=pd.DataFrame()
  simnum=0
  for i in len(classes[c]['z_bins'])-1:
    ctr=0
    while ctr<math.ceil(events_per_bin):
      #pick random event
      sn=np.random.choice(sn_list)
      df_LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
      with open(sn,'r') as f:
        meta=json.load(f)
      #simulate new environment
      z_new=10.**np.random.uniform(classes[c]['z_bins'][i],
                              classes[c]['z_bins'][i+1])
            #+np.random.normal(0, 0.001))
      ra_new=360*np.random.random_sample()
      dec_new=np.random.uniform(-30, 90)
      inj_stats=pd.concat([inj_stats,
                          (pd.DataFrame([[meta['z'],meta['R.A.'],meta['Declination'],
                                        z_new,ra_new,dec_new,sn.split('_meta')[0].split('/')[1]]],
                                        columns=['z_org','ra_org','dec_org',
                                                'z_new','ra_new','dec_new','src_event']))
                          ])
      #get new observing strategy
      start=np.random.uniform(df_LC['mjd'][df_LC['mag'].idxmin]-21.,
                              df_LC['mjd'][df_LC['mag'].idxmin]+7.) #before and after peak
      df_sim=pd.DataFrame()
      t=start
      while t-start<refitt.window:
        samp=df_samp.sample()
        df_sim=pd.concat([df_sim,pd.DataFrame(
                                            np.array([t+samp['t'].values[0],
                                            samp['bands'].values[0]]).T
                                            )
                          ])
        t+=float(kde.sample())
      df_sim=df_sim.rename(columns={0:'mjd',1:'passband'})
      #simulate
      df_sim,success=globals()['resimulate_'+classes[c]['method']](
                        sn.split('_meta')[0].split('/')[1],
                        df_sim,z_new,ra_new,dec_new)
      if not success:
        continue
      df_sim['mag_err']=df_sim.apply(
                          lambda x: (uncer_param_df['band']==x['passband']) &
                                    (pd.arrays.IntervalArray(
                                      uncer_param_df['interval']).contains(x['mag'])),
                                    axis=1).apply(
                          lambda x: stats.skewnorm.rvs(
                                      uncer_param_df[x]['a'],
                                      uncer_param_df[x]['loc'],
                                      uncer_param_df[x]['scale']
                                                      ), axis=1)
      flux=10.**((df_sim['mag']-df_sim['passband'].map(refitt.band_name_dict)
                    .map(refitt.ZTF_zp_dict))/-2.5)
      df_sim['SNR']=flux/abs(flux*df_sim['mag_err']*(np.log(10.)/2.5))
      first_phot=df_sim.iloc[df_sim['mjd'].idxmin()]
      if not (first_phot['mag']<21. and first_phot['SNR']>5.):
        continue
      df_sim=df_sim[(df_sim['mag']<21.) & (df_sim['mag']>13.5)] #this order of checking ensures the first photo is early
      if ((df_sim.shape[0]<4) or (df_sim['passband'].nunique()<2)):
        continue
      ctr+=1
      simnum+=1
      sim_stats=pd.concat([sim_stats,
                            (pd.DataFrame([[meta['z'],meta['R.A.'],meta['Declination'],
                                          z_new,ra_new,dec_new,sn.split('_meta')[0].split('/')[1]]],
                                          index=[simnum],
                                          columns=['z_org','ra_org','dec_org',
                                                  'z_new','ra_new','dec_new','src_event']))
                          ])
      df_sim=df_sim.drop(columns=['SNR']).reset_index(drop=True)
      with open(refitt.DATA_PATH+c+'/train/'+str(simnum)+'.json','w') as f:
        json.dump(df_sim.to_dict(orient='index'),f,indent=4)
  inj_stats.to_pickle(refitt.DATA_PATH+c+'/train/inj_stats.pkl')
  sim_stats.to_pickle(refitt.DATA_PATH+c+'/train/sim_stats.pkl')
  #stats
  fig=plt.figure()
  plt.hist(inj_stats['z_new'])
  plt.hist(sim_stats['z_new'])
  plt.xlabel('z')
  plt.ylabel('count')
  fig.savefig(refitt.DATA_PATH+c+'/train/sim_dist.png')
