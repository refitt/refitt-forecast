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
import sfdmap
dustmap = sfdmap.SFDMap('../sfddata-master')

from forecast import defs,utils,kernel

cls=str(sys.argv[1])
classes=defs.lib_classes
revisit=load('revisit_kde.joblib')
noise=load('noise_kde.joblib')
sampling=pd.read_pickle('sampling.pkl')
uncer_param_df=pd.read_pickle('uncer_params.pkl')

def resimulate_Ia(meta,LC,sim,z_new,ra_new,dec_new):
  success=True
  try:
    result,fitted_model=utils.fit_Ia_model(meta,LC,spec_z=True)
    if result.chisq>300.: success=False
  except RuntimeError:
    success=False
  if success:
    #put model in new conditions
    c=SkyCoord(ra_new,dec_new,unit='deg')
    ebv=dustmap.ebv(c) #ICRS frame
    fitted_model.set(z=z_new,mwebv=ebv)
    zp=sim['passband'].map(defs.ZTF_zp_dict)
    flux=fitted_model.bandflux(sim['passband'].map(defs.sncosmo_band_name_dict),
                              sim['mjd'],
                              zp=zp,
                              zpsys='ab')
    sim['mag']=zp-2.5*np.log10(flux+flux*noise.sample(n_samples=sim.shape[0])[:,0])
  return sim,success

def resimulate_GP(meta,LC,sim,z_new,*args):
  success=True
  z_org=float(meta['z'])
  # sample GP fit at restframe timestamps and mean wvls
  obs_src_time=LC['mjd'].min()+(LC['mjd']-LC['mjd'].min())/(1.+z_org)
  sim_src_time=sim['mjd'].min()+(sim['mjd']-sim['mjd'].min())/(1.+z_new)
  band_pred=kernel.get_band_info('ZTF_public')
  LC_GP=pd.DataFrame(data=np.vstack((np.repeat(sim_src_time,len(band_pred)),
                                     np.tile(band_pred,len(sim_src_time))
                                    )).T,
                       columns=['mjd','passband'])
  k_corr_scale=(1.+z_org)/(1.+z_new)
  kernel.GP_Boone(obs_src_time.values,
                       LC['mag'].values,
                       LC['mag_err'].values,
                       LC['passband'].values,
                       LC_GP,
                       k_corr_scale)
  # put back in observer frame
  LC_GP['mjd']=LC_GP['mjd'].min()+(LC_GP['mjd']-LC_GP['mjd'].min())*(1.+z_new)
  sim=pd.merge(sim,LC_GP,how='left',
                  left_on=['mjd','passband'],right_on=['mjd','passband']
                  )
  #dim/brighten source
  sim['mag']=sim['mag']+5.*np.log10(defs.cosmo.luminosity_distance(z_new)/
                              defs.cosmo.luminosity_distance(z_org))
  return sim,success

#get list of real SNe of cls
sn_list=[]
for sn in glob.glob(defs.DATA_PATH+'lib_gen/train_lcs/*_meta.json'):
  LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
  if LC[LC['mjd']<LC['mjd'][LC['mag'].idxmin()]].shape[0]==0:
    continue #need something on the rise to model properly
  with open(sn,'r') as f:
    meta=json.load(f)
  if meta['Type']==cls:
    sn_list.append(sn)

if not os.path.exists(defs.DATA_PATH+cls):
  os.makedirs(defs.DATA_PATH+cls)
inj_stats=pd.DataFrame()
sim_stats=pd.DataFrame()
simnum=0
for i in range(len(classes[cls]['logz_bins'])-1):
  ctr=0
  while ctr<defs.events_per_bin:
    #pick random event
    sn=np.random.choice(sn_list)
    LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index').sort_values(by=['mjd'])
    with open(sn,'r') as f:
      meta=json.load(f)
    #simulate new environment
    z_new=10.**np.random.uniform(classes[cls]['logz_bins'][i],
                            classes[cls]['logz_bins'][i+1])
          #+np.random.normal(0, 0.001))
    ra_new=360*np.random.random_sample()
    dec_new=np.random.uniform(-30, 90)
    inj_stats=pd.concat([inj_stats,
                        (pd.DataFrame([[meta['z'],meta['R.A.'],meta['Declination'],
                                      z_new,ra_new,dec_new,sn.split('_meta')[0].split('/')[-1]]],
                                      columns=['z_org','ra_org','dec_org',
                                              'z_new','ra_new','dec_new','src_event']))
                        ])
    #get new observing strategy
    start=np.random.uniform(LC['mjd'][LC['mag'].idxmin]-21.,
                            LC['mjd'][LC['mag'].idxmin]+7.) #before and after peak
    sim=pd.DataFrame()
    t=start
    while t-start<defs.window:
      sample=sampling.sample()
      sim=pd.concat([sim,pd.DataFrame(
                                      np.array([t+sample['t'].values[0],
                                      sample['bands'].values[0]]).T
                                      )
                        ])
      t+=float(revisit.sample())
    sim=sim.rename(columns={0:'mjd',1:'passband'}).sort_values(by=['mjd'])
    #simulate
    sim,success=globals()['resimulate_'+classes[cls]['method']](
                      meta,LC,sim,z_new,ra_new,dec_new)
    if not success:
      continue

    sim['mag_err']=sim.apply(
                        lambda x: (uncer_param_df['band']==x['passband']) &
                                  (pd.arrays.IntervalArray(
                                    uncer_param_df['interval']).contains(x['mag'])),
                                  axis=1).apply(
                        lambda x: stats.skewnorm.rvs(
                                    uncer_param_df[x]['a'],
                                    uncer_param_df[x]['loc'],
                                    uncer_param_df[x]['scale']
                                                    ), axis=1)
    # ^^^ will be empty if mag_err > model upper limit
    flux=10.**((sim['mag']-sim['passband'].map(defs.ZTF_zp_dict))/-2.5)
    sim['SNR']=flux/abs(flux*sim['mag_err']*(np.log(10.)/2.5))
    first_phot=sim.iloc[sim['mjd'].idxmin()]
    if first_phot['SNR'].size==0: #can happen if first photo is brighter than noise range modeled
      continue
    elif (first_phot['SNR']<5.):
      continue
    sim=sim[(sim['mag']<21.5) & (sim['mag']>13.5)]
    if (sim.shape[0]<4) or (sim['passband'].nunique()<2):
      continue
    ctr+=1
    simnum+=1
    sim_stats=pd.concat([sim_stats,
                          (pd.DataFrame([[meta['z'],meta['R.A.'],meta['Declination'],
                                        z_new,ra_new,dec_new,
                                        sim['mag'].min(),sn.split('_meta')[0].split('/')[-1]]],
                                        index=[simnum],
                                        columns=['z_org','ra_org','dec_org',
                                                'z_new','ra_new','dec_new',
                                                'peak_mag','src_event']))
                        ])
    sim=sim.drop(columns=['SNR']).reset_index(drop=True)
    with open(defs.DATA_PATH+cls+'/'+str(simnum)+'.json','w') as f:
      json.dump(sim.to_dict(orient='index'),f,indent=4)
inj_stats.to_pickle(defs.DATA_PATH+cls+'/inj_stats.pkl')
sim_stats.to_pickle(defs.DATA_PATH+cls+'/sim_stats.pkl')
