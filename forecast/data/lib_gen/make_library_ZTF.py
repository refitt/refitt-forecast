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
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import sncosmo
import sfdmap
dustmap = sfdmap.SFDMap('sfddata-master')

from forecast import defs, utils, kernel

cls=str(sys.argv[1])
classes=defs.lib_classes
kde=load('revisit_kde.joblib')
noise=load('noise_kde.joblib')
df_samp=pd.read_pickle('sampling.pkl')
uncer_param_df=pd.read_pickle('uncer_params.pkl')

def fit_Ia_model(meta,df_LC):
  flux,flux_err=utils.get_flux(df_LC)
  data=Table()
  data.meta['z']=float(meta['z'])
  data['mjd']=df_LC['mjd'].tolist()
  data['band']=df_LC['passband'].map(defs.sncosmo_band_name_dict).tolist()
  data['flux']=flux.tolist()
  data['flux_err']=flux_err.tolist()
  data['zp']=df_LC['passband'].map(defs.ZTF_zp_dict).tolist()
  data['zpsys']=['ab']*len(flux)
  c=SkyCoord(meta['R.A.'],meta['Declination'],
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

def resimulate_Ia(meta,df_LC,df_sim,z_new,ra_new,dec_new):
  success=True
  try:
    result,fitted_model=fit_Ia_model(meta,df_LC)
    if result.chisq>300.: success=False
  except RuntimeError:
    success=False
  if success:
    #put model in new conditions
    c=SkyCoord(ra_new,dec_new,unit='deg')
    ebv=dustmap.ebv(c) #ICRS frame
    fitted_model.set(z=z_new,mwebv=ebv)
    zp=df_sim['passband'].map(defs.ZTF_zp_dict)
    flux=fitted_model.bandflux(df_sim['passband'].map(defs.sncosmo_band_name_dict),
                              df_sim['mjd'],
                              zp=zp,
                              zpsys='ab')
    df_sim['mag']=zp-2.5*np.log10(flux+flux*noise.sample(n_samples=df_sim.shape[0])[:,0])
  return df_sim,success

def resimulate_GP(meta,df_LC,df_sim,z_new,*args):
  success=True
  z_org=float(meta['z'])
  # sample GP fit at restframe timestamps and mean wvls
  obs_src_time=df_LC['mjd'].min()+(df_LC['mjd']-df_LC['mjd'].min())/(1.+z_org)
  sim_src_time=df_sim['mjd'].min()+(df_sim['mjd']-df_sim['mjd'].min())/(1.+z_new)
  band_pred=kernel.get_band_info('ZTF_public')
  LC_GP=pd.DataFrame(data=np.vstack((np.repeat(sim_src_time,len(band_pred)),
                                     np.tile(band_pred,len(sim_src_time))
                                    )).T,
                       columns=['mjd','passband'])
  k_corr_scale=(1.+z_org)/(1.+z_new)
  kernel.GP_predict_ts(obs_src_time.values,
                       df_LC['mag'].values,
                       df_LC['mag_err'].values,
                       df_LC['passband'].values,
                       LC_GP,
                       k_corr_scale)
  # put back in observer frame
  LC_GP['mjd']=LC_GP['mjd'].min()+(LC_GP['mjd']-LC_GP['mjd'].min())*(1.+z_new)
  df_sim=pd.merge(df_sim,LC_GP,how='left',
                  left_on=['mjd','passband'],right_on=['mjd','passband']
                  )
  #dim/brighten source
  df_sim['mag']=df_sim['mag']+5.*np.log10(defs.cosmo.luminosity_distance(z_new)/
                              defs.cosmo.luminosity_distance(z_org))
  return df_sim,success

#get list of real SNe of cls
sn_list=[]
for sn in glob.glob(defs.DATA_PATH+'lib_gen/train_lcs/*_meta.json'):
  df_LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
  if df_LC[df_LC['mjd']<df_LC['mjd'][df_LC['mag'].idxmin()]].shape[0]==0:
    continue #conservative elimination
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
  while ctr<math.ceil(defs.events_per_bin):
    #pick random event
    sn=np.random.choice(sn_list)
    df_LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index').sort_values(by=['mjd'])
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
    start=np.random.uniform(df_LC['mjd'][df_LC['mag'].idxmin]-21.,
                            df_LC['mjd'][df_LC['mag'].idxmin]+7.) #before and after peak
    df_sim=pd.DataFrame()
    t=start
    while t-start<defs.window:
      samp=df_samp.sample()
      df_sim=pd.concat([df_sim,pd.DataFrame(
                                          np.array([t+samp['t'].values[0],
                                          samp['bands'].values[0]]).T
                                          )
                        ])
      t+=float(kde.sample())
    df_sim=df_sim.rename(columns={0:'mjd',1:'passband'}).sort_values(by=['mjd'])
    #simulate
    df_sim,success=globals()['resimulate_'+classes[cls]['method']](
                      meta,df_LC,df_sim,z_new,ra_new,dec_new)
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
    # ^^^ will be empty if mag_err > model upper limit
    flux=10.**((df_sim['mag']-df_sim['passband'].map(defs.ZTF_zp_dict))/-2.5)
    df_sim['SNR']=flux/abs(flux*df_sim['mag_err']*(np.log(10.)/2.5))
    first_phot=df_sim.iloc[df_sim['mjd'].idxmin()]
    if first_phot['SNR'].size==0: #can happen if first photo is brighter than noise range modeled
      continue
    elif (first_phot['SNR']<5.):
      continue
    df_sim=df_sim[(df_sim['mag']<21.5) & (df_sim['mag']>13.5)]
    if (df_sim.shape[0]<4) or (df_sim['passband'].nunique()<2):
      continue
    ctr+=1
    simnum+=1
    sim_stats=pd.concat([sim_stats,
                          (pd.DataFrame([[meta['z'],meta['R.A.'],meta['Declination'],
                                        z_new,ra_new,dec_new,
                                        df_sim['mag'].min(),sn.split('_meta')[0].split('/')[-1]]],
                                        index=[simnum],
                                        columns=['z_org','ra_org','dec_org',
                                                'z_new','ra_new','dec_new',
                                                'peak_mag','src_event']))
                        ])
    df_sim=df_sim.drop(columns=['SNR']).reset_index(drop=True)
    with open(defs.DATA_PATH+cls+'/'+str(simnum)+'.json','w') as f:
      json.dump(df_sim.to_dict(orient='index'),f,indent=4)
inj_stats.to_pickle(defs.DATA_PATH+cls+'/inj_stats.pkl')
sim_stats.to_pickle(defs.DATA_PATH+cls+'/sim_stats.pkl')
