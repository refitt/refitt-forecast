#dropping less than 4 data points
# or data in only one band
# not converting events with no rise

import numpy as np
import pandas as pd
import os, sys, glob
import json
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_squared_error, make_scorer

from scipy import stats
from scipy.stats import chi2_contingency
from pandas.plotting import scatter_matrix

bands=['g','r']
band_name={1:'ztfg',2:'ztfr'}
zp_dict={1:26.325,2:26.275} #from ZSDS_explanatory pg 67
k=5

sne=[]
for fname in glob.glob('ztf_lcs/*.txt'):
  sne.append(fname.split('.')[0])

meta=pd.DataFrame()
for y in range(2018,2021):
  meta=meta.append(pd.read_json('ztf_lcs/'+str(y)+'_sne_table.json')
                    .set_index('lasairname'))

info=['z','Type','R.A.','Declination']
for sn in sne:
  df = pd.DataFrame()
  for band in bands:
    df=df.append(pd.read_csv(sn+'.'+band+'.txt',skiprows=1,
                  delimiter=' ',
                  names=['mjd','passband','mag','mag_err']))
  if df.shape[0]<4 or df[df['passband']==1].empty or df[df['passband']==2].empty:
    continue
  df['locus_id']=sn.split('/')[1]
  df=df.sort_values(by=['mjd'],kind='mergesort')
  df=df.reset_index(drop=True)
  with open(sn+'.json','w') as f:
    json.dump(df.to_dict(orient='index'),f,indent=4)
  with open(sn+'_meta.json','w') as f:
    json.dump(meta.loc[sn.split('/')[1]][info].to_dict(),f,indent=4)

#estimate revisit intervals
# does not account for gaps due to moon
df=pd.DataFrame()
for sn in glob.glob('ztf_lcs/*_meta.json'):
  with open(sn,'r') as f:
    meta=json.load(f)
  df_LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
  df_LC=df_LC.sort_values(by=['mjd'],kind='mergesort')
  df_rev=df_LC[(df_LC['mjd'].diff()>1.) | (df_LC['mjd'].diff().isna())
              ][['mjd','mag']].rename(columns={'mjd':'gap'})
  df_rev['gap']=df_rev['gap'].diff()
  df_rev=df_rev.fillna(0)
  df_rev=df_rev[df_rev['gap']!=0.]
  df=pd.concat([df,df_rev])

visit=df['gap'].values
bandwidths=np.logspace(-5,-1,10)
train_scores,valid_scores=validation_curve(KernelDensity(kernel='gaussian'),
                                          visit[:,None],None,
                                          'bandwidth',#'kernel'],
                                          bandwidths,#kernels],
                                          cv=k)#, scoring=mse)

plt.figure()
plt.semilogx(bandwidths, np.mean(train_scores,axis=1))
plt.ylabel(str(k)+'-fold CV score')
plt.show()

bins=np.arange(0.,10.,0.01)
fig, ax = plt.subplots()
plt.hist(visit,bins=bins,density=True)
kde=KernelDensity(kernel='gaussian',bandwidth=2e-3).fit(visit[:,None])
x_plot=bins[:,np.newaxis]
dens=np.exp(kde.score_samples(x_plot))
plt.plot(x_plot[:,0],dens,lw=0.5)
plt.xlabel('days')
plt.savefig('revisit_gaussian_2e-3.png',dpi=300)

dump(kde,'revisit_kde.joblib')

#verification of independence of per visit attributes
# build contingency table (revisit finished 0-4 hrs Fig 9 Bellm 2019)
df_cont=pd.DataFrame()
for sn in glob.glob('ztf_lcs/*_meta.json'):
  with open(sn,'r') as f:
    meta=json.load(f)
  df_LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
  df_LC['mjd']=round(df_LC['mjd']-7/24) #putting mid observation at mn to help with rounding
  df_strat=pd.concat([group['passband'].reset_index(drop=True)
                        for name, group in df_LC.groupby('mjd')],
                        axis=1).T
  df_strat=df_strat.replace({1.0:'r',2.0:'g'})#,np.nan:'none'})
  df_strat=pd.get_dummies(df_strat)
  df_cont=pd.concat([df_cont,
                  df_strat.groupby(df_strat.reset_index(drop=True).index%2).sum()])

df_cont=df_cont.groupby(df_cont.reset_index(drop=True).index%2).sum()
chi2_contingency(df_cont)
# I cannot accept they are dependent

df_samp=pd.DataFrame()
for sn in glob.glob('ztf_lcs/*_meta.json'):
  with open(sn,'r') as f:
    meta=json.load(f)
  df_LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
  df_LC['mjd_round']=round(df_LC['mjd']-7/24) #putting mid observation at mn to help with rounding
  df_samp=pd.concat([df_samp,
                    pd.DataFrame([[(group['mjd']-group['mjd'].min()).values,
                                  group['passband'].values]
                        for name, group in df_LC.groupby('mjd_round')])
                    ])
df_samp.rename(columns=
              {0:'t',1:'bands'}
              ).reset_index(drop=True).to_pickle('sampling.pkl')

# uncertainity model
df_uncer=pd.DataFrame()
for sn in glob.glob('ztf_lcs/*_meta.json'):
  with open(sn,'r') as f:
    meta=json.load(f)
  df_LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
  df_uncer=pd.concat([df_uncer,df_LC[['mag','mag_err','passband']]])

mag_bins=[13.5,16.5,18.,18.5,19.,19.5,20,21.5]
bins=pd.cut(df_uncer['mag'], bins=mag_bins)
param_df=pd.DataFrame()
for name, group in df_uncer.groupby([bins,'passband']):
  uncer=group['mag_err'].values
  '''
  bandwidths=np.logspace(-5,-2,10)
  train_scores,valid_scores=validation_curve(KernelDensity(kernel='gaussian'),
                                            uncer[:,None],None,
                                            'bandwidth',#'kernel'],
                                            bandwidths,#kernels],
                                            cv=k)#, scoring=mse)

  plt.figure()
  plt.semilogx(bandwidths, np.mean(train_scores,axis=1)) #TF is the score here?
  plt.ylabel(str(k)+'-fold CV score')
  plt.show()
  '''
  bw=3e-4
  pbins=np.arange(0.,0.5,0.0025)
  fig, ax = plt.subplots()
  plt.hist(uncer,bins=pbins,density=True)
  #hist,bin_edges=np.histogram(uncer,bins=pbins)
  #print(hist.max(),name)
  kde=KernelDensity(kernel='gaussian',bandwidth=bw).fit(uncer[:,None])
  x_plot=pbins[:,np.newaxis]
  dens=np.exp(kde.score_samples(x_plot))
  plt.plot(x_plot[:,0],dens,lw=1)

  p=group['mag_err'].quantile(0.98)
  ae,loce,scalee=stats.skewnorm.fit(group[(group['mag_err']<p)]['mag_err'])
  p=stats.skewnorm.pdf(pbins,ae,loce,scalee)
  plt.plot(pbins,p,lw=2.)
  plt.xlabel('magnitude uncertainity')
  plt.savefig('uncer_skew_'+str(name)+'.png',dpi=300)

  param_df=pd.concat([param_df,pd.DataFrame([[name[1],name[0],ae,loce,scalee]])]
                    )

param_df.rename(columns={0:'band',1:'interval',2:'a',3:'loc',4:'scale'}
                ).reset_index(drop=True).to_pickle('uncer_params.pkl')

'''
def base(data,a2,a1,a0,b2,b1,b0,c1,c0):
  x,y=data
  obj=np.exp(((y-(a0+a1*np.exp(a2*x)))/(b0+b1*x+b2*x**2))**2.)
  return c0+c1*obj

for name, group in df_uncer.groupby('passband'):
  nbins=100
  H,xedges,yedges=np.histogram2d(group['mag'],group['mag_err'],bins=[nbins,nbins])
  fig=plt.figure()
  X,Y=np.meshgrid(xedges,yedges)
  plt.pcolormesh(X,Y,H.T,cmap='Greys')

  xcenter=(xedges[:-1]+xedges[1:])/2
  ycenter=(yedges[:-1]+yedges[1:])/2
  xx,yy=np.meshgrid(xcenter,ycenter)
  popt,pcov=curve_fit(base,(np.ravel(xx),np.ravel(yy)),np.ravel(H.T),p0=guess)
  zzfit=base((np.ravel(xx),np.ravel(yy)),*popt)
  plt.contour(xx,yy,np.reshape(zzfit,(nbins,nbins)),5,colors='r',alpha=0.3)
'''

#cleaning
LC_char=pd.DataFrame()
for sn in glob.glob('ztf_lcs/*_meta.json'):
  with open(sn,'r') as f:
    meta=json.load(f)
  df_LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
  LC_char=LC_char.append({'max_gap':df_LC['mjd'].diff().max(),
                'phot_den':df_LC.shape[0]/(df_LC['mjd'].max()-df_LC['mjd'].min()),
                'n_rise':df_LC[df_LC['mjd']<df_LC['mjd'][df_LC['mag'].idxmin()]].shape[0], #conservative
                'z':float(meta['z']),
                'type':meta['Type'],
                'ra':meta['R.A.'],
                'dec':meta['Declination'],
                'name':sn.split('_meta')[0].split('/')[1]
                },ignore_index=True)
LC_char=LC_char.set_index('name')

scatter_matrix(LC_char)

#remove nrise=0
LC_char=LC_char[LC_char['n_rise']>0]
print(LC_char.describe())
print(LC_char.groupby(['type']).size())

for name, group in LC_char.groupby('type'):
  ax=group['z'].hist(alpha=0.2,log=True,bins=np.arange(0.,1.,0.05),label=name)
ax.legend()

for index,sn in LC_char.iterrows():
  df_LC=pd.read_json('ztf_lcs/'+index+'.json',orient='index')
  flux,flux_err=convert_to_flux(df_LC)
  LC_fmt={}
  LC_fmt['meta']={'z':sn['z']}
  LC_fmt['data']={'mjd':df_LC['mjd'].tolist(),
                'band':df_LC['passband'].map(band_name).tolist(),
                'flux':flux.tolist(),
                'flux_err':flux_err.tolist(),
                'zp':df_LC['passband'].map(zp_dict).tolist(),
                'zpsys':['ab']*len(flux)}
  with open('sncosmo/'+index+'.json','w') as f:
      json.dump(LC_fmt,f,indent=4)

# cleaning other data
LC_char=pd.DataFrame()
for sn in glob.glob('group/*_meta.json'):
  with open(sn,'r') as f:
    meta=json.load(f)
  df_LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
  LC_char=LC_char.append({'max_gap':df_LC['mjd'].diff().max(),
                'phot_den':df_LC.shape[0]/(df_LC['mjd'].max()-df_LC['mjd'].min()),
                'n_rise':df_LC[df_LC['mjd']<df_LC['mjd'][df_LC['mag'].idxmin()]].shape[0], #conservative
                'z':float(meta['z']),
                'type':meta['Type'],
                'ra':meta['R.A.'],
                'dec':meta['Declination'],
                'name':sn.split('_meta')[0].split('/')[1]
                },ignore_index=True)
LC_char=LC_char.set_index('name')

#remove nrise=0
LC_char=LC_char[LC_char['n_rise']>0]
scatter_matrix(LC_char)
print(LC_char.describe())
print(LC_char.groupby(['type']).size())

for name, group in LC_char.groupby('type'):
  ax=group['z'].hist(alpha=0.2,log=True,bins=np.arange(0.,1.,0.05),label=name)
ax.legend()

#Note have dropped Ic Pec, Ib/c and changed all II P to II

from shutil import copyfile
for SN in LC_char.index:
  copyfile('group/'+SN+'.json', 'test/'+SN+'.json')
  copyfile('group/'+SN+'_meta.json', 'test/'+SN+'_meta.json')

#noise model
import sncosmo
import sfdmap
dustmap = sfdmap.SFDMap('/Users/test/Work/REFITT/sncosmo/sfddata-master') #do this only once
from astropy.coordinates import SkyCoord
from astropy import units as u

chi_sq_thresh=300
'''
#explore thresh
fits=pd.DataFrame()
for sn in sn_list:
  try:
    result,fitted_model=fit_model(sn)
    fits=pd.concat([fits,pd.DataFrame([[sn,result.chisq]])])
  except (RuntimeError, ValueError) as e:
    continue

for i, row in fits[(fits[1]>100) & (fits[1]<500)].sort_values(by=1).iterrows():
  result,fitted_model=fit_model(row[0])
  LC=sncosmo.read_lc('sncosmo/'+row[0]+'.json',format='json')
  sncosmo.plot_lc(LC,model=fitted_model,errors=result.errors,
                figtext=[sn+'\nchi_sq='+str(result.chisq)]);
'''

def initialize_model():
  dust=sncosmo.CCM89Dust() #r_v=A(V)/E(B−V); A(V) is total extinction in V
  model=sncosmo.Model(source='salt2',
                      effects=[dust],
                      effect_names=['mw'],
                      effect_frames=['obs'])
  return model

def fit_model(sn_name):
  data=sncosmo.read_lc('sncosmo/'+sn_name+'.json', format='json')
  with open('ztf_lcs/'+sn_name+'_meta.json','r') as f:
    sn_meta=json.load(f)
  c=SkyCoord(sn_meta['R.A.'],sn_meta['Declination'],
            unit=(u.hourangle, u.deg))
  model=initialize_model()
  ebv = dustmap.ebv(c) #ICRS frame
  model.set(z=data.meta['z'],mwebv=ebv)  # set the model's redshift.
  result, fitted_model=sncosmo.fit_lc(data,model,
                                    ['t0', 'x0', 'x1', 'c']) # parameters of model to vary
  return result, fitted_model

sn_list=[]
for sn in glob.glob('sncosmo/*.json'):
  sn_name=os.path.splitext(os.path.basename(sn))[0]
  with open('ztf_lcs/'+sn_name+'_meta.json','r') as f:
    meta=json.load(f)
  if meta['Type']=='Ia':
    sn_list.append(sn_name)

err_df=pd.DataFrame()
for sn in sn_list:
  try:
    result,fitted_model=fit_model(sn)
    if result.chisq<chi_sq_thresh:
      df_LC=pd.read_json('ztf_lcs/'+sn+'.json',orient='index')
      df_LC=df_LC.sort_values(by=['mjd'],kind='mergesort')
      df_LC['flux_sncosmo']=fitted_model.bandflux(df_LC['passband'].map(band_name),
                                        df_LC['mjd'],
                                        zp=df_LC['passband'].map(zp_dict),
                                        zpsys='ab')
      flux=10.**((df_LC['mag']-df_LC['passband'].map(zp_dict))/-2.5)
      err_df=pd.concat([err_df,
                      pd.DataFrame([df_LC['mag'],
                                    (flux-df_LC['flux_sncosmo'])
                                      /df_LC['flux_sncosmo']]).T])
  except (RuntimeError, ValueError) as e:
    continue

err_df=err_df.rename(columns={'Unnamed 0':'noise'})
errs=err_df[~err_df['noise'].isin([np.nan, np.inf, -np.inf])]['noise'].values

bandwidths=np.logspace(-5,-2,10)
train_scores,valid_scores=validation_curve(KernelDensity(kernel='gaussian'),
                                          errs[:,None],None,
                                          'bandwidth',#'kernel'],
                                          bandwidths,#kernels],
                                          cv=k)#, scoring=mse)

plt.figure()
plt.semilogx(bandwidths, np.mean(train_scores,axis=1)) #TF is the score here?
plt.ylabel(str(k)+'-fold CV score')
plt.show()

bw=2e-4
bins=np.arange(-0.5,0.5,0.001)
fig, ax = plt.subplots()
plt.hist(errs,bins=bins,density=True)
kde=KernelDensity(kernel='gaussian',bandwidth=bw).fit(errs[:,None])
x_plot=bins[:,np.newaxis]
dens=np.exp(kde.score_samples(x_plot))
plt.plot(x_plot[:,0],dens,lw=0.5)
plt.savefig('noise_gaussian_'+str(bw)+'.png',dpi=300)

dump(kde,'noise_kde.joblib')
