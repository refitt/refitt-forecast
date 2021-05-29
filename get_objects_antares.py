import sys,time
import json
import pandas as pd
from elasticsearch_dsl import Search
from antares_client.search import search
from antares_client.search import get_by_id
import marshmallow

import utils

min_photo=3
window=60 #window of action
gap=21
band_name_dict={'g':1,'R':2}

dest=str(sys.argv[1]) # is folder for destination
now=float(sys.argv[2])

query = (
    Search()
    .filter("range", **{"properties.num_mag_values": {"gte": min_photo}})
    .filter("range", **{"properties.newest_alert_observation_time": {"gte": now-gap}})#, "lte": now}})
    .filter("range", **{"properties.oldest_alert_observation_time": {"gte": now-window}})
    .filter("term", tags="refitt_newsources_snrcut")
    .to_dict()
)

df_stats=pd.DataFrame()
while True:
  try: 
    for locus in search(query):
      df_LC=locus.lightcurve
      df_ts=df_LC[df_LC['ant_survey']==1][['ant_mjd','ant_mag','ant_magerr','ant_passband']]
      df_ts['ant_passband']=df_ts['ant_passband'].replace(band_name_dict)
      ztf_id=locus.properties['ztf_object_id']
      df_ts['event_id']=ztf_id
      df_ts=df_ts.reset_index(drop=True).sort_values(by='ant_mjd').rename(columns={'ant_mjd':'mjd',
                                                                                   'ant_passband':'passband',
                                                                                   'ant_mag':'mag',
                                                                                   'ant_magerr':'mag_err'})
      age=now-df_ts['mjd'].min()
      obs_diff=[(group['mjd'].diff()>0.2).any() for pb,group in df_ts.groupby('passband')]
      if ((age<=window) and any(obs_diff)):
        meta={}
        meta['ra']=locus.ra
        meta['dec']=locus.dec
        try:
          meta['r_mag']=df_ts.iloc[
                        df_ts[df_ts['passband']==2]['mjd'].idxmax()
                        ]['mag']
          meta['r_mag_err']=df_ts.iloc[
                        df_ts[df_ts['passband']==2]['mjd'].idxmax()
                        ]['mag_err']
        except ValueError:
          meta['r_mag']=None
          meta['r_mag_err']=None
        try:
          meta['g_mag']=df_ts.iloc[
                        df_ts[df_ts['passband']==1]['mjd'].idxmax()
                        ]['mag']
          meta['g_mag_err']=df_ts.iloc[
                        df_ts[df_ts['passband']==1]['mjd'].idxmax()
                        ]['mag_err']
        except ValueError:
          meta['g_mag']=None
          meta['g_mag_err']=None
        with open(dest+'/'+ztf_id+'.json','w') as f:
          json.dump(df_ts.to_dict(orient='index'),f,indent=4)
        with open(dest+'/meta/'+ztf_id+'_meta.json','w') as f:
          json.dump(meta,f,indent=4)  
        df_stats=pd.concat([df_stats,pd.DataFrame([[meta['ra'],meta['dec'],
                                                 age,df_ts['mjd'].max(),df_ts.shape[0],
                                                 df_ts[df_ts['passband']==1].shape[0],
                                                 df_ts[df_ts['passband']==2].shape[0]]],
                                                 columns=['ra','dec','age','recent_photo',
                                                         'num_photo','num_photo_g','num_photo_r'])
                            ])
  except (json.decoder.JSONDecodeError) as e:
    utils.email('Timeout error with param '+str(e))
    utils.log('Timeout error with param '+str(e))
  except marshmallow.exceptions.ValidationError as e:
    utils.email('Light curve missing? with param '+str(e))
    utils.log('Light curve missing? with param '+str(e))
  except:
    utils.email('Unexpected error with param '+str(sys.exc_info()[0]))
    utils.log('Unexpected error with param '+str(sys.exc_info()[0]))
    raise
  break

import matplotlib.pyplot as plt
from pandas.plotting import table
from pandas.plotting import scatter_matrix
import astropy.units as u
from astropy.coordinates import SkyCoord

fig=plt.figure(figsize=(15,5))
ax=fig.add_subplot(221, projection='aitoff')
c=SkyCoord(df_stats['ra'],df_stats['dec'],unit='deg')
ax.grid(True)
ax.scatter(c.ra.wrap_at(180 * u.deg).radian, c.dec.radian)

cols=['age','recent_photo','num_photo','num_photo_g','num_photo_r']
ax=fig.add_subplot(222)
ax=table(ax,round(df_stats[cols].describe(),2),loc="center")
ax.auto_set_font_size(False)
ax.set_fontsize(10)
plt.axis("off")

for i,col in enumerate(cols):
  ax=fig.add_subplot(2,5,6+i)
  df_stats[col].hist(ax=ax)
  ax.set_xlabel(col)

plt.savefig(dest+'/daily_stats.png')
utils.log('Fetched {} objects from Antares. See summary at {}'.format(df_stats.shape[0],dest+'/daily_stats.png'))

