import json
import pandas as pd
import numpy as np
import math
import os,sys,glob,time

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import refitt

fig=plt.figure()
'''
c=SkyCoord(inj_stats['ra'],inj_stats['dec'],unit='deg')
plt.subplot(111, projection='aitoff')
plt.grid(True)
plt.scatter(c.ra.wrap_at(180 * u.deg).radian, c.dec.radian)

c=SkyCoord(sim_stats['ra'],sim_stats['dec'],unit='deg')
plt.subplot(111, projection='aitoff')
plt.grid(True)
plt.scatter(c.ra.wrap_at(180 * u.deg).radian, c.dec.radian)
'''

for c in refitt.lib_classes.keys():
  print(c)
  sim_stats=pd.read_pickle(refitt.refitt_loc+'/'+c+'/train/sim_stats.pkl')
  props=pd.DataFrame()
  for sim in glob.glob(refitt.refitt_loc+'/'+c+'/train/*.json'):
    df_sim=pd.read_json(sim,orient='index').sort_values(by=['mjd'])
    props=pd.concat([props,pd.DataFrame([[df_sim['mag'].min(),
                                         sim_stats.loc[int(os.path.basename(sim).split('.')[0])]['z_new']
                                         ]],columns=['peak','z'])])
  H,xedges,yedges=np.histogram2d(props['z'],props['peak']) 
  plt.imshow(H.T,interpolation='nearest',origin='lower',
             extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],aspect='auto')
  plt.savefig(c+'.png')

