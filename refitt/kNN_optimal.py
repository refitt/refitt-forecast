import numpy as np
import pandas as pd
import json
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import refitt

fig = plt.figure(1, figsize=(6., 4.)) #wh
fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.98, wspace=0., hspace=0.)
params = {'xtick.direction': 'in', 'ytick.direction': 'in'}
mpl.rcParams.update(params)
rows,cols=11,5 
gs = gridspec.GridSpec(rows,cols)
sax = []
for i in range(rows):
   for j in range(cols):
      sax.append(plt.subplot(gs[cols*i+j]))

k_opt_dict={}
for i,phase in enumerate(range(6,refitt.window+1)):
  obj=np.load(refitt.DATA_PATH+'obj_'+str(phase)+'.npy')
  sax[i].plot(obj[:,0],obj[:,1]/obj[:,2],label=phase)
  k_opt=obj[np.argmin(obj[:,1]/obj[:,2]),0]
  sax[i].annotate(str(phase)+';'+str(k_opt),xy=(0.5,0.5),xycoords='axes fraction')
  sax[i].axvline(x=k_opt, c='k', ls='-.',lw=1.0)
  sax[i].set_yticklabels([])
  sax[i].set_xlim(10.,50.)
  k_opt_dict[int(phase)]=int(k_opt)
fig.savefig(refitt.DATA_PATH+'obj.png',dpi=300)

with open(refitt.DATA_PATH+'kNN_optimal.json','w') as f:
  json.dump(k_opt_dict,f,indent=4)

