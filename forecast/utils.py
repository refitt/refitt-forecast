from email.mime.multipart import MIMEMultipart
from email.utils import COMMASPACE, formatdate
from email.mime.text import MIMEText
import smtplib
from datetime import datetime
import os,sys,glob
import numpy as np
import pickle
from sklearn.neighbors import BallTree

from forecast import defs, kernel

def email(text):
  server="127.0.0.1"
  msg = MIMEMultipart()
  send_from='refitt-planner@purdue.edu'
  send_to=['niharika.sravan@gmail.com']#,'bsubraya@purdue.edu']
  msg['From'] = send_from
  msg['To'] = COMMASPACE.join(send_to)
  msg['Date'] = formatdate(localtime=True)
  msg['Subject'] = 'Daily run exception'
  msg.attach(MIMEText(text))

  smtp = smtplib.SMTP(server)
  smtp.sendmail(send_from, send_to, msg.as_string())
  smtp.close()

def log(log):
  dir_path = os.path.dirname(os.path.realpath(__file__)) #FIXME: hardcoded
  log_str=datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+log+'\n'
  with open(dir_path+'/ZTF/daily_log','a') as f:
    f.write(log_str)

def make_balltrees(phase,survey='ZTF_public'):
  '''
  Helper function to create ball trees
  only meaningful to call after make_reps for phase is done
  will make balltrees and filename list at DATA_PATH
  '''
  X=np.array([])
  fname=np.array([])
  library_loc=kernel.select_library(survey,phase)
  for c in defs.lib_classes.keys():
    for event in glob.glob(defs.DATA_PATH+c+'/'+library_loc+'/*.npy'):
      NN_file=event.split(survey)
      LC_file=NN_file[0]+os.path.basename(NN_file[1]).split('_Xception')[0]+'.json'
      fname=np.append(fname,np.array(LC_file))# if fname.size else np.array(event)
      rep_comp=np.load(event)
      X=np.vstack([X,rep_comp[np.newaxis,:]]) if X.size else rep_comp[np.newaxis,:]
  np.save(defs.DATA_PATH+'balltree_AE_'+str(phase)+'_fnames',fname)

  tree=BallTree(X,leaf_size=2,metric='l1') #2 leaves are good
  with open(defs.DATA_PATH+'balltree_AE_'+str(phase)+'.pkl','wb') as f:
    pickle.dump(tree,f)

def get_flux(df_LC):
  """
  helper function to get fluxes from LC magnitudes using ZTF zero-points
  """
  flux=10.**((df_LC['mag']-df_LC['passband'].map(defs.ZTF_zp_dict))/-2.5)
  flux_err=abs(flux*df_LC['mag_err']*(np.log(10.)/2.5))
  return flux,flux_err

'''
def convert_to_mag(df_LC):
  df_LC['mag']=(df_LC['passband'].map(band_name_dict).map(ZTF_zp_dict)
                -2.5*np.log10(df_LC['flux']))
  df_LC['mag_err']=abs(2.5*df_LC['flux_err']/(df_LC['flux']*np.log(10.)))
  return df_LC.drop(columns=['flux','flux_err'])
'''

def fit_Ia_model(meta,LC,spec_z=None):
  import sncosmo
  import sfdmap
  from astropy.table import Table
  from astropy.coordinates import SkyCoord
  from astropy import units as u
  dustmap = sfdmap.SFDMap(defs.DATA_PATH+'/lib_gen/sfddata-master')
  
  flux,flux_err=get_flux(LC)
  data=Table()
  if spec_z: data.meta['z']=float(meta['z'])
  data['mjd']=LC['mjd'].tolist()
  data['band']=LC['passband'].map(defs.sncosmo_band_name_dict).tolist()
  data['flux']=flux.tolist()
  data['flux_err']=flux_err.tolist()
  data['zp']=LC['passband'].map(defs.ZTF_zp_dict).tolist()
  data['zpsys']=['ab']*len(flux)
  c=SkyCoord(meta['R.A.'],meta['Declination'],
            unit=(u.hourangle, u.deg))
  dust=sncosmo.CCM89Dust() #r_v=A(V)/E(B−V); A(V) is total extinction in V
  model=sncosmo.Model(source='salt2',
                      effects=[dust],
                      effect_names=['mw'],
                      effect_frames=['obs'])
  ebv=dustmap.ebv(c) #ICRS frame
  if not spec_z:
    model.set(mwebv=ebv)
    result,fitted_model=sncosmo.fit_lc(data,model,
                                      ['z','t0','x0','x1','c'],
                                      bounds={'z':(0.,0.2)})
  else:
    model.set(z=data.meta['z'],mwebv=ebv)  # set the model's redshift.
    result,fitted_model=sncosmo.fit_lc(data,model,
                                      ['t0','x0','x1','c'])

  return result,fitted_model

