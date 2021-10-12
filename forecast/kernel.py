#core
import numpy as np
import pdb
import math
import os,sys,glob
import pickle
import json
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
from typing import Tuple
#plotting
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#support
#from tslearn import metrics
import george
from scipy.stats import wasserstein_distance
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, PairwiseKernel
from scipy import stats, interpolate, spatial, signal
from scipy.optimize import minimize
import itertools
from astropy import units as u
from astropy.stats import biweight_location
from astropy.time import Time
from datetime import datetime

from forecast import defs

def select_library(survey,phase) -> str:
  return (survey+'_rep_phase_'+str(phase))

def get_band_info(survey) -> list:
  """
  returns band names for each survey
  """
  if survey=='ZTF_public':
    band_list=[1,2]
  #this option is not fully supported
  elif survey=='ZTF_full':
    band_list=[1,2,3]
  return band_list

class Transient():
  def __init__(self,ID:str,LC:pd.DataFrame,out_dir,
               current_mjd:float=Time(datetime.today()).mjd,survey:str='ZTF_public'):
    """
    REFITT LC class and supporting methods
    
    Arguments
    ---------
    ID: name of transient
    LC: dataframe with LC photometry
        columns ['mjd', 'passband', 'mag', 'mag_err']
        index integer
    out_dir: directory to write forecast json and png to. 
   
    Keyword Arguments:
    ------------------
    current_mjd: current clock time in MJD
                 specify if different from system
    survey: 'ZTF_public'
            currently only type that will work

    Examples:
    ---------
    To get predictions:
        will create a json file with a summary of predictions 
        and a figure showing forecasts in the folder with the LC json
    -----------------------
    >>> fname='/path/to/file/ZTF21abcdefg.json'
    >>> refitt.Transient.from_file(fname).predict_LC()
    
    """
    self.status=0
    ## status different from zero records a known issue where a forecast will fail
    self.ID=ID
    self.out_dir=out_dir
    self.now=current_mjd
    self.trigger=LC['mjd'].min()
    self.phase=int(self.now-self.trigger)
    if self.now < LC['mjd'].max():
      logging.warning("Current mjd {} is lesser than timestamp of most recent photometry.".format(self.now))
    elif self.phase < 0:
      print("Current mjd {} is lesser than timstamp of oldest photometry. I cannot proceed.".format(self.now))
      self.status=1
    elif self.phase<defs.horizon:
      print("Current mjd {} is lesser than horizon. I am not trained for this. I cannot proceed".format(self.now))
      self.status=2
    elif self.phase>defs.window:
      print("Current mjd {} is greater than window. I am not trained for this. I cannot proceed".format(self.now))
      self.status=2
    self.LC=LC[LC['mjd']<=self.now]
    self.future_LC=LC[LC['mjd']>self.now] #is empty if live
    self.z=None
    self.ra=None
    self.dec=None
    self.spec_class=None
    self.survey=survey
    if self.status<1:
      try: self.LC_GP=self.fit_GP()
      except ValueError: 
        print('Gaussian Process fit failed. I cannot proceed.')
        self.status=3 #GP fail when only one data point

  @classmethod
  def from_file(cls,fname:str,**options) -> 'Transient':
      """
      helper function to build Transient object instance from file
      will assume system time as current MJD

      Arguments
      ---------
      fname: path to file 
      **options: passed to the `pandas.read_json`

      Example:
      --------
      To make forecast for system time for fname and place outputs in directory where file is
      >>> refitt.Transient.from_file(fname).predict_LC()

      """
      ID=os.path.basename(os.path.splitext(fname)[0]) #ZTFID
      out_dir=os.path.dirname(fname)
      LC=pd.read_json(fname,orient='index',**options).sort_values(by=['mjd'])
      #getattr(pd, f'read_{connector}')(source, **options)
      return cls(ID,LC,out_dir)
  
  def fit_GP(self,k_corr_scale:float=1.,method='Boone') -> pd.DataFrame:
    """
    fits self.LC with Gaussian Process up to self.now
    """
    time=self.LC['mjd'].values
    mag=self.LC['mag'].values
    mag_err=self.LC['mag_err'].values
    passband=self.LC['passband'].values
    band_pred=get_band_info(self.survey)
    time_arr=np.arange(self.trigger,self.now+defs.resol,defs.resol)

    LC_GP=pd.DataFrame(data=np.vstack((np.repeat(time_arr,len(band_pred)),
                                       np.tile(band_pred,len(time_arr))
                                      )).T,columns=['mjd','passband'])
    globals()['GP_'+method](time,mag,mag_err,passband,LC_GP,k_corr_scale)
    return LC_GP

  def create_AE_rep(self) -> 'Transient':
    """
    creates attribute self.AE_rep that stores output vector for the AE
    """
    def gaussian(x,mu,sig):
      return np.exp(-np.power(x-mu,2.)/(2*np.power(sig,2.)))
    from keras.applications.xception import Xception
    from keras.applications.xception import preprocess_input
    model=Xception(weights='imagenet',include_top=False,pooling='avg')
    Xception_h,Xception_w=299,299
    mag_bins=np.linspace(25,defs.bright,Xception_h) #13.6 at 3 sigma
    band_list=get_band_info(self.survey)
    if len(band_list)<3:
      num_bands=3
    else:
      num_bands=len(band_list)
    #making 200x200xnum_bands vector of light curve
    #note that since ZTF public only has two bands,
    #one band will remain empty below
    AE_input=np.zeros((Xception_h,Xception_w,num_bands))
    for i,b in enumerate(band_list):
      ts=self.LC_GP[self.LC_GP['passband']==b]
      bin_means,bin_edges,binnumber=stats.binned_statistic(ts['mjd'],ts['mag'],
                                                           statistic='mean',
                                                           bins=Xception_w,
                                                           range=[self.trigger-defs.pad,
                                                                  self.trigger+self.phase+defs.pad])
      bin_errs,bin_edges,binnumber=stats.binned_statistic(ts['mjd'],ts['mag_err'],
                                                          #sigma of addition of two sigmas
                                                          statistic=lambda x:np.sqrt(np.sum(np.power(x,2)))/len(x),
                                                          bins=Xception_w,
                                                          range=[self.trigger-defs.pad,
                                                                 self.trigger+self.phase+defs.pad])
      bin_means=np.nan_to_num(bin_means)
      bin_errs=np.nan_to_num(bin_errs)
      #for each timestamp bin resolved above, evaluate the gaussian
      for j in range(Xception_w):
        AE_input[:,j,i]=gaussian(mag_bins,bin_means[j],bin_errs[j])
    features=[]
    #making 6C(num_bands) combinations of vector
    combs=itertools.combinations(range(num_bands),3)
    for comb in combs:
      x=preprocess_input(np.expand_dims(AE_input[:,:,comb],axis=0))
      features.append(model.predict(x).flatten())
    self.AE_rep=np.concatenate(features)
    return self 
  
  def find_NNs(self,kNN:int,start:int=0) -> 'Transient':
    '''
    create attribute self.ref_list with a (kNN,1) DataFrame with file locations for most similar library LCs
    needs object instance to have an AE_rep attribute (use method create_AE_rep)

    Arguments:
    ----------
    kNN: number of neighbours
    
    Keyword Arguments:
    ------------------
    start: starting index for the closest neighbour
           helps avoid self during training with LOOCV
           Default=0

    '''
    with open(defs.DATA_PATH+'balltree_AE_'+str(self.phase)+'.pkl','rb') as f:
      tree=pickle.load(f)
    fname=np.load(defs.DATA_PATH+'balltree_AE_'+str(self.phase)+'_fnames.npy')
    dist,ind=tree.query(self.AE_rep[np.newaxis,:],kNN+start)
    #if ((class_info) and (class_info in lib_classes.keys())):
    #   NNs=update_NNs(NNs,kNN,class_info,classifier)
    #find modal class iterate over their NNs only
    ref_list=pd.DataFrame()
    for pos in ind[0][start:]:
      NN_file=fname[pos]
      ref_list=pd.concat([ref_list,pd.DataFrame([NN_file])],axis=1)
    self.ref_list=ref_list.T.reset_index(drop=True)
    return self

  @staticmethod
  def get_class(ref_list:pd.DataFrame,kNN:int=-1,reset_index:bool=True,obj_class=None) -> Tuple[pd.DataFrame, list]:
    '''
    recieves a (1,N) DataFrame of filenames and returns subset with modal class(es) and modal class(es)
    passing an obj_class will return subset with that class; helps to use knowledge from spec class

    Arguments:
    ----------
    ref_list: DataFrame of similar library LCs
              !!generated by find_NNs!!

    Keyword Arguments:
    ------------------
    kNN: specify if want to look at only subset of provided ref_list
    reset_index: False if want to keep rank of neighbour
      both of the above helps reuse calculations to speedup training
    TODO: perhaps only provide one?
    '''
    ref_list=ref_list.iloc[:kNN]
    if (obj_class in defs.lib_classes.keys()): c_max=[defs.DATA_PATH+obj_class]
    else: c_max=ref_list[0].apply(lambda x: os.path.dirname(str(x))).mode().values
    c_rel=pd.concat([ref_list[ref_list[0].str.contains(c_str)] for c_str in c_max])
    if reset_index: c_rel=c_rel.reset_index(drop=True)
    cguess=[os.path.basename(c) for c in c_max]
    cguess=[cguess,c_rel.shape[0]/(ref_list.shape[0]*len(cguess))]
    return c_rel,cguess
  
  def predict_kNN(self,time_predict:np.ndarray,c_rel:pd.DataFrame) -> 'Transient':
    """
    given DataFrame of library LC filenames creates self.mag_predict and self.time_predict for magnitude 
    estimated from each
    relies on align_CC to align self and Transient(libLC)

    Arguments:
    ----------
    time_predict: mjds at which magnitudes needs to be estimated
    c_rel: DataFrame of filenames of library LCs
           !!obtained by staticmethod get_class if forcing class!!
    """
    band_list=get_band_info(self.survey)
    mag_predict=np.zeros((c_rel.shape[0],len(band_list),len(time_predict)))
    for NN,fl in c_rel.iterrows():
      with open(fl[0]) as f:
        df_LC_NN=pd.read_json(f,orient='index').sort_values(by=['mjd'])
      NN_obj=Transient(fl[0],df_LC_NN,os.path.dirname(fl[0]),current_mjd=df_LC_NN['mjd'].min()+defs.window)
      shift_x,shift_y=align_CC(self,NN_obj)
      for i,b in enumerate(band_list):
        a_band_ref=NN_obj.LC_GP['passband'].map(lambda x: x==b)
        mag_predict[NN,i,:]=(NN_obj.LC_GP[a_band_ref]['mag']-shift_y)#/d
    self.mag_predict=mag_predict
    self.time_predict=time_predict
    return self

  def summarize_kNN(self,c_rel_ind:np.ndarray) -> 'Transient':
    """
    creates attributes self.mag_predict_mean and self.mag_predict_sigma with 
    mean and std of magnitude from all neighbours, respectively

    should follow use of predict_kNN because needs self.time_predict and self.mag_predict

    Keyword Arguments:
    ------------------
    c_rel_ind: boolean mask when NN with modal class
               Normally is all True because predict_kNN used only modal classes
    """
    band_list=get_band_info(self.survey)
    mag_predict_mean=np.zeros((len(band_list),len(self.time_predict)))
    mag_predict_sigma=np.zeros((len(band_list),len(self.time_predict)))
    for i,b in enumerate(band_list):
      for j in range(len(self.time_predict)):
        rel_ind=np.logical_and(np.logical_not(np.isnan(self.mag_predict[:,i,j])),c_rel_ind)
        # if only one neighbour with non NaN predictions or not enough neighbours, mostly bad neighbours found
        if (np.sum(rel_ind)<3): self.status=9
        mag_pred_clean=self.mag_predict[rel_ind,i,j]
        mag_predict_mean[i,j],mag_predict_sigma[i,j]=self.predict_stat_LC(mag_pred_clean)
    self.mag_predict_mean=mag_predict_mean
    self.mag_predict_sigma=mag_predict_sigma
    return self

  @staticmethod
  def predict_stat_LC(mag_predict:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """
    computes predicted signal stats at each epoch for all neighbours
    used by summarize_kNN
    """
    mag_mean_kNN=np.mean(mag_predict)
    mag_sigma_kNN=np.sqrt(np.sum((mag_mean_kNN-mag_predict)**2.)/
                           (len(mag_predict)-1))
    ##squared weighting, per NN needs to be read from NN_dict from find_NNs
    ## and passed here
    #weight[k]=NNs[str(NN)]['distance']**2.
    ##needs to be computed by predict_LC, cleaned of nans and passed here
    #flux_err_predict[NN-1,i,:]=(flux_pred_std-shift_y)
    #this is weighted mean and propogated uncertainities (from weighted mean)
    #flux_mean_k=np.sum(np.multiply(flux_predict, weight/np.sum(weight)))
    #flux_sigma_k=np.sqrt(np.sum(np.multiply(np.square(flux_err_predict),
         #    np.square(weight/np.sum(weight)))))
    return mag_mean_kNN,mag_sigma_kNN

  def predict_LC(self) -> 'Transient':
    """
    given a Transient object makes all function calls to make predictions and
    associated output files
    
    Example:
    >>> fname='/path/to/file/ZTF21abcdefg.json' #or 'ZTF21abcdefg.json
    >>> LC=pd.read_json(fname,orient='index').sort_values(by=['mjd'])
    >>> ID=os.path.basename(os.path.splitext(fname)[0])
    >>> out_dir=os.path.dirname(fname)
    >>> sn=kernel.Transient(ID,LC,out_dir) #,current_mjd= to use a different time
    >>> sn.predict_LC()

    """
    k=kNN_at_tst(self)
    time_predict=np.arange(self.trigger,self.trigger+defs.window+defs.resol,defs.resol)
    self.create_AE_rep().find_NNs(k)
    c_rel,cguess=self.get_class(self.ref_list,k)
    self.cguess=cguess
    self.predict_kNN(time_predict,c_rel).summarize_kNN(c_rel.shape[0]*[True]).report()
    #FIXME: if status !=0 say something...
    return self

  def report(self):
    """
    creates a json file with a summary of prediction and a figure showing forecasts
    """
    phase_dict={-1:'rising',1:'fading',0:'plateau'}
    preds={}
    preds['ztf_id']=self.ID
    preds['instrument']=self.survey
    preds['time_since_trigger']=self.phase
    preds['current_time']=self.now
    preds['num_obs']=self.LC.shape[0]
    preds['class']=self.cguess
    past=np.where(self.time_predict<=self.now)
    future=np.where(self.time_predict>self.now)
    band_list=get_band_info(self.survey)
    mag_mean,mag_sigma=self.mag_predict_mean,self.mag_predict_sigma
    for i,b in enumerate(band_list):
      preds['phase_'+defs.band_name_dict[b]]=phase_dict[np.sign(
                                                          round((mag_mean[i,future[0][0]]-
                                                          mag_mean[i,past[0][0]])/defs.resol,1))]
      preds['next_mag_mean_'+defs.band_name_dict[b]]=mag_mean[i,future[0][0]]
      preds['next_mag_sigma_'+defs.band_name_dict[b]]=mag_sigma[i,future[0][0]]
      tpeak=self.time_predict[np.argmin(mag_mean[i,:])]
      tpeaklow=self.time_predict[np.argmin(mag_mean[i,:]+mag_sigma[i,:])]
      tpeakhigh=self.time_predict[np.argmin(mag_mean[i,:]-mag_sigma[i,:])]
      preds['time_to_peak_'+defs.band_name_dict[b]]=[tpeak-self.now,
                                                    max(tpeakhigh-tpeak,tpeaklow-tpeak),
                                                    min(tpeakhigh-tpeak,tpeaklow-tpeak)]
    preds['time_arr']=self.time_predict.tolist()
    #computing mean daily model confidence
    mdc=0.
    for i,b in enumerate(band_list):
        mdc+=np.sum(mag_sigma[i,:])
        preds['mag_mean_'+defs.band_name_dict[b]]=mag_mean[i,:].tolist()
        preds['mag_sigma_'+defs.band_name_dict[b]]=mag_sigma[i,:].tolist()
    mdmc=mdc/(len(self.time_predict)*len(band_list))
    preds['mdmc']=mdmc
    #computing mean daily observed error
    obs_error=0.
    for j in range(self.LC.shape[0]):
       obs=self.LC.iloc[j,:]
       idx=int((obs['mjd']-self.trigger)/defs.resol)
       obs_error+=abs(mag_mean[band_list.index(obs['passband']),idx]
                      -obs['mag'])
    preds['moe']=obs_error/self.LC.shape[0]
    with open(self.out_dir+'/'+self.ID+'_prediction.json','w') as f:
      json.dump(preds,f,indent=4)
    self.prediction_fig(preds)
    return

  def prediction_fig(self,preds):
    """
    makes a plot of predictions
    """
    fig = plt.figure(1, figsize=(6., 3.))
    params={'xtick.labelsize':8,'ytick.labelsize':8,
            'xtick.direction':'in','ytick.direction':'in',
            'hatch.linewidth':0.5}
    mpl.rcParams.update(params)
    fig.subplots_adjust(left=0.09,right=0.95,bottom=0.13,top=0.8,
                        wspace=0.,hspace=0.)
    rows,cols=1,2
    gs = gridspec.GridSpec(rows,cols)
    sax = []
    for r in range(rows):
      for c in range(cols):
        sax.append(plt.subplot(gs[cols*r+c]))
    ylow=[]
    yhigh=[]

    band_list=get_band_info(self.survey)
    for i,b in enumerate(band_list):
      mag_mean=np.array(preds['mag_mean_'+defs.band_name_dict[b]])
      mag_sigma=np.array(preds['mag_sigma_'+defs.band_name_dict[b]])
      a_band_ref=self.LC['passband'].map(lambda x: x==b)
      sax[i].errorbar(self.LC[a_band_ref]['mjd'],
                      self.LC[a_band_ref]['mag'],
                      yerr=self.LC[a_band_ref]['mag_err'],
                      fmt='o',c=defs.band_colors[b],ecolor=defs.band_colors[b])
      try:
        a_band_ref=self.future_LC['passband'].map(lambda x: x==b)
        sax[i].errorbar(self.future_LC[a_band_ref]['mjd'],
           self.future_LC[a_band_ref]['mag'],
           yerr=self.future_LC[a_band_ref]['mag_err'],
           fmt='o',c=defs.band_colors[b],ecolor=defs.band_colors[b])
      except:
        pass
      ###PREDICTION
      past=np.where(self.time_predict<=self.trigger+self.phase,True,False)
      future=np.where(self.time_predict>=self.trigger+self.phase,True,False)
      sax[i].plot(self.time_predict[past],mag_mean[past],c=defs.band_colors[b])
      sax[i].fill_between(self.time_predict[past],
                          mag_mean[past]-mag_sigma[past],
                          mag_mean[past]+mag_sigma[past],
                          facecolor=defs.band_colors[b],alpha=0.5)
      sax[i].plot(self.time_predict[future],mag_mean[future],c=defs.band_colors[b],
                  alpha=0.5)
      sax[i].fill_between(self.time_predict[future],
                          mag_mean[future]-mag_sigma[future],
                          mag_mean[future]+mag_sigma[future],
                          facecolor=defs.band_colors[b],alpha=0.25)
      sax[i].axvline(x=self.trigger+self.phase,c='k',lw=0.5)
      sax[i].set_xticks(np.arange(self.trigger,self.trigger+defs.window,10))
      sax[i].set_xticklabels(['0','10','20','30','40','50'])
      sax[i].set_xlabel('days since trigger')
      if i%cols==0:
        sax[i].set_ylabel('mag')
      else:
        sax[i].set_yticklabels([])
      sax[i].set_xlim(self.trigger-defs.pad,self.trigger+defs.window)
      ymin,ymax=sax[i].get_ylim()
      ylow.append(ymin)
      yhigh.append(ymax)
    for i,b in enumerate(band_list):
      sax[i].set_ylim(max(yhigh),min(ylow))
      sax[i].annotate('observed',xy=(0.05,0.05), xycoords='axes fraction',size=7)
      sax[i].annotate('predicted',xy=(0.8,0.05), xycoords='axes fraction',size=7)
      sax[i].annotate(defs.band_name_dict[b],xy=(0.9,0.9),
                      xycoords='axes fraction',color=defs.band_colors[b])
      sax[i].grid(lw=0.1,color='grey')
      twin_sax0=sax[i].twiny()
      twin_sax0.set_xlim(self.trigger-defs.pad,self.trigger+defs.window)
      dates=np.arange(math.ceil(self.trigger/10)*10,math.floor(self.trigger+defs.window),10)
      twin_sax0.set_xticks(dates)
      #twin_sax0.set_xticklabels(Time(dates,format='mjd',out_subfmt='date').iso,rotation=45,ha='left')
      twin_sax0.set_xticklabels(Time(
                                     Time(dates,format='mjd').iso,
                                     format='iso',out_subfmt='date'
                                    ).iso,rotation=45,ha='left')
      twin_sax0.get_xaxis().tick_top()
    fig.savefig(self.out_dir+'/'+self.ID+'_prediction.png')
    plt.close()
    return

def GP_Boone(time,mag,mag_err,passband,LC_GP,k_corr_scale):
    def neg_ln_like(p):
      gp.set_parameter_vector(p)
      return -gp.log_likelihood(mag)

    def grad_neg_ln_like(p):
      gp.set_parameter_vector(p)
      return -gp.grad_log_likelihood(mag)

    length_scale=20 #guess 
    central_wvl={0:357.0,1:476.7,2:621.5,3:754.5,4:870.75,5:1004.0} #for LSST
    dim=[central_wvl[int(pb)] for pb in passband]
    signal_to_noise_arr=(np.abs(mag)/
                        np.sqrt(mag_err**2+(1e-2*np.max(mag))**2))
    scale=np.abs(mag[signal_to_noise_arr.argmax()])
    # setup GP model
    kernel=((0.5*scale)**2*george.kernels.Matern32Kernel( #changed by J.R.
                                   [length_scale**2,6000**2], ndim=2))
    kernel.freeze_parameter('k2:metric:log_M_1_1') #Fixed length scale in wavelength
    kernel.freeze_parameter('k1:log_constant') #Fixed overvall scale
    x_data=np.vstack([time,dim]).T
    gp=george.GP(kernel,mean=biweight_location(mag)) #changed by J.R.
    # train
    gp.compute(x_data,mag_err)
    result=minimize(neg_ln_like,
                    gp.get_parameter_vector(),
                    jac=grad_neg_ln_like)
    gp.set_parameter_vector(result.x)
    #predict
    ts_pred=np.zeros((LC_GP.shape[0],2))
    ts_pred[:,0]=LC_GP['mjd']
    ts_pred[:,1]=LC_GP['passband'].map(central_wvl)*k_corr_scale
    LC_GP['mag'],mag_var=gp.predict(mag,ts_pred,return_var=True)
    LC_GP['mag_err']=np.sqrt(mag_var)
    return

def GP_free(time,mag,mag_err,passband,LC_GP,k_corr_scale):
    def neg_ln_like(p):
      gp.set_parameter_vector(p)
      return -gp.log_likelihood(mag)

    def grad_neg_ln_like(p):
      gp.set_parameter_vector(p)
      return -gp.grad_log_likelihood(mag)

    central_wvl={1:471.70,2:653.74,3:687.20} #for ZTF
    dim=[central_wvl[int(pb)] for pb in passband]
    signal_to_noise_arr=(np.abs(mag)/
                        np.sqrt(mag_err**2+(1e-2*np.max(mag))**2))
    scale=np.abs(mag[signal_to_noise_arr.argmax()])
    # setup GP model
    kernel=((0.5*scale)**2*george.kernels.Matern32Kernel([20**2,6000**2], ndim=2))
    kernel.freeze_parameter('k1:log_constant') #Fixed overall scale
    x_data=np.vstack([time,dim]).T
    gp=george.GP(kernel,mean=21.5)
    # train
    gp.compute(x_data,mag_err)
    result=minimize(neg_ln_like,
                    gp.get_parameter_vector(),
                    jac=grad_neg_ln_like)
    gp.set_parameter_vector(result.x)
    #predict
    ts_pred=np.zeros((LC_GP.shape[0],2))
    ts_pred[:,0]=LC_GP['mjd']
    ts_pred[:,1]=LC_GP['passband'].map(central_wvl)*k_corr_scale
    LC_GP['mag'],mag_var=gp.predict(mag,ts_pred,return_var=True)
    LC_GP['mag_err']=np.sqrt(mag_var)
    return

def GP_Villar_Cranmer(time,mag,mag_err,passband,LC_GP):
    ####
    # Not production ready. Warnings:
    # A notion of kcorrection not implemented
    # distance_matrix is queried by passband number so u=1, g=2 and so on.
    ####
    import speclite.filters
    #filters = ['ztf-g','ztf-r','ztf-i']
    #data = {f: speclite.filters.load_filter(f"{defs.DATA_PATH}{f}.ecsv") for f in filters}
    filters = 'ugrizy'
    data = {f: speclite.filters.load_filters(f'lsst2016-{f}')[0] for f in filters}
    wavelength = np.linspace(3000, 10000, 10000) * u.AA
    normalized = {f: data[f](wavelength) / data[f](wavelength).sum() for f in filters}
    
    def get_distance(filter1, filter2):
        dist = wasserstein_distance(u_values = wavelength.value, 
                                    v_values = wavelength.value, 
                                    u_weights = normalized[filter1], 
                                    v_weights = normalized[filter2])
        return dist
     
    distance_matrix = np.array([[get_distance(col, row) for col in filters] for row in filters])
    distance_matrix /= np.average(distance_matrix)

    mag_var = np.var(mag)
    def metric(x1, x2, p):
        band1 = (x1[1].astype(int))
        band2 = (x2[1].astype(int))
        time_distance = x2[0] - x1[0]
        photometric_distance = distance_matrix[band1, band2]
        return (
            mag_var*np.exp(-photometric_distance**2/(2*p[0]**2) - time_distance**2/(2*p[1]**2))
            )

    def fit_gp(X, y, p):
        cur_metric = lambda x1, x2, gamma: metric(x1, x2, p)
        kernel = PairwiseKernel(metric=cur_metric)
        gp = GaussianProcessRegressor(kernel=kernel,
                                    alpha=y[:,1]**2,
                                    normalize_y=True)
        gp.fit(X,y[:, 0])
        return gp

    cX = np.stack((time,passband),axis=1).astype(np.float32)
    cy = np.stack((mag,mag_err),axis=1).astype(np.float32)

    def try_lt_lp(p):
        summed_log_like = 0.0
        gp=fit_gp(cX,cy,p)
        summed_log_like += gp.log_marginal_likelihood()
        return -summed_log_like
 
    res = minimize(lambda x: try_lt_lp(np.exp(x)),
         [2.3, 0.0], bounds = [(-3, 4.5), (-6, 4)])
    best_lengths = np.exp(res.x)
    gp=fit_gp(cX,cy,best_lengths)
    ts_pred=np.zeros((LC_GP.shape[0],2))
    ts_pred[:,0]=LC_GP['mjd']
    ts_pred[:,1]=LC_GP['passband']
    LC_GP['mag'],mag_var=gp.predict(ts_pred,return_std=True)#gp.predict(mag,ts_pred,return_var=True)
    LC_GP['mag_err']=np.sqrt(mag_var)
    return
 
#deprecated for ZTF: uncomment tslearn.metrics to use
def align_DTW(obs_LC,df_LC_ref,instrument):
  y_offset=[]
  x_offset=[]
  bad_NN_flag=False
  band_list=get_band_info(instrument)
  for i,b in enumerate(band_list):
    a_band_ref=obs_LC['passband'].map(lambda x: x==b)
    subseq_ts=np.vstack((obs_LC[a_band_ref]['mjd'].values,
                         obs_LC[a_band_ref]['mag'].values)).T

    a_band_match=df_LC_ref['passband'].map(lambda x: x==b)
    seq_ts=np.vstack((df_LC_ref[a_band_match]['mjd'].values,
                      df_LC_ref[a_band_match]['mag'].values)).T

    #if only one data point no point doing DTW
    if (np.shape(seq_ts)[0]<2 or np.shape(subseq_ts)[0]<1 or 
      np.shape(subseq_ts)[0]>np.shape(seq_ts)[0]):
      continue

    path,dist=metrics.dtw_subsequence_path(subseq_ts, seq_ts)
    # for shortest path use optimal pairs to get suggested for shift along
    # time and mag axes
    for j in path:
      y_offset.append(seq_ts[j[1],1]-subseq_ts[j[0],1])
      x_offset.append(seq_ts[j[1],0]-subseq_ts[j[0],0])
  if len(y_offset)==0.: # DTW skipped for all bands
    bad_NN_flag=True
  # return median offset
  shift_y=np.median(y_offset)
  shift_x=np.median(x_offset)
  '''
  #using modal shifts 
  kde=stats.gaussian_kde(x_offset)
  x_range=np.arange(min(x_offset),max(x_offset),1e-3)
  dens=kde(x_range)
  shift_x=x_range[np.argmax(dens)]

  kde=stats.gaussian_kde(y_offset)
  y_range=np.arange(min(y_offset),max(y_offset),1e-4)
  dens=kde(y_range)
  shift_y=y_range[np.argmax(dens)]
  '''
  return shift_x,shift_y,bad_NN_flag

def align_CC(transient1,transient2):
  band_list=get_band_info(transient1.survey)
  delta_t=np.zeros(len(band_list))
  delta_mag=np.zeros(len(band_list))
  for i,b in enumerate(band_list):
    a_band_obs=transient1.LC_GP['passband'].map(lambda x: x==b)
    mag_mean_obs=transient1.LC_GP[a_band_obs]['mag']
    a_band_ref=transient2.LC_GP['passband'].map(lambda x: x==b)
    mag_mean_ref=transient2.LC_GP[a_band_ref]['mag']

    cc_val=[]
    cc_del=[]
    lag_values=np.arange(-1.*transient1.phase+defs.resol,defs.window,defs.resol)
    cc_arr=signal.correlate(mag_mean_obs,mag_mean_ref)
    cc_del=lag_values[np.argmax(cc_arr)] #in mjd
    delta_t[i]=transient2.trigger-transient1.trigger+cc_del
    delta_mag[i]=np.amin(mag_mean_ref[:math.ceil((transient1.phase+cc_del)/defs.resol)])-np.amin(mag_mean_obs) 
  shift_x=np.mean(delta_t)
  shift_y=np.mean(delta_mag)
  return shift_x,shift_y

def kNN_at_tst(transient):
  '''
  Finds optimal k for transient epoch
  files created by kNN_optimal.py
  '''
  fname={'LSST':'k_optimal.json',
         'ZTF_public':'kNN_optimal.json'}
  with open(defs.DATA_PATH+fname[transient.survey], 'r') as f:
    k_opt_dict=json.load(f)
  kNN=k_opt_dict[str(transient.phase)]
  return kNN

#NOT UPDATED
def get_classifier_perf(classifier,c):
   if classifier=='default':
      if c=='42':
         TPR=0.6
         FPR=0.1
      elif c=='62':
         TPR=0.5
         FPR=0.05
   return TPR, FPR

#NOT UPDATED
def update_prob(prior_c,classifier,c):
   TPR,FPR=get_classifier_perf(classifier,c)
   post_prob=TPR*prior_c/(TPR*prior_c+FPR*(1-prior_c))
   return post_prob

#NOT UPDATED
def update_NNs(NNs,k,c,classifier):
   num_class=0
   for NN in range(k):
      if NNs[NN]['class']==c:
         num_class+=1
   prob_c=num_class/float(k)
   prob_c=update_prob(prob_c,classifier,c)
   num_c=round(prob_c*k)
   num_class=0
   while num_class<num_c:
      if not (NNs[num_class]['class']==c):
         for i in range(num_class+1,len(NNs.keys())):
            if (NNs[i]['class']==c):
               NN_temp = NNs[i]
               for j in range(num_class,i):
                  NNs[j+1]=NNs[j]
               NNs[num_class] = NN_temp
               break
      num_class+=1
   return NNs

if __name__ == '__main__':
  fname=str(sys.argv[1])
  obj=Transient.from_file(fname)
  if obj.status==0:
    obj.predict_LC() # make prediction for system time

