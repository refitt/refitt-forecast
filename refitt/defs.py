import pkg_resources
import numpy as np

DATA_PATH = pkg_resources.resource_filename('refitt', 'data/')
#this is frame for image sent to CAE
depth=21 #LSST will be 25
bright=12
horizon=6 #start of forecasts
window=60 #window of action
min_photo=3
gap=21 #maximum lapse since last photometry
pad=2. #for plotting and CAE
resol=0.02 #GP, CC alignment and moe

ZTF_zp_dict={1:26.325,2:26.275,3:25.660} #from ZSDS_explanatory pg 67
band_name_dict={0:'u',1:'g',2:'r',3:'i',4:'z',5:'y'}
sncosmo_band_name_dict={1:'ztfg',2:'ztfr',3:'ztfi'}
antares_band_name_dict={'g':1,'R':2}
band_colors={1:'limegreen',2:'orangered',3:'goldenrod'}#,'limegreen','darkturquoise',
             #'mediumslateblue','orchid']
lib_classes={
    'Ia':{'logz_bins':np.arange(-2.7,-0.69,0.1),'method':'Ia'},
    'II':{'logz_bins':np.arange(-2.7,-0.89,0.1),'method':'GP'},
    'IIn':{'logz_bins':np.arange(-2.7,-0.89,0.1),'method':'GP'},
    'IIb':{'logz_bins':np.arange(-2.7,-0.89,0.1),'method':'GP'},
    'Ib':{'logz_bins':np.arange(-2.7,-0.89,0.1),'method':'GP'},
    'Ic':{'logz_bins':np.arange(-2.7,-0.89,0.1),'method':'GP'},
    'Ic-BL':{'logz_bins':np.arange(-2.7,-0.89,0.1),'method':'GP'},
    'SLSN-I':{'logz_bins':np.arange(-1.7,-0.29,0.1),'method':'GP'},
    'SLSN-II':{'logz_bins':np.arange(-1.7,-0.29,0.1),'method':'GP'}
     }

from astropy.cosmology import FlatLambdaCDM
cosmo=FlatLambdaCDM(H0=70,Om0=0.3,Tcmb0=2.725)

events_per_bin=500

