# -*- coding: utf-8 -*-
'''
Compare CBH from lidar and ceilometer
'''

import os
cd=os.getcwd()
from matplotlib import pyplot as plt
import xarray as xr
import glob
import numpy as np
import matplotlib
import matplotlib.dates as mdates
import warnings

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 300

warnings.filterwarnings('ignore')

#%% Inputs
sources={'Ceilometer':'wfip3/barg.ceil.z01.cbh/*nc',
         'Lidar':'wfip3/barg.lidar.z03.cbh/*nc'}

colors={'Ceilometer':'c','Lidar':'k'}
markersizes={'Ceilometer':5,'Lidar':3}

#graphics
days_plot=7

#%% Initialization
data={}
for s in sources:
    files=glob.glob(os.path.join(cd,'data',sources[s]))
    data[s]=[]
    for f in files:
        data_f=xr.open_dataset(f)
        tnum=data_f.time.values+data_f.base_time.values
        data_f['time']=tnum*np.timedelta64(1,'s')+np.datetime64('1970-01-01T00:00:00')
    
        if len(data[s])>0:
            data[s]=xr.concat([data[s],data_f],dim='time')
        else:
            data[s]=data_f
    


site=sources[list(sources.keys())[0]].split('/')[1].split('.')[0]
os.makedirs(os.path.join(cd,f'figures/cbh_comp_{site}'),exist_ok=True)

#%% Plots
plt.close('all')
date_fmt = mdates.DateFormatter('%Y-%m-%d')
time=data[list(sources.keys())[0]].time.values
dtime=np.timedelta64(days_plot,'D')
time_bins=np.arange(time[0],time[-1]+dtime,dtime)

for t1,t2 in zip(time_bins[:-1],time_bins[1:]):
    fig=plt.figure(figsize=(18,5))
    
    empty=0
    for s in sources:
        sel=(data[s].time>=t1)*(data[s].time<=t2)
        if np.sum(sel)==0:
            empty+=1
            continue
        plt.plot(data[s].time[sel],data[s].first_cbh.where(data[s].first_cbh>=0)[sel],'.',label=s,markersize=markersizes[s],color=colors[s],alpha=0.75)
    if empty<len(sources):
        plt.xlabel('Time (UTC)')
        plt.ylabel('CBH [m]')
        plt.ylim([0,12000])
        plt.gca().xaxis.set_major_formatter(date_fmt)        
        plt.grid()
        plt.legend(loc='upper right')
        plt.title(f'CBH comparison at {site}')
        plt.savefig(os.path.join(cd,f'figures/cbh_comp_{site}/{str(t1)[:10].replace("-","")}.{str(t2)[:10].replace("-","")}.cbh.png'))
        plt.close()