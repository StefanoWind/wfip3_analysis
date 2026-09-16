# -*- coding: utf-8 -*-
"""
Read barge movement file
"""

import os
cd=os.getcwd()
import numpy as np
from matplotlib import pyplot as plt
from doe_dap_dl import DAP
import warnings
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
username='sletizia'
password='pass_DAP1506@'

sites=["Martha's Vineyard","Cape Cod","NOAA Ship"]

channels={"Martha's Vineyard":['wfip3/mvco.assist.z01.00','wfip3/mvco.ceil.z01.00','wfip3/mvco.met.z01.00'],
          "Cape Cod":         ['wfip3/caco.assist.z01.00','wfip3/caco.lidar.z02.a0'],
          "NOAA Ship":        ['wfip3/noaa_ship.assist.z01.00','wfip3/noaa_ship.ceil.z01.b0']}

ext={'wfip3/mvco.assist.z01.00':'assistcha',
     'wfip3/mvco.ceil.z01.00':'',
     'wfip3/mvco.met.z01.00':'',
     'wfip3/caco.assist.z01.00':'assistno12cha',
     'wfip3/caco.lidar.z02.a0':'user5',
     'wfip3/noaa_ship.assist.z01.00':'assistno12cha',
     'wfip3/noaa_ship.ceil.z01.b0':''}
     
dtype={'wfip3/mvco.assist.z01.00':'cdf',
    'wfip3/mvco.ceil.z01.00':'dat',
    'wfip3/mvco.met.z01.00':'dat',
    'wfip3/caco.assist.z01.00':'cdf',
    'wfip3/caco.lidar.z02.a0':'nc',
    'wfip3/noaa_ship.assist.z01.00':'cdf',
    'wfip3/noaa_ship.ceil.z01.b0':'nc'}

sdate='20231001000000'#start date for data search
edate='20251231000000'#end date for data search
hours=168

#%% Functions
def strtime_to_dt64(strtime):
    return np.datetime64(f'{strtime[:4]}-{strtime[4:6]}-{strtime[6:8]}T{strtime[8:10]}:{strtime[10:12]}:{strtime[12:14]}')

def dt64_to_str(dt64):
    return str(dt64).replace('-','').replace('T','').replace(':','')

def dap_search(channel,sdate,edate,ftype,ext1,hours=30):
    '''
    Wrapper for a2e.search to avoid timeout:
        Inputs: channel name, start date, end date, file format, extention in WDH name, number of days scanned at each loop
        Outputs: list of files mathing the criteria
    '''
    
    if ~np.isnan(hours):
        time_bins=np.arange(strtime_to_dt64(sdate),strtime_to_dt64(edate)+np.timedelta64(hours,'h')/2,np.timedelta64(hours,'h'))
    else:
        time_bins=[strtime_to_dt64(sdate),strtime_to_dt64(edate)]
        
    search_all=[]
    for t1,t2 in zip(time_bins[:-1],time_bins[1:]):
        
        sd=dt64_to_str(t1)
        ed=dt64_to_str(t2)
        
        if ext1=='':
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [sd,ed]
                },
                'file_type':ftype,
            }
            print(f'Searching: {channel} from {sd} to {ed} for format {ftype}')
        else:
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [sd,ed]
                },
                'file_type':ftype,
                'ext1': ext1
            }
            print(f'Searching: {channel} from {sd} to {ed} for format {ftype} and ext1 {ext1}')
    
        search=a2e.search(_filter)
        
        if search is None:
            print('Invalid authentication')
            return None
        else:
            search_all+=search
    
    return search_all

#%% Initalization
a2e = DAP('wdh.energy.gov',confirm_downloads=False)
a2e.setup_cert_auth(username=username, password=password)

#%% Main
time_file={}
for site in sites:
    time_file[site]={}
    for channel in channels[site]:
        
        files=dap_search(channel, sdate, edate, dtype[channel], ext[channel],hours)
    
        time_file[site][channel]=np.array([datetime.strptime(f["date_time"],"%Y%m%d%H%M%S") for f in files])

#%% Plots
palette=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948']
all_channels=sorted({channel for site in sites for channel in channels[site]})
colors={channel:palette[i%len(palette)] for i,channel in enumerate(all_channels)}

date_fmt=mdates.DateFormatter('%b %Y')
fig,axs=plt.subplots(len(sites),1,figsize=(16,3*len(sites)),sharex=True,squeeze=False)
for ax,site in zip(axs[:,0],sites):
    site_channels=channels[site]
    for i,channel in enumerate(site_channels):
        t=time_file[site][channel]
        ax.plot(t,np.zeros(len(t))+i,'.',markersize=10,color=colors[channel])
    ax.set_ylim(-0.5,len(site_channels)-0.5)
    ax.set_yticks(range(len(site_channels)))
    ax.set_yticklabels(site_channels)
    ax.set_title(site)
    ax.grid(True,color='#e1e0d9')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax.xaxis.set_major_formatter(date_fmt)
plt.tight_layout()

