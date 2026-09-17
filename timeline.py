# -*- coding: utf-8 -*-
"""
Read barge movement file
"""

import os
cd=os.getcwd()
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from doe_dap_dl import DAP
import utm
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

sites=["Martha's Vineyard","Cape Cod","Rhode Island","Barge"]

channels={"Martha's Vineyard":['wfip3/mvco.assist.z01.00','wfip3/mvco.ceil.z01.00'],
          "Cape Cod":         ['wfip3/caco.assist.z01.00','wfip3/caco.lidar.z02.a0','wfip3/caco.ceil.z01.b0','wfip3/caco.met.z01.00'],
          "Rhode Island":     ['wfip3/rhod.assist.z01.00','wfip3/rhod.lidar.z01.a0','wfip3/rhod.met.z01.00'],
          "Barge":            ['wfip3/barg.assist.z01.00','wfip3/barg.ceil.z01.b0','wfip3/barg.ecflux.z01.a0']}

ext={'wfip3/mvco.assist.z01.00':'assistcha',
     'wfip3/mvco.ceil.z01.00':'',
     'wfip3/caco.assist.z01.00':'assistno12cha',
     'wfip3/caco.lidar.z02.a0':'fpt',
     'wfip3/caco.ceil.z01.b0':'',
     'wfip3/caco.met.z01.00':'',
     'wfip3/rhod.assist.z01.00':'assistcha',
     'wfip3/rhod.lidar.z01.a0':'',
     'wfip3/rhod.met.z01.00':'',
     'wfip3/barg.assist.z01.00':'assistno12cha',
     'wfip3/barg.ceil.z01.b0':'',
     'wfip3/barg.ecflux.z01.a0':''}
     
     
dtype= {'wfip3/mvco.assist.z01.00':'cdf',
        'wfip3/mvco.ceil.z01.00':'dat',
        'wfip3/caco.assist.z01.00':'cdf',
        'wfip3/caco.lidar.z02.a0':'nc',
        'wfip3/caco.ceil.z01.b0':'nc',
        'wfip3/caco.met.z01.00':'nc',
        'wfip3/rhod.assist.z01.00':'cdf',
        'wfip3/rhod.lidar.z01.a0':'nc',
        'wfip3/rhod.met.z01.00':'csv',
        'wfip3/barg.assist.z01.00':'cdf',
        'wfip3/barg.ceil.z01.b0':'nc',
        'wfip3/barg.ecflux.z01.a0':'nc'}

sdate='20240801000000'#start date for data search
edate='20240805000000'#end date for data search
hours=168

#barge info
barge_gps_source='data/WHOI_WFIP3_barge_bowstern_GPS_22-Oct-2024.dat'
barge_gps_headers='yyyy mm dd HH MM SS bow_lon bow_lat stern_lon stern_lat'
lat0=40.9015#[deg]
lon0=-70.787#[deg]
max_dist=600#[m]

#graphics
colors={'assist':'r','ceil':'b','met':'k','lidar':'g','ecflux':'k'}

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

barge_gps=pd.read_csv(barge_gps_source, delim_whitespace=True,header=None, names=barge_gps_headers.split(' '))
barge_gps=barge_gps.replace(999,np.nan)
barge_gps=barge_gps.dropna(subset=['bow_lat','bow_lon'])

barge_gps_time=np.array([np.datetime64(f'{Y}-{m:02d}-{d:02d}T{H:02d}:{M:02d}:{S:02d}') for Y,m,d,H,M,S in \
                    zip(barge_gps['yyyy'],barge_gps['mm'],barge_gps['dd'],barge_gps['HH'],barge_gps['MM'],barge_gps['SS'])])

XY0=utm.from_latlon(lat0,lon0)
XY=utm.from_latlon(barge_gps['bow_lat'].values,barge_gps['bow_lon'].values)

inplace=(((XY[0]-XY0[0])**2+(XY[0]-XY0[0])**2)**0.5<max_dist)+0
barge_start=np.where(np.diff(inplace)>0)[0]
barge_end=np.where(np.diff(inplace)<0)[0]

#%% Plots
all_channels=sorted({channel for site in sites for channel in channels[site]})

date_fmt=mdates.DateFormatter('%b %Y')
fig,axs=plt.subplots(len(sites),1,figsize=(16,3*len(sites)),sharex=True,squeeze=False)
for ax,site in zip(axs[:,0],sites):
    site_channels=channels[site]
    if site=='Barge':
        for ctr,(s,e) in enumerate(zip(barge_start,barge_end)):
            if ctr==0:
                ax.axvspan(barge_gps_time[s], barge_gps_time[e],facecolor='lightblue', hatch='//',edgecolor='gray',alpha=0.5,label='Barge on station')
            else:
                ax.axvspan(barge_gps_time[s], barge_gps_time[e],facecolor='lightblue', hatch='//',edgecolor='gray',alpha=0.5)
    for i,channel in enumerate(site_channels):
        t=time_file[site][channel]
        ax.plot(t,np.zeros(len(t))+i,'.',markersize=10,color=colors[channel.split('.')[1]])
    ax.set_ylim(-0.5,len(site_channels)-0.5)
    ax.set_yticks(range(len(site_channels)))
    ax.set_yticklabels(site_channels)
    ax.set_title(site)
    ax.grid(True,color='#e1e0d9')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax.xaxis.set_major_formatter(date_fmt)
    if site=='Barge':
        ax.legend(draggable=True)
plt.tight_layout()

