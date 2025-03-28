# -*- coding: utf-8 -*-
"""
Read barge movement file
"""

import os
cd=os.getcwd()
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/Main/utils')
import utils as utl
import numpy as np
from matplotlib import pyplot as plt
from doe_dap_dl import DAP
import utm
import pandas as pd
import warnings
import matplotlib.dates as mdates
import matplotlib
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
source='data/WHOI_WFIP3_barge_bowstern_GPS_22-Oct-2024.dat'
headers='yyyy mm dd HH MM SS bow_lon bow_lat stern_lon stern_lat'
username='sletizia'
password='pass_DAP1506@'
channels=['wfip3/barg.assist.z01.00','wfip3/barg.assist.tropoe.z01.c0']
ext={'wfip3/barg.assist.z01.00':'assistsummary','wfip3/barg.assist.tropoe.z01.c0':''}
dtype={'wfip3/barg.assist.z01.00':'cdf','wfip3/barg.assist.tropoe.z01.c0':'nc'}
lat0=40.9015#[deg]
lon0=-70.787#[deg]
max_dist=600#[m]
sdate='20240520000000'#start date for data search
edate='20241001000000'#end date for data search

#grpahics
colors={'wfip3/barg.assist.z01.00':'r','wfip3/barg.assist.tropoe.z01.c0':'g'}
labels={'wfip3/barg.assist.z01.00':'ASSIST functional','wfip3/barg.assist.tropoe.z01.c0':'TROPoe v12'}

#%% Functions
def dap_search(channel,sdate,edate,file_type,ext1,time_search=30):
    '''
    Wrapper for a2e.search to avoid timeout:
        Inputs: channel name, start date, end date, file format, extention in WDH name, number of days scanned at each loop
        Outputs: list of files mathing the criteria
    '''
    dates_num=np.arange(utl.datenum(sdate,'%Y%m%d%H%M%S'),utl.datenum(edate,'%Y%m%d%H%M%S'),time_search*24*3600)
    dates=[utl.datestr(d,'%Y%m%d%H%M%S') for d in dates_num]+[edate]
    search_all=[]
    for d1,d2 in zip(dates[:-1],dates[1:]):
        
        if ext1!='':
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [d1,d2]
                },
                'file_type': file_type,
                'ext1':ext1, 
            }
        else:
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [d1,d2]
                },
                'file_type': file_type,
            }
        
        search=a2e.search(_filter)
        
        if search is None:
            print('Invalid authentication')
            return None
        else:
            search_all+=search
    
    return search_all

#%% Initalization
Data = pd.read_csv(source, delim_whitespace=True,header=None, names=headers.split(' '))
Data=Data.replace(999,np.nan)
# Data=Data.drop(columns=['stern_lon','stern_lat'])
Data=Data.dropna(subset=['bow_lat','bow_lon'])

time=np.array([np.datetime64(f'{Y}-{m:02d}-{d:02d}T{H:02d}:{M:02d}:{S:02d}') for Y,m,d,H,M,S in \
                    zip(Data['yyyy'],Data['mm'],Data['dd'],Data['HH'],Data['MM'],Data['SS'])])
    
XY0=utm.from_latlon(lat0,lon0)
    
#%% Main
XY=utm.from_latlon(Data['bow_lat'].values,Data['bow_lon'].values)

a2e = DAP('a2e.energy.gov',confirm_downloads=False)
a2e.setup_cert_auth(username=username, password=password)

files={}
time_file={}
for c in channels:
    files[c]=dap_search(c, sdate,edate, dtype[c], ext[c])

    time_file[c]=np.array([np.datetime64(f'{f["data_date"][:4]}-{f["data_date"][4:6]}-{f["data_date"][6:8]}T00:00:00') for f in files[c]])

inplace=(((XY[0]-XY0[0])**2+(XY[0]-XY0[0])**2)**0.5<max_dist)+0
start=np.where(np.diff(inplace)>0)[0]
end=np.where(np.diff(inplace)<0)[0]

#%% Plots
plt.figure(figsize=(16,4))
ctr=0
for s,e in zip(start,end):
    if ctr==0:
        plt.axvspan(time[s], time[e],facecolor='lightblue', hatch='//',edgecolor='gray',alpha=0.5,label='Barge on station')
    else:
        plt.axvspan(time[s], time[e],facecolor='lightblue', hatch='//',edgecolor='gray',alpha=0.5)
    ctr+=1

ctr=0
for c in channels:
    plt.plot(time_file[c],np.zeros(len(time_file[c]))-ctr,'.',markersize=10,color=colors[c],label=labels[c])
    ctr+=1
date_fmt = mdates.DateFormatter('%b %Y')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))  # First day of each month
plt.gca().xaxis.set_major_formatter(date_fmt) 
plt.yticks([])
plt.grid()
plt.legend()