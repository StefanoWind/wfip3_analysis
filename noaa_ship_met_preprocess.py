# -*- coding: utf-8 -*-
'''
Split the merged NOAA ship met file into daily files and plot overall time series
'''

import os
cd=os.path.dirname(__file__)
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 300
plt.close('all')

#%% Inputs
source=os.path.join(cd,'data','wfip3','noaa_ship.met.merged.r01.c0','noaa_ship.met.merged.r01.c0.20241202.000000.10m.nc')

#%% Initialization
outdir=os.path.dirname(source)
os.makedirs(os.path.join(cd,'figures','noaa_ship_met'),exist_ok=True)

Data=xr.open_dataset(source)

#%% Main

#split into daily files
days=pd.to_datetime(Data.time.values).normalize()
for day in np.unique(days):
    Data_day=Data.isel(time=days==day)
    date_str=pd.Timestamp(day).strftime('%Y%m%d')
    outfile=os.path.join(outdir,f'noaa_ship.met.r01.c0.{date_str}.000000.nc')
    Data_day.to_netcdf(outfile)

#%% Plots
date_fmt=mdates.DateFormatter('%Y-%m-%d')

fig,ax=plt.subplots(3,1,figsize=(18,10),sharex=True)

ax[0].plot(Data.time,Data.tair,label='tair')
ax[0].plot(Data.time,Data.tair_10,label='tair_10')
ax[0].plot(Data.time,Data.tair_2,label='tair_2')
ax[0].set_ylabel(r'Air temperature [$^\circ$C]')
ax[0].legend()
ax[0].grid()

ax[1].plot(Data.time,Data.rhair,label='rhair')
ax[1].plot(Data.time,Data.rhair_2,label='rhair_2')
ax[1].plot(Data.time,Data.rhair_10,label='rhair_10')
ax[1].set_ylabel('Relative humidity [%]')
ax[1].legend()
ax[1].grid()

ax[2].plot(Data.time,Data.pair_10,label='pair_10')
ax[2].plot(Data.time,Data.psealevel,label='psealevel')
ax[2].set_ylabel('Pressure [mbar]')
ax[2].set_xlabel('Time (UTC)')
ax[2].xaxis.set_major_formatter(date_fmt)
ax[2].legend()
ax[2].grid()

fig.suptitle('NOAA ship surface meteorology')
plt.tight_layout()
plt.savefig(os.path.join(cd,'figures','noaa_ship_met','timeseries.png'))
