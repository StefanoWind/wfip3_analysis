# -*- coding: utf-8 -*-
'''
Plot time series of pressure, temperature, and relative humidity from CACO met station
'''

import os
cd=os.getcwd()
from matplotlib import pyplot as plt
import xarray as xr
import glob
import matplotlib
import matplotlib.dates as mdates
import numpy as np
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.size'] = 12
plt.close('all')

#%% Inputs
source=os.path.join(cd,'data','wfip3','caco.met.z01.00','*.nc')
thresholds={'air_pressure':[900,1040],
            'air_temperature':[-15,40],
            'relative_humidity':[0,100]}
window=7#[#] points in despiking median window
max_mad={'air_pressure':2,#[hPa]
          'air_temperature':2,#[C]
          'relative_humidity':10}#[%]

#%% Functions
def diff_from_median(x,window):
    '''deviation of x from the running median over `window` points, centered except at the edges (still `window` points, but shifted so no fewer points are used)'''
    x=np.asarray(x)
    n=len(x)
    half=window//2
    med=np.zeros(n)
    windows=np.lib.stride_tricks.sliding_window_view(x,window)
    med[half:n-half]=np.nanmedian(windows,axis=1)
    med[:half]=np.nanmedian(x[:window])
    med[n-half:]=np.nanmedian(x[-window:])
    return x-med

#%% Initialization
files=sorted(glob.glob(source))
Data=xr.open_mfdataset(files)

#%% Main
for v in thresholds:
    Data[f'{v}_mad']=np.abs(xr.DataArray(diff_from_median(Data[v].values,window),coords={'time':Data.time.values}))

    Data[f'{v}_good']=Data[v].where(Data[v]>=thresholds[v][0])\
                             .where(Data[v]<=thresholds[v][1])\
                             .where(Data[f'{v}_mad']<=max_mad[v])
    Data[f'{v}_bad']= Data[v].where(Data[v]>=thresholds[v][0],thresholds[v][0])\
                             .where(Data[v]<=thresholds[v][1],thresholds[v][1])\
                             .where(np.isnan(Data[f'{v}_good'])*(~np.isnan(Data[v])))

#%% Plots
date_fmt=mdates.DateFormatter('%Y-%m-%d')

fig,ax=plt.subplots(3,1,figsize=(18,10),sharex=True)


ax[0].plot(Data.time,Data.air_pressure_bad,'xr',markersize=10)
ax[0].plot(Data.time,Data.air_pressure_good,'.-g',markersize=1)
ax[0].set_ylabel('Pressure [hPa]')
ax[0].grid()

ax[1].plot(Data.time,Data.air_temperature_bad,'xr',markersize=10)
ax[1].plot(Data.time,Data.air_temperature_good,'.-g',markersize=1)
ax[1].set_ylabel(r'Temperature [$^\circ$C]')
ax[1].grid()

ax[2].plot(Data.time,Data.relative_humidity_bad,'xr',markersize=10)
ax[2].plot(Data.time,Data.relative_humidity_good,'.-g',markersize=1)
ax[2].set_ylabel('Relative humidity [%]')
ax[2].set_xlabel('Time (UTC)')
ax[2].xaxis.set_major_formatter(date_fmt)
ax[2].grid()

fig.suptitle('CACO surface meteorology')
plt.tight_layout()
plt.show()
