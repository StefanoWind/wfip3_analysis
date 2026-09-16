# -*- coding: utf-8 -*-
"""
Compare TROPoe versions
"""

import os
cd=os.path.dirname(__file__)
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
import glob
from doe_dap_dl import DAP

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi'] = 500

#%% Inputs
channels=['wfip3/barg.assist.tropoe.z01.c0','wfip3/barg.assist.tropoe.z01.c1']
sdate='2024-06-12'
edate='2024-06-17'
username='sletizia'
password='pass_DAP1506@'

#%% Initialization
a2e = DAP('a2e.energy.gov',confirm_downloads=False)
a2e.setup_cert_auth(username=username, password=password)

for channel in channels:
    
    _filter = {'Dataset': channel,
        'date_time': {'between':  [sdate.replace('-','')+'000000',edate.replace('-','')+'235959']},
        'file_type': 'nc'}
    
    a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel), replace=False)
    
#%% Main
files1=glob.glob(os.path.join(cd,'data',channels[0],'*nc'))
Data1=xr.open_mfdataset(files1)
    
files2=glob.glob(os.path.join(cd,'data',channels[1],'*nc'))
Data2=xr.open_mfdataset(files2)        

dT=Data2.temperature-Data1.temperature
dr=Data2.waterVapor-Data1.waterVapor

#check attributes
for a in Data1.attrs:
    if str(Data1.attrs[a]) != str(Data2.attrs[a]):
        print(f'{a}: {Data1.attrs[a]} vs. {Data2.attrs[a]}')


#%% Plots
plt.close('all')
plt.figure(figsize=(20,15))
plt.subplot(3,1,1)
plt.pcolor(Data1.time,Data1.height*1000,Data1.temperature.T,vmin=5,vmax=30,cmap='hot')
plt.ylim([0,2000])
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.title(channels[0])
plt.colorbar(label=r'$T$ [$^\circ$C]')

plt.subplot(3,1,2)
plt.pcolor(Data2.time,Data2.height*1000,Data2.temperature.T,vmin=5,vmax=30,cmap='hot')
plt.ylim([0,2000])
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.title(channels[1])
plt.colorbar(label=r'$T$ [$^\circ$C]')

plt.subplot(3,1,3)
plt.pcolor(dT.time,dT.height*1000,dT.T,vmin=-1,vmax=1,cmap='seismic')
plt.ylim([0,2000])
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.title(f'{channels[1]}-{channels[0]}')
plt.colorbar(label=r'$\Delta T$ [$^\circ$C]')
plt.tight_layout()

plt.figure(figsize=(20,15))
plt.subplot(3,1,1)
plt.pcolor(Data1.time,Data1.height*1000,Data1.waterVapor.T,vmin=0,vmax=15,cmap='Blues')
plt.ylim([0,2000])
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.title(channels[0])
plt.colorbar(label=r'$r$ [g Kg$^{-1}$]')

plt.subplot(3,1,2)
plt.pcolor(Data2.time,Data2.height*1000,Data2.waterVapor.T,vmin=0,vmax=15,cmap='Blues')
plt.ylim([0,2000])
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.title(channels[1])
plt.colorbar(label=r'$r$ [g Kg$^{-1}$]')

plt.subplot(3,1,3)
plt.pcolor(dr.time,dr.height*1000,dr.T,vmin=-1,vmax=1,cmap='seismic')
plt.ylim([0,2000])
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.title(f'{channels[1]}-{channels[0]}')
plt.colorbar(label=r'$\Delta r$ [g Kg$^{-1}$]')
plt.tight_layout()