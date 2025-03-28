# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''
Compare TROPoe results and met on barge
'''

import os
cd=os.getcwd()
from matplotlib import pyplot as plt
import xarray as xr
import glob
import numpy as np
import matplotlib
import yaml
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.size'] = 12
plt.close('all')


#%% Inputs

#dataset
source_config=os.path.join(cd,'configs','config.yaml')

z_irs=3#[m] height of ASSIST asl
time_shift_met=np.timedelta64(10, 'm')#shift met data to center of avg window
M_h20=18.015#[g/mol] molar mass of water 
M_a=28.97#[g/mol] molar mass of air
max_height=100#[m] max height of TROPoe results of interest

#graphics
days_plot=7

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

files_trp=glob.glob(config['path_trp_comp'])
files_met=glob.glob(config['path_met_comp'])

os.makedirs(os.path.join(cd,'figures/temp_barg'),exist_ok=True)
os.makedirs(os.path.join(cd,'figures/r_barg'),exist_ok=True)
  
#%% Main

#read met data
Data_met=xr.open_mfdataset(files_met,decode_times=False)
time_met=(Data_met.time.values-719529)*60*60*24*np.timedelta64(1,'s')+np.datetime64('1970-01-01T00:00:00')+time_shift_met
z1_met=Data_met.met_z.isel(rh_t_heights=0).values[0]
z2_met=Data_met.met_z.isel(rh_t_heights=-1).values[0]

#temperature
T1_met=Data_met.air_temp_c.isel(air_temp_sensors=0).values
T2_met=Data_met.air_temp_c.isel(air_temp_sensors=-1).values
dT_dz_met=(T2_met-T1_met)/(z2_met-z1_met)

#mixing ratio
rh1_met=Data_met.rh.isel(rhT_sensors=0).values
P1_met=Data_met['air_press_mb'].isel(num_air_press=0).values*100
es1_met=611.2*np.exp(17.62*T1_met/(T1_met+243.12))#Teten's formula
e1_met=es1_met*rh1_met/100
r1_met=M_h20/M_a*e1_met/(P1_met-e1_met)*1000

rh2_met=Data_met.rh.isel(rhT_sensors=-1).values
P2_met=Data_met['air_press_mb'].isel(num_air_press=-1).values*100
es2_met=611.2*np.exp(17.62*T2_met/(T2_met+243.12))
e2_met=es2_met*rh2_met/100
r2_met=M_h20/M_a*e2_met/(P2_met-e2_met)*1000

dr_dz_met=(r2_met-r1_met)/(z2_met-z1_met)

#read TROPoe data
Data_trp=xr.open_mfdataset(files_trp).sel(height=slice(0,max_height))

#qc TROPoe data
Data_trp['cbh'][(Data_trp['lwp']<config['min_lwp']).compute()]=Data_trp['height'].max()#remove clouds with low lwp

qc_gamma=Data_trp['gamma']<=config['max_gamma']
qc_rmsa=Data_trp['rmsa']<=config['max_rmsa']
qc_cbh=Data_trp['height']<=Data_trp['cbh']
qc=qc_gamma*qc_rmsa*qc_cbh
Data_trp['temperature_qc']=Data_trp['temperature'].where(qc)#filter temperature
Data_trp['r_qc']=  Data_trp['waterVapor'].where(qc)#filter mixing ratio
Data_trp['rh_qc']=  Data_trp['rh'].where(qc)#filter RH ratio

#extract TROPoe data
time_trp=Data_trp.time.values
T1_trp=Data_trp['temperature_qc'].interp(height=(z1_met-z_irs)/1000).values
T2_trp=Data_trp['temperature_qc'].interp(height=(z2_met-z_irs)/1000).values
dT_dz_trp=(T2_trp-T1_trp)/(z2_met-z1_met)

r1_trp=Data_trp['r_qc'].interp(height=(z1_met-z_irs)/1000).values
r2_trp=Data_trp['r_qc'].interp(height=(z2_met-z_irs)/1000).values
dr_dz_trp=(r2_trp-r1_trp)/(z2_met-z1_met)

#check RH->r
rh1_trp=Data_trp['rh_qc'].interp(height=(z1_met-z_irs)/1000).values
es1_trp=611.2*np.exp(17.62*T1_trp/(T1_trp+243.12))
e1_trp=es1_trp*rh1_trp/100
P1_trp=Data_trp['pressure'].interp(height=(z1_met-z_irs)/1000).values*100
r1_trp_check=M_h20/M_a*e1_trp/(P1_trp-e1_trp)*1000

print(f'Max difference in WVMR is {np.round(np.nanmax(np.abs(r1_trp_check-r1_trp)),1)} g/Kg')

#%% Plots
plt.close('all')
date_fmt = mdates.DateFormatter('%Y-%m-%d')
dtime=np.timedelta64(days_plot,'D')
time_bins=np.arange(time_met[0],time_met[-1]+dtime,dtime)

for t1,t2 in zip(time_bins[:-1],time_bins[1:]):
    
    plt.figure(figsize=(18,8))
    plt.subplot(2,1,1)
    sel=(time_met>=t1)*(time_met<=t2)
    plt.plot(time_met[sel],T2_met[sel],label='Met')
    sel=(time_trp>=t1)*(time_trp<=t2)
    plt.plot(time_trp[sel],T2_trp[sel],label='TROPoe')
    plt.ylabel(f'Temperature ({z2_met} m)'+' [$^\circ$C]')
    plt.grid()
    plt.title('Temperature comparison at barge')
    plt.ylim([0,35])
    plt.legend()
    
    plt.subplot(2,1,2)
    sel=(time_met>=t1)*(time_met<=t2)
    plt.plot(time_met[sel],dT_dz_met[sel])
    sel=(time_trp>=t1)*(time_trp<=t2)
    plt.plot(time_trp[sel],dT_dz_trp[sel])
    plt.xlabel('Time (UTC)')
    plt.ylabel(f'Temperature gradient ({z2_met} m -{z1_met} m)'+' [$^\circ$C m${-1}$]')
    plt.gca().xaxis.set_major_formatter(date_fmt)        
    plt.grid()
    plt.title('Temperature gradient comparison at barge')
    plt.ylim([-0.15,0.15])
    plt.savefig(os.path.join(cd,f'figures/temp_barg/{str(t1)[:10].replace("-","")}.temp.png'))
    plt.close()
    
for t1,t2 in zip(time_bins[:-1],time_bins[1:]):
    
    plt.figure(figsize=(18,8))
    plt.subplot(2,1,1)
    sel=(time_met>=t1)*(time_met<=t2)
    plt.plot(time_met[sel],r2_met[sel],label='Met')
    sel=(time_trp>=t1)*(time_trp<=t2)
    plt.plot(time_trp[sel],r2_trp[sel],label='TROPoe')
    plt.ylabel(f'Mixing ratio ({z2_met} m)'+' [%]')
    plt.grid()
    plt.title('Mixing ratio comparison at barge')
    plt.ylim([0,20])
    plt.legend()
    
    plt.subplot(2,1,2)
    sel=(time_met>=t1)*(time_met<=t2)
    plt.plot(time_met[sel],dr_dz_met[sel])
    sel=(time_trp>=t1)*(time_trp<=t2)
    plt.plot(time_trp[sel],dr_dz_trp[sel])
    plt.xlabel('Time (UTC)')
    plt.ylabel(f'Mixing ratio ({z2_met} m -{z1_met} m)'+' [% m${-1}$]')
    plt.gca().xaxis.set_major_formatter(date_fmt)        
    plt.grid()
    plt.title('Mixing ratio gradient comparison at barge')
    plt.ylim([-0.1,0.1])
    plt.savefig(os.path.join(cd,f'figures/r_barg/{str(t1)[:10].replace("-","")}.r.png'))
    plt.close()



