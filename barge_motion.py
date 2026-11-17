# -*- coding: utf-8 -*-
'''
Read barge motion (MicroStrain 3DM-GX5 MIP binary) IMU data and compare with ASSIST imaginary radiance
'''

import os
cd=os.getcwd()
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import welch
from scipy.stats import spearmanr,norm
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
if len(sys.argv)==1:
    source_imu=os.path.join(cd,'data/wfip3/barg.motion.z01.00/*imu.bin')
    source_irs=os.path.join(cd,'data/wfip3/barg.assist.z01.a0/*nc')
else:
    source_imu=sys.argv[1]
    source_irs=sys.argv[2]

window='10min'#averaging window (length of IMU files)
min_n_irs=20#minimum number of ASSIST samples per window (~28 expected)
variables={'roll':'$^\\circ$','pitch':'$^\\circ$',
           'acc_x':'g','acc_y':'g','acc_z':'g',
           'gyro_x':'$^\\circ$ s$^{-1}$','gyro_y':'$^\\circ$ s$^{-1}$','gyro_z':'$^\\circ$ s$^{-1}$'}#IMU variables of interest and units

fs_imu=10#[Hz] IMU sampling frequency
nperseg=2048#Welch segment length (204.8 s)
min_n_imu=5000#minimum number of IMU samples per window (~6000 expected at 10 Hz)
n_bins=20#number of bins per axis in 2D histograms

# MIP IMU packet layout, fields in the order logged (big-endian) [LORD MicroStrain, 3DM-GX5-45 Data Communications Protocol Manual, 2018]
IMU_SET=0x80
FIELDS=[
    (0x0C,[('roll','>f4'),('pitch','>f4'),('yaw','>f4')]),#[rad]
    (0x05,[('gyro_x','>f4'),('gyro_y','>f4'),('gyro_z','>f4')]),#[rad/s]
    (0x04,[('acc_x','>f4'),('acc_y','>f4'),('acc_z','>f4')]),#[g]
    (0x17,[('pressure','>f4')]),#[mbar]
    (0x06,[('mag_x','>f4'),('mag_y','>f4'),('mag_z','>f4')]),#[Gauss]
    (0x12,[('gps_tow','>f8'),('gps_week','>u2'),('gps_flags','>u2')]),
    ]
PACKET=np.dtype([('header','u1',4)]
                +[item for desc,names in FIELDS for item in [(f'len_{desc}','u1'),(f'desc_{desc}','u1')]+names]
                +[('checksum','u1',2)])

GPS_EPOCH=np.datetime64('1980-01-06T00:00:00')
LEAP_SECONDS=np.timedelta64(18,'s')#GPS-UTC offset since 2017-01-01

#%% Functions
def read_mip(file):
    raw=np.fromfile(file,dtype=np.uint8)
    n=PACKET.itemsize
    starts=np.flatnonzero((raw[:-n+1]==0x75)&(raw[1:len(raw)-n+2]==0x65)&(raw[2:len(raw)-n+3]==IMU_SET)&(raw[3:len(raw)-n+4]==n-6))
    packets=raw[starts[:,None]+np.arange(n)]

    # Fletcher-16 checksum [LORD MicroStrain, 2018]
    body=packets[:,:-2].astype(np.int64)
    valid=(body.sum(axis=1)%256==packets[:,-2])&((body*np.arange(n-2,0,-1)).sum(axis=1)%256==packets[:,-1])
    rec=np.ascontiguousarray(packets[valid]).view(PACKET).ravel()
    for desc,names in FIELDS:
        rec=rec[rec[f'desc_{desc}']==desc]
    return rec

def spearman_ci(x,y,alpha=0.05):
    # Fisher z-transform with Spearman standard error [Bonett and Wright, 2000]
    rho=spearmanr(x,y,nan_policy='omit').statistic
    n=np.sum(~np.isnan(x)&~np.isnan(y))
    se=np.sqrt((1+rho**2/2)/(n-3))
    z=norm.ppf(1-alpha/2)
    return rho,np.tanh(np.arctanh(rho)-z*se),np.tanh(np.arctanh(rho)+z*se)

def to_dataset(rec):
    names=[name for desc,fields in FIELDS for name,_ in fields]
    data={name:rec[name].astype(float) for name in names}
    time=GPS_EPOCH+(data['gps_week']*7*86400+data['gps_tow'])*1e9*np.timedelta64(1,'ns')-LEAP_SECONDS
    return xr.Dataset({k:('time',v) for k,v in data.items()},coords={'time':time})

#%% Main

#read IMU one file at a time, keeping per-window sums and per-file PSD only (memory)
files_imu=sorted(glob.glob(source_imu))
sums=[]
psd={v:[] for v in variables}
for f in files_imu:
    d=to_dataset(read_mip(f))[list(variables)]
    d=d.assign({v:np.degrees(d[v]) for v in ['roll','pitch','gyro_x','gyro_y','gyro_z']})

    df=d.to_dataframe()
    t_win=df.index.floor(window)
    sums.append(pd.concat({'sum':df.groupby(t_win).sum(),'sum2':(df**2).groupby(t_win).sum(),'n':df.groupby(t_win).count()},axis=1))

    #power spectral density per continuous file (Hann window, 50% overlap) [Welch, 1967]
    d=d.sortby('time').drop_duplicates('time')
    if d.time.size<nperseg:
        continue
    time_reg=np.arange(d.time.values[0],d.time.values[-1],np.timedelta64(int(1000/fs_imu),'ms'))
    d_reg=d.interp(time=time_reg)
    for v in variables:
        f_psd,P=welch(d_reg[v].values,fs=fs_imu,nperseg=nperseg,detrend='linear')
        psd[v].append(P)
psd=xr.Dataset({v:('freq',np.mean(psd[v],axis=0)) for v in variables},coords={'freq':f_psd})

#combine windows across files (population std, as xarray default)
sums=pd.concat(sums).groupby(level=0).sum()
imu_mean=sums['sum']/sums['n']
imu_std=np.sqrt(np.maximum(sums['sum2']/sums['n']-imu_mean**2,0))

#read ASSIST
files_irs=sorted(glob.glob(source_irs))
Data_irs=xr.open_mfdataset(files_irs,data_vars='minimal',coords='minimal',compat='override')
rad_imag=Data_irs.mean_imaginary_rad_985_990.where(Data_irs.qc_mean_imaginary_rad_985_990==0).compute()

#statistics over common time windows
irs=rad_imag.resample(time=window)
stats=xr.Dataset({'rad_imag_mean':irs.mean(),'rad_imag_std':irs.std(),'rad_imag_n':irs.count()})
imu_stats=pd.DataFrame({'imu_n':sums['n']['roll']})
for v in variables:
    imu_stats[f'{v}_mean']=imu_mean[v]
    imu_stats[f'{v}_std']=imu_std[v]
imu_stats.index.name='time'
stats=xr.merge([stats,xr.Dataset.from_dataframe(imu_stats)],join='inner')
stats=stats.where((stats.rad_imag_n>=min_n_irs)&(stats.imu_n>=min_n_imu),drop=True)
print(f'{stats.time.size} valid {window} windows')

#output
with pd.ExcelWriter(os.path.join(cd,'data',f'barge_motion_rad_imag_stats_{window}.xlsx')) as writer:
    stats.to_dataframe().to_excel(writer,sheet_name='stats')
    psd.to_dataframe().to_excel(writer,sheet_name='psd')

#%% Plots
os.makedirs(os.path.join(cd,'figures','barge_motion'),exist_ok=True)
hist={(stat,v):np.histogram2d(stats[f'{v}_{stat}'].values,stats[f'rad_imag_{stat}'].values,bins=n_bins)
      for stat in ['mean','std'] for v in variables}
vmax=max(H.max() for H,_,_ in hist.values())
for stat in ['mean','std']:
    fig=plt.figure(figsize=(20,10))
    for i,v in enumerate(variables):
        ax=plt.subplot(2,4,i+1)
        H,x_edges,y_edges=hist[(stat,v)]
        pc=plt.pcolor(x_edges,y_edges,np.ma.masked_equal(H.T,0),cmap='hot',norm=LogNorm(vmin=1,vmax=vmax))
        rho,ci_low,ci_high=spearman_ci(stats[f'{v}_{stat}'].values,stats[f'rad_imag_{stat}'].values)
        plt.title(rf'$\rho$ = {rho:.2f} [{ci_low:.2f}, {ci_high:.2f}]',fontsize=14)
        plt.xlabel(f'{stat.capitalize()} of {v} [{variables[v]}]')
        if i%4==0:
            plt.ylabel(f'{stat.capitalize()} of imaginary radiance \n (985-990 cm$^{{-1}}$) [r.u.]')
        plt.grid()
    plt.tight_layout(rect=[0,0,0.92,1])
    cax=fig.add_axes([0.93,0.08,0.015,0.88])
    fig.colorbar(pc,cax=cax,label='Count')
    plt.savefig(os.path.join(cd,'figures','barge_motion',f'rad_imag_vs_motion_{stat}_{window}.png'))

fig=plt.figure(figsize=(20,7))
for i,(group,unit) in enumerate(zip([['roll','pitch'],['acc_x','acc_y','acc_z'],['gyro_x','gyro_y','gyro_z']],
                                    ['$^\\circ$','g','$^\\circ$ s$^{-1}$'])):
    ax=plt.subplot(1,3,i+1)
    for v,color in zip(group,['b','r','g']):
        i_peak=int(psd[v][1:].argmax())+1
        plt.loglog(psd.freq[1:],psd[v][1:],color=color,label=f'{v} (peak period = {1/psd.freq.values[i_peak]:.1f} s)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(f'PSD [({unit})$^2$ Hz$^{{-1}}$]')
    plt.legend()
    plt.grid(which='both',alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(cd,'figures','barge_motion','psd_motion.png'))
