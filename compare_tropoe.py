# -*- coding: utf-8 -*-
'''
Compare TROPoe retrievals across the sites listed in config['sources_comparison']
'''

import os
cd=os.path.dirname(__file__)
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import glob
import yaml
import itertools
import re
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 300

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')

heights=[0,5,10,20]#height array indices of interest
height_max=2000#[m] max height for the diurnal difference heatmap
time_tolerance=pd.Timedelta('5min')#max time difference for matching points across sites
variables=['temperature','waterVapor','gamma','cbh','rmsa','lwp']
var_names={'temperature':'Temperature','waterVapor':'Mixing ratio'}
units={'temperature':r'$^\circ$C','waterVapor':'g kg$^{-1}$'}

#%% Initialization
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

os.makedirs(os.path.join(cd,'figures/tropoe_comp'),exist_ok=True)

#%% Main

#load data
data_full={}#full height profile up to height_max, used for the diurnal heatmap
data={}#sparse heights, used for the scatter and time series plots
for name,path in config['sources_comparison'].items():
    files=sorted(glob.glob(os.path.join(cd,path,'*nc')))
    print(f'{name}: {len(files)} files found')
    if len(files)==0:
        continue
    ds=xr.open_mfdataset(files,data_vars='minimal',coords='minimal',compat='override',
                          preprocess=lambda d: d[list(variables)])
    ds=ds.sel(height=slice(0,height_max/1000)).compute()

    #QC flags
    ds['cbh'][(ds['lwp']<config['min_lwp']).compute()]=ds['height'].max()#remove clouds with low lwp
    qc_gamma=ds['gamma']<=config['max_gamma']
    qc_rmsa=ds['rmsa']<=config['max_rmsa']
    qc_cbh=ds['height']<=ds['cbh']
    qc=qc_gamma*qc_rmsa*qc_cbh
    ds['temperature']=ds['temperature'].where(qc)#filter temperature
    ds['waterVapor']= ds['waterVapor'].where(qc)#filter mixing ratio

    data_full[name]=ds
    data[name]=ds.isel(height=heights)

#pairwise scatter comparisons
for n1,n2 in itertools.combinations(data,2):
    for var,unit in units.items():
        fig,axs=plt.subplots(2,2,figsize=(11,11))
        for ax,i_h in zip(axs.flatten(),range(len(heights))):
            h=int(np.round(data[n1].height.isel(height=i_h).values*1000))#[m] actual height at this index
            s1=data[n1][var].isel(height=i_h).to_series().rename('x')
            s2=data[n2][var].isel(height=i_h).to_series().rename('y')
            df=pd.merge_asof(s1.sort_index().reset_index(),s2.sort_index().reset_index(),
                              on='time',tolerance=time_tolerance,direction='nearest').dropna()
            x=df['x'].values
            y=df['y'].values

            ax.plot(x,y,'.k',alpha=0.3,markersize=5)
            if len(x)>1:
                p=np.polyfit(x,y,1)
                x_fit=np.array([np.nanmin(x),np.nanmax(x)])
                ax.plot(x_fit,np.polyval(p,x_fit),'r')
                r=np.corrcoef(x,y)[0,1]
                ax.text(0.05,0.95,f'y={p[0]:.2f}x+{p[1]:.2f}\nR={r:.2f}\nN={len(x)}',
                        transform=ax.transAxes,va='top')
            ax.set_xlabel(f'{n1} [{unit}]')
            ax.set_ylabel(f'{n2} [{unit}]')
            ax.set_title(f'{h} m')
            ax.grid()

        fig.suptitle(f'{var_names[var]}: {n1} vs {n2}')
        plt.tight_layout()
        n1_clean=re.sub(r'\W+','_',n1)
        n2_clean=re.sub(r'\W+','_',n2)
        # plt.savefig(os.path.join(cd,f'figures/tropoe_comp/{n1_clean}_vs_{n2_clean}.{var}.png'))
        # plt.close(fig)

#time series at each height
for var,unit in units.items():
    fig,axs=plt.subplots(2,2,figsize=(14,10),sharex=True)
    for ax,i_h in zip(axs.flatten(),range(len(heights))):
        h=int(np.round(list(data.values())[0].height.isel(height=i_h).values*1000))#[m] actual height at this index
        for name in data:
            ax.plot(data[name].time,data[name][var].isel(height=i_h),'.',markersize=3,alpha=0.5,label=name)
        ax.set_ylabel(f'{var_names[var]} [{unit}]')
        ax.set_title(f'{h} m')
        ax.grid()
    axs[0,0].legend()
    for ax in axs[-1,:]:
        ax.set_xlabel('Time (UTC)')
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    fig.suptitle(f'{var_names[var]} time series')
    plt.tight_layout()
    # plt.savefig(os.path.join(cd,f'figures/tropoe_comp/timeseries.{var}.png'))
    # plt.close(fig)

#diurnal mean difference heatmap (all heights up to height_max)
for n1,n2 in itertools.combinations(data_full,2):
    ds2_aligned=data_full[n2].reindex(time=data_full[n1].time,method='nearest',tolerance=time_tolerance)
    heights_m=data_full[n1].height.values*1000

    fig,axs=plt.subplots(1,2,figsize=(14,6))
    for ax,(var,unit) in zip(axs,units.items()):
        diff=data_full[n1][var]-ds2_aligned[var]
        diff_hourly=diff.groupby('time.hour').mean('time').transpose('height','hour')
        vmax=np.nanmax(np.abs(diff_hourly.values))

        im=ax.pcolormesh(np.arange(24),heights_m,diff_hourly.values,shading='nearest',cmap='seismic',vmin=-vmax,vmax=vmax)
        cb=plt.colorbar(im,ax=ax)
        cb.set_label(f'{var_names[var]} difference [{unit}]')
        ax.set_xlabel('Hour (UTC)')
        ax.set_ylabel('Height[m a.g.l.]')
        ax.set_title(var_names[var])
        ax.set_xticks(np.arange(0,24,3))

    fig.suptitle(f'Mean difference by hour of day: {n1} - {n2}')
    plt.tight_layout()
    n1_clean=re.sub(r'\W+','_',n1)
    n2_clean=re.sub(r'\W+','_',n2)
    # plt.savefig(os.path.join(cd,f'figures/tropoe_comp/{n1_clean}_vs_{n2_clean}.diurnal_diff.png'))
    # plt.close(fig)
