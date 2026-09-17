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
import utm
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 300
plt.close('all')

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')

heights=[0,7,26]#height array indices of interest
height_max=2000#[m] max height for the diurnal difference heatmap
time_tolerance=pd.Timedelta('5min')#max time difference for matching points across sites
variables=['temperature','waterVapor','gamma','cbh','rmsa','lwp']
var_names={'temperature':r'$T$','waterVapor':r'$r$'}
units={'temperature':r'$^\circ$C','waterVapor':'g kg$^{-1}$'}
limits={'temperature':[-20,40],'waterVapor':[0,20]}

#barge on-station GPS filtering
barge_gps_source=os.path.join(cd,'data','WHOI_WFIP3_barge_bowstern_GPS_22-Oct-2024.dat')
barge_gps_headers='yyyy mm dd HH MM SS bow_lon bow_lat stern_lon stern_lat'
barge_lat0,barge_lon0=40.9015,-70.787#[deg] nominal barge position
barge_max_dist=600#[m] max distance from the nominal position to be considered "on station"
barge_gps_tolerance=pd.Timedelta('15min')#max gap between a TROPoe sample and a GPS fix

#%% Initialization
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

os.makedirs(os.path.join(cd,'figures/tropoe_comp'),exist_ok=True)

#barge position: flag the times when it was on station, and the corresponding time spans
barge_gps=pd.read_csv(barge_gps_source,sep=r'\s+',header=None,names=barge_gps_headers.split(' '))
barge_gps=barge_gps.replace(999,np.nan).dropna(subset=['bow_lat','bow_lon'])
barge_gps_time=pd.to_datetime(dict(year=barge_gps['yyyy'],month=barge_gps['mm'],day=barge_gps['dd'],
                                    hour=barge_gps['HH'],minute=barge_gps['MM'],second=barge_gps['SS']))
XY0=utm.from_latlon(barge_lat0,barge_lon0)
XY=utm.from_latlon(barge_gps['bow_lat'].values,barge_gps['bow_lon'].values)
barge_on_station=pd.Series(np.sqrt((XY[0]-XY0[0])**2+(XY[1]-XY0[1])**2)<barge_max_dist,
                            index=pd.DatetimeIndex(barge_gps_time)).sort_index()

flag=barge_on_station.values.astype(int)
starts=np.where(np.diff(flag)>0)[0]+1
ends=np.where(np.diff(flag)<0)[0]
if flag[0]==1:
    starts=np.r_[0,starts]
if flag[-1]==1:
    ends=np.r_[ends,len(flag)-1]
barge_on_station_spans=[(barge_on_station.index[s],barge_on_station.index[e]) for s,e in zip(starts,ends)]

#%% Main

#load data
data_full={}#full height profile up to height_max, used for the diurnal heatmap
data={}#sparse heights, used for the scatter and time series plots
for name,path in config['sources_comparison'].items():
    files=sorted(glob.glob(os.path.join(cd,path,'*.nc')))
    good_files=[]
    for f in files:
        try:
            with xr.open_dataset(f):
                pass
            good_files.append(f)
        except Exception as e:
            print(f'  skipping unreadable file {os.path.basename(f)}: {e}')
    files=good_files
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

def exclude_barge_off_station(ds):
    '''NaN out temperature/waterVapor while the barge was off station'''
    on_station=barge_on_station.reindex(ds.time.values,method='nearest',tolerance=barge_gps_tolerance).fillna(False)
    on_station=xr.DataArray(on_station.values,coords={'time':ds.time},dims='time')
    ds=ds.copy()
    ds['temperature']=ds['temperature'].where(on_station)
    ds['waterVapor']=ds['waterVapor'].where(on_station)
    return ds

#data with the barge's off-station samples excluded: used by every plot except the time series,
#which shows the barge's full record and instead marks the on-station periods
data_full_excl={name:(exclude_barge_off_station(ds) if name=='Barge' else ds) for name,ds in data_full.items()}
data_excl={name:ds.isel(height=heights) for name,ds in data_full_excl.items()}

#pairwise scatter comparisons: one figure per height/variable, grid of sites
#(lower triangle = linear regression, upper triangle = histogram of difference)
sites=list(data_excl.keys())
n_sites=len(sites)
if n_sites>1:
    for var,unit in units.items():
        for i_h in range(len(heights)):
            h=int(np.round(data_excl[sites[0]].height.isel(height=i_h).values*1000))#[m] actual height at this index

            #match data across all site pairs, and find the common histogram count range
            diff_bins=np.arange(-2,2.1,0.1)
            pairs={}
            max_count=0
            for i,j in itertools.combinations(range(n_sites),2):
                s1=data_excl[sites[i]][var].isel(height=i_h).to_series().rename('x')
                s2=data_excl[sites[j]][var].isel(height=i_h).to_series().rename('y')
                df=pd.merge_asof(s1.sort_index().reset_index(),s2.sort_index().reset_index(),
                                  on='time',tolerance=time_tolerance,direction='nearest').dropna()
                pairs[(i,j)]=df
                counts,_=np.histogram((df['y']-df['x']).values,bins=diff_bins)
                max_count=max(max_count,counts.max() if len(counts)>0 else 0)

            fig,axs=plt.subplots(n_sites,n_sites,figsize=(2.5*n_sites+2,2.5*n_sites+2))
            for i in range(n_sites):
                for j in range(n_sites):
                    ax=axs[i,j]
                    if i==j:
                        ax.axis('off')
                        continue
                    elif i>j:#lower triangle: linear regression (x=sites[j], y=sites[i]), same axis limits everywhere
                        df=pairs[(j,i)]
                        x=df['x'].values
                        y=df['y'].values
                        ax.plot(x,y,'.k',alpha=0.3,markersize=5)
                        ax.plot(limits[var],limits[var],'--',color='gray',linewidth=1)#1:1 line
                        if len(x)>1:
                            p=np.polyfit(x,y,1)
                            ax.plot(limits[var],np.polyval(p,limits[var]),'r')
                            r2=np.corrcoef(x,y)[0,1]**2
                            ax.text(0.05,0.95,f'y={p[0]:.2f}x+{p[1]:.2f}\n$R^2$={r2:.2f}\nN={len(x)}',
                                    transform=ax.transAxes,va='top',fontsize=8)
                        ax.set_xlim(limits[var])
                        ax.set_ylim(limits[var])
                        ax.grid()
                        xlabel=f'{var_names[var]} at {sites[j]} [{unit}]'
                        ylabel=f'{var_names[var]} at {sites[i]} [{unit}]'
                    else:#upper triangle: histogram of the difference between sites, with bias and RMSE stamped on top
                        df=pairs[(i,j)]
                        diff=(df['y']-df['x']).values
                        ax.hist(diff,bins=diff_bins,color='gray')
                        if len(diff)>0:
                            bias=np.nanmean(diff)
                            rmse=np.sqrt(np.nanmean(diff**2))
                            ax.text(0.05,0.95,f'Bias={bias:.2f}\nRMSE={rmse:.2f}',
                                    transform=ax.transAxes,va='top',fontsize=8)
                        ax.set_title(f'{sites[j]}$-${sites[i]}',fontsize=9)
                        ax.set_xlim(diff_bins[0],diff_bins[-1])
                        ax.set_ylim(0,max_count*1.05 if max_count>0 else 1)
                        ax.grid()
                        xlabel=f'$\\Delta$ {var_names[var]} [{unit}]'
                        ylabel='Count'

                    #y label/ticks only on the leftmost non-diagonal cell of the row,
                    #x label/ticks only on the bottommost non-diagonal cell of the column,
                    #and both on every cell touching the diagonal
                    show_y=j==min(k for k in range(n_sites) if k!=i) or j==i+1
                    show_x=i==max(k for k in range(n_sites) if k!=j) or i==j-1
                    if show_x:
                        ax.set_xlabel(xlabel)
                    else:
                        ax.tick_params(labelbottom=False)
                    if show_y:
                        ax.set_ylabel(ylabel)
                    else:
                        ax.tick_params(labelleft=False)

            fig.suptitle(f'{var_names[var]} at {h} m')
            plt.tight_layout()
            plt.savefig(os.path.join(cd,f'figures/tropoe_comp/{var}.{h}m.regression_grid.png'))
            plt.close(fig)

#time series at each height, stacked vertically
tmin=min(data[name].time.min().values for name in data)
tmax=max(data[name].time.max().values for name in data)
for var,unit in units.items():
    fig,axs=plt.subplots(len(heights),1,figsize=(10,3*len(heights)),sharex=True)
    axs=np.atleast_1d(axs)
    for ax,i_h in zip(axs,range(len(heights))):
        h=int(np.round(list(data.values())[0].height.isel(height=i_h).values*1000))#[m] actual height at this index
        for ctr,(s,e) in enumerate(barge_on_station_spans):
            ax.axvspan(s,e,facecolor='lightblue',hatch='//',edgecolor='gray',alpha=0.3,zorder=0,
                       label='Barge on station' if (i_h==0 and ctr==0) else None)
        for name in data:
            ax.plot(data[name].time,data[name][var].isel(height=i_h),'.',markersize=3,alpha=0.5,label=name)
        ax.set_xlim(tmin,tmax)
        ax.set_ylabel(f'{var_names[var]} [{unit}]')
        ax.set_title(f'{h} m')
        ax.grid()
        if i_h<len(heights)-1:
            ax.tick_params(labelbottom=False)
    axs[0].legend()
    axs[-1].set_xlabel('Time (UTC)')
    axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[-1].xaxis.get_major_locator()))

    fig.suptitle(f'{var_names[var]} time series')
    plt.tight_layout()
    plt.savefig(os.path.join(cd,f'figures/tropoe_comp/timeseries.{var}.png'))
    plt.close(fig)

#diurnal mean difference heatmaps: single figure, grid of sites
#(lower triangle = temperature difference, upper triangle = mixing ratio difference)
sites_full=list(data_full_excl.keys())
n_sites_full=len(sites_full)
if n_sites_full>1:
    fig,axs=plt.subplots(n_sites_full,n_sites_full,figsize=(3.5*n_sites_full+2,3.5*n_sites_full+2))
    im_temp=None
    im_water=None
    for i in range(n_sites_full):
        for j in range(n_sites_full):
            ax=axs[i,j]
            if i==j:
                ax.axis('off')
                continue

            var='temperature' if i>j else 'waterVapor'
            unit=units[var]
            ds_i=data_full_excl[sites_full[i]]
            ds_j_aligned=data_full_excl[sites_full[j]].reindex(time=ds_i.time,method='nearest',tolerance=time_tolerance)
            heights_m=ds_i.height.values*1000
            diff=ds_i[var]-ds_j_aligned[var]
            diff_hourly=diff.groupby('time.hour').mean('time').transpose('height','hour')

            im=ax.pcolormesh(diff_hourly.hour.values,heights_m,diff_hourly.values,shading='nearest',cmap='seismic',vmin=-2,vmax=2)
            ax.set_title(f'{sites_full[i]}$-${sites_full[j]}',fontsize=9)
            ax.set_xlim(0,23)
            ax.set_xticks(np.arange(0,24,6))

            #y label/ticks only on the leftmost non-diagonal cell of the row,
            #x label/ticks only on the bottommost non-diagonal cell of the column,
            #and both on every cell touching the diagonal
            show_y=j==min(k for k in range(n_sites_full) if k!=i) or j==i+1
            show_x=i==max(k for k in range(n_sites_full) if k!=j) or i==j-1
            if show_x:
                ax.set_xlabel('Hour (UTC)')
            else:
                ax.tick_params(labelbottom=False)
            if show_y:
                ax.set_ylabel('Height [m a.g.l.]')
            else:
                ax.tick_params(labelleft=False)

            if i>j:
                im_temp=im
            else:
                im_water=im

    fig.suptitle('Mean difference by hour of day (row site $-$ column site)')
    fig.subplots_adjust(hspace=0.5,wspace=0.5,right=0.88,bottom=0.12)
    grid_left,grid_top=axs[0,0].get_position().x0,axs[0,0].get_position().y1
    grid_right,grid_bottom=axs[-1,-1].get_position().x1,axs[-1,-1].get_position().y0
    if im_temp is not None:#horizontal colorbar spanning the full width, at the bottom
        cax_temp=fig.add_axes([grid_left,0.04,grid_right-grid_left,0.02])
        fig.colorbar(im_temp,cax=cax_temp,orientation='horizontal',
                     label=f'{var_names["temperature"]} difference [{units["temperature"]}]')
    if im_water is not None:#vertical colorbar spanning the full height, on the right
        cax_water=fig.add_axes([0.91,grid_bottom,0.02,grid_top-grid_bottom])
        fig.colorbar(im_water,cax=cax_water,label=f'{var_names["waterVapor"]} difference [{units["waterVapor"]}]')
    plt.savefig(os.path.join(cd,'figures/tropoe_comp/diurnal_diff_grid.png'))
    plt.close(fig)

#site maps of overall bias/RMSE relative to a reference site: one figure per variable per statistic,
#grid of reference site (columns) x height (rows), each subplot maps all sites colored by the statistic
import cartopy.io.shapereader as shpreader
from shapely.geometry import box as shapely_box

layout=pd.read_excel(os.path.join(cd,'data','WFIP3_layout.xlsx')).set_index('Site')
layout=layout.loc[[s for s in sites if s in layout.index]]

if n_sites>1 and len(layout)==n_sites:
    lons=np.array([layout.loc[s,'Longitude'] for s in sites])
    lats=np.array([layout.loc[s,'Latitude'] for s in sites])
    margin=0.1#[deg] padding around the sites' bounding box
    xlim=(lons.min()-margin,lons.max()+margin)
    ylim=(lats.min()-margin,lats.max()+margin)
    aspect=1/np.cos(np.deg2rad(lats.mean()))#approximate equal-distance aspect ratio at this latitude

    #simple coastline, clipped to the sites' bounding box
    bbox=shapely_box(xlim[0],ylim[0],xlim[1],ylim[1])
    coast_path=shpreader.natural_earth(resolution='10m',category='physical',name='coastline')
    coast_lines=[]
    for rec in shpreader.Reader(coast_path).records():
        clipped=rec.geometry.intersection(bbox)
        if clipped.is_empty:
            continue
        for g in (clipped.geoms if hasattr(clipped,'geoms') else [clipped]):
            coast_lines.append(np.asarray(g.coords))

    for var,unit in units.items():
        #bias and RMSE of every site relative to every other site, at every height of interest
        bias=np.zeros((len(heights),n_sites,n_sites))
        rmse=np.zeros((len(heights),n_sites,n_sites))
        for i_h in range(len(heights)):
            for i,j in itertools.combinations(range(n_sites),2):
                s1=data_excl[sites[i]][var].isel(height=i_h).to_series().rename('x')
                s2=data_excl[sites[j]][var].isel(height=i_h).to_series().rename('y')
                df=pd.merge_asof(s1.sort_index().reset_index(),s2.sort_index().reset_index(),
                                  on='time',tolerance=time_tolerance,direction='nearest').dropna()
                diff=(df['y']-df['x']).values
                bias[i_h,i,j]=np.nanmean(diff)
                bias[i_h,j,i]=-bias[i_h,i,j]
                rmse[i_h,i,j]=rmse[i_h,j,i]=np.sqrt(np.nanmean(diff**2))

        for stat_name,stat,cmap in [('Bias',bias,'seismic'),('RMSE',rmse,'Reds')]:
            off_diag=~np.eye(n_sites,dtype=bool)
            vmax=np.nanmax(np.abs(stat[:,off_diag]))
            vmin=-vmax if stat_name=='Bias' else 0

            fig,axs=plt.subplots(len(heights),n_sites,figsize=(3.2*n_sites,3.2*len(heights)),squeeze=False)
            for i_h in range(len(heights)):
                h=int(np.round(data_excl[sites[0]].height.isel(height=i_h).values*1000))
                for i_ref in range(n_sites):
                    ax=axs[i_h,i_ref]
                    for line in coast_lines:
                        ax.plot(line[:,0],line[:,1],color='k',linewidth=0.8,zorder=1)
                    other=np.arange(n_sites)!=i_ref
                    sc=ax.scatter(lons[other],lats[other],c=stat[i_h,i_ref,other],cmap=cmap,vmin=vmin,vmax=vmax,
                                  s=150,edgecolor='k',zorder=3)
                    ax.scatter(lons[i_ref],lats[i_ref],color='k',s=250,marker='*',zorder=4)
                    for x,y,name in zip(lons,lats,sites):
                        ax.annotate(name,(x,y),xytext=(4,4),textcoords='offset points',fontsize=7)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_aspect(aspect)
                    ax.set_title(f'Ref: {sites[i_ref]}, {h} m',fontsize=9)
                    if i_h==len(heights)-1:
                        ax.set_xlabel('Longitude [deg]')
                    else:
                        ax.tick_params(labelbottom=False)
                    if i_ref==0:
                        ax.set_ylabel('Latitude [deg]')
                    else:
                        ax.tick_params(labelleft=False)

            fig.suptitle(f'{stat_name} of {var_names[var]} relative to reference site')
            fig.subplots_adjust(right=0.9)
            grid_top=axs[0,0].get_position().y1
            grid_bottom=axs[-1,-1].get_position().y0
            cax=fig.add_axes([0.93,grid_bottom,0.02,grid_top-grid_bottom])#colorbar spanning the full height
            fig.colorbar(sc,cax=cax,label=f'{stat_name} [{unit}]')
            plt.savefig(os.path.join(cd,f'figures/tropoe_comp/{var}.{stat_name.lower()}_map.png'))
            plt.close(fig)
