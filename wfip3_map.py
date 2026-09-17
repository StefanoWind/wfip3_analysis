# -*- coding: utf-8 -*-
"""
Plot WFIP3 site layout over aerial imagery
"""

import os
cd=os.getcwd()
import truststore
truststore.inject_into_ssl()#use OS certificate store (needed behind corporate TLS-inspecting proxies, e.g. Netskope)
import pandas as pd
import geopandas as gpd
import contextily as cx
from matplotlib import pyplot as plt
import warnings
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 14

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
source='data/WFIP3_layout.xlsx'
margin=0.25#[deg] padding around the sites' bounding box
lon_range=(-80,-60)#[deg] sanity bounds on longitude, used to flag bad entries
lat_range=(35,45)#[deg] sanity bounds on latitude, used to flag bad entries

#%% Initialization
layout=pd.read_excel(source)

valid=layout['Longitude'].between(*lon_range)&layout['Latitude'].between(*lat_range)
if (~valid).any():
    print(f"Dropping sites with out-of-range coordinates: {layout.loc[~valid,'Site'].tolist()}")
layout=layout[valid].reset_index(drop=True)

#%% Main
gdf=gpd.GeoDataFrame(layout,geometry=gpd.points_from_xy(layout.Longitude,layout.Latitude),crs='EPSG:4326')
gdf_web=gdf.to_crs(epsg=3857)

corners=gpd.GeoDataFrame(geometry=gpd.points_from_xy(
    [layout.Longitude.min()-margin,layout.Longitude.max()+margin],
    [layout.Latitude.min()-margin, layout.Latitude.max()+margin]),crs='EPSG:4326')
corners_web=corners.to_crs(epsg=3857)

fig,ax=plt.subplots(figsize=(10,10))
gdf_web.plot(ax=ax,color='r',markersize=100,edgecolor='k',alpha=1,zorder=3)
for x,y,site in zip(gdf_web.geometry.x,gdf_web.geometry.y,layout['Site']):
    ax.annotate(site,(x,y),xytext=(6,6),textcoords='offset points',fontsize=12,color='w',fontweight='bold',zorder=3)

ax.set_xlim(list(corners_web.geometry.x))
ax.set_ylim(list(corners_web.geometry.y))

cx.add_basemap(ax,source=cx.providers.Esri.WorldImagery)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('WFIP3 site layout')
plt.tight_layout()
