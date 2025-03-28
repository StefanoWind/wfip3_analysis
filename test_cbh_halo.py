# -*- coding: utf-8 -*-
'''
Test CBH from single lidar scan
'''

import os
cd=os.getcwd()
import sys
from matplotlib import pyplot as plt
import xarray as xr
import glob
import numpy as np
import matplotlib
import matplotlib.dates as mdates
import warnings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

warnings.filterwarnings('ignore')

#%% Inputs
source=os.path.join(cd,'data/wfip3/barg.lidar.z03.a0/barg.lidar.z03.a0.20240609.010016.stare.nc')

path_cbh='C:/Users/SLETIZIA/codes/TROPoe_processor/utils'
path_utils='C:/Users/SLETIZIA/OneDrive - NREL/Desktop/Main/utils'

#%% Initialization
sys.path.append(path_cbh)
sys.path.append(path_utils)

import utils as utl
import cbh_halo as cbh

#%% Main
time_avg,cbh_avg=cbh.compute_cbh(source,utl,averages=60)