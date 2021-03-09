# Author: Xia Lin
# Date:   Feb 2021
# Thanks to François Massonnet for the help on metrics calculation!
# Contents:
# 1) A function that interpolates the input ice concentration data into the NSIDC-0051 grid, 
#    the input data is a numpy array but a NetCDF file 
# 2) A function that computes sea ice concentration errors between two datasets
#    on the mean state, interannual variability and trend in order to get the metrics;
#    The input data is a numpy array but a NetCDF file 
# 3) The heatmap function  
# 4) The annotate heatmap function
# 5) A script deals with the input NetCDF data, and then calls the function 1) 
# 6) A script calls the function 2) and then computes the ice concentration metrics
# 7) A script calls the functions 3) and 4) to plot the ice concentration metrics

# ------------------------------------
# PART 1) The interpolation function  |
# ------------------------------------
def compute_interp(lat,lon,field):
  ''' Input: -latitude, longitude of the original grid
             -ice variable on the original grid

      Output: Interpolated the ice variable to the NSIDC-0051 polar stereographic grid
  '''
  if np.min(lat) > 0:
    # Northern Hemisphere
    # load lat-lon of the target grid
    access_pr_file = '/sea ice data/OBS/siconc/NSIDC-0051/siconc_r1i1p1_mon_197901-201712_nh-psn25.nc'
    dset = xr.open_dataset(access_pr_file)
    NHlat_curv = np.array(dset['latitude'][:,:])
    NHlon_curv = np.array(dset['longitude'][:,:])
    #Create a pyresample object holding the origin grid:
    orig_def = pyresample.geometry.SwathDefinition(lons=lon, lats=lat)
    #Create another pyresample object for the target (curvilinear) grid:
    targ_def = pyresample.geometry.SwathDefinition(lons=NHlon_curv, lats=NHlat_curv)
    #siconc_nearest interp
    NHconcentration = np.zeros((336,304,448))
    NHconcentration = np.array([pyresample.kd_tree.resample_nearest(orig_def, np.array(field[i, :, :]), targ_def, radius_of_influence = 500000, fill_value=None) for i in range(336)])
    idx = np.where(NHconcentration > 1000.00)
    NHconcentration[idx] = np.nan
    return NHconcentration
  
  elif np.max(lat) < 0:
    # Southern Hemisphere
    # load lat-lon of the target grid
    access_pr_file = '/sea ice data/OBS/siconc/NSIDC-0051/siconc_r1i1p1_mon_197901-201712_sh-pss25.nc'
    dset = xr.open_dataset(access_pr_file)
    SHlat_curv = np.array(dset['latitude'][:,:])
    SHlon_curv = np.array(dset['longitude'][:,:])
    #Create a pyresample object holding the origin grid:
    orig_def = pyresample.geometry.SwathDefinition(lons=lon, lats=lat)
    #Create another pyresample object for the target (curvilinear) grid:
    targ_def = pyresample.geometry.SwathDefinition(lons=SHlon_curv, lats=SHlat_curv)
    #sic_nearest interp
    SHconcentration = np.zeros((336,316,332))
    SHconcentration = np.array([pyresample.kd_tree.resample_nearest(orig_def, np.array(field[i, :, :]), targ_def, radius_of_influence = 500000,  fill_value=None) for i in range(336)])
    idx = np.where(SHconcentration > 1000.00)
    SHconcentration[idx] = np.nan
    return SHconcentration
  
  else:
    # Northern Hemisphere
    # load lat-lon of the target grid
    access_pr_file = '/sea ice data/OBS/siconc/NSIDC-0051/siconc_r1i1p1_mon_197901-201712_nh-psn25.nc'
    dset = xr.open_dataset(access_pr_file)
    NHlat_curv = np.array(dset['latitude'][:,:])
    NHlon_curv = np.array(dset['longitude'][:,:])
    #Create a pyresample object holding the origin grid:
    orig_def = pyresample.geometry.SwathDefinition(lons=lon, lats=lat)
    #Create another pyresample object for the target (curvilinear) grid:
    targ_def = pyresample.geometry.SwathDefinition(lons=NHlon_curv, lats=NHlat_curv)
    #siconc_nearest interp
    NHconcentration = np.zeros((336,304,448))
    NHconcentration = np.array([pyresample.kd_tree.resample_nearest(orig_def, np.array(field[i, :, :]), targ_def, radius_of_influence = 500000, fill_value=None) for i in range(336)])
    idx = np.where(NHconcentration > 1000.00)
    NHconcentration[idx] = np.nan

    # Southern Hemisphere
    # load lat-lon of the target grid
    access_pr_file = '/sea ice data/OBS/siconc/NSIDC-0051/siconc_r1i1p1_mon_197901-201712_sh-pss25.nc'
    dset = xr.open_dataset(access_pr_file)
    SHlat_curv = np.array(dset['latitude'][:,:])
    SHlon_curv = np.array(dset['longitude'][:,:])
    #Create a pyresample object holding the origin grid:
    orig_def = pyresample.geometry.SwathDefinition(lons=lon, lats=lat)
    #Create another pyresample object for the target (curvilinear) grid:
    targ_def = pyresample.geometry.SwathDefinition(lons=SHlon_curv, lats=SHlat_curv)
    #sic_nearest interp
    SHconcentration = np.zeros((336,316,332))
    SHconcentration = np.array([pyresample.kd_tree.resample_nearest(orig_def, np.array(field[i, :, :]), targ_def, radius_of_influence = 500000,  fill_value=None) for i in range(336)])
    idx = np.where(SHconcentration > 1000.00)
    SHconcentration[idx] = np.nan
    return NHconcentration, SHconcentration

# -----------------------------------------------
# PART 2) The ice concentration errors function  |   
# -----------------------------------------------
def compute_siconc_metrics(concentration, concentration1, cellarea):
  ''' Input: - sea ice concentration (%) in the Arctic or Antarctic from two datasets
             - cellarea: array of grid cell areas in the Arctic or Antarctic (square meters)

      Output: Errors between two ice concentration datasets of mean cycle, anomaly variance and trend in the Arctic or Antarctic
  '''
  #Mean cycle, anomaly varaiance & trend
  #=========================================
  print('             ')
  print('1. MEAN CYCLE')
  print('=============')
  #1. Evaluation of the mean seasonal cycle
  #Here we conpute the mean monhly concentrations over the whole period
  nt, ny, nx = concentration.shape
  #Calculate the mean cycle...
  conc = np.array([np.nanmean(concentration[m::12,:,:], axis=0) for m in range(12)])
  conc1 = np.array([np.nanmean(concentration1[m::12,:,:], axis=0) for m in range(12)])
  #Now we prepare, for each cell, a variable containing the abs value of errors
  #Calculate errors on mean cycle...
  error_mean_conc=np.array([abs(conc[m,:,:] - conc1[m,:,:]) for m in range(12)])
  #Calculate the global error by hemisphere
  #We don't want take into account cells that are ice free (e.g. in tropics ) or with nan values. So creat two masks.
  mask=np.zeros(((12,ny,nx)))
  for jt in range(12):
    for jy in np.arange(ny):
      for jx in np.arange(nx):
        if conc[jt,jy,jx] == 0 and conc1[jt,jy,jx] == 0 :  
          mask[jt,jy,jx] = 0
        elif np.isnan(conc[jt,jy,jx]) or np.isnan(conc1[jt,jy,jx]):
          mask[jt,jy,jx] = 0
        else:
          mask[jt,jy,jx] = 1.0
  #Calculate the global error by hemisphere
  error_mean_monthly = np.array([(np.nansum(error_mean_conc[m,:,:]*cellarea*mask[m,:,:])/np.nansum(cellarea*mask[m,:,:])) for m in range(12)])
  ndpm=[31,28,31,30,31,30,31,31,30,31,30,31];
  error_mean=np.sum(error_mean_monthly*ndpm)/np.sum(ndpm)
  print(error_mean)
 
  #=======================================================================
  print('                ')
  print('ANOMALY VARIANCE')
  print('================')
  #2.Evaluate the variance of anomalies. The anomalies are defined as the signal minus the mean seasonal cycle
  nt, ny, nx = concentration.shape
  #Compute anomalies
  conc = np.array([np.nanmean(concentration[m::12,:,:], axis=0) for m in range(12)])
  conc1 = np.array([np.nanmean(concentration1[m::12,:,:], axis=0) for m in range(12)])
  ano_conc = np.array([(concentration[j,:,:] - conc[j%12,:,:]) for j in range(nt)])
  ano_conc1 = np.array([(concentration1[j,:,:] - conc1[j%12,:,:]) for j in range(nt)])
  #Compute std of anomalies
  std_ano_conc=np.nanstd(ano_conc,axis=0)
  std_ano_conc1=np.nanstd(ano_conc1,axis=0)
  #Compute errors on std
  error_std_conc=np.array(abs(std_ano_conc-std_ano_conc1))
  #Creat masks
  mask_std_conc=np.zeros((ny,nx))
  for jy in np.arange(ny):
    for jx in np.arange(nx):
      if std_ano_conc[jy,jx] == 0 and std_ano_conc1[jy,jx] ==0 :
        mask_std_conc[jy,jx] = 0
      elif np.isnan(std_ano_conc[jy,jx]) or np.isnan(std_ano_conc1[jy,jx]):
        mask_std_conc[jy,jx] = 0
      else:
        mask_std_conc[jy,jx] = 1.0
  #Compute global error on std
  error_std=np.nansum(error_std_conc*cellarea*mask_std_conc)/np.nansum(cellarea*mask_std_conc)
  print(error_std)
  
  #========================================================================
  print('     ')
  print('TREND')
  print('=====')
  #3.Evaluate the skill in each cell to reproduce the trend 
  months = np.arange(1, 337, 1)
  nt, ny, nx = concentration.shape
  #compute the trend of each cell
  trend_ano_conc=np.zeros((ny,nx))
  trend_ano_conc1=np.zeros((ny,nx))
  for jy in np.arange(ny):
    for jx in np.arange(nx):
      ano_concc=np.array(ano_conc[:,jy,jx])
      ano_concc1=np.array(ano_conc1[:,jy,jx])
      if (np.isnan(np.mean(ano_concc))) or (np.isnan(np.mean(ano_concc1))):
        trend_ano_conc[jy,jx]=np.nan
        trend_ano_conc1[jy,jx]=np.nan
      else:
        coeff = np.polyfit(months, ano_concc, 1)#1linear function
        trend_ano_conc[jy,jx]=coeff[0]
        coeff1 = np.polyfit(months, ano_concc1, 1)
        trend_ano_conc1[jy,jx]=coeff1[0]
  #Calculate error on trend
  error_trend_conc=12*10*abs(trend_ano_conc-trend_ano_conc1)#/decade
  #creat masks
  mask_trend_conc=np.zeros((ny,nx))
  for jy in np.arange(ny):
    for jx in np.arange(nx):
      if (trend_ano_conc[jy,jx] == 0) and (trend_ano_conc1[jy,jx]) ==0 :
        mask_trend_conc[jy,jx] = 0
      elif (np.isnan(trend_ano_conc[jy,jx])) or (np.isnan(trend_ano_conc1[jy,jx])):
        mask_trend_conc[jy,jx] = 0
      else:
        mask_trend_conc[jy,jx] = 1.0
  #Compute global error
  error_trend=np.nansum(error_trend_conc*cellarea*mask_trend_conc)/np.nansum(cellarea*mask_trend_conc)
  print(error_trend)
  return error_mean, error_std, error_trend

# ------------------------------
# PART 3) The heatmap function  |
# ------------------------------
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    #ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")#30

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

# ---------------------------------------
# PART 4) The annotate heatmap function  |
# ---------------------------------------
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            #kw.update(color=textcolors[0])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# --------------------------------------------------
# PART 5) A script deals with the input NetCDF data |
# --------------------------------------------------   
import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
#for interpolation (you will have to install pyresample first)
import pyresample
import netCDF4 as nc
from netCDF4 import Dataset
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.basemap import Basemap, addcyclic
# Read data and interp into same grid
# ----------------------------
# Load NSIDC0051 NH &SH siconc
# ----------------------------
access_pr_file = '/sea ice data/OBS/siconc/NSIDC-0051/siconc_r1i1p1_mon_197901-201712_nh-psn25.nc'
dset = xr.open_dataset(access_pr_file)
NHlat_curv = np.array(dset['latitude'][:,:])
NHlon_curv =np.array(dset['longitude'][:,:])
NHconcentration = np.array(dset['siconc'][12:348,:,:])
NHcellarea = np.array(dset['areacello'][:,:])

access_pr_file = '/sea ice data/OBS/siconc/NSIDC-0051/siconc_r1i1p1_mon_197901-201712_sh-pss25.nc'
dset = xr.open_dataset(access_pr_file)
SHlat_curv = np.array(dset['latitude'][:,:])
SHlon_curv =np.array(dset['longitude'][:,:])
SHconcentration = np.array(dset['siconc'][12:348,:,:])
SHcellarea = np.array(dset['areacello'][:,:])
NHconcentration[91,:,:]=np.nanmean(NHconcentration[7::12,:,:], axis=0)
NHconcentration[143,:,:]=np.nanmean(NHconcentration[11::12,:,:], axis=0)
SHconcentration[91,:,:]=np.nanmean(SHconcentration[7::12,:,:], axis=0)
SHconcentration[143,:,:]=np.nanmean(SHconcentration[11::12,:,:], axis=0)
np.savez('NSIDC0051_1980_2007_siconc.npz', NHlat_curv, NHlon_curv, NHconcentration, SHlat_curv, SHlon_curv, SHconcentration, NHcellarea, SHcellarea)                                               

#------------------------------------------------------------------------------------------------------------------------------------
# Load OSI450 NH &SH siconc and interp into NSIDC0051 the Polar stereographic projection at a grid cell size of 25 x 25 km
#------------------------------------------------------------------------------------------------------------------------------------
# Northern Hemisphere
access_pr_file = '/sea ice data/OBS/siconc/OSI-450/ice_conc_nh_ease2-250_cdr-v2p0_mon_198001-201512.nc'
dset = xr.open_dataset(access_pr_file)
lat = np.array(dset['lat'][:,:])
lon =np.array(dset['lon'][:,:])
field = np.array(dset['ice_conc'][0:336,:,:])
NHconcentration=compute_interp(lat,lon,field)
# Southern Hemisphere
access_pr_file = '/sea ice data/OBS/siconc/OSI-450/ice_conc_sh_ease2-250_cdr-v2p0_mon_198001-201512.nc'
dset = xr.open_dataset(access_pr_file)
lat = np.array(dset['lat'][:,:])
lon =np.array(dset['lon'][:,:])
field = np.array(dset['ice_conc'][0:336,:,:])
SHconcentration=compute_interp(lat,lon,field)
NHconcentration[75,:,:]=np.nanmean(NHconcentration[3::12,:,:], axis=0)
NHconcentration[76,:,:]=np.nanmean(NHconcentration[4::12,:,:], axis=0)
SHconcentration[75,:,:]=np.nanmean(SHconcentration[3::12,:,:], axis=0)
SHconcentration[76,:,:]=np.nanmean(SHconcentration[4::12,:,:], axis=0)
np.savez('OSI450_1980_2007_siconc.npz', NHlat_curv, NHlon_curv, NHconcentration, SHlat_curv, SHlon_curv, SHconcentration)                                               

#---------------------------------------------------------------------------------------------------------------------------------
# Load 1980-2007 model output data and interp into NSIDC0051 the Polar stereographic projection at a grid cell size of 25 x 25 km |
#---------------------------------------------------------------------------------------------------------------------------------
path='/sea ice data/OMIP/All OMIP data/r1i1p1f1/siconc/1980_2007/'
files=os.listdir(path)
for file in files:
  print(file)
  dset = xr.open_dataset(os.path.join(path,file))
  if file.startswith('siconc_SImon_IPSL'):
    lat=np.array(dset['nav_lat'][:,:])
    lon=np.array(dset['nav_lon'][:,:])
    field = np.array(np.squeeze(dset['siconc'][:,:,:]))
  elif file.startswith('siconc_SImon_GFDL-CM4'):
    lat0=np.array(dset['lat'])
    lon0=np.array(dset['lon'])
    field0 = np.array(np.squeeze(dset['siconc'][:,:,:]))
    idx1=np.where(lon0<0)
    lon0[idx1]=lon0[idx1]+360
    lon=np.zeros((1080,1440))
    lat=np.zeros((1080,1440))
    field=np.zeros(((336,1080,1440)))
    lon[:,0:960]=lon0[:,480:1440]
    lon[:,960:1440]=lon0[:,0:480]
    lat[:,0:960]=lat0[:,480:1440]
    lat[:,960:1440]=lat0[:,0:480]
    field[:,:,0:960]=field0[:,:,480:1440]
    field[:,:,960:1440]=field0[:,:,0:480]
  elif file.startswith('siconc_SImon_GFDL-OM4p5B'):
    lat0=np.array(dset['lat'])
    lon0=np.array(dset['lon'])
    field0 = np.array(np.squeeze(dset['siconc'][:,:,:]))
    idx1=np.where(lon0<0)
    lon0[idx1]=lon0[idx1]+360
    lon=np.zeros((576,720))
    lat=np.zeros((576,720))
    field=np.zeros(((336,576,720)))
    lon[:,0:480]=lon0[:,240:720]
    lon[:,480:720]=lon0[:,0:240]
    lat[:,0:480]=lat0[:,240:720]
    lat[:,480:720]=lat0[:,0:240]
    field[:,:,0:480]=field0[:,:,240:720]
    field[:,:,480:720]=field0[:,:,0:240]
  else:
    lat=np.array(dset['latitude'][:,:])
    lon=np.array(dset['longitude'][:,:])
    field = np.array(np.squeeze(dset['siconc'][:,:,:]))
  
  idx=np.where((abs(field)>10000))
  field[idx]=np.nan
  idx1=np.where(abs(lon)>10000)
  lon[idx1]=np.nan
  lat[idx1]=np.nan
  a = lon.shape[0]
  b = lon.shape[1]
  for i in range(a):
    for j in range(b):
      if lon[i,j]>180:
        lon[i,j] = lon[i,j]-360

  siconc=compute_interp(lat,lon,field)
  NHconcentration=siconc[0]
  SHconcentration=siconc[1]
  np.savez(file[13:31]+'_1980_2007_siconc.npz', NHlat_curv, NHlon_curv, NHconcentration, SHlat_curv, SHlon_curv, SHconcentration)

# --------------------------------------------------------
# PART 6) A script computes the ice concentration metrics |
# --------------------------------------------------------
#typical errors-differences between two observations
a=np.load('NSIDC0051_1980_2007_siconc.npz')
NHconcentration1=a['arr_2']/100 
SHconcentration1=a['arr_5']/100
NHcellarea=a['arr_6']
SHcellarea=a['arr_7']
a=np.load('OSI450_1980_2007_siconc.npz')
NHconcentration2=a['arr_2']/100
SHconcentration2=a['arr_5']/100
NHtyerror=compute_siconc_metrics(NHconcentration1, NHconcentration2, NHcellarea)
SHtyerror=compute_siconc_metrics(SHconcentration1, SHconcentration2, SHcellarea)
name=['CMCC-CM2-HR4_omip2_1980_2007_siconc.npz', 'CMCC-CM2-SR5_omip1_1980_2007_siconc.npz', 'CMCC-CM2-SR5_omip2_1980_2007_siconc.npz', 'EC-Earth3_omip1_r1_1980_2007_siconc.npz','EC-Earth3_omip2_r1_1980_2007_siconc.npz', 'GFDL-CM4_omip1_r1i_1980_2007_siconc.npz', 'GFDL-OM4p5B_omip1__1980_2007_siconc.npz', 'IPSL-CM6A-LR_omip1_1980_2007_siconc.npz',  'MIROC6_omip1_r1i1p_1980_2007_siconc.npz', 'MIROC6_omip2_r1i1p_1980_2007_siconc.npz', 'MRI-ESM2-0_omip1_r_1980_2007_siconc.npz', 'MRI-ESM2-0_omip2_r_1980_2007_siconc.npz', 'NorESM2-LM_omip1_r_1980_2007_siconc.npz', 'NorESM2-LM_omip2_r_1980_2007_siconc.npz']

NHerror_mean1=np.zeros(14)
SHerror_mean1=np.zeros(14)
NH_error_std1=np.zeros(14)
SH_error_std1=np.zeros(14)
NH_error_trend1=np.zeros(14)
SH_error_trend1=np.zeros(14)
Metrics_siconc=np.zeros((17, 6))
for obs in range(2):
  if obs==0:#models vs NSIDC-0051
    i='NSIDC0051'
    for num in range(14):
      a=np.load(path + name[num])
      NHconcentration=a['arr_2']/100
      SHconcentration=a['arr_5']/100
      NHMetrics=compute_siconc_metrics(NHconcentration, NHconcentration1, NHcellarea)
      SHMetrics=compute_siconc_metrics(SHconcentration, SHconcentration1, SHcellarea)
      NHerror_mean1[num]=NHMetrics[0]#NHerror_mean
      SHerror_mean1[num]=SHMetrics[0]#SHerror_mean
      NH_error_std1[num]=NHMetrics[1]#NH_error_std
      SH_error_std1[num]=SHMetrics[1]#SH_error_std
      NH_error_trend1[num]=NHMetrics[2]#NH_error_trend
      SH_error_trend1[num]=SHMetrics[2]#SH_error_trend
  else:#models vs OSI-450
    i='OSI450' 
    for num in range(14):
      a=np.load(path + name[num])
      NHconcentration=a['arr_2']/100
      SHconcentration=a['arr_5']/100
      NHMetrics=compute_siconc_metrics(NHconcentration, NHconcentration2, NHcellarea)
      SHMetrics=compute_siconc_metrics(SHconcentration, SHconcentration2, SHcellarea)
      NHerror_mean1[num]=NHMetrics[0]#NHerror_mean
      SHerror_mean1[num]=SHMetrics[0]#SHerror_mean
      NH_error_std1[num]=NHMetrics[1]#NH_error_std
      SH_error_std1[num]=SHMetrics[1]#SH_error_std
      NH_error_trend1[num]=NHMetrics[2]#NH_error_trend
      SH_error_trend1[num]=SHMetrics[2]#SH_error_trend

  Metrics_siconc[0:14,0]=NHerror_mean1/NHtyerror[0]
  Metrics_siconc[0:14,1]=NH_error_std1/NHtyerror[1]
  Metrics_siconc[0:14,2]=NH_error_trend1/NHtyerror[2]
  Metrics_siconc[0:14,3]=SHerror_mean1/SHtyerror[0]
  Metrics_siconc[0:14,4]=SH_error_std1/SHtyerror[1]
  Metrics_siconc[0:14,5]=SH_error_trend1/SHtyerror[2]
  Metrics_siconc[14,0]=np.mean(NHerror_mean1)/NHtyerror[0]
  Metrics_siconc[14,1]=np.mean(NH_error_std1)/NHtyerror[1]
  Metrics_siconc[14,2]=np.mean(NH_error_trend1)/NHtyerror[2]
  Metrics_siconc[14,3]=np.mean(SHerror_mean1)/SHtyerror[0]
  Metrics_siconc[14,4]=np.mean(SH_error_std1)/SHtyerror[1]
  Metrics_siconc[14,5]=np.mean(SH_error_trend1)/SHtyerror[2] 
  Metrics_siconc[15,:]=(Metrics_siconc[1,:]+Metrics_siconc[3,:]+Metrics_siconc[8,:]+Metrics_siconc[10,:]+Metrics_siconc[12,:])/5#OMIP1 mean
  Metrics_siconc[16,:]=(Metrics_siconc[2,:]+Metrics_siconc[4,:]+Metrics_siconc[9,:]+Metrics_siconc[11,:]+Metrics_siconc[13,:])/5#OMIP2 mean
  np.savez('siconc_metrics_'+str(i)+'.npz', Metrics_siconc, NHerror_mean1, SHerror_mean1, NH_error_std1, SH_error_std1, NH_error_trend1, SH_error_trend1)

# --------------------------------------------------------------
# PART 7) A script plots the ice concentration metrics (heatmap)|
# --------------------------------------------------------------
Models=['CMCC-CM2-HR4/2','CMCC-CM2-SR5/1','CMCC-CM2-SR5/2','EC-Earth3/1','EC-Earth3/2','GFDL-CM4/1','GFDL-OM4p5B/1','IPSL-CM6A-LR/1','MIROC6/1','MIROC6/2','MRI-ESM2-0/1','MRI-ESM2-0/2','NorESM2-LM/1','NorESM2-LM/2','Model mean','Model mean/1','Model mean/2']
Variables=['Mean Conc. North','Std Ano Conc. North','Trend Ano Conc. North','Mean Conc. South','Std Ano Conc. South','Trend Ano Conc. South']
for obs in range(2):
  if obs==0:#NSIDC-0051
    a=np.load('siconc_metrics_NSIDC0051.npz')
  else:#OSI450
    a=np.load('siconc_metrics_OSI450.npz')
  values=a['arr_0']
  dpi=100
  squaresize = 220
  figwidth = 18*squaresize/float(dpi)
  figheight = 6*squaresize/float(dpi)
  fig,ax1 = plt.subplots(1, figsize=(figwidth, figheight), dpi=dpi)
  im,cbar = heatmap(values, Models,Variables, ax=ax1, cmap="OrRd", vmin=1, vmax=5) 
  texts = annotate_heatmap(im, valfmt="{x:.2f}",size=16,threshold=3.5)
  cbar.remove()
  ax1.set_xticklabels(['Mean Conc. North','Std Ano Conc. North','Trend Ano Conc. North','Mean Conc. South','Std Ano Conc. South','Trend Ano Conc. South'])
  plt.setp(ax1.get_xticklabels(), fontname='Arial', fontsize=16)
  plt.setp(ax1.get_yticklabels(), fontname='Arial', fontsize=16)
  cax = fig.add_axes([0.75, 0.113, 0.01, 0.765])
  cbar = fig.colorbar(im, cax=cax,ticks=[1,1.5,2,2.5,3,3.5,4,4.5,5], orientation="vertical")
  cbar.ax.yaxis.set_ticks_position('both')
  cbar.ax.tick_params(direction='in',length=2,labelsize=16)
  if obs==0:#NSIDC-0051
    ax1.set_title("(a) Models vs NSIDC-0051", fontname='Arial', fontsize=16)
    plt.savefig('./Fig2a_Metrics_siconc_NSIDC0051.png', bbox_inches = "tight", dpi = 500)
  else:#OSI450
    ax1.set_title("(b) Models vs OSI-450", fontname='Arial', fontsize=16)
    plt.savefig('./Fig2b_Metrics_siconc_OSI450.png', bbox_inches = "tight", dpi = 500)




