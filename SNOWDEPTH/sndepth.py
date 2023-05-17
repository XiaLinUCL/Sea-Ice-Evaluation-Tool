# Author: Xia Lin
# Date:   Feb 2021
# Contents:
# 1) A function that interpolates the input snow depth data into the NSIDC-0051 grid, 
#    the input data is a numpy array but a NetCDF file 
# 2) A function that computes snow depth errors between two datasets on the mean state in order to get the metrics;    
#    The input data is a numpy array but a NetCDF file 
# 3) The heatmap function  
# 4) The annotate heatmap function (The heatmap and annotate heatmap functions were written 
#    by following https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html)
# 5) A script deals with the input NetCDF data, and then calls the function 1) 
# 6) A script calls the function 2) and then computes the snow depth metrics
# 7) A script calls the functions 3) and 4) to plot the snow depth metrics（Fig. 7b）
# 8) A script plots the February (Arctic) and September (Antarctic) mean snow depth differences (Figs. A7-A8);

# -----------------------------------
# PART 1) The interpolation function |
# -----------------------------------
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

# ----------------------------------------
# PART 2) The snow depth errors function  |   
# ----------------------------------------
def compute_sisnthick_metrics(thickness0, thickness1):
  ''' Input: - snow depth (m) in the Arctic or Antarctic from two datasets
      Output: Errors between two snow depth datasets of mean cycle in the Arctic or Antarctic
  '''
  nt, ny, nx = thickness1.shape
  #Calculate the global error by hemisphere
  #We don't want take into account cells that are ice free (e.g. in tropics ) or with nan values. So creat two masks.
  mask=np.zeros(((nt,ny,nx)))
  for jt in range(nt):
    for jy in np.arange(ny):
      for jx in np.arange(nx):
        if thickness0[jt,jy,jx] == 0 and thickness1[jt,jy,jx] == 0 :  
          mask[jt,jy,jx] = 0
        elif np.isnan(thickness0[jt,jy,jx]) or np.isnan(thickness1[jt,jy,jx]):
          mask[jt,jy,jx] = 0
        else:
          mask[jt,jy,jx] = 1.0
  
  #Now we prepare, for each cell, a variable containing the abs value of errors
  #Calculate errors on mean cycle... vs Envisat
  error_mean_thick=np.array([abs(thickness0[m,:,:] - thickness1[m,:,:]) for m in range(nt)])
  error_mean_monthly = np.array([(np.nansum(error_mean_thick[m,:,:]*mask[m,:,:])/np.nansum(mask[m,:,:])) for m in range(nt)])
  ndpm=[31,28,31,30,30,31]
  error_mean=np.sum(NHerror_mean_monthly*ndpm)/np.sum(ndpm)
  
  return error_mean

# ------------------------------
# PART 3) The heatmap function  |
# ------------------------------
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
  ''' Input: -data: A 2D numpy array of shape (N, M).
             -row_labels: A list or array of length N with the labels for the rows.
             -col_labels: A list or array of length M with the labels for the columns.
             -ax: A `matplotlib.axes.Axes` instance to which the heatmap is plotted. If
                  not provided, use current axes or create a new one.  Optional.
             -cbar_kw: A dictionary with arguments to `matplotlib.Figure.colorbar`. Optional.
             -cbarlabel: The label for the colorbar. Optional.
             -**kwargs: All other arguments are forwarded to `imshow`.
      Output: Create a heatmap from a numpy array and two lists of labels.
  '''
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
    ''' Input: -im: The AxesImage to be labeled.
               -data: Data used to annotate.  If None, the image's data is used. Optional.
               -valfmt: The format of the annotations inside the heatmap.  This should either use the string 
                        format method, e.g. "$ {x:.2f}", or be a `matplotlib.ticker.Formatter`. Optional.       
               -textcolors: A pair of colors.  The first is used for values below a threshold,
                            the second for those above. Optional.
               -threshold: Value in data units according to which the colors from textcolors are applied. 
                           If None (the default) uses the middle of the colormap as separation. Optional.          
               -**kwargs: All other arguments are forwarded to each call to `text` used to create the text labels.
        Output: Annotate a heatmap
    '''
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
#for interpolation (you will have to install pyresample first)
import pyresample
import netCDF4 as nc
from netCDF4 import Dataset
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
from mpl_toolkits.basemap import Basemap, addcyclic
#----------------------------
#Load NSIDC0051 NH & SH grid |
#----------------------------
access_pr_file = '/sea ice data/OBS/siconc/NSIDC-0051/siconc_r1i1p1_mon_197901-201712_nh-psn25.nc'
dset = xr.open_dataset(access_pr_file)
NHlat_curv = np.array(dset['latitude'][:,:])
NHlon_curv =np.array(dset['longitude'][:,:])
NHcellarea = np.array(dset['areacello'][:,:])
access_pr_file = '/sea ice data/OBS/siconc/NSIDC-0051/siconc_r1i1p1_mon_197901-201712_sh-pss25.nc'
dset = xr.open_dataset(access_pr_file)
SHlat_curv = np.array(dset['latitude'][:,:])
SHlon_curv =np.array(dset['longitude'][:,:])
SHcellarea = np.array(dset['areacello'][:,:])

#------------------------------------------------------------------------------------------------------------------------------
# Load Envisat NH & SH snowdepth and interp into NSIDC0051 the Polar stereographic projection at a grid cell size of 25 x 25 km|
#------------------------------------------------------------------------------------------------------------------------------
# Northern Hemisphere 1-4,11,12  SH:5-10 2003-2007
f = Dataset('/sea ice data/OBS/sithick/CTOH_AVISO+_Envisat_Cryosat/2seaice_products_tfmra/ease2_grids_12500_smth25000/nh/SIT_NH_2002_2012_ENV_SnowW99m.ease2_12500_smth25000.nc', mode = "r")
lat = f.variables["latitude"][:,:]
lon = f.variables["longitude"][:,:]
time = f.variables["time"]
field = f.variables['snow_depth'][2:32,:,:] 
field_unc = f.variables['snow_depth_unc'][2:32,:,:] 
idx = np.where(field < 0)
field[idx] = np.nan
idx = np.where(field_unc < 0)
field_unc[idx] = np.nan
idx = np.where(field > 10)##large values
field[idx] = np.nan
a = lon.shape[0]
b = lon.shape[1]
for i in range(a):
  for j in range(b):
    if lon[i,j]>180:
      lon[i,j] = lon[i,j]-360
NHsndepth=compute_interp(lat,lon,field)
NHsndepth_unc=compute_interp(lat,lon,field_unc)
# Southern Hemisphere
f = Dataset('/sea ice data/OBS/sithick/CTOH_AVISO+_Envisat_Cryosat/2seaice_products_tfmra/ease2_grids_12500_smth25000/sh/SIT_SH_2002_2011_ENV_SnowAMSR.ease2_12500_smth25000.nc', mode = "r")
lat = f.variables["latitude"][:,:]
lon = f.variables["longitude"][:,:]
time = f.variables["time"]
field = f.variables['snow_depth'][1:31,:,:] 
field_unc = f.variables['snow_depth_unc'][1:31,:,:] 
idx = np.where(field < 0)
field[idx] = np.nan
idx = np.where(field_unc < 0)
field_unc[idx] = np.nan
a = lon.shape[0]
b = lon.shape[1]
for i in range(a):
  for j in range(b):
    if lon[i,j]>180:
      lon[i,j] = lon[i,j]-360
SHsndepth=compute_interp(lat,lon,field)
SHsndepth_unc=compute_interp(lat,lon,field_unc)
np.savez('Envisat_2003_2007_sisnthick.npz', NHlat_curv, NHlon_curv, NHsndepth, NHsndepth_unc, SHlat_curv, SHlon_curv, SHsndepth, SHsndepth_unc)                                               

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Load SnowModel-LG reconstructed NH & SH snowdepth and interp into NSIDC0051 the Polar stereographic projection at a grid cell size of 25 x 25 km|
#-------------------------------------------------------------------------------------------------------------------------------------------------
#SnowModel-LG/ERA5 2003-2007
dirname='/sea ice data/OBS/sisnthick/SnowModel-LG/'  
filename = 'snod_era5_monthly.gdat'
inputfile = os.path.join(dirname, filename)
with open(inputfile, 'rb') as fid:
  snowdep = np.fromfile(fid, np.float32).reshape((456,361,361))#1980.8-2018.7
  idx = np.where(snowdep < -9000.00)
  snowdep[idx] = np.nan  
  field=snowdep[269:329,:,:]#2003-2007
f = Dataset('/sea ice data/OBS/sisnthick/SnowModel-LG/icemotion_daily_nh_25km_19781101_19781231_v4.1.nc', mode = "r")
lat = f.variables["latitude"][:,:]
lon = f.variables["longitude"][:,:]
a = lon.shape[0]
b = lon.shape[1]
for i in range(a):
  for j in range(b):
    if lon[i,j]>180:
      lon[i,j] = lon[i,j]-360
NHsndepth=compute_interp(lat,lon,field)
np.savez('SMLGreconstructed_2003_2007_sisnthick.npz', NHlat_curv, NHlon_curv, NHsndepth)

#---------------------------------------------------------------------------------------------------------------------------------
# Load 1980-2007 model output data and interp into NSIDC0051 the Polar stereographic projection at a grid cell size of 25 x 25 km |
#---------------------------------------------------------------------------------------------------------------------------------
path='/sea ice data/OMIP/All OMIP data/r1i1p1f1/sisnthick/1980-2007/'
files=os.listdir(path)
for file in files:
  print(file)
  dset = xr.open_dataset(os.path.join(path,file))
  if file.startswith('sisnthick_SImon_IPSL'):
    lat=np.array(dset['nav_lat'][:,:])
    lon=np.array(dset['nav_lon'][:,:])
    field = np.array(np.squeeze(dset['sisnthick'][276:336,:,:]))#2003-2007
  elif file.startswith('sisnthick_SImon_GFDL-CM4'):
    lat0=np.array(dset['lat'])
    lon0=np.array(dset['lon'])
    field0 = np.array(np.squeeze(dset['sisnthick'][276:336,:,:]))
    idx1=np.where(lon0<0)
    lon0[idx1]=lon0[idx1]+360
    lon=np.zeros((1080,1440))
    lat=np.zeros((1080,1440))
    field=np.zeros(((60,1080,1440)))
    lon[:,0:960]=lon0[:,480:1440]
    lon[:,960:1440]=lon0[:,0:480]
    lat[:,0:960]=lat0[:,480:1440]
    lat[:,960:1440]=lat0[:,0:480]
    field[:,:,0:960]=field0[:,:,480:1440]
    field[:,:,960:1440]=field0[:,:,0:480]
  elif file.startswith('sisnthick_SImon_GFDL-OM4p5B'):
    lat0=np.array(dset['lat'])
    lon0=np.array(dset['lon'])
    [lon,lat]=np.meshgrid(lon0,lat0)
    field = np.array(np.squeeze(dset['sisnthick'][276:336,:,:]))
  else:
    lat=np.array(dset['latitude'][:,:])
    lon=np.array(dset['longitude'][:,:])
    field = np.array(np.squeeze(dset['sisnthick'][276:336,:,:]))
  
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

  sisnthick=compute_interp(lat,lon,field)
  NHsnthick=sisnthick[0]
  SHsnthick=sisnthick[1]
  np.savez(file[14:32]+'_2003_2007_sisnthick.npz', NHlat_curv, NHlon_curv, NHsnthick, SHlat_curv, SHlon_curv, SHsnthick)

# -------------------------------------------------
# PART 6) A script computes the snow depth metrics |
# -------------------------------------------------
# Envisat
a=np.load('Envisat_2003_2007_sisnthick.npz')
NHthick1=a['arr_2']
NHthick1_unc=a['arr_3']#uncertainty
SHthick1=a['arr_6']
SHthick1_unc=a['arr_7']
# Typical error from Envisat SIT uncertainty
# Here we conpute the mean monthly thickness over the whole period
nt, ny, nx = NHthick1.shape
nt1, ny1, nx1 = SHthick1.shape
# Calculate the mean cycle...
NHthickness1 = np.array([np.nanmean(NHthick1[m::6,:,:], axis=0) for m in range(6)])
NHthickness1_unc = np.array([np.nanmean(NHthick1_unc[m::6,:,:], axis=0) for m in range(6)])
SHthickness1 = np.array([np.nanmean(SHthick1[m::6,:,:], axis=0) for m in range(6)])
SHthickness1_unc = np.array([np.nanmean(SHthick1_unc[m::6,:,:], axis=0) for m in range(6)])

#Calculate the global error by hemisphere
#We don't want take into account cells that are ice free (e.g. in tropics ) or with nan values. So creat two masks.
NHmask=np.zeros(((6,ny,nx)))
for jt in range(6):
  for jy in np.arange(ny):
    for jx in np.arange(nx):
      if NHthickness1[jt,jy,jx] == 0 :  
        NHmask[jt,jy,jx] = 0
      elif np.isnan(NHthickness1[jt,jy,jx]):
        NHmask[jt,jy,jx] = 0
      else:
        NHmask[jt,jy,jx] = 1.0
SHmask=np.zeros(((6,ny1,nx1)))
for jt in range(6):
  for jy in np.arange(ny1):
    for jx in np.arange(nx1):
      if SHthickness1[jt,jy,jx] == 0 :
        SHmask[jt,jy,jx] = 0
      elif np.isnan(SHthickness1[jt,jy,jx]):
        SHmask[jt,jy,jx] = 0
      else:
        SHmask[jt,jy,jx] = 1.0

#Calculate the global error by hemisphere
NHunc = np.array([(np.nansum(NHthickness1_unc[m,:,:]*NHmask[m,:,:])/np.nansum(NHmask[m,:,:])) for m in range(6)])
SHunc = np.array([(np.nansum(SHthickness1_unc[m,:,:]*SHmask[m,:,:])/np.nansum(SHmask[m,:,:])) for m in range(6)])
NHndpm=[31,28,31,30,30,31];
SHndpm=[31,30,31,31,30,31];
NHtyerror=np.sum(NHunc*NHndpm)/np.sum(NHndpm)
SHtyerror=np.sum(SHunc*SHndpm)/np.sum(SHndpm)

name=['CMCC-CM2-HR4_omip2_2003_2007_sisnthick.npz', 'CMCC-CM2-SR5_omip1_2003_2007_sisnthick.npz', 'CMCC-CM2-SR5_omip2_2003_2007_sisnthick.npz', 'EC-Earth3_omip1_r1_2003_2007_sisnthick.npz','EC-Earth3_omip2_r1_2003_2007_sisnthick.npz', 'GFDL-CM4_omip1_r1i_2003_2007_sisnthick.npz', 'GFDL-OM4p5B_omip1__2003_2007_sisnthick.npz', 'IPSL-CM6A-LR_omip1_2003_2007_sisnthick.npz',  'MIROC6_omip1_r1i1p_2003_2007_sisnthick.npz', 'MIROC6_omip2_r1i1p_2003_2007_sisnthick.npz', 'MRI-ESM2-0_omip1_r_2003_2007_sisnthick.npz', 'MRI-ESM2-0_omip2_r_2003_2007_sisnthick.npz', 'NorESM2-LM_omip1_r_2003_2007_sisnthick.npz', 'NorESM2-LM_omip2_r_2003_2007_sisnthick.npz']
NHerror_mean1=np.zeros(14)
SHerror_mean1=np.zeros(14)
Metrics_sithick=np.zeros((17, 2))
for num in range(14):
  a=np.load(name[num])
  NHthick=a['arr_2']
  SHthick=a['arr_5']
  #Calculate the mean cycle...
  NHthickness = np.array([np.nanmean(NHthick[m::12,:,:], axis=0) for m in range(12)])
  NHthickness0 = np.zeros((6,304,448))
  NHthickness0[0:4,:,:] = NHthickness[0:4,:,:]
  NHthickness0[4:6,:,:] = NHthickness[10:12,:,:]
  SHthickness = np.array([np.nanmean(SHthick[m::12,:,:], axis=0) for m in range(12)])
  SHthickness0 = np.zeros((6,316,332))
  SHthickness0[0:6,:,:] = SHthickness[4:10,:,:]
  NHerror_mean1[num]=compute_sisnthick_metrics(NHthickness0, NHthickness1)
  SHerror_mean1[num]=compute_sisnthick_metrics(SHthickness0, SHthickness1)
Metrics_sithick[0:14,0]=NHerror_mean1/NHtyerror
Metrics_sithick[0:14,1]=SHerror_mean1/SHtyerror
Metrics_sithick[14,0]=np.mean(NHerror_mean1)/NHtyerror
Metrics_sithick[14,1]=np.mean(SHerror_mean1)/SHtyerror
Metrics_sithick[15,:]=(Metrics_sithick[1,:]+Metrics_sithick[3,:]+Metrics_sithick[8,:]+Metrics_sithick[10,:]+Metrics_sithick[12,:])/5#OMIP1 mean
Metrics_sithick[16,:]=(Metrics_sithick[2,:]+Metrics_sithick[4,:]+Metrics_sithick[9,:]+Metrics_sithick[11,:]+Metrics_sithick[13,:])/5#OMIP2 mean
np.savez('metric_sisnthick_EnviSat.npz', Metrics_sithick, NHerror_mean1, SHerror_mean1) 

#snowmodel-LG
a=np.load('SMLGreconstructed_2003_2007_sisnthick.npz')
NHthick1=a['arr_2']
NHthickness1 = np.array([np.nanmean(NHthick1[m::12,:,:], axis=0) for m in range(12)])
NHthickness2 = np.zeros((6,304,448))
NHthickness2[0:4,:,:] = NHthickness1[0:4,:,:]
NHthickness2[4:6,:,:] = NHthickness1[10:12,:,:]

NHerror_mean1=np.zeros(14)
Metrics_sithick=np.zeros((17, 1))
for num in range(14):
  a=np.load(path + name[num])
  NHthick=a['arr_2']
  #Calculate the mean cycle...
  NHthickness = np.array([np.nanmean(NHthick[m::12,:,:], axis=0) for m in range(12)])
  NHthickness0 = np.zeros((6,304,448))
  NHthickness0[0:4,:,:] = NHthickness[0:4,:,:]
  NHthickness0[4:6,:,:] = NHthickness[10:12,:,:]
  NHerror_mean1[num]=compute_sisnthick_metrics(NHthickness0, NHthickness2)
Metrics_sithick[0:14,0]=NHerror_mean1/NHtyerror
Metrics_sithick[14,0]=np.mean(NHerror_mean1)/NHtyerror
Metrics_sithick[15,:]=(Metrics_sithick[1,:]+Metrics_sithick[3,:]+Metrics_sithick[8,:]+Metrics_sithick[10,:]+Metrics_sithick[12,:])/5#OMIP1 mean
Metrics_sithick[16,:]=(Metrics_sithick[2,:]+Metrics_sithick[4,:]+Metrics_sithick[9,:]+Metrics_sithick[11,:]+Metrics_sithick[13,:])/5#OMIP2 mean
np.savez('metric_sisnthick_SnowModelLG.npz', Metrics_sithick, NHerror_mean1, SHerror_mean1) 

# ---------------------------------------------------------
# PART 7) A script plots the snow depth metrics (Figs. 7b) | 
# ---------------------------------------------------------
Models=['CMCC-CM2-HR4/J','CMCC-CM2-SR5/C','CMCC-CM2-SR5/J','EC-Earth3/C','EC-Earth3/J','GFDL-CM4/C','GFDL-OM4p5B/C','IPSL-CM6A-LR/C','MIROC6/C','MIROC6/J','MRI-ESM2-0/C','MRI-ESM2-0/J','NorESM2-LM/C','NorESM2-LM/J','Model mean','Model mean/C','Model mean/J']
Variables=['Mean Snow Depth NH','Mean Snow Depth SH','Mean Snow Depth NH']
#Envisat&SnowModelLG
values=np.zeros((17, 3))
a=np.load('metric_sisnthick_EnviSat.npz')
values[:,0:2]=a['arr_0']
a=np.load('/metric_sisnthick_SnowModelLG.npz')
values[:,2]=a['arr_0']
dpi=100
squaresize = 220
figwidth = 18*squaresize/float(dpi)
figheight = 6*squaresize/float(dpi)
fig,ax1 = plt.subplots(1, figsize=(figwidth, figheight), dpi=dpi)
im,cbar = heatmap(values, Models,Variables, ax=ax1, cmap="OrRd", vmin=0, vmax=3) 
texts = annotate_heatmap(im, valfmt="{x:.2f}",size=16,threshold=2)
cbar.remove()
ax1.set_xticklabels(['Mean Snow Depth NH','Mean Snow Depth SH','Mean Snow Depth NH'])
plt.setp(ax1.get_xticklabels(), fontname='Arial', fontsize=16)
plt.setp(ax1.get_yticklabels(), fontname='Arial', fontsize=16)
cax = fig.add_axes([0.75, 0.113, 0.01, 0.765])
cbar = fig.colorbar(im, cax=cax,ticks=[0.5,0.7,0.9,1.1,1.3], orientation="vertical")
cbar.ax.yaxis.set_ticks_position('both')
cbar.ax.tick_params(direction='in',length=2,labelsize=16)
ax1.set_title("(b) Snow depth: models vs. Envisat & SnowModel-LG", fontname='Arial', fontsize=16)
plt.savefig('./Figure7b.png', bbox_inches = "tight", dpi = 500)

# ----------------------------------------------------------------------------------------------------------------
# PART 8) A script plots the February (Arctic) and September (Antarctic) mean snow depth differences (Figs. A7-A8)|
# ----------------------------------------------------------------------------------------------------------------
# 2003-2007 Envisat: Arctic:1-4,11,12 February  Antarctic: 5-10 Sept. 
# ICESat: 13 measurement campaigns for the Arctic and 11 for the Antarctic, 
#         limited to February-March, March-April, May-June, October-November with each roughly 33 days. 
# Prepare the data
a=np.load('Envisat_2003_2007_sisnthick.npz')
NHlat_curv=a['arr_0']
NHlon_curv=a['arr_1']
NHthick=a['arr_2']
NHthick_unc=a['arr_3']
SHlat_curv=a['arr_4']
SHlon_curv=a['arr_5']
SHthick=a['arr_6']
SHthick_unc=a['arr_7']
#Here we conpute the mean monthly concentrations over the whole period
NHthick1 = np.array([np.nanmean(NHthick[m::6,:,:], axis=0) for m in range(6)])
SHthick1 = np.array([np.nanmean(SHthick[m::6,:,:], axis=0) for m in range(6)])
NHthick_unc1 = np.array([np.nanmean(NHthick_unc[m::6,:,:], axis=0) for m in range(6)])
SHthick_unc1 = np.array([np.nanmean(SHthick_unc[m::6,:,:], axis=0) for m in range(6)])

# Define a figure properties
varin='sisnthick'
units='m'

name=['Envisat','CMCC-CM2-SR5/C','CMCC-CM2-SR5/J','CMCC-CM2-HR4/J','EC-Earth3/C','EC-Earth3/J','GFDL-CM4/C','MIROC6/C','MIROC6/J','GFDL-OM4p5B/C','MRI-ESM2-0/C','MRI-ESM2-0/J','IPSL-CM6A-LR/C','NorESM2-LM/C','NorESM2-LM/J',]
files=['Envisat_2003_2007_sisnthick.npz','CMCC-CM2-SR5_omip1_2003_2007_sisnthick.npz', 'CMCC-CM2-SR5_omip2_2003_2007_sisnthick.npz', 'CMCC-CM2-HR4_omip2_2003_2007_sisnthick.npz','EC-Earth3_omip1_r1_2003_2007_sisnthick.npz','EC-Earth3_omip2_r1_2003_2007_sisnthick.npz', 'GFDL-CM4_omip1_r1i_2003_2007_sisnthick.npz','MIROC6_omip1_r1i1p_2003_2007_sisnthick.npz', 'MIROC6_omip2_r1i1p_2003_2007_sisnthick.npz','GFDL-OM4p5B_omip1__2003_2007_sisnthick.npz', 'MRI-ESM2-0_omip1_r_2003_2007_sisnthick.npz', 'MRI-ESM2-0_omip2_r_2003_2007_sisnthick.npz', 'IPSL-CM6A-LR_omip1_2003_2007_sisnthick.npz', 'NorESM2-LM_omip1_r_2003_2007_sisnthick.npz', 'NorESM2-LM_omip2_r_2003_2007_sisnthick.npz']
for jt in range(2):
  if jt==0:
    jt1=1
    hemisphere = "n"
    lat=NHlat_curv
    lon=NHlon_curv
  elif jt==1:
    jt1=8
    hemisphere = "s"
    lat=SHlat_curv 
    lon=SHlon_curv

  fig=plt.figure(figsize=(5, 9))
  gs1 = gridspec.GridSpec(5, 3)
  gs1.update(wspace=0.04, hspace=0.06) # set the spacing between axes. 
  for num in range(15):  
    axes=plt.subplot(gs1[num])
    a=np.load(path + files[num]) 
    NHthick=a['arr_2']
    SHthick=a['arr_5']#!!!!!

    if num==0:
      NHfield=NHthick1[1,:,:]
      SHfield=SHthick1[4,:,:]
    else:
      #Here we conpute the mean monthly concentrations over the whole period
      NHthick2 = np.array([np.nanmean(NHthick[m::12,:,:], axis=0) for m in range(12)])
      NHfield=NHthick2[1,:,:]-NHthick1[1,:,:]   
      SHthick2 = np.array([np.nanmean(SHthick[m::12,:,:], axis=0) for m in range(12)]) 
      SHfield=SHthick2[8,:,:]-SHthick1[4,:,:] 

    # Create the colorbar
    # Load the colormap
    if num==0:
      lb=0
      ub=0.38
      nsteps=0.02
      colmap ='Reds'
      extmethod = "max"
    else:
      lb=-0.25001  
      ub=0.25001
      nsteps=0.025
      colmap ='RdBu_r'
      extmethod = "both"
    clevs = np.arange(lb, ub, nsteps)
    cmap = eval("plt.cm." + colmap)
    # Colors for values outside the colorbar
    # Get first and last colors of the colormap
    first_col = cmap(0)[0:3]
    last_col  = cmap(255)[0:3]
    first_col1 = cmap(10)[0:3]
    # Create "over" and "under" colors.
    # Take respectively the latest and first color and
    # darken them by 50%
    col_under = [i / 2.0 for i in first_col]
    col_over  = [i / 2.0 for i in last_col ]
 
    # Create a projection 
    # Determine if we are in the northern hemisphere or in the Southern hemisphere.
    if hemisphere == "n":
      boundlat = 50
      l0 = 0.
    elif hemisphere == "s":
      boundlat = -50
      l0 = 180.
    else:
      sys.exit("(map_polar_field) Hemisphere unkown")

    # Projection name
    projname = hemisphere + "plaea" 
    map = Basemap(projection = projname,lat_0=90, boundinglat = boundlat, lon_0 = l0, resolution = 'l', round=True, ax=axes)
    x, y = map(lon, lat)

    # Plot the figure 
    if hemisphere == "n":
      field = np.squeeze(NHfield[:, :])
    elif hemisphere == "s":
      field = np.squeeze(SHfield[:, :])
    # Draw coastlines, country boundaries, fill continents, meridians & parallels
    map.drawcoastlines(linewidth = 0.15)
    map.fillcontinents(color = 'silver', lake_color = 'white')
    map.drawmeridians(np.arange(0, 360, 60), linewidth=0.9, latmax=80, color='silver')
    map.drawparallels(np.arange(60, 90, 10), linewidth=0.9, labels=[True,True,True,True], color='silver',fontsize=1)
    if hemisphere == "s":
      map.drawparallels(np.arange(-90, -55, 10), linewidth=0.9, labels=[True,True,True,True], color='silver',fontsize=1) 
    map.drawlsmask(ocean_color='white')
    meridians = np.arange(0, 360, 60)
    circle = map.drawmapboundary(linewidth=1, color='black')
    circle.set_clip_on(False)
    # Create a contourf object called "cs"
    if num==0:
      cs = map.contourf(x, y, field, clevs, cmap = cmap, vmin=lb, vmax=ub, latlon = False, extend = extmethod)
      cs.cmap.set_under('white')
      cax = fig.add_axes([0.93, 0.25, 0.03, 0.49])
      cbar = fig.colorbar(cs, cax=cax,ticks=[0,0.04,0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36])
      cbar.ax.yaxis.set_ticks_position('both')
      cbar.ax.tick_params(direction='in',length=2)
      cbar.set_label('Snow depth (m)' ,fontname='Arial',fontsize=8, fontweight='bold')
      for l in cbar.ax.get_yticklabels():
        l.set_fontproperties('Arial') 
        l.set_fontsize(8)
        l.set_fontweight('bold') 
    else: 
      norm = colors.DivergingNorm(vmin=lb, vcenter=0, vmax=ub)
      cs = map.contourf(x, y, field, clevs, cmap = cmap, vmin=lb, vmax=ub, norm = norm, latlon = False, extend = extmethod)
      cs.cmap.set_under(col_under)
      cs.cmap.set_over(col_over)
    axes.set_title(name[num],fontname = 'Arial', fontsize = 7.5,fontweight = 'bold', loc = 'center',y = 1.0, pad = 3)
    
  cax = fig.add_axes([1.05, 0.25, 0.03, 0.49])
  cbar = fig.colorbar(cs, cax=cax,ticks=[-0.2,-0.1,0,0.1,0.2])
  cbar.ax.yaxis.set_ticks_position('both')
  cbar.ax.tick_params(direction='in',length=2)
  cbar.set_label('Snow depth difference (vs. Envisat, m)' ,fontname = 'Arial',fontsize = 8, fontweight = 'bold')
  for l in cbar.ax.get_yticklabels():
    l.set_fontproperties('Arial') 
    l.set_fontsize(8) 
    l.set_fontweight('bold')

  # Save figure 
  filename    = 'FigureA' + str(jt + 7).zfill(1) 
  imageformat = "png"  
  dpi         = 500     
  plt.savefig(filename + "." + imageformat, bbox_inches = "tight", dpi = dpi)
  print('Figure ' + filename + "." + imageformat + " printed")
  plt.close("fig")
