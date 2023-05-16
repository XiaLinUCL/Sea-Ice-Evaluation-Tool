# Author: Xia Lin
# Date:   Feb 2021
# Contents:
# 1) A function rotates the input ice drift data on ORCA grid into the polar stereographic grid;
#    History: Based on Olivier Lecomte and François Massonnet's Python script
# 2) A function interpolates the input ice drift data into the NSIDC-0051 grid, and also corrects the ice drift;
# 3) A function computes sea ice drift magnitude (MKE) errors between two datasets in order to get the metrics;
# 4) A function computes ice-motion vector correlation coefficients between two datasets to get the metrics;
#    The input data is a numpy array but a NetCDF file;
# 5) The heatmap function;  
# 6) The annotate heatmap function (The heatmap and annotate heatmap functions were written 
#    by following https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html);
# 7) A script deals with the input NetCDF data, and then calls the function 1); 
# 8) A script calls the function 2) and then computes the ice-motion MKE metrics;
# 9) A script calls the function 3) and then computes the ice-motion vector correlation coefficients metrics;
# 10) A script plots the ice vector correlation coefficients in the Arctic and Antarctic (Figs. 8-9);
# 11) A script calls the functions 5) and 6) to plot the ice drift metrics (Fig. 10);
# 12) A script plots the February and September mean ice-motion mean kinetic energy differences in the Arctic and Antarctic (Figs. A9-A12);

# -------------------------------
# PART 1) The rotation function  |
# -------------------------------
def deg2rad(angle):
  """
  Conversion from degrees to radians
  input   : angle: a float or a numpy array
  output  : equivalent in radians
  """
  return 2.0 * np.pi / 360.0 * angle

def compute_rotate(glamt,gphit,u,v,glamv,gphiv,hems):
  ''' 
  Rotation of vectors from given source grid to reference
  frame. The NEMO ORCA source grid is handled.
      Input: -glamt: Longitude at T-points
             -gphit: Latitude  at T-points
             -glamv: Longitude at V-points
             -gphiv: Latitude  at V-points
             -u: zonal ice velocity on the original grid
             -v: meridional ice velocity on the original grid
             -hems: Northern or Southern Hemisphere
      Output: Rotated ice vectors 
  '''
  if len(u.shape) != 3 or len(v.shape) != 3:
    sys.exit("Velocity has not 3 dimensions")

  # Get dimensions
  nt, ny, nx = u.shape
  if v.shape != u.shape:
    sys.exit("Shapes do not conform")

  if glamt.shape != (ny, nx):
    sys.exit("Grid does not have the same dimensions as the vector data")

  # Create arrays of longitude and latitude shifted down by one row
  glamvp = np.concatenate((np.full((1, nx), np.nan), glamv[:-1, :]))
  gphivp = np.concatenate((np.full((1, nx), np.nan), gphiv[:-1, :]))

  NPx = 0.0 - 2.0 * np.tan(np.pi / 4.0 - deg2rad(gphit / 2.0)) * np.cos(deg2rad(glamt))
  NPy = 0.0 - 2.0 * np.tan(np.pi / 4.0 - deg2rad(gphit / 2.0)) * np.sin(deg2rad(glamt))
  NP = NPx ** 2 + NPy ** 2
  Dvx = 2.0 * np.tan(np.pi / 4.0 - deg2rad(gphiv / 2.0)) * np.cos(deg2rad(glamv)) - 2.0 * np.tan(np.pi / 4.0 - deg2rad(gphivp / 2.0)) * np.cos(deg2rad(glamvp)) 
  Dvy = 2.0 * np.tan(np.pi / 4.0 - deg2rad(gphiv / 2.0)) * np.sin(deg2rad(glamv)) - 2.0 * np.tan(np.pi / 4.0 - deg2rad(gphivp / 2.0)) * np.sin(deg2rad(glamvp))                     
  NP_Dv = np.sqrt((Dvx ** 2 + Dvy ** 2) * NP)

  # Computation of cosine and sine of angles between source and target grid
  gcost = (NPx * Dvx + NPy * Dvy) / NP_Dv
  gsint = (NPx * Dvy - NPy * Dvx) / NP_Dv

  # Projection of input components to target frame
  # Note that numpy handles the fact that v and the sin/cos have different dimensions, 
  # by iterating over the dimension that only exists in v or u
  if hems == 'n': #Arctic
    vy = v * gcost + u * gsint 
    ux = u * gcost - v * gsint
  else:#Antarctic added
    vy = -v * gcost - u * gsint 
    ux = -u * gcost + v * gsint
  
  return ux, vy

# --------------------------------------------------
# PART 2) The interpolation and correction function |
# --------------------------------------------------
def compute_interp_correct(lon, lat, ux, vy, hems):
  ''' Input: -latitude, longitude of the original grid
             -ice vectors on the original grid
      Output: Interpolated and corrected ice vectors 
  '''
  #Interp into NSIDC-0051 grid
  #Load NSIDC-0051 NH &SH grid and siconc
  access_pr_file = '/sea ice data/OBS/siconc/NSIDC-0051/siconc_r1i1p1_mon_197901-201712_nh-psn25.nc'
  dset = xr.open_dataset(access_pr_file)
  NHlat1 = np.array(dset['latitude'][:,:])
  NHlon1 =np.array(dset['longitude'][:,:])
  NHconcentration1 = np.array(dset['siconc'][288:348,:,:])
  access_pr_file = '/sea ice data/OBS/siconc/NSIDC-0051/siconc_r1i1p1_mon_197901-201712_sh-pss25.nc'
  dset = xr.open_dataset(access_pr_file)
  SHlat1 = np.array(dset['latitude'][:,:])
  SHlon1 =np.array(dset['longitude'][:,:])
  SHconcentration1 = np.array(dset['siconc'][288:348,:,:])

  lon_curv=lon
  lat_curv=lat
  a = lon_curv.shape[0]
  b = lon_curv.shape[1]
  for i in range(a):
    for j in range(b):
      if lon_curv[i,j]>180:
        lon_curv[i,j] = lon_curv[i,j]-360

  #Create a pyresample object holding the origin grid:
  orig_def = pyresample.geometry.SwathDefinition(lons=lon_curv, lats=lat_curv)
  #Create another pyresample object for the target (curvilinear) grid:
  if hems == 'n': #Arctic
    targ_def = pyresample.geometry.SwathDefinition(lons=NHlon1, lats=NHlat1)
    ux1 = np.zeros((60,304,448))
    vy1 = np.zeros((60,304,448))
  else:#Antarctic
    targ_def = pyresample.geometry.SwathDefinition(lons=SHlon1, lats=SHlat1)
    ux1 = np.zeros((60,316,332))
    vy1 = np.zeros((60,316,332))
  #sidrift_nearest interp
  ux1 = np.array([pyresample.kd_tree.resample_nearest(orig_def, ux[i, :, :], targ_def, radius_of_influence = 500000, fill_value=None) for i in range(60)])
  idx = np.where(ux1 > 1000.00)
  ux1[idx] = np.nan
  vy1 = np.array([pyresample.kd_tree.resample_nearest(orig_def, vy[i, :, :], targ_def, radius_of_influence = 500000, fill_value=None) for i in range(60)])
  idx = np.where(vy1 > 1000.00)
  vy1[idx] = np.nan

  #creat landmask of NSIDC-0051
  if hems == 'n': #Arctic
    land=np.zeros(((60,304,448)))
    idx = np.where(np.isnan(NHconcentration1))
    land[idx] = 1
    #delete strange point
    idx = np.where((NHlat1>80) & (NHlon1>75) & (NHlon1<90))
    for i in range(60):
      SQs=land[i,:,:]
      SQs[idx] = 0
      land[i,:,:]=SQs
  else: #Antarctic
    land=np.zeros(((60,316,332)))
    idx = np.where(np.isnan(SHconcentration1))
    land[idx] = 1

  #Removed any data closer than 75 km (25km grid) to the coast  
  #Removed any data with sea-ice concentrations below 50%
  if hems == 'n': #Arctic
    land1=np.zeros(((60,304,448)))
    land2=np.zeros(((60,304,448))) 
    for t in range(60):
      for i in range(304):
        for j in range(444):
          if abs(land[t,i,j]-land[t,i,j+1])==1:
            land1[t,i,j-3:j+4]=1
    for t in range(60):
      for j in range(448):
        for i in range(300):
          if abs(land[t,i,j]-land[t,i+1,j])==1:
            land2[t,i-3:i+4,j]=1
    idx=np.where((land==1) | (land2==1) | (land1==1) | (NHconcentration1<50))#corrected
    ux1[idx]=np.nan
    vy1[idx]=np.nan
  else: #Antarctic
    land1=np.zeros(((60,316,332)))
    land2=np.zeros(((60,316,332))) 
    for t in range(60):
      for i in range(316):
        for j in range(328):
          if abs(land[t,i,j]-land[t,i,j+1])==1:
            land1[t,i,j-3:j+4]=1
    for t in range(60):
      for j in range(332):
        for i in range(312):
          if abs(land[t,i,j]-land[t,i+1,j])==1:
            land2[t,i-3:i+4,j]=1
    idx=np.where((land==1) | (land2==1) | (land1==1) | (SHconcentration1<50))#corrected
    ux1[idx]=np.nan
    vy1[idx]=np.nan

  return ux1, vy1

# -------------------------------------------------------
# PART 3) The ice-motion magnitude (MKE) errors function |     
# -------------------------------------------------------
def compute_MKE_metrics(MKE, MKE1):
  ''' Input: - Mean Kinetic Energy (MKE) in the Arctic or Antarctic from two datasets
      Output: Magnitude errors between two ice drift datasets in the Arctic or Antarctic
  '''
  nt, ny, nx = MKE.shape
  #Calculate the mean cycle...
  conc = np.array([np.nanmean(MKE[m::12,:,:], axis=0) for m in range(12)])
  conc1 = np.array([np.nanmean(MKE1[m::12,:,:], axis=0) for m in range(12)])
  
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
  error_mean_monthly = np.array([(np.nansum(error_mean_conc[m,:,:]*mask[m,:,:])/np.nansum(mask[m,:,:])) for m in range(12)])
  ndpm=[31,28,31,30,31,30,31,31,30,31,30,31];
  error_mean=np.sum(error_mean_monthly*ndpm)/np.sum(ndpm)
  
  return error_mean

# ----------------------------------------------------------------
# PART 4) The ice-motion vector correlation coefficients function |     
# ----------------------------------------------------------------
def compute_vectorcorr_metrics(u1, u2, v1, v2):
  ''' 
  Vector correlation coefficients based on crosby (1993)
      Input: - Ice vectors in the Arctic or Antarctic from two datasets
      Output: Vector correlation coefficients between two ice drift datasets in the Arctic or Antarctic
  '''
  nt, ny, nx = u1.shape
  idx=np.where(np.isnan(u1) | np.isnan(v1) | np.isnan(u2) | np.isnan(v2))#corrected
  u1[idx]=np.nan
  v1[idx]=np.nan
  u2[idx]=np.nan
  v2[idx]=np.nan

  u_1=ma.masked_invalid(u1)
  v_1=ma.masked_invalid(v1)
  u_2=ma.masked_invalid(u2)
  v_2=ma.masked_invalid(v2)

  #variance u  covariance uv
  u1u2=np.zeros((ny,nx))
  u1u1=np.zeros((ny,nx))
  u2u2=np.zeros((ny,nx))
  u1v1=np.zeros((ny,nx))
  v1v1=np.zeros((ny,nx))
  u1v2=np.zeros((ny,nx))
  v2v2=np.zeros((ny,nx))
  u2v2=np.zeros((ny,nx))
  v1v2=np.zeros((ny,nx))
  v1u2=np.zeros((ny,nx))
  for i in range(ny):
    for j in range(nx):
      u1u2[i,j] = ma.cov(u_1[:,i,j],u_2[:,i,j],rowvar=True)[0,1]
      u1u1[i,j] = ma.cov(u_1[:,i,j],u_2[:,i,j],rowvar=True)[0,0]
      u2u2[i,j] = ma.cov(u_1[:,i,j],u_2[:,i,j],rowvar=True)[1,1]
      u1v1[i,j] = ma.cov(u_1[:,i,j],v_1[:,i,j],rowvar=True)[0,1]
      v1v1[i,j] = ma.cov(u_1[:,i,j],v_1[:,i,j],rowvar=True)[1,1] 
      u1v2[i,j] = ma.cov(u_1[:,i,j],v_2[:,i,j],rowvar=True)[0,1]
      v2v2[i,j] = ma.cov(u_1[:,i,j],v_2[:,i,j],rowvar=True)[1,1]
      u2v2[i,j] = ma.cov(u_2[:,i,j],v_2[:,i,j],rowvar=True)[0,1]
      v1v2[i,j] = ma.cov(v_1[:,i,j],v_2[:,i,j],rowvar=True)[0,1]
      v1u2[i,j] = ma.cov(v_1[:,i,j],u_2[:,i,j],rowvar=True)[0,1]

  f0=u1u1*(u2u2*(v1v2)**2+v2v2*(v1u2)**2)+v1v1*(u2u2*(u1v2)**2+v2v2*(u1u2)**2)+2*(u1v1*u1v2*v1u2*u2v2)+2*(u1v1*u1u2*v1v2*u2v2)
  f1=-2*(u1u1*v1u2*v1v2*u2v2)-2*(v1v1*u1u2*u1v2*u2v2)-2*(u2u2*u1v1*u1v2*v1v2)-2*(v2v2*u1v1*u1u2*v1u2)
  f=f0+f1
  g=(u1u1*v1v1-u1v1**2)*(u2u2*v2v2-u2v2**2)
  idx=np.where(g==0)
  g[idx]=np.nan
  corr=f/g/2
  #if number less than 11, the correlation coefficient is not valid
  #keep significant ice-motion vector correlation coefficients at a level of 99%
  n=np.ones(((nt,ny,nx)))
  idx=np.where(np.isnan(u_1) | np.isnan(v_1) | np.isnan(u_2) | np.isnan(v_2))
  n[idx]=0
  n1 = np.array(np.squeeze([np.nansum(n,axis=0)]))
  Q=n1*corr
  idx=np.where(n1<11)
  corr[idx]=np.nan
  idx=np.where(Q<8)  
  corr[idx]=np.nan
  corrmean=np.nanmean(corr)
  
  return corr, corrmean

# ------------------------------
# PART 5) The heatmap function  |
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
# PART 6) The annotate heatmap function  |
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
# PART 7) A script deals with the input NetCDF data |
# --------------------------------------------------   
import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import numpy.ma as ma
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
# Load NSIDC-0051 NH &SH grid 
access_pr_file = '/sea ice data/OBS/siconc/NSIDC-0051/siconc_r1i1p1_mon_197901-201712_nh-psn25.nc'
dset = xr.open_dataset(access_pr_file)
NHlat1 = np.array(dset['latitude'][:,:])
NHlon1 =np.array(dset['longitude'][:,:])
access_pr_file = '/sea ice data/OBS/siconc/NSIDC-0051/siconc_r1i1p1_mon_197901-201712_sh-pss25.nc'
dset = xr.open_dataset(access_pr_file)
SHlat1 = np.array(dset['latitude'][:,:])
SHlon1 =np.array(dset['longitude'][:,:])
# ----------------------------------------------------------
# Load ICDC_NSIDC ice drift and interp into NSIDC0051 girds |
# ----------------------------------------------------------
for obs in range(2):
  if obs==0:
    hems='n'
    filename = '/sea ice data/OBS/sea ice drift/ICDC_NSIDC/NSIDC__SeaIceDrift__monthly__25km__198001-200912-NH__v04.1__UHH-ICDC_fv0.01.nc'
  else:
    hems='s'
    filename = '/sea ice data/OBS/sea ice drift/ICDC_NSIDC/NSIDC__SeaIceDrift__monthly__25km__198001-200912-SH__v04.1__UHH-ICDC_fv0.01.nc'
  dset = nc.Dataset(filename)
  lat = np.array(dset['lat'][:])
  lon = np.array(dset['lon'][:])
  ux = np.array(np.squeeze(dset['ux'][276:336,:,:]))*(24*3600/10**3)*1.357##m/s->km/day correction factor
  vy = np.array(np.squeeze(dset['vy'][276:336,:,:]))*(24*3600/10**3)*1.357##m/s->km/day
  #Remove data with a spurious, exact value of zero.
  idx=np.where((ux>990) | (ux==0) | (vy==0))#corrected
  ux[idx]=np.nan
  vy[idx]=np.nan
  metrics=compute_interp_correct(lon, lat, ux, vy, hems)
  ux1=metrics[0]
  vy1=metrics[1]
  if obs==0:
    np.savez('ICDC_NSIDC_NH_200301_200712_siuv_corrected.npz', NHlon1, NHlat1, ux1, vy1)
  else:
    np.savez('ICDC_NSIDC_SH_200301_200712_siuv_corrected.npz', SHlon1, SHlat1, ux1, vy1)
#------------------------------------------------------------
# Load KIMURA NH &SH sidrift and interp into NSIDC0051 girds
#------------------------------------------------------------
def mean_func(data):
    # note: apply always moves core dimensions to the end
    return xr.apply_ufunc(np.nanmean, data,
                       input_core_dims=[['time']],
                       kwargs={'axis': -1})

for obs in range(2):
  if obs==0:
    hems='n'
    filename = '/sea ice data/OBS/sea ice drift/KIMURA/KIMURA_NH_2003_2007_daily.nc'
    filename1 = '/sea ice data/OBS/sea ice drift/KIMURA/KIMURA_NH_2003_2007_monthlysum_number.nc'
  else:
    hems='s'
    filename = '/sea ice data/OBS/sea ice drift/KIMURA/KIMURA_SH_2003_2007_daily.nc'
    filename1 = '/sea ice data/OBS/sea ice drift/KIMURA/KIMURA_SH_2003_2007_monthlysum_number.nc'
  
  data = xr.open_dataset(filename)
  data1=data.resample(time='1MS').apply(mean_func)
  t = np.array(data1['time'][:])
  lat = np.array(data1['latitude'][:,:])
  lon = np.array(data1['longitude'][:,:])
  ux = np.array(np.squeeze(data1['u'][:,:,:]*(24*3600/10**5)))#cm/s->km/day
  vy = np.array(np.squeeze(data1['v'][:,:,:]*(24*3600/10**5)))
  #the number of days with valid ice motion estimates > 10 days with valid daily drift data required
  dset = nc.Dataset(filename1)
  n = np.array(np.squeeze(dset['number'][:,:,:]))
  idx=np.where(n<11)
  ux[idx]=np.nan
  vy[idx]=np.nan
  #Remove data with a spurious, exact value of zero.
  idx=np.where((ux==0) | (vy==0))#corrected
  ux[idx]=np.nan
  vy[idx]=np.nan
  metrics=compute_interp_correct(lon, lat, ux, vy, hems)
  ux1=metrics[0]
  vy1=metrics[1]
  if obs==0:
    np.savez('KIMURA_NH_200301_200712_siuv_corrected.npz', NHlon1, NHlat1, ux1, vy1)
  else:
    np.savez('KIMURA_SH_200301_200712_siuv_corrected.npz', SHlon1, SHlat1, ux1, vy1)
#----------------------------------
# Load 2003-2007 model output data |
#----------------------------------
filename = '/sea ice data/OMIP/All OMIP data/r1i1p1f1/sidrift/1980-2007/siu_SImon_IPSL-CM6A-LR_omip1_r1i1p1f1_gn_198001-200712.nc'
dset = nc.Dataset(filename)
lat = np.array(dset['nav_lat'][:,:])
lon = np.array(dset['nav_lon'][:,:])
u = np.array(np.squeeze(dset['siu'][276:336,:,:]))
filename = '/sea ice data/OMIP/All OMIP data/r1i1p1f1/sidrift/1980-2007/siv_SImon_IPSL-CM6A-LR_omip1_r1i1p1f1_gn_198001-200712.nc'
dset = nc.Dataset(filename)
v = np.array(np.squeeze(dset['siv'][276:336,:,:]))
idx=np.where((abs(u)>10000) | (abs(v)>10000) | (u==0) | (v==0))
u[idx]=np.nan
v[idx]=np.nan
idx1=np.where(abs(lon)>10000)
lon[idx1]=np.nan
lat[idx1]=np.nan
np.savez('IPSL-CM6A-LR_200301_200712_siuv.npz', lon, lat, u, v)

filename = '/sea ice data/OMIP/All OMIP data/r1i1p1f1/sidrift/1980-2007/siu_SImon_GFDL-CM4_omip1_r1i1p1f1_gn_198001-200712.nc'
dset = nc.Dataset(filename)
lat = np.array(dset['lat'][:,:])
lon = np.array(dset['lon'][:,:])
u = np.array(np.squeeze(dset['siu'][276:336,:,:]))
filename = '/sea ice data/OMIP/All OMIP data/r1i1p1f1/sidrift/1980-2007/siv_SImon_GFDL-CM4_omip1_r1i1p1f1_gn_198001-200712.nc'
dset = nc.Dataset(filename)
v = np.array(np.squeeze(dset['siv'][276:336,:,:]))
idx=np.where((abs(u)>10000) | (abs(v)>10000) | (u==0) | (v==0))
u[idx]=np.nan
v[idx]=np.nan
idx1=np.where(abs(lon)>10000)
lon[idx1]=np.nan
lat[idx1]=np.nan
idx1=np.where(lon<0)
lon[idx1]=lon[idx1]+360
lon1=np.zeros((1080,1440))
lat1=np.zeros((1080,1440))
u1=np.zeros(((60,1080,1440)))
v1=np.zeros(((60,1080,1440)))
lon1[:,0:960]=lon[:,480:1440]
lon1[:,960:1440]=lon[:,0:480]
lat1[:,0:960]=lat[:,480:1440]
lat1[:,960:1440]=lat[:,0:480]
u1[:,:,0:960]=u[:,:,480:1440]
u1[:,:,960:1440]=u[:,:,0:480]
v1[:,:,0:960]=v[:,:,480:1440]
v1[:,:,960:1440]=v[:,:,0:480]
np.savez('GFDL-CM4_200301_200712_siuv.npz', lon1, lat1, u1, v1)

filename = '/sea ice data/OMIP/All OMIP data/r1i1p1f1/sidrift/1980-2007/siu_SImon_GFDL-OM4p5B_omip1_r1i1p1f1_gn_198001-200712.nc'
dset = nc.Dataset(filename)
lat = np.array(dset['lat'][:,:])
lon = np.array(dset['lon'][:,:])
u = np.array(np.squeeze(dset['siu'][276:336,:,:]))
filename = '/sea ice data/OMIP/All OMIP data/r1i1p1f1/sidrift/1980-2007/siv_SImon_GFDL-OM4p5B_omip1_r1i1p1f1_gn_198001-200712.nc'
dset = nc.Dataset(filename)
v = np.array(np.squeeze(dset['siv'][276:336,:,:]))
idx=np.where((abs(u)>10000) | (abs(v)>10000) | (u==0) | (v==0))
u[idx]=np.nan
v[idx]=np.nan
idx1=np.where(abs(lon)>10000)
lon[idx1]=np.nan
lat[idx1]=np.nan
idx1=np.where(lon<0)
lon[idx1]=lon[idx1]+360
lon1=np.zeros((576,720))
lat1=np.zeros((576,720))
u1=np.zeros(((60,576,720)))
v1=np.zeros(((60,576,720)))
lon1[:,0:480]=lon[:,240:720]
lon1[:,480:720]=lon[:,0:240]
lat1[:,0:480]=lat[:,240:720]
lat1[:,480:720]=lat[:,0:240]
u1[:,:,0:480]=u[:,:,240:720]
u1[:,:,480:720]=u[:,:,0:240]
v1[:,:,0:480]=v[:,:,240:720]
v1[:,:,480:720]=v[:,:,0:240]
np.savez('GFDL-OM4p5B_200301_200712_siuv.npz', lon, lat, u, v)

path='/sea ice data/OMIP/All OMIP data/r1i1p1f1/sidrift/1980-2007/'
name=['siu_SImon_CMCC-CM2-HR4_omip2_r1i1p1f1_gn_198001-200712.nc','siu_SImon_CMCC-CM2-SR5_omip1_r1i1p1f1_gn_198001-200712.nc','siu_SImon_CMCC-CM2-SR5_omip2_r1i1p1f1_gn_198001-200712.nc','siu_SImon_EC-Earth3_omip1_r1i1p1f1_gn_198001-200712.nc','siu_SImon_EC-Earth3_omip2_r1i1p1f1_gn_198001-200712.nc',
'siu_SImon_MIROC6_omip1_r1i1p1f1_gn_198001-200712.nc','siu_SImon_MIROC6_omip2_r1i1p1f1_gn_198001-200712.nc','siu_SImon_MRI-ESM2-0_omip1_r1i1p1f1_gn_198001-200712.nc','siu_SImon_MRI-ESM2-0_omip2_r1i1p1f1_gn_198001-200712.nc','siu_SImon_NorESM2-LM_omip1_r1i1p1f1_gn_198001-200712.nc','siu_SImon_NorESM2-LM_omip2_r1i1p1f1_gn_198001-200712.nc']
name1=['siv_SImon_CMCC-CM2-HR4_omip2_r1i1p1f1_gn_198001-200712.nc','siv_SImon_CMCC-CM2-SR5_omip1_r1i1p1f1_gn_198001-200712.nc','siv_SImon_CMCC-CM2-SR5_omip2_r1i1p1f1_gn_198001-200712.nc','siv_SImon_EC-Earth3_omip1_r1i1p1f1_gn_198001-200712.nc','siv_SImon_EC-Earth3_omip2_r1i1p1f1_gn_198001-200712.nc',
'siv_SImon_MIROC6_omip1_r1i1p1f1_gn_198001-200712.nc','siv_SImon_MIROC6_omip2_r1i1p1f1_gn_198001-200712.nc','siv_SImon_MRI-ESM2-0_omip1_r1i1p1f1_gn_198001-200712.nc','siv_SImon_MRI-ESM2-0_omip2_r1i1p1f1_gn_198001-200712.nc','siv_SImon_NorESM2-LM_omip1_r1i1p1f1_gn_198001-200712.nc','siv_SImon_NorESM2-LM_omip2_r1i1p1f1_gn_198001-200712.nc']

for num in range(11):
  filename=(path + name[num])
  dset = nc.Dataset(filename)
  lat = np.array(dset['latitude'][:,:])
  lon = np.array(dset['longitude'][:,:])
  u = np.array(np.squeeze(dset['siu'][276:336,:,:]))
  filename = (path + name1[num])
  dset = nc.Dataset(filename)
  v = np.array(np.squeeze(dset['siv'][276:336,:,:]))
  idx=np.where((abs(u)>10000) | (abs(v)>10000) | (u==0) | (v==0))
  u[idx]=np.nan
  v[idx]=np.nan
  idx1=np.where(abs(lon)>10000)
  lon[idx1]=np.nan
  lat[idx1]=np.nan
  np.savez(name[num][10:28]+'_200301_200712_siuv.npz', lon, lat, u, v)
#----------------------------------------------------------------------------------
# rotate and interp model outputs into NSIDC0051 grid, and correct the ice vertors |
#----------------------------------------------------------------------------------
name=['CMCC-CM2-HR4_omip2_200301_200712_siuv.npz','CMCC-CM2-SR5_omip1_200301_200712_siuv.npz','CMCC-CM2-SR5_omip2_200301_200712_siuv.npz','EC-Earth3_omip1_r1_200301_200712_siuv.npz','EC-Earth3_omip2_r1_200301_200712_siuv.npz','GFDL-CM4_200301_200712_siuv.npz','GFDL-OM4p5B_200301_200712_siuv.npz','IPSL-CM6A-LR_200301_200712_siuv.npz','MIROC6_omip1_r1i1p_200301_200712_siuv.npz','MIROC6_omip2_r1i1p_200301_200712_siuv.npz','MRI-ESM2-0_omip1_r_200301_200712_siuv.npz','MRI-ESM2-0_omip2_r_200301_200712_siuv.npz','NorESM2-LM_omip1_r_200301_200712_siuv.npz','NorESM2-LM_omip2_r_200301_200712_siuv.npz']
for num in range(14):
  a=np.load(name[num])
  glamt=a['arr_0']# Longitude at T-points
  gphit=a['arr_1']# Latitude  at T-points
  u=a['arr_2'][:,:,:]*(24*3600/10**3) #m/s->km/day  #2003-2007 
  v=a['arr_3'][:,:,:]*(24*3600/10**3) #m/s->km/day  #2003-2007 
  glamv=a['arr_0']# Longitude from models - same for T-point and V-points here
  gphiv=a['arr_1']# Latitude from models - same for T-point and V-points here
  lon=a['arr_0']
  lat=a['arr_1']
  for i in range(2):
    if i==0:
      hems='n'
    else:
      hems='s'
    uv1=compute_rotate(glamt,gphit,u,v,glamv,gphiv,hems)
    u1=uv1[0]
    v1=uv1[1]
    uv2=compute_interp_correct(lon, lat, u1, v1, hems)
    u2=uv2[0]
    v2=uv2[1]
    if i==0:
      np.savez(name[num][0:37]+'_NH_corrected.npz', NHlon1, NHlat1, u2, v2)
    else:
      np.savez(name[num][0:37]+'_SH_corrected.npz', SHlon1, SHlat1, u2, v2)

# -----------------------------------------------------
# PART 8) A script computes the ice-motion MKE metrics |
# -----------------------------------------------------
#typical errors-differences between two observations
a=np.load('ICDC_NSIDC_NH_200301_200712_siuv_corrected.npz')
NHlon=a['arr_0']
NHlat=a['arr_1']
u1=a['arr_2'][:,:,:]/(24*3600/10**3)#km/day -> m/s    
v1=a['arr_3'][:,:,:]/(24*3600/10**3)   
NHMKE1=(u1**2+v1**2)/2
a=np.load('ICDC_NSIDC_SH_200301_200712_siuv_corrected.npz')
SHlon=a['arr_0']
SHlat=a['arr_1']
u1=a['arr_2'][:,:,:]/(24*3600/10**3)#km/day -> m/s    
v1=a['arr_3'][:,:,:]/(24*3600/10**3)#km/day -> m/s  
SHMKE1=(u1**2+v1**2)/2

a=np.load('KIMURA_NH_200301_200712_siuv_corrected.npz')
u2=a['arr_2'][0:60,:,:]/(24*3600/10**3)  
v2=a['arr_3'][0:60,:,:]/(24*3600/10**3) 
NHMKE2=(u2**2+v2**2)/2
a=np.load('KIMURA_SH_200301_200712_siuv1_corrected.npz')
u2=a['arr_2'][0:60,:,:]/(24*3600/10**3)  
v2=a['arr_3'][0:60,:,:]/(24*3600/10**3) 
SHMKE2=(u2**2+v2**2)/2
NHtyerror=compute_MKE_metrics(NHMKE1, NHMKE2)
SHtyerror=compute_MKE_metrics(SHMKE1, SHMKE2)

name0=['CMCC-CM2-HR4_omip2_200301_200712_siuv_NH_corrected.npz','CMCC-CM2-SR5_omip1_200301_200712_siuv_NH_corrected.npz','CMCC-CM2-SR5_omip2_200301_200712_siuv_NH_corrected.npz','EC-Earth3_omip1_r1_200301_200712_siuv_NH_corrected.npz','EC-Earth3_omip2_r1_200301_200712_siuv_NH_corrected.npz','GFDL-CM4_200301_200712_siuv_NH_corrected.npz','GFDL-OM4p5B_200301_200712_siuv_NH_corrected.npz','IPSL-CM6A-LR_200301_200712_siuv_NH_corrected.npz','MIROC6_omip1_r1i1p_200301_200712_siuv_NH_corrected.npz','MIROC6_omip2_r1i1p_200301_200712_siuv_NH_corrected.npz','MRI-ESM2-0_omip1_r_200301_200712_siuv_NH_corrected.npz','MRI-ESM2-0_omip2_r_200301_200712_siuv_NH_corrected.npz','NorESM2-LM_omip1_r_200301_200712_siuv_NH_corrected.npz','NorESM2-LM_omip2_r_200301_200712_siuv_NH_corrected.npz']
name=['CMCC-CM2-HR4_omip2_200301_200712_siuv_SH_corrected.npz','CMCC-CM2-SR5_omip1_200301_200712_siuv_SH_corrected.npz','CMCC-CM2-SR5_omip2_200301_200712_siuv_SH_corrected.npz','EC-Earth3_omip1_r1_200301_200712_siuv_SH_corrected.npz','EC-Earth3_omip2_r1_200301_200712_siuv_SH_corrected.npz','GFDL-CM4_200301_200712_siuv_SH_corrected.npz','GFDL-OM4p5B_200301_200712_siuv_SH_corrected.npz','IPSL-CM6A-LR_200301_200712_siuv_SH_corrected.npz','MIROC6_omip1_r1i1p_200301_200712_siuv_SH_corrected.npz','MIROC6_omip2_r1i1p_200301_200712_siuv_SH_corrected.npz','MRI-ESM2-0_omip1_r_200301_200712_siuv_SH_corrected.npz','MRI-ESM2-0_omip2_r_200301_200712_siuv_SH_corrected.npz','NorESM2-LM_omip1_r_200301_200712_siuv_SH_corrected.npz','NorESM2-LM_omip2_r_200301_200712_siuv_SH_corrected.npz']
NHerror_mean1=np.zeros(14)
SHerror_mean1=np.zeros(14)
Metrics_sidrift=np.zeros((17, 2))
for obs in range(2):
  for num in range(14):
      a=np.load(name0[num])
      uu=a['arr_2'][:,:,:]/(24*3600/10**3)
      vv=a['arr_3'][:,:,:]/(24*3600/10**3)
      NHMKE=(uu**2+vv**2)/2
      a=np.load(name[num])  
      uu=a['arr_2'][:,:,:]/(24*3600/10**3)
      vv=a['arr_3'][:,:,:]/(24*3600/10**3)
      SHMKE=(uu**2+vv**2)/2
    if obs==0:#models vs ICDC-NSIDC
      i='ICDC-NSIDC'
      NHerror_mean1[num]=compute_MKE_metrics(NHMKE, NHMKE1)
      SHerror_mean1[num]=compute_MKE_metrics(SHMKE, SHMKE1)
    else:
      i='KIMURA'
      NHerror_mean1[num]=compute_MKE_metrics(NHMKE, NHMKE2)
      SHerror_mean1[num]=compute_MKE_metrics(SHMKE, SHMKE2)

  Metrics_sidrift[0:14,0]=NHerror_mean1/NHtyerror
  Metrics_sidrift[0:14,1]=SHerror_mean1/SHtyerror 
  Metrics_sidrift[14,0]=np.mean(NHerror_mean1)/NHtyerror
  Metrics_sidrift[14,1]=np.mean(SHerror_mean1)/SHtyerror
  Metrics_sidrift[15,:]=(Metrics_sidrift[1,:]+Metrics_sidrift[3,:]+Metrics_sidrift[8,:]+Metrics_sidrift[10,:]+Metrics_sidrift[12,:])/5#OMIP1 mean
  Metrics_sidrift[16,:]=(Metrics_sidrift[2,:]+Metrics_sidrift[4,:]+Metrics_sidrift[9,:]+Metrics_sidrift[11,:]+Metrics_sidrift[13,:])/5#OMIP2 mean
  np.savez('sidrift_metrics_MKE_'+str(i)+'.npz', Metrics_sidrift, NHerror_mean1, SHerror_mean1)

# ---------------------------------------------------------------------------------
# PART 9) A script computes the ice-motion vector correlation coefficients metrics |
# ---------------------------------------------------------------------------------
#typical errors-differences between two observations
a=np.load('ICDC_NSIDC_NH_200301_200712_siuv_corrected.npz')
NHlon=a['arr_0']
NHlat=a['arr_1']
NHu1=a['arr_2'][:,:,:]/(24*3600/10**3)#km/day -> m/s    
NHv1=a['arr_3'][:,:,:]/(24*3600/10**3)   
a=np.load('ICDC_NSIDC_SH_200301_200712_siuv_corrected.npz')
SHlon=a['arr_0']
SHlat=a['arr_1']
SHu1=a['arr_2'][:,:,:]/(24*3600/10**3)#km/day -> m/s    
SHv1=a['arr_3'][:,:,:]/(24*3600/10**3)#km/day -> m/s  

a=np.load('KIMURA_NH_200301_200712_siuv_corrected.npz')
NHu2=a['arr_2'][0:60,:,:]/(24*3600/10**3)  
NHv2=a['arr_3'][0:60,:,:]/(24*3600/10**3) 
a=np.load('KIMURA_SH_200301_200712_siuv1_corrected.npz')
SHu2=a['arr_2'][0:60,:,:]/(24*3600/10**3)  
SHv2=a['arr_3'][0:60,:,:]/(24*3600/10**3) 

NHtyerror=compute_vectorcorr_metrics(NHu1, NHu2, NHv1, NHv2)
SHtyerror=compute_vectorcorr_metrics(SHu1, SHu2, SHv1, SHv2)
NHtyerror1=NHtyerror[0]
SHtyerror1=SHtyerror[0]

name0=['CMCC-CM2-HR4_omip2_200301_200712_siuv_NH_corrected.npz','CMCC-CM2-SR5_omip1_200301_200712_siuv_NH_corrected.npz','CMCC-CM2-SR5_omip2_200301_200712_siuv_NH_corrected.npz','EC-Earth3_omip1_r1_200301_200712_siuv_NH_corrected.npz','EC-Earth3_omip2_r1_200301_200712_siuv_NH_corrected.npz','GFDL-CM4_200301_200712_siuv_NH_corrected.npz','GFDL-OM4p5B_200301_200712_siuv_NH_corrected.npz','IPSL-CM6A-LR_200301_200712_siuv_NH_corrected.npz','MIROC6_omip1_r1i1p_200301_200712_siuv_NH_corrected.npz','MIROC6_omip2_r1i1p_200301_200712_siuv_NH_corrected.npz','MRI-ESM2-0_omip1_r_200301_200712_siuv_NH_corrected.npz','MRI-ESM2-0_omip2_r_200301_200712_siuv_NH_corrected.npz','NorESM2-LM_omip1_r_200301_200712_siuv_NH_corrected.npz','NorESM2-LM_omip2_r_200301_200712_siuv_NH_corrected.npz']
name=['CMCC-CM2-HR4_omip2_200301_200712_siuv_SH_corrected.npz','CMCC-CM2-SR5_omip1_200301_200712_siuv_SH_corrected.npz','CMCC-CM2-SR5_omip2_200301_200712_siuv_SH_corrected.npz','EC-Earth3_omip1_r1_200301_200712_siuv_SH_corrected.npz','EC-Earth3_omip2_r1_200301_200712_siuv_SH_corrected.npz','GFDL-CM4_200301_200712_siuv_SH_corrected.npz','GFDL-OM4p5B_200301_200712_siuv_SH_corrected.npz','IPSL-CM6A-LR_200301_200712_siuv_SH_corrected.npz','MIROC6_omip1_r1i1p_200301_200712_siuv_SH_corrected.npz','MIROC6_omip2_r1i1p_200301_200712_siuv_SH_corrected.npz','MRI-ESM2-0_omip1_r_200301_200712_siuv_SH_corrected.npz','MRI-ESM2-0_omip2_r_200301_200712_siuv_SH_corrected.npz','NorESM2-LM_omip1_r_200301_200712_siuv_SH_corrected.npz','NorESM2-LM_omip2_r_200301_200712_siuv_SH_corrected.npz']
NHerror_mean1=np.zeros(14)
SHerror_mean1=np.zeros(14)
NHerror=np.zeros(((14,304,448)))
SHerror=np.zeros(((14,316,332)))
Metrics_sidrift=np.zeros((17, 2))
for obs in range(2):
  for num in range(14):
    a=np.load(name0[num])
    NHuu=a['arr_2'][:,:,:]/(24*3600/10**3)
    NHvv=a['arr_3'][:,:,:]/(24*3600/10**3) 
    a=np.load(name[num])  
    SHuu=a['arr_2'][:,:,:]/(24*3600/10**3)
    SHvv=a['arr_3'][:,:,:]/(24*3600/10**3)
    if obs==0:#models vs ICDC-NSIDC
      i='ICDC-NSIDC' 
      NHcorr=compute_vectorcorr_metrics(NHuu, NHu1, NHvv, NHv1)
      SHcorr=compute_vectorcorr_metrics(SHuu, SHu1, SHvv, SHv1)
      NHerror_mean1[num]=NHcorr[1]
      SHerror_mean1[num]=SHcorr[1]
      NHerror[num,:,:]=NHcorr[0]
      SHerror[num,:,:]=SHcorr[0]
    else:
      i='KIMURA'
      NHcorr=compute_vectorcorr_metrics(NHuu, NHu1, NHvv, NHv1)
      SHcorr=compute_vectorcorr_metrics(SHuu, SHu2, SHvv, SHv2)
      NHerror_mean1[num]=NHcorr[1]
      SHerror_mean1[num]=SHcorr[1]
      NHerror[num,:,:]=NHcorr[0]
      SHerror[num,:,:]=SHcorr[0]

  Metrics_sidrift[0:14,0]=NHtyerror[1]/NHerror_mean1  #lower values higher skills
  Metrics_sidrift[0:14,1]=SHtyerror[1]/SHerror_mean1
  Metrics_sidrift[14,0]=NHtyerror[1]/np.mean(NHerror_mean1)
  Metrics_sidrift[14,1]=SHtyerror[1]/np.mean(SHerror_mean1)
  Metrics_sidrift[15,:]=(Metrics_sidrift[1,:]+Metrics_sidrift[3,:]+Metrics_sidrift[8,:]+Metrics_sidrift[10,:]+Metrics_sidrift[12,:])/5#OMIP1 mean
  Metrics_sidrift[16,:]=(Metrics_sidrift[2,:]+Metrics_sidrift[4,:]+Metrics_sidrift[9,:]+Metrics_sidrift[11,:]+Metrics_sidrift[13,:])/5#OMIP2 mean
  np.savez('sidrift_metrics_vectorcorr_'+str(i)+'.npz', Metrics_sidrift, NHerror_mean1, SHerror_mean1, NHerror, SHerror, NHtyerror1, SHtyerror1)

# -------------------------------------------------------------------------------------------------------
# PART 10) A script plots the ice vector correlation coefficients in the Arctic and Antarctic (Figs. 8-9)|
# -------------------------------------------------------------------------------------------------------
name1=['ICDC-NSIDCv4.1','CMCC-CM2-SR5/C','CMCC-CM2-SR5/J','CMCC-CM2-HR4/J','EC-Earth3/C','EC-Earth3/J','GFDL-CM4/C','MIROC6/C','MIROC6/J','GFDL-OM4p5B/C','MRI-ESM2-0/C','MRI-ESM2-0/J','IPSL-CM6A-LR/C','NorESM2-LM/C','NorESM2-LM/J']
name2=['ICDCNSIDC','CMCC-CM2-SR501','CMCC-CM2-SR502','CMCC-CM2-HR402','EC-Earth301','EC-Earth302','GFDL-CM401','MIROC601','MIROC602','GFDL-OM4p5B01','MRI-ESM2-001','MRI-ESM2-002','IPSL-CM6A-LR01','NorESM2-LM01','NorESM2-LM02']

a=np.load('sidrift_metrics_vectorcorr_KIMURA.npz')
NHvalues[0,:,:]=a['arr'][5]
NHvalues[1:15,:,:]=a['arr'][3]
SHvalues[0,:,:]=a['arr'][6]
SHvalues[1:15,:,:]=a['arr'][4]

for hems in range(2):
  if hems==0:
    lon=NHlon
    lat=NHlat
    corr=NHvalues
    hemisphere='n' 
  else:
    lon=SHlon
    lat=SHlat
    corr=SHvalues
    hemisphere='s' 

  fig=plt.figure(figsize=(5.5, 10))
  gs1 = gridspec.GridSpec(5, 3)
  gs1.update(wspace=0.06, hspace=0.08)
  for num in range(15):
    axes=plt.subplot(gs1[num]) 
    # Define a figure name
    colmap = "viridis"
    extmethod = "neither"
    varin='SIdrift'
    lb=0.1
    ub=0.9001

  # -----------------------
  #  Create the colorbar |
  # -----------------------
    nsteps=0.1
    clevs = np.arange(lb, ub, nsteps)
    extmethod = "both"
  # Load the colormap
    cmap = eval("plt.cm." + colmap)

  # Colors for values outside the colorbar
  # Get first and last colors of the colormap
    first_col = cmap(0)[0:3]
    last_col  = cmap(255)[0:3]
    first_col1 = cmap(10)[0:3]#added by Xia
  # Create "over" and "under" colors.
  # Take respectively the latest and first color and
  # darken them by 50%
    col_under = [i / 2 for i in first_col]
    col_over  = [i / 2 for i in last_col ]
 
  # --------------------
  # Create a projection |
  # --------------------
  # Determine if we are in the northern hemisphere or in the Southern hemisphere.
    if hemisphere == "n":
      boundlat = 50.
      l0 = 0.
    elif hemisphere == "s":
      boundlat = -50.
      l0 = 180.
    else:
      sys.exit("(map_polar_field) Hemisphere unkown")

  # Projection name
    projname = hemisphere + "plaea" 
    map = Basemap(projection = projname, boundinglat = boundlat, lon_0 = l0, resolution = 'l', round=True)
    x, y = map(lon, lat)
    yy = np.arange(0, y.shape[0], 5)
    xx = np.arange(0, x.shape[1], 5)
    points = np.meshgrid(yy, xx)

  # ----------------
  # Plot the figure |
  # ----------------
    field = np.squeeze(corr[num, :, :])
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
    cs = map.contourf(x, y, field, clevs, cmap = cmap, vmin=0.1, vmax=ub, latlon = False, extend = extmethod)
    cs.cmap.set_over(col_over)
    axes.set_title(name1[num],fontname='Arial', fontsize=7.5,fontweight='bold', loc='center',y=1.0, pad=3)

  cax = fig.add_axes([0.93, 0.25, 0.03, 0.49])
  cbar = fig.colorbar(cs, cax=cax,ticks=[0,0.2,0.4,0.6,0.8,1])
  cbar.ax.yaxis.set_ticks_position('both')
  cbar.ax.tick_params(direction='in',length=2)
  cbar.set_label('Ice-motion vector correlation coefficient (vs. KIMURA)', fontname ='Arial', fontsize = 8, fontweight='bold')
  for l in cbar.ax.get_yticklabels():
    l.set_fontproperties('Arial') 
    l.set_fontsize(8) 
    l.set_fontweight('bold')
 
  # Save figure 
  filename    = 'Figure' + str(hems + 8).zfill(1)     
  plt.savefig(filename + ".png", bbox_inches = "tight", dpi = 500)
  plt.close("fig")

# -------------------------------------------------------
# PART 11) A script plots the ice drift metrics (Fig. 10)|
# -------------------------------------------------------
Models=['CMCC-CM2-HR4/J','CMCC-CM2-SR5/C','CMCC-CM2-SR5/J','EC-Earth3/C','EC-Earth3/J','GFDL-CM4/C','GFDL-OM4p5B/C','IPSL-CM6A-LR/C','MIROC6/C','MIROC6/J','MRI-ESM2-0/C','MRI-ESM2-0/J','NorESM2-LM/C','NorESM2-LM/J','Model mean','Model mean/C','Model mean/J']
Variables=['Mean Kin. En. NH', 'Mean Kin. En. SH','Mean Kin. En. NH', 'Mean Kin. En. SH']
values=np.zeros((17, 4))
a=np.load('sidrift_metrics_MKE_ICDC-NSIDC.npz')
values[:,0:2]=a['arr_0']
a=np.load('sidrift_metrics_MKE_KIMURA.npz')
values[:,2:4]=a['arr_0']
dpi=100
squaresize = 220
figwidth = 18*squaresize/float(dpi)
figheight = 6*squaresize/float(dpi)
fig,ax1 = plt.subplots(1, figsize=(figwidth, figheight), dpi=dpi)
im,cbar = heatmap(values, Models,Variables, ax=ax1, cmap="OrRd", vmin=1, vmax=5) 
texts = annotate_heatmap(im, valfmt="{x:.2f}",size=16,threshold=4)
cbar.remove()
ax1.set_xticklabels(['Mean Kin. En. NH', 'Mean Kin. En. SH','Mean Kin. En. NH', 'Mean Kin. En. SH'])
plt.setp(ax1.get_xticklabels(), fontname='Arial', fontsize=16)
plt.setp(ax1.get_yticklabels(), fontname='Arial', fontsize=16)
cax = fig.add_axes([0.75, 0.113, 0.01, 0.765])
cbar = fig.colorbar(im, cax=cax,ticks=[1,2,3,4,5], orientation="vertical")
cbar.ax1.yaxis.set_ticks_position('both')
cbar.ax1.tick_params(direction='in',length=2,labelsize=16)
ax1.set_title("(a) Drift magnitude: \n models vs. NSIDC & KIMURA", fontname='Arial', fontsize=16)
plt.savefig('./Figure10a.png', bbox_inches = "tight", dpi = 500)

Variables=['Vect. Corr. NH','Vect. Corr. SH','Vect. Corr. NH','Vect. Corr. SH']
values=np.zeros((17, 4))
a=np.load('sidrift_metrics_vectorcorr_ICDC-NSIDC.npz')
values[:,0:2]=a['arr_0']
a=np.load('sidrift_metrics_vectorcorr_KIMURA.npz')
values[:,2:4]=a['arr_0']
dpi=100
squaresize = 220
figwidth = 18*squaresize/float(dpi)
figheight = 6*squaresize/float(dpi)
fig,ax1 = plt.subplots(1, figsize=(figwidth, figheight), dpi=dpi)
im,cbar = heatmap(values, Models,Variables, ax=ax1, cmap="OrRd", vmin=1, vmax=1.3) 
texts = annotate_heatmap(im, valfmt="{x:.2f}",size=16,threshold=1.24)
cbar.remove()
ax1.set_xticklabels(['Vect. Corr. NH','Vect. Corr. SH','Vect. Corr. NH','Vect. Corr. SH'])
plt.setp(ax1.get_xticklabels(), fontname='Arial', fontsize=16)
plt.setp(ax1.get_yticklabels(), fontname='Arial', fontsize=16)
cax = fig.add_axes([0.75, 0.113, 0.01, 0.765])
cbar = fig.colorbar(im, cax=cax,ticks=[1,1.1,1.2,1.3], orientation="vertical")
cbar.ax1.yaxis.set_ticks_position('both')
cbar.ax1.tick_params(direction='in',length=2,labelsize=16)
ax1.set_title("(b) Drift direction: \n models vs. NSIDC & KIMURA", fontname='Arial', fontsize=16)
plt.savefig('./Figure10b.png', bbox_inches = "tight", dpi = 500)

# ---------------------------------------------------------------------------------------
# PART 12) A script plots the February and September mean ice-motion mean kinetic energy |
#          differences in the Arctic and Antarctic (Figs. A9-A12);                       |
# ---------------------------------------------------------------------------------------
a=np.load('KIMURA_NH_200301_200712_siuv_corrected') 
NHlon_curv=a['arr_0']
NHlat_curv=a['arr_1']
u=a['arr_2'][0:60,:,:]/(24*3600/10**3)  
v=a['arr_3'][0:60,:,:]/(24*3600/10**3) 
NHMKE=(u**2+v**2)/2
NHMKE1 = np.array([np.nanmean(NHMKE[m::12,:,:], axis=0) for m in range(12)])
a=np.load('KIMURA_SH_200301_200712_siuv_corrected')
SHlon_curv=a['arr_0']
SHlat_curv=a['arr_1']
u=a['arr_2'][0:60,:,:]/(24*3600/10**3)  
v=a['arr_3'][0:60,:,:]/(24*3600/10**3) 
SHMKE=(u**2+v**2)/2
SHMKE1 = np.array([np.nanmean(SHMKE[m::12,:,:], axis=0) for m in range(12)])

# Define a figure properties
varin='MKE'
units='($\mathregular{m^2}$/$\mathregular{s^2}$)'

name=['ICDC-NSIDCv4.1','CMCC-CM2-SR5/C','CMCC-CM2-SR5/J','CMCC-CM2-HR4/J','EC-Earth3/C','EC-Earth3/J','GFDL-CM4/C','MIROC6/C','MIROC6/J','GFDL-OM4p5B/C','MRI-ESM2-0/C','MRI-ESM2-0/J','IPSL-CM6A-LR/C','NorESM2-LM/C','NorESM2-LM/J']#!!!!!!
filesNH=['ICDC_NSIDC_NH_200301_200712_siuv_corrected.npz','CMCC-CM2-SR5_omip1_200301_200712_siuv_corrected.npz','CMCC-CM2-SR5_omip2_200301_200712_siuv_corrected.npz','CMCC-CM2-HR4_omip2_200301_200712_siuv_corrected.npz','EC-Earth3_omip1_r1_200301_200712_siuv_corrected.npz','EC-Earth3_omip2_r1_200301_200712_siuv_corrected.npz','GFDL-CM4_200301_200712_siuv_corrected.npz','MIROC6_omip1_r1i1p_200301_200712_siuv_corrected.npz','MIROC6_omip2_r1i1p_200301_200712_siuv_corrected.npz','GFDL-OM4p5B_200301_200712_siuv_corrected.npz','MRI-ESM2-0_omip1_r_200301_200712_siuv_corrected.npz','MRI-ESM2-0_omip2_r_200301_200712_siuv_corrected.npz','IPSL-CM6A-LR_200301_200712_siuv_corrected.npz','NorESM2-LM_omip1_r_200301_200712_siuv_corrected.npz','NorESM2-LM_omip2_r_200301_200712_siuv_corrected.npz']
filesSH=['ICDC_NSIDC_SH_200301_200712_siuv_corrected.npz','CMCC-CM2-SR5_omip1_200301_200712_siuv_SH_corrected.npz','CMCC-CM2-SR5_omip2_200301_200712_siuv_SH_corrected.npz','CMCC-CM2-HR4_omip2_200301_200712_siuv_SH_corrected.npz','EC-Earth3_omip1_r1_200301_200712_siuv_SH_corrected.npz','EC-Earth3_omip2_r1_200301_200712_siuv_SH_corrected.npz','GFDL-CM4_200301_200712_siuv_SH_corrected.npz','MIROC6_omip1_r1i1p_200301_200712_siuv_SH_corrected.npz','MIROC6_omip2_r1i1p_200301_200712_siuv_SH_corrected.npz','GFDL-OM4p5B_200301_200712_siuv_SH_corrected.npz','MRI-ESM2-0_omip1_r_200301_200712_siuv_SH_corrected.npz','MRI-ESM2-0_omip2_r_200301_200712_siuv_SH_corrected.npz','IPSL-CM6A-LR_200301_200712_siuv_SH_corrected.npz','NorESM2-LM_omip1_r_200301_200712_siuv_SH_corrected.npz','NorESM2-LM_omip2_r_200301_200712_siuv_SH_corrected.npz']

for jt in range(4):
  if jt==0:
    jt1=8
    hemisphere = "n"
    lat=NHlat_curv
    lon=NHlon_curv
    files=filesNH
    MKE0=NHMKE1
  elif jt==1:
    jt1=1
    hemisphere = "n"
    lat=NHlat_curv
    lon=NHlon_curv
    files=filesNH
    MKE0=NHMKE1
  elif jt==2:
    jt1=1
    hemisphere = "s"
    lat=SHlat_curv 
    lon=SHlon_curv
    files=filesSH
    MKE0=SHMKE1
  elif jt==3:
    jt1=8
    hemisphere = "s"
    lat=SHlat_curv 
    lon=SHlon_curv
    files=filesSH
    MKE0=SHMKE1

  fig=plt.figure(figsize=(5, 9))
  gs1 = gridspec.GridSpec(5, 3)
  gs1.update(wspace=0.04, hspace=0.06) # set the spacing between axes. 
  for num in range(15):  
    axes=plt.subplot(gs1[num])
    if num==0:
      a=np.load(files[num])
      u=a['arr_2'][0:60,:,:]*1.357
      v=a['arr_3'][0:60,:,:]*1.357
    else:
      a=np.load(files[num])
      u=a['arr_2'][0:60,:,:]/(24*3600/10**3)  
      v=a['arr_3'][0:60,:,:]/(24*3600/10**3) 
    MKE=(u**2+v**2)/2
    MKE1 = np.array([np.nanmean(MKE[m::12,:,:], axis=0) for m in range(12)])
    field=MKE1-MKE0

    # Create the colorbar
    # Load the colormap
    lb=-0.008
    ub=0.00800001
    nsteps=0.001
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
    field1 = np.squeeze(field[jt1, :, :])
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
    norm = colors.DivergingNorm(vmin=lb, vcenter=0, vmax=ub)
    cs = map.contourf(x, y, field1, clevs, cmap = cmap, vmin=lb, vmax=ub, norm = norm, latlon = False, extend = extmethod)
    cs.cmap.set_under(col_under)#white')#
    cs.cmap.set_over(col_over)
    axes.set_title(name[num],fontname='Arial', fontsize=7.5,fontweight='bold', loc='center',y=1.0, pad=3)
  
  cax = fig.add_axes([0.93, 0.25, 0.03, 0.49])
  cbar = fig.colorbar(cs, cax=cax,ticks=[-0.008,-0.004,0,0.004,0.008])
  cbar.ax.yaxis.set_ticks_position('both')
  cbar.ax.tick_params(direction='in',length=2)
  cbar.set_label('Ice-motion MKE difference (vs. KIMURA, $\mathregular{m^2}$/$\mathregular{s^2}$)' ,fontname = 'Arial', fontsize = 8 , fontweight = 'bold')
  for l in cbar.ax.get_yticklabels():
    l.set_fontproperties('Arial') 
    l.set_fontsize(8) 
    l.set_fontweight('bold')
 
  # Save figure 
  filename    = 'FigureA' + str(jt + 9).zfill(1) ! 
  imageformat = "png"  
  dpi         = 500     
  plt.savefig(filename + "." + imageformat, bbox_inches = "tight", dpi = dpi)
  print('Figure ' + filename + "." + imageformat + " printed")
  plt.close("fig")
