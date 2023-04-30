# Author: Xia Lin
# Date:   Feb 2021
# Contents:
# 1) A function that computes sea ice extent 
# 2) A function that computes ice extent errors between two datasets
#    on the mean state, interannual variability and trend in order to get the metrics;
#    The input data is a numpy array but a NetCDF file 
# 3) The heatmap function  
# 4) The annotate heatmap function (The heatmap and annotate heatmap functions were written 
#    by following https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html)
# 5) A script calls the function 1) to calculate the ice extent from ice concentration, 
#    and then plots the mean seasonal cycle of ice extent in the Arctic and Antarctic (Fig. 3)
# 6) A script plots the monthly anomalies of ice extent from the observational 
#    and model means in the Arctic and Antarctic (Fig. 4)  
# 7) A script calls the function 1) to calculate the ice extent from ice concentration
#    and then calls the function 2) to compute the ice extent metrics
# 8) A script calls the functions 3) and 4) to plot the ice extent metrics (Figs. 5a,b)

# ---------------------------------------------
# PART 1) The ice extent calculation function  |   
# ---------------------------------------------
def compute_extent(concentration, cellarea, threshold = 0.15, mask = 1):
  '''Input: - sea ice concentration 
            - cellarea: array of grid cell areas (square meters)
            - Threshold over which to consider cell as ice covered
            - mask (1 on ocean, 0 on continent)

     Output: Sea ice extent in the region defined by the mask
  '''
  import sys
  import numpy as np
  
  if np.max(concentration) < 10.0:
    sys.exit("(compute_extent): concentration seems to not be in percent")

  if len(concentration.shape) == 3:
    nt, ny, nx = concentration.shape
    ext = np.asarray([np.nansum( (concentration[jt, :, :] > threshold) * cellarea * mask) / 1e12 for jt in range(nt)])
  elif len(concentration.shape) == 2:
    ext = np.nansum((concentration > threshold) * cellarea * mask) / 1e12
  else:
    sys.exit("(compute_extent): concentration has not 2 nor 3 dimensions")

  return ext

# ----------------------------------------
# PART 2) The ice extent errors function  |   
# ----------------------------------------
def compute_siext_metrics(extent, extent1):
  '''Input: - sea ice extent (10^6 km2) in the Arctic or Antarctic from two datasets
            
     Output: Errors between two ice extent datasets of mean cycle, anomaly variance and trend in the Arctic or Antarctic
  '''
  cycle = np.array([np.nanmean(extent[m::12]) for m in range(12)])
  cycle1 = np.array([np.nanmean(extent1[m::12]) for m in range(12)])
  ano = np.array([extent[j] - cycle[j%12] for j in range(len(extent))])
  ano1 = np.array([extent1[j] - cycle1[j%12] for j in range(len(extent1))])
  years = np.arange(1980, 2008, 1.0/12)
  parameter = np.polyfit(years, ano, 1)#1 linear function
  parameter1 = np.polyfit(years, ano1, 1)#1 linear function

  ndpm=[31,28,31,30,31,30,31,31,30,31,30,31];
  error_mean=np.sum(abs(cycle-cycle1)*ndpm)/np.sum(ndpm)
  error_std=abs(np.std(ano)-np.std(ano1))
  error_trend=abs(parameter[0]-parameter1[0])*10*12 #10^6 km2/decade 

  return error_mean, error_std, error_trend

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

# ----------------------------------------------------------------------------------------------------
# PART 5) A script to plot the mean seasonal cycle of ice extent in the Arctic and Antarctic (Fig. 3) |
# ----------------------------------------------------------------------------------------------------
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

np.random.seed(3) #to fix the colors of CMIP6
colors = [np.random.random(3) for j in range(16)]
#NSIDC0051
a=np.load('NSIDC0051_1980_2007_siconc.npz')
NHconcentration1=a['arr_2']/100 
SHconcentration1=a['arr_5']/100
NHcellarea=a['arr_6']
SHcellarea=a['arr_7']
NHextent1=compute_extent(NHconcentration1, NHcellarea, threshold = 0.15, mask = 1)
SHextent1=compute_extent(SHconcentration1, SHcellarea, threshold = 0.15, mask = 1)
#seasonal cycle
NHcycle1 = np.array([np.nanmean(NHextent1[m::12]) for m in range(12)])
SHcycle1 = np.array([np.nanmean(SHextent1[m::12]) for m in range(12)])
NHano1 = np.array([NHextent1[j] - NHcycle1[j%12] for j in range(len(NHextent1))])
SHano1 = np.array([SHextent1[j] - SHcycle1[j%12] for j in range(len(SHextent1))])

#OSI450
a=np.load('OSI450_1980_2007_siconc.npz')
NHconcentration2=a['arr_2']/100
SHconcentration2=a['arr_5']/100
NHextent2=compute_extent(NHconcentration2, NHcellarea, threshold = 0.15, mask = 1)
SHextent2=compute_extent(SHconcentration2, SHcellarea, threshold = 0.15, mask = 1)
#seasonal cycle
NHcycle2 = np.array([np.nanmean(NHextent2[m::12]) for m in range(12)])
SHcycle2 = np.array([np.nanmean(SHextent2[m::12]) for m in range(12)])
NHano2 = np.array([NHextent2[j] - NHcycle2[j%12] for j in range(len(NHextent2))])
SHano2 = np.array([SHextent2[j] - SHcycle2[j%12] for j in range(len(SHextent2))])

fig,((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(12,9))
months = np.arange(1,13)
ax1.grid(linestyle=':', zorder=1)
ax1.scatter(months, NHcycle2, color = 'black', label='NSIDC-0051', marker='x', s=28, zorder=35)
ax1.scatter(months, NHcycle1, color = 'c', label='OSI-450', marker='+', s=28, zorder=30)
name=['CMCC-CM2-HR4_omip2_1980_2007_siconc.npz', 'CMCC-CM2-SR5_omip1_1980_2007_siconc.npz', 'CMCC-CM2-SR5_omip2_1980_2007_siconc.npz', 'EC-Earth3_omip1_r1_1980_2007_siconc.npz','EC-Earth3_omip2_r1_1980_2007_siconc.npz', 'GFDL-CM4_omip1_r1i_1980_2007_siconc.npz', 'GFDL-OM4p5B_omip1__1980_2007_siconc.npz', 'IPSL-CM6A-LR_omip1_1980_2007_siconc.npz',  'MIROC6_omip1_r1i1p_1980_2007_siconc.npz', 'MIROC6_omip2_r1i1p_1980_2007_siconc.npz', 'MRI-ESM2-0_omip1_r_1980_2007_siconc.npz', 'MRI-ESM2-0_omip2_r_1980_2007_siconc.npz', 'NorESM2-LM_omip1_r_1980_2007_siconc.npz', 'NorESM2-LM_omip2_r_1980_2007_siconc.npz']
name1=['CMCC-CM2-HR4/J','CMCC-CM2-SR5/C','CMCC-CM2-SR5/J','EC-Earth3/C','EC-Earth3/J','GFDL-CM4/C','GFDL-OM4p5B/C','IPSL-CM6A-LR/C','MIROC6/C','MIROC6/J','MRI-ESM2-0/C','MRI-ESM2-0/J','NorESM2-LM/C','NorESM2-LM/J']
months = np.arange(1,13)
MNHcycle=np.zeros((14,12))
for num in range(14):
  a=np.load(path + name[num])
  NHconcentration=a['arr_2']/100
  NHextent=compute_extent(NHconcentration, NHcellarea, threshold = 0.15, mask = 1)
  NHcycle = np.array([np.nanmean(NHextent[m::12]) for m in range(12)])
  MNHcycle[num,:] = NHcycle
MNHcycle0= np.nanmean(MNHcycle,axis=0)
MNHcycle11 = (MNHcycle[1,:]+MNHcycle[3,:]+MNHcycle[8,:]+MNHcycle[10,:]+MNHcycle[12,:])/5
MNHcycle22 = (MNHcycle[2,:]+MNHcycle[4,:]+MNHcycle[9,:]+MNHcycle[11,:]+MNHcycle[13,:])/5
ax1.plot(months, MNHcycle0, color = 'firebrick', label='Model mean',linewidth=2, zorder=20)
ax1.plot(months, MNHcycle11, color = 'red', label='Model mean/C',linewidth=2, zorder=20)
ax1.plot(months, MNHcycle22, color = 'red', label='Model mean/J',linewidth=2,linestyle='-.', zorder=20)
ax1.set_ylabel('Arctic ice extent ($\mathregular{10^6}$ $\mathregular{km^2}$)', fontsize=13, fontweight = 'bold')
xticks=ax1.set_xticks(np.arange(1,13,1))
yticks=ax1.set_yticks(np.arange(6,17,2)) 
ax1.text(11, 15.5, '(a)', fontsize=13, fontweight = 'bold')
labels = [item.get_text() for item in ax1.get_xticklabels()]
labels = ['J','F','M','A','M','J','J','A','S','O','N','D']
ax1.set_xticklabels(labels,fontname='Arial', fontsize=13)
for tick in ax1.get_yticklabels():
  tick.set_fontname('Arial')
  tick.set_fontsize(13)

ax2.grid(linestyle=':', zorder=1)
ax2.scatter(months, SHcycle2, color = 'black', label='NSIDC-0051', marker='x', s=28, zorder=35)
ax2.scatter(months, SHcycle1, color = 'c', label='OSI-450', marker='+', s=28, zorder=30)
months = np.arange(1,13)
MSHcycle=np.zeros((14,12))
for num in range(14):
  a=np.load(path + name[num])
  SHconcentration=a['arr_5']/100
  SHextent=compute_extent(SHconcentration, SHcellarea, threshold = 0.15, mask = 1)
  SHcycle = np.array([np.nanmean(SHextent[m::12]) for m in range(12)])
  MSHcycle[num,:] = SHcycle
MSHcycle0= np.nanmean(MSHcycle,axis=0)
MSHcycle11 = (MSHcycle[1,:]+MSHcycle[3,:]+MSHcycle[8,:]+MSHcycle[10,:]+MSHcycle[12,:])/5
MSHcycle22 = (MSHcycle[2,:]+MSHcycle[4,:]+MSHcycle[9,:]+MSHcycle[11,:]+MSHcycle[13,:])/5
ax2.plot(months, MSHcycle0, color = 'firebrick', label='Model mean',linewidth=2, zorder=20)
ax2.plot(months, MSHcycle11, color = 'red', label='Model mean/C',linewidth=2, zorder=20)
ax2.plot(months, MSHcycle22, color = 'red', label='Model mean/J',linewidth=2,linestyle='-.', zorder=20)
ax2.set_ylabel('Antarctic ice extent ($\mathregular{10^6}$ $\mathregular{km^2}$)', fontsize=13, fontweight = 'bold')
ax2.legend(prop={'family':'Arial', "size":13, 'weight':'bold'})
ax2.set_xticks(np.arange(1,13,1))
ax2.set_yticks(np.arange(0,21,2))
ax2.text(11, 19, '(b)', fontsize=13, fontweight = 'bold')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, borderaxespad = 0.,prop={'family':'Arial', "size":12, 'weight':'bold'})#'center left'
labels = [item.get_text() for item in ax2.get_xticklabels()]
labels = ['J','F','M','A','M','J','J','A','S','O','N','D']
ax2.set_xticklabels(labels,fontname='Arial', fontsize=13)
for tick in ax2.get_yticklabels():
  tick.set_fontname('Arial')
  tick.set_fontsize(13)

months = np.arange(1,13)
ax3.grid(linestyle=':', zorder=1)
name=['CMCC-CM2-HR4_omip2_1980_2007_siconc.npz', 'CMCC-CM2-SR5_omip1_1980_2007_siconc.npz', 'CMCC-CM2-SR5_omip2_1980_2007_siconc.npz', 'EC-Earth3_omip1_r1_1980_2007_siconc.npz','EC-Earth3_omip2_r1_1980_2007_siconc.npz', 'GFDL-CM4_omip1_r1i_1980_2007_siconc.npz', 'GFDL-OM4p5B_omip1__1980_2007_siconc.npz', 'IPSL-CM6A-LR_omip1_1980_2007_siconc.npz',  'MIROC6_omip1_r1i1p_1980_2007_siconc.npz', 'MIROC6_omip2_r1i1p_1980_2007_siconc.npz', 'MRI-ESM2-0_omip1_r_1980_2007_siconc.npz', 'MRI-ESM2-0_omip2_r_1980_2007_siconc.npz', 'NorESM2-LM_omip1_r_1980_2007_siconc.npz', 'NorESM2-LM_omip2_r_1980_2007_siconc.npz']
name1=['CMCC-CM2-HR4/J','CMCC-CM2-SR5/C','CMCC-CM2-SR5/J','EC-Earth3/C','EC-Earth3/J','GFDL-CM4/C','GFDL-OM4p5B/C','IPSL-CM6A-LR/C','MIROC6/C','MIROC6/J','MRI-ESM2-0/C','MRI-ESM2-0/J','NorESM2-LM/C','NorESM2-LM/J']
months = np.arange(1,13)
MNHcycle=np.zeros((14,12))
for num in range(14):
  a=np.load(path + name[num])
  NHconcentration=a['arr_2']/100
  NHextent=compute_extent(NHconcentration, NHcellarea, threshold = 0.15, mask = 1)
  NHcycle = np.array([np.nanmean(NHextent[m::12]) for m in range(12)])
  MNHcycle[num,:] = NHcycle
  if (num==2):
    ax3.plot(months, NHcycle, color = colors[num+2], label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10) 
  elif (num==3):
    ax3.plot(months, NHcycle, color = 'darkgreen', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==4):
    ax3.plot(months, NHcycle, color = 'darkgreen', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10)
  elif (num==8):
    ax3.plot(months, NHcycle, color = 'darkorange', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==9):
    ax3.plot(months, NHcycle, color = 'darkorange', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10)
  elif (num==10):
    ax3.plot(months, NHcycle, color = 'tab:grey', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==11):
    ax3.plot(months, NHcycle, color = 'tab:grey', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10)
  elif (num==12):
    ax3.plot(months, NHcycle, color = 'gold', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==13):
    ax3.plot(months, NHcycle, color = 'gold', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10)
  elif (num==5):
    ax3.plot(months, NHcycle, color = 'blue', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  else:
    ax3.plot(months, NHcycle, color = colors[num+3], label=name1[num],linewidth=2,alpha=0.8, zorder=10) 
ax3.set_xlabel('Month',fontsize=13, fontweight = 'bold')
ax3.set_ylabel('Arctic ice extent ($\mathregular{10^6}$ $\mathregular{km^2}$)', fontsize=13, fontweight = 'bold')
xticks=ax3.set_xticks(np.arange(1,13,1))
yticks=ax3.set_yticks(np.arange(2,19,2)) 
ax3.text(11, 17, '(c)', fontsize=13, fontweight = 'bold')
labels = [item.get_text() for item in ax3.get_xticklabels()]
labels = ['J','F','M','A','M','J','J','A','S','O','N','D']
ax3.set_xticklabels(labels,fontname='Arial', fontsize=13)
for tick in ax3.get_yticklabels():
  tick.set_fontname('Arial')
  tick.set_fontsize(13)

ax4.grid(linestyle=':', zorder=1)
months = np.arange(1,13)
MSHcycle=np.zeros((14,12))
for num in range(14):
  a=np.load(path + name[num])
  SHconcentration=a['arr_5']/100
  SHextent=compute_extent(SHconcentration, SHcellarea, threshold = 0.15, mask = 1)
  SHcycle = np.array([np.nanmean(SHextent[m::12]) for m in range(12)])
  MSHcycle[num,:] = SHcycle
  if (num==2):
    ax4.plot(months, SHcycle, color = colors[num+2], label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10) 
  elif (num==3):
    ax4.plot(months, SHcycle, color = 'darkgreen', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==4):
    ax4.plot(months, SHcycle, color = 'darkgreen', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10)
  elif (num==8):
    ax4.plot(months, SHcycle, color = 'darkorange', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==9):
    ax4.plot(months, SHcycle, color = 'darkorange', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10)
  elif (num==10):
    ax4.plot(months, SHcycle, color = 'tab:grey', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==11):
    ax4.plot(months, SHcycle, color = 'tab:grey', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10)
  elif (num==12):
    ax4.plot(months, SHcycle, color = 'gold', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==13):
    ax4.plot(months, SHcycle, color = 'gold', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10)
  elif (num==5):
    ax4.plot(months, SHcycle, color = 'blue', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  else:
    ax4.plot(months, SHcycle, color = colors[num+3], label=name1[num],linewidth=2,alpha=0.8, zorder=10)
ax4.set_xlabel('Month',fontsize=13, fontweight = 'bold')
ax4.set_ylabel('Antarctic ice extent ($\mathregular{10^6}$ $\mathregular{km^2}$)', fontsize=13, fontweight = 'bold')
ax4.set_xticks(np.arange(1,13,1))
ax4.set_yticks(np.arange(0,24,2))
ax4.text(11, 20.5, '(d)', fontsize=13, fontweight = 'bold')
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, prop={'family':'Arial',"size":12, 'weight':'bold'})
labels = [item.get_text() for item in ax4.get_xticklabels()]
labels = ['J','F','M','A','M','J','J','A','S','O','N','D']
ax4.set_xticklabels(labels,fontname='Arial', fontsize=13)
for tick in ax4.get_yticklabels():
  tick.set_fontname('Arial')
  tick.set_fontsize(13)
plt.savefig("./Figure3.png", bbox_inches = "tight", dpi = 500)                         
plt.close()

# -------------------------------------------------------------------------
# PART 6) A script to plot the monthly anomalies of ice extent from the    |
#         observational and model mean in the Arctic and Antarctic (Fig. 4)|
# -------------------------------------------------------------------------
#Anomalies & Trend
#Multi-model mean vs observations
NHextent3=(NHextent1+NHextent2)/2
SHextent3=(SHextent1+SHextent2)/2
NHcycle1 = np.array([np.nanmean(NHextent3[m::12]) for m in range(12)])
SHcycle1 = np.array([np.nanmean(SHextent3[m::12]) for m in range(12)])
NHano1 = np.array([NHextent3[j] - NHcycle1[j%12] for j in range(len(NHextent3))])
SHano1 = np.array([SHextent3[j] - SHcycle1[j%12] for j in range(len(SHextent3))])

MNHextent=np.zeros((14,336))
MSHextent=np.zeros((14,336))
for num in range(14):
  a=np.load(path + name[num])
  NHconcentration=a['arr_2']/100
  NHextent=compute_extent(NHconcentration, NHcellarea, threshold = 0.15, mask = 1)
  MNHextent[num,:] = NHextent
  SHconcentration=a['arr_5']/100
  SHextent=compute_extent(SHconcentration, SHcellarea, threshold = 0.15, mask = 1)
  MSHextent[num,:] = SHextent
MNHextent1 = (MNHextent[1,:]+MNHextent[3,:]+MNHextent[8,:]+MNHextent[10,:]+MNHextent[12,:])/5
MSHextent1 = (MSHextent[1,:]+MSHextent[3,:]+MSHextent[8,:]+MSHextent[10,:]+MSHextent[12,:])/5
MNHextent2 = (MNHextent[2,:]+MNHextent[4,:]+MNHextent[9,:]+MNHextent[11,:]+MNHextent[13,:])/5
MSHextent2 = (MSHextent[2,:]+MSHextent[4,:]+MSHextent[9,:]+MSHextent[11,:]+MSHextent[13,:])/5
MNHcycle1 = np.array([np.nanmean(MNHextent1[m::12]) for m in range(12)])
MNHano1 = np.array([MNHextent1[j] - MNHcycle1[j%12] for j in range(len(MNHextent1))])
MSHcycle1 = np.array([np.nanmean(MSHextent1[m::12]) for m in range(12)])
MSHano1 = np.array([MSHextent1[j] - MSHcycle1[j%12] for j in range(len(MSHextent1))])
MNHcycle2 = np.array([np.nanmean(MNHextent2[m::12]) for m in range(12)])
MNHano2 = np.array([MNHextent2[j] - MNHcycle2[j%12] for j in range(len(MNHextent2))])
MSHcycle2 = np.array([np.nanmean(MSHextent2[m::12]) for m in range(12)])
MSHano2 = np.array([MSHextent2[j] - MSHcycle2[j%12] for j in range(len(MSHextent2))])

fig,((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(12,6))
years = np.arange(1980, 2008, 1.0/12)
parameter1 = np.polyfit(years, NHano1, 1)
f1 = np.poly1d(parameter1)
ax1.plot(years, NHano1, color = 'black', label='OBS-NH')
ax1.plot(years, f1(years), color = 'black',linestyle='--')
parameter = np.polyfit(years, MNHano1, 1)
f = np.poly1d(parameter)
NHstd_ano1=np.nanstd(NHano1,axis=0)
MNHstd_ano1=np.nanstd(MNHano1,axis=0)
ax1.plot(years, MNHano1, color = 'tab:green')
ax1.plot(years, f(years), color = 'tab:green',linestyle='--')
ax1.text(2007, 2.5, '(a)', fontsize=13, fontweight = 'bold')
ax1.text(1980, -2.0, 'OBS-NH', color ='black',fontsize=11, fontweight = 'bold') 
ax1.text(1987, -2.0, 'Std.:0.54 Trend: -0.53', color ='black',fontsize=11, fontweight = 'bold') 
ax1.text(1980, -2.5, 'OMIP1-NH',color ='tab:green',fontsize=11, fontweight = 'bold')
ax1.text(1987, -2.5, 'Std.:0.61 Trend: -0.47',color ='tab:green',fontsize=11, fontweight = 'bold')
tick_spacing = 1
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax1.set_xticklabels(['','','1980','','','','','1985','','','','','1990','','','','','1995','','','','','2000','','','','','2005'])
ax1.set_yticks(np.arange(-3,4,1))
ax1.grid(linestyle=':')
ax1.tick_params(labelsize=13, direction='in')
ax1.set_ylabel('$\mathregular{10^6}$ $\mathregular{km^2}$', fontsize=13, fontweight = 'bold')

years = np.arange(1980, 2008, 1.0/12)
parameter1 = np.polyfit(years, NHano1, 1)
f1 = np.poly1d(parameter1)
ax2.plot(years, NHano1, color = 'black', label='OBS-NH')
ax2.plot(years, f1(years), color = 'black',linestyle='--')
parameter = np.polyfit(years, MNHano2, 1)
f = np.poly1d(parameter)
NHstd_ano1=np.nanstd(NHano1,axis=0)
MNHstd_ano2=np.nanstd(MNHano2,axis=0)
ax2.plot(years, MNHano2, color = 'tab:orange')
ax2.plot(years, f(years), color = 'tab:orange',linestyle='--')
ax2.text(2007, 2.5, '(b)', fontsize=13, fontweight = 'bold')
ax2.text(1980, -2.0, 'OBS-NH', color ='black',fontsize=11, fontweight = 'bold') 
ax2.text(1980, -2.5, 'OMIP2-NH',color ='tab:orange',fontsize=11, fontweight = 'bold')
ax2.text(1987, -2.5, 'Std.:0.49 Trend: -0.42',color ='tab:orange',fontsize=11, fontweight = 'bold')
tick_spacing = 1
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax2.set_xticklabels(['','','1980','','','','','1985','','','','','1990','','','','','1995','','','','','2000','','','','','2005'])
ax2.set_yticks(np.arange(-3,4,1))
ax2.grid(linestyle=':')
ax2.tick_params(labelsize=13, direction='in')
ax2.set_ylabel('$\mathregular{10^6}$ $\mathregular{km^2}$', fontsize=13, fontweight = 'bold')
#SH
parameter1 = np.polyfit(years, SHano1, 1)
f1 = np.poly1d(parameter1)
ax3.plot(years, SHano1, color = 'black', label='OBS-SH')
ax3.plot(years, f1(years), color = 'black',linestyle='--')
parameter = np.polyfit(years, MSHano1, 1)
f = np.poly1d(parameter)
SHstd_ano1=np.nanstd(SHano1,axis=0)
MSHstd_ano1=np.nanstd(MSHano1,axis=0)
ax3.plot(years, MSHano1, color = 'tab:green')
ax3.plot(years, f(years), color = 'tab:green',linestyle='--')
ax3.text(2007, 2.5, '(c)', fontsize=13, fontweight = 'bold')
ax3.text(1980, -2.0, 'OBS-SH', color ='black',fontsize=11, fontweight = 'bold') 
ax3.text(1987, -2.0, 'Std.:0.44 Trend: 0.15', color ='black',fontsize=11, fontweight = 'bold') 
ax3.text(1980, -2.5, 'OMIP1-SH',color ='tab:green',fontsize=11, fontweight = 'bold')
ax3.text(1987, -2.5, 'Std.:0.52 Trend: 0.01',color ='tab:green',fontsize=11, fontweight = 'bold')
tick_spacing = 1
ax3.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax3.set_xticklabels(['','','1980','','','','','1985','','','','','1990','','','','','1995','','','','','2000','','','','','2005'])
ax3.set_yticks(np.arange(-3,4,1))
ax3.grid(linestyle=':')
ax3.tick_params(labelsize=13, direction='in')
ax3.set_ylabel('$\mathregular{10^6}$ $\mathregular{km^2}$', fontsize=13, fontweight = 'bold')
ax3.set_xlabel('year', fontsize=13, fontweight = 'bold')

years = np.arange(1980, 2008, 1.0/12)
parameter1 = np.polyfit(years, SHano1, 1)
f1 = np.poly1d(parameter1)
ax4.plot(years, SHano1, color = 'black', label='OBS-SH')
ax4.plot(years, f1(years), color = 'black',linestyle='--')
parameter = np.polyfit(years, MSHano2, 1)
f = np.poly1d(parameter)
SHstd_ano1=np.nanstd(SHano1,axis=0)
MSHstd_ano2=np.nanstd(MSHano2,axis=0)
ax4.plot(years, MSHano2, color = 'tab:orange')
ax4.plot(years, f(years), color = 'tab:orange',linestyle='--')
ax4.text(2007, 2.5, '(d)', fontsize=13, fontweight = 'bold')
ax4.text(1980, -2.0, 'OBS-SH', color ='black',fontsize=11, fontweight = 'bold') 
ax4.text(1980, -2.5, 'OMIP2-SH',color ='tab:orange',fontsize=11, fontweight = 'bold')
ax4.text(1987, -2.5, 'Std.:0.38 Trend: 0.02',color ='tab:orange',fontsize=11, fontweight = 'bold')
tick_spacing = 1
ax4.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax4.set_xticklabels(['','','1980','','','','','1985','','','','','1990','','','','','1995','','','','','2000','','','','','2005'])
ax4.set_yticks(np.arange(-3,4,1))
ax4.grid(linestyle=':')
ax4.tick_params(labelsize=13, direction='in')
ax4.set_ylabel('$\mathregular{10^6}$ $\mathregular{km^2}$', fontsize=13, fontweight = 'bold')
ax4.set_xlabel('year', fontsize=13, fontweight = 'bold')
plt.savefig("./Figure4.png", dpi = 500, bbox_inches = "tight")
plt.close()

# ---------------------------------------------------------
# PART 7) A script computes the ice extent and its metrics |
# ---------------------------------------------------------
#typical errors-differences between two observations
a=np.load('NSIDC0051_1980_2007_siconc.npz')
NHconcentration1=a['arr_2']/100
SHconcentration1=a['arr_5']/100
NHcellarea=a['arr_6']
SHcellarea=a['arr_7']
NHextent1=compute_extent(NHconcentration1, NHcellarea, threshold = 0.15, mask = 1)
SHextent1=compute_extent(SHconcentration1, SHcellarea, threshold = 0.15, mask = 1)
a=np.load('OSI450_1980_2007_siconc.npz')
NHconcentration2=a['arr_2']/100 
SHconcentration2=a['arr_5']/100
NHextent2=compute_extent(NHconcentration2, NHcellarea, threshold = 0.15, mask = 1)
SHextent2=compute_extent(SHconcentration2, SHcellarea, threshold = 0.15, mask = 1)
NHtyerror=compute_siext_metrics(NHextent1, NHextent2)
SHtyerror=compute_siext_metrics(SHextent1, SHextent2)
name=['CMCC-CM2-HR4_omip2_1980_2007_siconc.npz', 'CMCC-CM2-SR5_omip1_1980_2007_siconc.npz', 'CMCC-CM2-SR5_omip2_1980_2007_siconc.npz', 'EC-Earth3_omip1_r1_1980_2007_siconc.npz','EC-Earth3_omip2_r1_1980_2007_siconc.npz', 'GFDL-CM4_omip1_r1i_1980_2007_siconc.npz', 'GFDL-OM4p5B_omip1__1980_2007_siconc.npz', 'IPSL-CM6A-LR_omip1_1980_2007_siconc.npz',  'MIROC6_omip1_r1i1p_1980_2007_siconc.npz', 'MIROC6_omip2_r1i1p_1980_2007_siconc.npz', 'MRI-ESM2-0_omip1_r_1980_2007_siconc.npz', 'MRI-ESM2-0_omip2_r_1980_2007_siconc.npz', 'NorESM2-LM_omip1_r_1980_2007_siconc.npz', 'NorESM2-LM_omip2_r_1980_2007_siconc.npz']

NHerror_mean1=np.zeros(14)
SHerror_mean1=np.zeros(14)
NH_error_std1=np.zeros(14)
SH_error_std1=np.zeros(14)
NH_error_trend1=np.zeros(14)
SH_error_trend1=np.zeros(14)
Metrics_siext=np.zeros((17, 6))
for obs in range(2):
  if obs==0:#models vs NSIDC-0051
    i='NSIDC0051'
    for num in range(14):
      a=np.load(path + name[num])
      NHconcentration=a['arr_2']/100
      SHconcentration=a['arr_5']/100
      NHextent=compute_extent(NHconcentration, NHcellarea, threshold = 0.15, mask = 1)
      SHextent=compute_extent(SHconcentration, SHcellarea, threshold = 0.15, mask = 1)
      NHMetrics=compute_siext_metrics(NHextent, NHextent1)
      SHMetrics=compute_siext_metrics(SHextent, SHextent1)
      NHerror_mean1[num]=NHMetrics[0]#NHerror_mean
      SHerror_mean1[num]=SHMetrics[0]#SHerror_mean
      NH_error_std1[num]=NHMetrics[1]#NH_error_std
      SH_error_std1[num]=SHMetrics[1]#SH_error_std
      NH_error_trend1[num]=NHMetrics[2]#NH_error_trend
      SH_error_trend1[num]=SHMetrics[2]#SH_error_trend
  else：#models vs OSI-450
    i='OSI450' 
    for num in range(14):
      a=np.load(path + name[num])
      NHconcentration=a['arr_2']/100
      SHconcentration=a['arr_5']/100
      NHextent=compute_extent(NHconcentration, NHcellarea, threshold = 0.15, mask = 1)
      SHextent=compute_extent(SHconcentration, SHcellarea, threshold = 0.15, mask = 1)
      NHMetrics=compute_siext_metrics(NHextent, NHextent2)
      SHMetrics=compute_siext_metrics(SHextent, SHextent2)
      NHerror_mean1[num]=NHMetrics[0]#NHerror_mean
      SHerror_mean1[num]=SHMetrics[0]#SHerror_mean
      NH_error_std1[num]=NHMetrics[1]#NH_error_std
      SH_error_std1[num]=SHMetrics[1]#SH_error_std
      NH_error_trend1[num]=NHMetrics[2]#NH_error_trend
      SH_error_trend1[num]=SHMetrics[2]#SH_error_trend

  Metrics_siext[0:14,0]=NHerror_mean1/NHtyerror[0]
  Metrics_siext[0:14,1]=NH_error_std1/NHtyerror[1]
  Metrics_siext[0:14,2]=NH_error_trend1/NHtyerror[2]
  Metrics_siext[0:14,3]=SHerror_mean1/SHtyerror[0]
  Metrics_siext[0:14,4]=SH_error_std1/SHtyerror[1]
  Metrics_siext[0:14,5]=SH_error_trend1/SHtyerror[2]
  Metrics_siext[14,0]=np.mean(NHerror_mean1)/NHtyerror[0]
  Metrics_siext[14,1]=np.mean(NH_error_std1)/NHtyerror[1]
  Metrics_siext[14,2]=np.mean(NH_error_trend1)/NHtyerror[2]
  Metrics_siext[14,3]=np.mean(SHerror_mean1)/SHtyerror[0]
  Metrics_siext[14,4]=np.mean(SH_error_std1)/SHtyerror[1]
  Metrics_siext[14,5]=np.mean(SH_error_trend1)/SHtyerror[2] 
  Metrics_siext[15,:]=(Metrics_siext[1,:]+Metrics_siext[3,:]+Metrics_siext[8,:]+Metrics_siext[10,:]+Metrics_siext[12,:])/5#OMIP1 mean
  Metrics_siext[16,:]=(Metrics_siext[2,:]+Metrics_siext[4,:]+Metrics_siext[9,:]+Metrics_siext[11,:]+Metrics_siext[13,:])/5#OMIP2 mean
  np.savez('siext_metrics_'+str(i)+'.npz', Metrics_siext, NHerror_mean1, SHerror_mean1, NH_error_std1, SH_error_std1, NH_error_trend1, SH_error_trend1)

# ----------------------------------------------------------
# PART 8) A script plots the ice extent metrics (Figs. 5a,b)|
# ----------------------------------------------------------
Models=['CMCC-CM2-HR4/J','CMCC-CM2-SR5/C','CMCC-CM2-SR5/J','EC-Earth3/C','EC-Earth3/J','GFDL-CM4/C','GFDL-OM4p5B/C','IPSL-CM6A-LR/C','MIROC6/C','MIROC6/J','MRI-ESM2-0/C','MRI-ESM2-0/J','NorESM2-LM/C','NorESM2-LM/J','Model mean','Model mean/C','Model mean/J']
Variables=['Mean Ext. NH','Std Ano Ext. NH','Trend Ano Ext. NH','Mean Ext. SH','Std Ano Ext. SH','Trend Ano Ext. SH']
for obs in range(2):
  if obs==0:#NSIDC-0051
    a=np.load('siext_metrics_NSIDC0051.npz')
    values=a['arr_0']
    values[10,1]=0.1
  else:#OSI450
    a=np.load('siext_metrics_OSI450.npz')
    values=a['arr_0']
  dpi=100
  squaresize = 220
  figwidth = 18*squaresize/float(dpi)
  figheight = 6*squaresize/float(dpi)
  fig,ax1 = plt.subplots(1, figsize=(figwidth, figheight), dpi=dpi)
  im,cbar = heatmap(values, Models,Variables, ax=ax1, cmap="OrRd", vmin=0, vmax=50) 
  texts = annotate_heatmap(im, valfmt="{x:.1f}",size=14.5,threshold=40)      
  cbar.remove()
  ax1.set_xticklabels(['Mean Ext. NH','Std Ano Ext. NH','Trend Ano Ext. NH','Mean Ext. SH','Std Ano Ext. SH','Trend Ano Ext. SH'])#,fontname='Arial', fontsize=12)
  plt.setp(ax1.get_xticklabels(), fontname='Arial', fontsize=16)
  plt.setp(ax1.get_yticklabels(), fontname='Arial', fontsize=16)
  cax = fig.add_axes([0.75, 0.113, 0.01, 0.765])
  cbar = fig.colorbar(im, cax=cax, orientation="vertical")
  cbar.ax.yaxis.set_ticks_position('both')
  cbar.ax.tick_params(direction='in', length=2,labelsize=16)
  if obs==0:#NSIDC-0051
    ax1.set_title("(a) Extent: models vs. NSIDC-0051", fontname='Arial', fontsize=16)
    plt.savefig('./Figure5a.png', bbox_inches = "tight", dpi = 500)
  else:#OSI450
    ax1.set_title("(b) Extent: models vs OSI-450", fontname='Arial', fontsize=16)
    plt.savefig('./Figure5b.png', bbox_inches = "tight", dpi = 500)


