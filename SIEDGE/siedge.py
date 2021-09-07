# Author: Xia Lin
# Date:   Feb 2021
# Contents:
# 1) A function that computes Intergrated Ice Edge Error (IIEE) between two datasets 
#    and the monthly mean in order to get the metrics;
#    The input data is a numpy array but a NetCDF file 
# 2) The heatmap function  
# 3) The annotate heatmap function (The heatmap and annotate heatmap functions were written 
#    by following https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html)
# 4) A script calls the function 1) to calculate the IIEE from ice concentration,
#    and plots the mean seasonal cycle of IIEE in the Arctic and Antarctic (Fig. 6)
# 5) A script calls the function 1) to calculate the IIEE from ice concentration
#    and to compute the ice edge location metrics
# 6) A script calls the functions 2) and 3) to plot the ice edge metrics (Fig. 5c)

# ---------------------------------------------------------
# PART 1) The Intergrated Ice Edge Error (IIEE) function  |   
# ---------------------------------------------------------
def compute_siedge_metrics(concentration, concentration1, cellarea):
  ''' Input: - sea ice concentration (%) in the Arctic or Antarctic from two datasets
             - cellarea: array of grid cell areas in the Arctic or Antarctic (square meters)

      Output: Intergrated Ice Edge Error (IIEE) in the Arctic or Antarctic
  ''' 
  nt, ny, nx = concentration.shape
  concmask=np.zeros(((nt,ny,nx)))
  conc1mask=np.zeros(((nt,ny,nx)))
  idx = np.where(concentration >=0.15)
  concmask[idx] = 1
  idx = np.where(concentration1 >=0.15)
  conc1mask[idx] = 1
  IIEE=np.array([np.nansum(np.abs(concmask[jt,:,:]-conc1mask[jt,:,:])*cellarea[:,:]) for jt in range(nt)])/10**12
  AEE=np.abs(np.array([np.nansum((concmask[jt,:,:]-conc1mask[jt,:,:])*cellarea[:,:]) for jt in range(nt)])/10**12)
  ME=IIEE-AEE

  cycle = np.array([np.nanmean(IIEE[m::12]) for m in range(12)])
  ndpm=[31,28,31,30,31,30,31,31,30,31,30,31]
  error_mean=np.sum(cycle*ndpm)/np.sum(ndpm) 

  return IIEE, AEE, ME, error_mean

# ------------------------------
# PART 2) The heatmap function  |
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
# PART 3) The annotate heatmap function  |
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

# -------------------------------------------------------------------------------------
# PART 4) A script to plot the mean seasonal cycle of IIEE in the Arctic and Antarctic |
# -------------------------------------------------------------------------------------
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

np.random.seed(3) 
colors = [np.random.random(3) for j in range(16)]
a=np.load('NSIDC0051_1980_2007_siconc.npz')
NHconcentration1=a['arr_2']/100 
SHconcentration1=a['arr_5']/100
NHcellarea=a['arr_6']
SHcellarea=a['arr_7']
a=np.load('OSI450_1980_2007_siconc.npz')
NHconcentration2=a['arr_2']/100
SHconcentration2=a['arr_5']/100
NHtyerror=compute_siedge_metrics(NHconcentration1, NHconcentration2, NHcellarea)
SHtyerror=compute_siedge_metrics(SHconcentration1, SHconcentration2, SHcellarea)
NHIIEE=NHtyerror[0]
SHIIEE=NHtyerror[0]
NHcycle1 = np.array([np.nanmean(NHIIEE[m::12]) for m in range(12)])
SHcycle1 = np.array([np.nanmean(SHIIEE[m::12]) for m in range(12)])

name=['CMCC-CM2-HR4_omip2_1980_2007_siconc.npz', 'CMCC-CM2-SR5_omip1_1980_2007_siconc.npz', 'CMCC-CM2-SR5_omip2_1980_2007_siconc.npz', 'EC-Earth3_omip1_r1_1980_2007_siconc.npz','EC-Earth3_omip2_r1_1980_2007_siconc.npz', 'GFDL-CM4_omip1_r1i_1980_2007_siconc.npz', 'GFDL-OM4p5B_omip1__1980_2007_siconc.npz', 'IPSL-CM6A-LR_omip1_1980_2007_siconc.npz',  'MIROC6_omip1_r1i1p_1980_2007_siconc.npz', 'MIROC6_omip2_r1i1p_1980_2007_siconc.npz', 'MRI-ESM2-0_omip1_r_1980_2007_siconc.npz', 'MRI-ESM2-0_omip2_r_1980_2007_siconc.npz', 'NorESM2-LM_omip1_r_1980_2007_siconc.npz', 'NorESM2-LM_omip2_r_1980_2007_siconc.npz']
name1=['CMCC-CM2-HR4/J','CMCC-CM2-SR5/C','CMCC-CM2-SR5/J','EC-Earth3/C','EC-Earth3/J','GFDL-CM4/C','GFDL-OM4p5B/C','IPSL-CM6A-LR/C','MIROC6/C','MIROC6/J','MRI-ESM2-0/C','MRI-ESM2-0/J','NorESM2-LM/C','NorESM2-LM/J']
#'NSIDC-0051',
months = np.arange(1,13)
MNHIIEE1=np.zeros((14,12))
MSHIIEE1=np.zeros((14,12))
for num in range(14):
  print(num)
  a=np.load(name[num])
  NHconcentration=a['arr_2']/100
  SHconcentration=a['arr_5']/100
  NHedge=compute_siedge_metrics(NHconcentration, NHconcentration1, NHcellarea)
  SHedge=compute_siedge_metrics(SHconcentration, SHconcentration1, SHcellarea)
  NHIIEE = NHedge[0]
  SHIIEE = SHedge[0]
  MNHIIEE = np.array([np.nanmean(NHIIEE[m::12]) for m in range(12)])
  MSHIIEE = np.array([np.nanmean(SHIIEE[m::12]) for m in range(12)])
  MNHIIEE1[num,:] = MNHIIEE
  MSHIIEE1[num,:] = MSHIIEE

fig,((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(12,9))
ax1.grid(linestyle=':', zorder=1)
months = np.arange(1,13)
MIIEE10= np.nanmean(MNHIIEE1,axis=1)
MIIEE11 = (MNHIIEE1[1,:]+MNHIIEE1[3,:]+MNHIIEE1[8,:]+MNHIIEE1[10,:]+MNHIIEE1[12,:])/5
MIIEE12 = (MNHIIEE1[2,:]+MNHIIEE1[4,:]+MNHIIEE1[9,:]+MNHIIEE1[11,:]+MNHIIEE1[13,:])/5
ax1.plot(months, MIIEE10, color = 'firebrick', label='Model mean',linewidth=2, zorder=20)
ax1.plot(months, MIIEE11, color = 'red', label='Model mean/C',linewidth=2, zorder=20)
ax1.plot(months, MIIEE12, color = 'red', label='Model mean/J',linewidth=2,linestyle='-.', zorder=20)
ax1.scatter(months, NHcycle1, color = 'c', label='OSI-450', marker='+', s=28, zorder=30)
ax1.set_ylabel('Arctic IIEE (vs. NSIDC-0051, $\mathregular{10^6}$ $\mathregular{km^2}$)', fontsize=11, fontweight = 'bold')
xticks=ax1.set_xticks(np.arange(1,13,1))
yticks=ax1.set_yticks(np.arange(0,5,1)) 
ax1.text(11, 3.6, '(a)', fontsize=13, fontweight = 'bold')
labels = [item.get_text() for item in ax1.get_xticklabels()]
labels = ['J','F','M','A','M','J','J','A','S','O','N','D']
ax1.set_xticklabels(labels,fontname='Arial', fontsize=13)
for tick in ax1.get_yticklabels():
  tick.set_fontname('Arial')
  tick.set_fontsize(13)

ax2.grid(linestyle=':', zorder=1)
months = np.arange(1,13)
MIIEE10= np.nanmean(MSHIIEE1,axis=1)
MIIEE11 = (MSHIIEE1[1,:]+MSHIIEE1[3,:]+MSHIIEE1[8,:]+MSHIIEE1[10,:]+MSHIIEE1[12,:])/5
MIIEE12 = (MSHIIEE1[2,:]+MSHIIEE1[4,:]+MSHIIEE1[9,:]+MSHIIEE1[11,:]+MSHIIEE1[13,:])/5
ax2.plot(months, MIIEE10, color = 'firebrick', label='Model mean',linewidth=2, zorder=20)
ax2.plot(months, MIIEE11, color = 'red', label='Model mean/C',linewidth=2, zorder=20)
ax2.plot(months, MIIEE12, color = 'red', label='Model mean/J',linewidth=2,linestyle='-.', zorder=20)
ax2.scatter(months, SHcycle1, color = 'c', label='OSI-450', marker='+', s=28, zorder=30)
ax2.set_ylabel('Antarctic IIEE (vs. NSIDC-0051, $\mathregular{10^6}$ $\mathregular{km^2}$)', fontsize=11, fontweight = 'bold')
ax2.set_xticks(np.arange(1,13,1))
ax2.set_yticks(np.arange(0,7,1))
ax2.text(11, 5.5, '(b)', fontsize=13, fontweight = 'bold')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,prop={'family':'Arial', "size":12, 'weight':'bold'})#'center left'
labels = [item.get_text() for item in ax2.get_xticklabels()]
labels = ['J','F','M','A','M','J','J','A','S','O','N','D']
ax2.set_xticklabels(labels,fontname='Arial', fontsize=13)
for tick in ax2.get_yticklabels():
  tick.set_fontname('Arial')
  tick.set_fontsize(13)

ax3.grid(linestyle=':', zorder=1)
months = np.arange(1,13)
for num in range(14):
  if (num==2):
    ax3.plot(months, MNHIIEE1[num,:], color = colors[num+2], label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10) 
  elif (num==3):
    ax3.plot(months, MNHIIEE1[num,:], color = 'darkgreen', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==4):
    ax3.plot(months, MNHIIEE1[num,:], color = 'darkgreen', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10) 
  elif (num==8):
    ax3.plot(months, MNHIIEE1[num,:], color = 'darkorange', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==9):
    ax3.plot(months, MNHIIEE1[num,:], color = 'darkorange', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10) 
  elif (num==10):
    ax3.plot(months, MNHIIEE1[num,:], color = 'tab:grey', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==11):
    ax3.plot(months, MNHIIEE1[num,:], color = 'tab:grey', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10) 
  elif (num==12):
    ax3.plot(months, MNHIIEE1[num,:], color = 'gold', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==13):
    ax3.plot(months, MNHIIEE1[num,:], color = 'gold', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10) 
  elif (num==5):
    ax3.plot(months, MNHIIEE1[num,:], color = 'blue', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  else:
    ax3.plot(months, MNHIIEE1[num,:], color = colors[num+3], label=name1[num],linewidth=2,alpha=0.8, zorder=10)  
ax3.set_xlabel('Month', fontsize=12, fontweight = 'bold')
ax3.set_ylabel('Arctic IIEE (vs. NSIDC-0051, $\mathregular{10^6}$ $\mathregular{km^2}$)', fontsize=11, fontweight = 'bold')
xticks=ax3.set_xticks(np.arange(1,13,1))
yticks=ax3.set_yticks(np.arange(0,6,1)) 
ax3.text(11, 5.1, '(c)', fontsize=13, fontweight = 'bold')
labels = [item.get_text() for item in ax3.get_xticklabels()]
labels = ['J','F','M','A','M','J','J','A','S','O','N','D']
ax3.set_xticklabels(labels,fontname='Arial', fontsize=13)
for tick in ax3.get_yticklabels():
  tick.set_fontname('Arial')
  tick.set_fontsize(13)

ax4.grid(linestyle=':', zorder=1)#, linewidth=2)
months = np.arange(1,13)
for num in range(14):
  if (num==2):
    ax4.plot(months, MSHIIEE1[num,:], color = colors[num+2], label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10) 
  elif (num==3):
    ax4.plot(months, MSHIIEE1[num,:], color = 'darkgreen', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==4):
    ax4.plot(months, MSHIIEE1[num,:], color = 'darkgreen', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10) 
  elif (num==8):
    ax4.plot(months, MSHIIEE1[num,:], color = 'darkorange', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==9):
    ax4.plot(months, MSHIIEE1[num,:], color = 'darkorange', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10) 
  elif (num==10):
    ax4.plot(months, MSHIIEE1[num,:], color = 'tab:grey', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==11):
    ax4.plot(months, MSHIIEE1[num,:], color = 'tab:grey', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10)
  elif (num==12):
    ax4.plot(months, MSHIIEE1[num,:], color = 'gold', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  elif (num==13):
    ax4.plot(months, MSHIIEE1[num,:], color = 'gold', label=name1[num],linewidth=2,linestyle='-.',alpha=0.8, zorder=10)
  elif (num==5):
    ax4.plot(months, MSHIIEE1[num,:], color = 'blue', label=name1[num],linewidth=2,alpha=0.8, zorder=10)
  else:
    ax4.plot(months, MSHIIEE1[num,:], color = colors[num+3], label=name1[num],linewidth=2,alpha=0.8, zorder=10)
ax4.set_xlabel('Month', fontsize=12, fontweight = 'bold')
ax4.set_ylabel('Antarctic IIEE (vs. NSIDC-0051, $\mathregular{10^6}$ $\mathregular{km^2}$)', fontsize=11, fontweight = 'bold')
ax4.set_xticks(np.arange(1,13,1))
ax4.set_yticks(np.arange(0,8,1))
ax4.text(11, 7.1, '(d)', fontsize=13, fontweight = 'bold')
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,prop={'family':'Arial', "size":12, 'weight':'bold'})#'center left'
labels = [item.get_text() for item in ax4.get_xticklabels()]
labels = ['J','F','M','A','M','J','J','A','S','O','N','D']
ax4.set_xticklabels(labels,fontname='Arial', fontsize=13)
for tick in ax4.get_yticklabels():
  tick.set_fontname('Arial')
  tick.set_fontsize(13)
plt.savefig("./Figure6.png", bbox_inches = "tight", dpi = 500)
plt.close()
  
# ----------------------------------------------------------------------
# PART 5) A script computes the ice edge location error and its metrics |
# ----------------------------------------------------------------------
#typical errors-differences between two observations
a=np.load('NSIDC0051_1980_2007_siconc.npz')
NHconcentration1=a['arr_2']/100 
SHconcentration1=a['arr_5']/100
NHcellarea=a['arr_6']
SHcellarea=a['arr_7']
a=np.load('OSI450_1980_2007_siconc.npz')
NHconcentration2=a['arr_2']/100
SHconcentration2=a['arr_5']/100
NHtyerror=compute_siedge_metrics(NHconcentration1, NHconcentration2, NHcellarea)
SHtyerror=compute_siedge_metrics(SHconcentration1, SHconcentration2, SHcellarea)

name=['CMCC-CM2-HR4_omip2_1980_2007_siconc.npz', 'CMCC-CM2-SR5_omip1_1980_2007_siconc.npz', 'CMCC-CM2-SR5_omip2_1980_2007_siconc.npz', 'EC-Earth3_omip1_r1_1980_2007_siconc.npz','EC-Earth3_omip2_r1_1980_2007_siconc.npz', 'GFDL-CM4_omip1_r1i_1980_2007_siconc.npz', 'GFDL-OM4p5B_omip1__1980_2007_siconc.npz', 'IPSL-CM6A-LR_omip1_1980_2007_siconc.npz',  'MIROC6_omip1_r1i1p_1980_2007_siconc.npz', 'MIROC6_omip2_r1i1p_1980_2007_siconc.npz', 'MRI-ESM2-0_omip1_r_1980_2007_siconc.npz', 'MRI-ESM2-0_omip2_r_1980_2007_siconc.npz', 'NorESM2-LM_omip1_r_1980_2007_siconc.npz', 'NorESM2-LM_omip2_r_1980_2007_siconc.npz']
NHerror_mean1=np.zeros(14)
SHerror_mean1=np.zeros(14)
NHerror_mean2=np.zeros(14)
SHerror_mean2=np.zeros(14)
Metrics_siedge=np.zeros((17, 4))
for num in range(14):
  a=np.load(path + name[num])
  NHconcentration=a['arr_2']/100
  SHconcentration=a['arr_5']/100
  #models vs NSIDC-0051
  NHMetrics=compute_siedge_metrics(NHconcentration, NHconcentration1, NHcellarea)
  SHMetrics=compute_siedge_metrics(SHconcentration, SHconcentration1, SHcellarea)
  NHerror_mean1[num]=NHMetrics[3]#NHerror_mean
  SHerror_mean1[num]=SHMetrics[3]#SHerror_mean
  #models vs OSI-450 
  NHMetrics=compute_siedge_metrics(NHconcentration, NHconcentration2, NHcellarea)
  SHMetrics=compute_siedge_metrics(SHconcentration, SHconcentration2, SHcellarea)
  NHerror_mean2[num]=NHMetrics[3]#NHerror_mean
  SHerror_mean2[num]=SHMetrics[3]#SHerror_mean
  
Metrics_siedge[0:14,0]=NHerror_mean1/NHtyerror[3]
Metrics_siedge[0:14,1]=SHerror_mean1/SHtyerror[3]
Metrics_siedge[0:14,2]=NHerror_mean2/NHtyerror[3]
Metrics_siedge[0:14,3]=SHerror_mean2/SHtyerror[3]
Metrics_siedge[14,0]=np.mean(NHerror_mean1)/NHtyerror[3]
Metrics_siedge[14,1]=np.mean(SHerror_mean1)/SHtyerror[3]
Metrics_siedge[14,2]=np.mean(NHerror_mean2)/NHtyerror[3]
Metrics_siedge[14,3]=np.mean(SHerror_mean2)/SHtyerror[3]
Metrics_siedge[15,:]=(Metrics_siedge[1,:]+Metrics_siedge[3,:]+Metrics_siedge[8,:]+Metrics_siedge[10,:]+Metrics_siedge[12,:])/5#OMIP1 mean
Metrics_siedge[16,:]=(Metrics_siedge[2,:]+Metrics_siedge[4,:]+Metrics_siedge[9,:]+Metrics_siedge[11,:]+Metrics_siedge[13,:])/5#OMIP2 mean
np.savez('siedge_metrics_NSIDC0051&OSI-450.npz', Metrics_siedge, NHerror_mean1, SHerror_mean1, NHerror_mean2, SHerror_mean2)

# -----------------------------------------------------
# PART 6) A script plots the ice edge metrics (heatmap)|
# -----------------------------------------------------
Models=['CMCC-CM2-HR4/J','CMCC-CM2-SR5/C','CMCC-CM2-SR5/J','EC-Earth3/C','EC-Earth3/J','GFDL-CM4/C','GFDL-OM4p5B/C','IPSL-CM6A-LR/C','MIROC6/C','MIROC6/J','MRI-ESM2-0/C','MRI-ESM2-0/J','NorESM2-LM/C','NorESM2-LM/J','Model mean','Model mean/C','Model mean/J']
Variables=['Mean Edge NH','Mean Edge SH','Mean Edge NH','Mean Edge SH']
a=np.load('siedge_metrics_NSIDC0051&OSI-450.npz')
values=a['arr_0']
dpi=100
squaresize = 220
figwidth = 18*squaresize/float(dpi)
figheight = 6*squaresize/float(dpi)
fig,ax1 = plt.subplots(1, figsize=(figwidth, figheight), dpi=dpi)
im,cbar = heatmap(values, Models,Variables, ax=ax1, cmap="OrRd", vmin=1, vmax=6) 
texts = annotate_heatmap(im, valfmt="{x:.2f}",size=16,threshold=4)
ax1.set_xticklabels(['Mean Edge NH','Mean Edge SH','Mean Edge NH','Mean Edge SH'])#,fontname='Arial', fontsize=12)
plt.setp(ax1.get_xticklabels(), fontname='Arial', fontsize=16)
plt.setp(ax1.get_yticklabels(), fontname='Arial', fontsize=16)
cax = fig.add_axes([0.75, 0.113, 0.01, 0.765])
cbar = fig.colorbar(im, cax=cax,ticks=[1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6], orientation="vertical")
cbar.ax.yaxis.set_ticks_position('both')
cbar.ax.tick_params(direction='in',length=2,labelsize=16)
ax1.set_title("(c) Edge: models vs. NSIDC-0051 & OSI-450 ", fontname='Arial', fontsize=16)
plt.savefig('./Figure5c.png', bbox_inches = "tight", dpi = 500)
