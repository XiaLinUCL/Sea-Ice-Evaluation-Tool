1 Information and contact

The Sea Ice Evaluation Tool version 1.0 (SITool v1.0) is a performance metrics and diagnostics tool developed to evaluate the model skills in simulating the bi-polar sea ice concentration, extent, edge location, thickness, snow depth, and ice drift. This tool contains scripts to process model outputs from CMIP6 OMIP and two sets of observational references. This user guide aims at providing technical information for using this tool. More details on this study are provided in the paper submitted to Geoscientific Model Development (Lin et al. 2021). In case any questions arise, please do not hesitate to contact Xia Lin (xia.lin@uclouvain.be).

2 Technical requirements

The SITool v1.0 is based on Python 3.8.3. You have to install needed python packages before the calculation and plot, such as install Dataset from netCDF4 to read the NetCDF files, install pyresample to do the interpolation and install Basemap from mpl_toolkits.basemap to plot nice maps. 

3 Getting the SITool and preparing the data

You can download the SITool v1.0 from the following URL: https://github.com/XiaLinUCL/Sea-Ice-Evaluation-Tool.

Data should be obtained separately, which is available online. CMIP6 OMIP data are freely available from the Earth System Grid Federation (https://esgf-node.llnl.gov/search/cmip6/). Two sets of observational references used to do the comparison are detailed in Lin et al. (2021).

4 Contents in the SITool

All the methods are detailed in each script, such as the function used to interpolate the input data into the polar stereographic 25 km resolution grid, the heatmap function and the functions used to compute metrics. The contents in each folder are detailed below:

SICONC: deal with ice concentration data, compute and plot the ice concentration metrics;

SIEXT: compute the ice extent from ice concentration, plot the mean seasonal cycle and monthly anomalies of ice extent, compute and plot the ice extent metrics;

SIEDGE: compute Intergrated Ice Edge Error (IIEE) from ice concentration, plot the mean seasonal cycle of IIEE, compute and plot the ice edge location metrics;

SITHICK: deal with ice thickness data, compute and plot the ice thickness metrics;

SNOWDEPTH: deal with snow depth data, compute and plot the snow depth metrics;

SIDRIFT: deal with ice drift data, compute and plot the ice-motion magnitude (MKE) and direction (vector correlation coefficient) metrics, plot the ice-vector correlation coefficient;

5 Run the scripts

Make sure the data are prepared in the right directory;

Launch Python 3.8.3 in the terminal, and then run the python scripts.

References

X. Lin, F. Massonnet, T. Fichefet, and M. Vancoppenolle. SITool (v1.0) - a new evaluation tool for large-scale sea ice simulations: application to CMIP6 OMIP. Geophysical Model Development, under review.
