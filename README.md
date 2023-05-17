1 Information and contact

The Sea Ice Evaluation Tool version 1.0 (SITool v1.0) is a performance metrics and diagnostics tool developed to evaluate the model skills in simulating the sea ice concentration, extent, edge location, thickness, snow depth, and ice drift in the Arctic and Antarctic. This tool contains scripts to process model outputs from CMIP6 OMIP and two sets of observational references for six sea ice variables. This user guide aims at providing technical information for using this tool. More details on this study are provided in the paper submitted to Geoscientific Model Development (Lin et al. 2021). In case any questions arise, please do not hesitate to contact Xia Lin (xia.lin@uclouvain.be or xia.lin104@gmail.com).

2 Technical requirements

The SITool v1.0 is based on Python 3.8.3. You have to install needed python packages before the calculation and plot, such as install Dataset from netCDF4 to read the NetCDF files, install pyresample to do the interpolation and install Basemap from mpl_toolkits.basemap to plot nice maps.The package versions can be found in the environment file.

3 Getting the SITool and preparing the data

You can download the SITool v1.0 from the following URL: https://github.com/XiaLinUCL/Sea-Ice-Evaluation-Tool.

Data should be obtained separately, which is available online. CMIP6 OMIP data are freely available from the Earth System Grid Federation (https://esgf-node.llnl.gov/search/cmip6/). Two sets of observational references used to do the comparison are detailed in Lin et al. (2021).

4 Contents in the SITool

All the methods are detailed in each script, such as the function used to interpolate the input data into the polar stereographic 25 km resolution grid, the heatmap function and the functions used to compute metrics. The contents in each folder are detailed below:

SICONC: deal with ice concentration data, compute and plot the ice concentration metrics, plot the February and September mean ice concentration differences;

SIEXT: compute the ice extent from ice concentration, plot the mean seasonal cycle and monthly anomalies of ice extent, compute and plot the ice extent metrics;

SIEDGE: compute Intergrated Ice Edge Error (IIEE) from ice concentration, plot the mean seasonal cycle of IIEE, compute and plot the ice edge location metrics;

SITHICK: deal with ice thickness data, compute and plot the ice thickness metrics, plot the February (Arctic) and September (Antarctic) mean ice thickness differences;

SNOWDEPTH: deal with snow depth data, compute and plot the snow depth metrics, plot the February (Arctic) and September (Antarctic) mean snow depth differences;

SIDRIFT: deal with ice drift data, compute and plot the ice-motion magnitude (MKE) and direction (vector correlation coefficient) metrics, plot the ice-vector correlation coefficient, plot the February and September mean ice-motion mean kenetic energy differences;

5 Run the scripts

Make sure the data are prepared in the right directory;

Launch Python 3.8.3 in the terminal, and then run the python scripts.

References

Lin, X., Massonnet, F., Fichefet, T., and Vancoppenolle, M.: SITool (v1.0) – a new evaluation tool for large-scale sea ice simulations: application to CMIP6 OMIP, Geosci. Model Dev., 14, 6331–6354, https://doi.org/10.5194/gmd-14-6331-2021, 2021. 
