The Sea Ice Evaluation Tool (SITool) is a performance metrics and diagnostics tool developed to evaluate the model skills in simulating the bi-polar sea ice concentration, extent, edge location, thickness, snow depth, and ice drift.

This repository contains scripts to process model outputs from CMIP6 OMIP. Data should be obtained separately, which is available online.
CMIP6 OMIP data are freely available from the Earth System Grid Federation. Observational references used in this paper are detailed in the paper submitted to Geoscientific Model Development (Lin et al. 2021).

Contents
---------
SICONC:    deal with ice concentration data, compute and plot the ice concentration metrics
SIEXT:     compute the ice extent from ice concentration, plot the mean seasonal cycle and monthly anomalies of ice extent, compute and plot the ice extent metrics
SIEDGE:    computes Intergrated Ice Edge Error (IIEE) from ice concentration, plot the mean seasonal cycle of IIEE, compute and plot the ice edge location metrics
SITHICK:   deal with ice thickness data, compute and plot the ice thickness metrics
SNOWDEPTH: deal with snow depth data, compute and plot the snow depth metrics
SIDRIFT:   deal with ice drift data, compute and plot the ice-motion magnitude (MKE) and direction (vector correlation coefficient) metrics, plot the ice-vector correlation coefficient

Usage Notes
----------
Detailed in each script.

Contact
--------
Xia Lin (xia.lin@uclouvain.be)

