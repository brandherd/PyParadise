#!/bin/bash
start=$(date +'%s')
ParadiseApp.py NGC2691.COMB.RSS.fits NGC2691.COMB.RSS 6.0 --SSP_par parameters_stellar
echo "Stellar Continuum Fitting took $(($(date +'%s') - $start)) seconds"
start=$(date +'%s')
ParadiseApp.py NGC2691.COMB.RSS.fits NGC2691.COMB.RSS 6.0 --line_par parameters_eline 
echo "Emission Line Fitting took $(($(date +'%s') - $start)) seconds"
start=$(date +'%s')
ParadiseApp.py NGC2691.COMB.RSS.fits NGC2691.COMB.RSS 6.0 --line_par parameters_eline --bootstraps 100 --modkeep 80 --SSP_par parameters_stellar
echo "Bootstrapping took $(($(date +'%s') - $start)) seconds"
start=$(date +'%s')
ParadisePlot.py NGC2691.COMB.RSS.fits NGC2691.COMB.RSS 1 --redshift 0.0135 --mask_cont excl.cont --fit_line lines.fit --figname NGC2691.spec_fit.png
echo "Plotting took $(($(date +'%s') - $start)) seconds"
