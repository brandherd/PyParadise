#!/usr/bin/env python
import argparse
from Paradise import *
import numpy
import sys
from matplotlib import pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""This program allows to plot the fitting of a single spectrum from an RSS or CUBE file to check the results from PyParadise.""",
formatter_class=argparse.ArgumentDefaultsHelpFormatter, prog='SpectrumPlotting')
    parser.add_argument("input", type=str, help="""Input data file used for PyParadise run""")
    parser.add_argument("prefix", type=str, help="""Prefix used for nameing all PyParadise output files.""")
    parser.add_argument("pixel", type=str, help="""Indicate the pixel to be plotted which is either an integer for an RSS (starting with 1) or a comma seperate tuple x,y for a input cube.""")
    parser.add_argument("--redshift", default=0.0, type=float, help="""Provide the redshift if rest-frame spectral windows are desired to show.""")
    parser.add_argument("--mask_cont", default=None, type=str, help="""File name to the continuum spectral exclude file to be presented on the figure.""")
    parser.add_argument("--fit_line", default=None, type=str, help="""File name to the spectra line fitting selection file to be presented on the figure.""")
    parser.add_argument("--figname", type=str, default=None, help="""File name of the output figure if it should be stored. Default is only to display the figure.""")
    args = parser.parse_args()
    
    try:
        rss = loadSpectrum(args.input)
    except IOError:
        print("Input data not found. Please check the file names.")
        sys.exit()
    try:
        cont = loadSpectrum('%s.cont_model.fits'%(args.prefix))
        res = loadSpectrum('%s.cont_res.fits'%(args.prefix))
    except IOError:
        print("PyParadise output data cannot be found. Please check the prefix for the file names.")
        sys.exit()
    try:
        line = loadSpectrum('%s.eline_model.fits'%(args.prefix))
        line_model_present=True
    except IOError:
        print("No emission line model found. Plot without the line model.")
        line_model_present=False
    
    try:
        if ',' in args.pixel:
            cube_x = int(args.pixel.split(',')[0])-1
            cube_y = int(args.pixel.split(',')[1])-1
        else:
            rss_fiber = int(args.pixel)-1
    except:
        print("Wrong format to select the spectrum. Please check your data format and syntax for the spectrum coordinates")
        sys.exit()
        
    z=args.redshift
    select = rss._error >1e3
    rss._error[select]=0

    i=0
    plt.style.use('seaborn')
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.edgecolor'] = 'k'
    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_axes([0.09,0.55,0.9,0.43])
    ax2 = fig.add_axes([0.09,0.1,0.9,0.43])
    if rss._datatype == 'CUBE':
        ax1.plot(rss._wave,rss._data[:,cube_y,cube_x],'-k',lw=1.5,label='data')
        ax1.plot(rss._wave,rss._error[:,cube_y,cube_x],'-g',lw=1.5,label='error')
        if rss._mask is not None:
            ax1.plot(rss._wave,rss._mask[:,cube_y,cube_x]*0.1,'-b',lw=1.5,label='badpix')
        ax1.plot(cont._wave,cont._data[:,cube_y,cube_x],'-r',lw=1.5,label='best-fit cont')
    elif rss._datatype == 'RSS':
        ax1.plot(rss._wave,rss._data[rss_fiber,:],'-k',lw=1.5,label='data')
        ax1.plot(rss._wave,rss._error[rss_fiber,:],'-g',lw=1.5,label='error')
        if rss._mask is not None:
            ax1.plot(rss._wave,rss._mask[rss_fiber,:]*0.1,'-b',lw=1.5,label='badpix')
        ax1.plot(cont._wave,cont._data[rss_fiber,:],'-r',lw=1.5,label='best-fit cont')
    
    leg=ax1.legend(loc='upper left',fontsize=16)
    leg.draw_frame(True)
    
    if args.mask_cont is not None:
        cont_mask = CustomMasks(args.mask_cont)
        for i in range(len(cont_mask['rest_frame'])):
            ax1.axvspan(cont_mask['rest_frame'][i][0]*(1+z),cont_mask['rest_frame'][i][1]*(1+z),color='k',alpha=0.2)
        for i in range(len(cont_mask['observed_frame'])):
            ax1.axvspan(cont_mask['observed_frame'][i][0],cont_mask['observed_frame'][i][1],color='r',alpha=0.2)
    
    if line_model_present:
        if rss._datatype == 'CUBE':
            ax2.plot(res._wave,res._data[:,cube_y,cube_x],'-g',lw=1.5,label='best-fit residuals')
            ax2.plot(line._wave,line._data[:,cube_y,cube_x],'-b',lw=1.5,label='emission-line model')
        elif rss._datatype == 'RSS':
            ax2.plot(res._wave,res._data[rss_fiber,:],'-g',lw=1.5,label='best-fit residuals')
            ax2.plot(line._wave,line._data[rss_fiber,:],'-b',lw=1.5,label='emission-line model')
        leg2 = ax2.legend(loc='upper left',fontsize=16)
        leg2.draw_frame(True)
        if args.fit_line is not None:
            line_mask = CustomMasks(args.fit_line)
        for i in range(len(line_mask['rest_frame'])):
            ax2.axvspan(line_mask['rest_frame'][i][0]*(1+z),line_mask['rest_frame'][i][1]*(1+z),color='b',alpha=0.2)
    ax1.set_ylabel(r'$f_\lambda$',fontsize=16)
    ax1.minorticks_on()
    ax1.tick_params(axis='both',which='major',direction='in',width=1.5,length=6,labelsize=16)
    ax1.tick_params(axis='both',which='minor',direction='in',width=1.5,length=3)
    ax1.set_xticklabels([])
    ax2.minorticks_on()
    ax2.tick_params(axis='both',which='major',direction='in',width=1.5,length=6,labelsize=16)
    ax2.tick_params(axis='both',which='minor',direction='in',width=1.5,length=3)

    ax2.set_xlabel('observed-frame wavelength [$\AA$]',fontsize=18)
    ax2.set_ylabel(r'$f_\lambda$',fontsize=16)
    if args.figname is not None:
        plt.savefig(args.figname)
    plt.show()
    
