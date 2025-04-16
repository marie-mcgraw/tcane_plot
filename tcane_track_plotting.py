import matplotlib.pyplot as plt
import seaborn as sns
import cartopy as ct
import cartopy.feature as cfeature
import mahalanobis
import cmasher as cmr
import data_info
import matplotlib as mpl
import numpy as np
import compute_predictions
import pandas as pd
import palettable
from plots import plot_probability_ellipses
import cartopy as ct
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import scipy.stats as sps
import os,glob,sys
import seaborn as sns
import cartopy.crs as ccrs
import mahalanobis
import tcane_data_funcs
from math import floor, ceil

ELLIPSE_COLOR = {1: 'darkviolet', 2: 'goldenrod', 3: 'dodgerblue'}
NAUTICAL_MILE_TO_KM = 1.852
FIGURE_WIDTH = 32
THETA = np.linspace(0, 2 * np.pi, 1000)
#
contours = (.1, .25, .5, .75, .9,)
KM_TO_DEG = 1.0 / 111.
NMI_TO_DEG = 1.0 / 60.
#
COLOR_DEFAULT = palettable.colorbrewer.diverging.RdYlBu_9_r.mpl_colors
COLORMAP_DEFAULT = palettable.colorbrewer.diverging.RdYlBu_9_r.get_mpl_colormap()
colors = cmr.take_cmap_colors(
            "cmr.pride", len(contours), cmap_range=(0.2, 0.8), return_fmt="hex")
#
def plot_circle(ax, radius, color):
    x = radius * np.cos(THETA)
    y = radius * np.sin(THETA)
    ax.plot(x, y, '-', color=color, linewidth=4)


def plot_ellipse(ax, sigma_u, sigma_v, rho, color):
    # r = np.sqrt(-2.0 * (np.log(1.0 - 0.67))) for Pr(capture) = 67%
    r = 1.489068584398725

    x = r * sigma_u * np.cos(THETA)
    y = r * sigma_v * (rho * np.cos(THETA) + np.sqrt(1 - rho * rho) * np.sin(THETA))
    ax.plot(x, y, '-', color=color, linewidth=4)

##
def get_plot_vars_TRACK(X_out,X_in,fore_sel):
    X_plot = X_out.set_index(['FHOUR','TTYPE']).xs((fore_sel),level=1)[['LONN','LATN','DATE','MU_U','MU_V','SIGMA_U','SIGMA_V','RHO','NAME']]
    X_plot[['OFDX','OFDY']] = X_in.set_index(['FHOUR'])[['OFDX','OFDY']]
    X_plot['ftime(hr)'] = X_plot.index
    X_plot = X_plot.rename(columns={'MU_U':'mu_u','MU_V':'mu_v','SIGMA_U':'sigma_u','SIGMA_V':'sigma_v','RHO':'rho','NAME':'Name'})
    fore_vecs = np.arange(12,X_plot.dropna(how='any').index.max()+1,12)
    X_p2 = X_plot.loc[fore_vecs]
    X_p2['rho'] = X_p2['rho'].astype(float)
    X_p2['ttype'] = fore_sel
    X_plot['ttype'] = fore_sel
    return X_p2,X_plot
#
def get_plot_lims_fore(X):
    #
    X.columns = map(lambda x: str(x).upper(), X.columns)
    #
    KM_TO_DEG = 1.0 / 111.
    Xx = X[X['FHOUR']>0]
    sigma_X = KM_TO_DEG*Xx['SIGMA_U'].astype(float).values
    sigma_Y = KM_TO_DEG*Xx['SIGMA_V'].astype(float).values
    #
    y_min = Xx["LATN"].astype(float).values + KM_TO_DEG * Xx["MU_V"].astype(float).values - sigma_Y
    y_max = Xx["LATN"].astype(float).values + KM_TO_DEG * Xx["MU_V"].astype(float).values + sigma_Y
    #
    x_min = Xx["LONN"].astype(float).values + KM_TO_DEG * Xx["MU_U"].astype(float).values - sigma_X
    x_max = Xx["LONN"].astype(float).values + KM_TO_DEG * Xx["MU_U"].astype(float).values + sigma_X
    #
    x_lims = [int(round(x_min.min()/5.0)*5.0),int(round(x_max.max()/5.0)*5.0)]
    y_lims = [int(round(y_min.min()/5.0)*5.0),int(round(y_max.max()/5.0)*5.0)]
    return x_lims,y_lims,sigma_X,sigma_Y
#
def make_track_plt(ax,Xi,X_out,fore_sel,show_all=False,contours=(.1,.25,.5,.75,.9,),alpha=0.4,):
    if show_all:
        leadtimes = np.arange(12,121,12)
        X = Xi.set_index(['ttype']).xs(fore_sel)
    else:
        leadtimes = np.arange(12,121,12)
        ind_keep = [12,24,36,48,60,72,96,120]
        Xis = Xi[Xi.reset_index().set_index(['FHOUR']).index.isin(ind_keep)]
        X = Xis.reset_index().set_index(['ttype']).xs(fore_sel)
    x_lims,y_lims,details,ax0 = plot_probability_ellipses(
        X,
        ax=ax,
        leadtimes=leadtimes,
        contours=contours,
        extent=None,
        alpha=0.4,
        vector=True,
        )
#
    # 
    x_spread = round((max(x_lims) - min(x_lims))/5)*5
    y_spread = round((max(y_lims) - min(y_lims))/5)*5
    x_margin = x_spread/3
    y_margin = y_spread/3
    #
    plot0_lon = float(X_out.set_index(['FHOUR','TTYPE']).loc[(0,'late')]['LONN'])
    plot0_lat = float(X_out.set_index(['FHOUR','TTYPE']).loc[(0,'late')]['LATN'])
    #
    ax.plot(plot0_lon,plot0_lat,'x',color='k',markersize=7,transform=ct.crs.PlateCarree(central_longitude=0.))
    ax.text(plot0_lon+0.5,plot0_lat+0.25,'0',fontsize=10,transform=ct.crs.PlateCarree(central_longitude=0.))
    #ax.set_extent([-95,-72,15,37.5])
    # ax.set_extent([X['LONN'].min()-11,X['LONN'].max()+5,X['LATN'].min()-5,X['LATN'].max()+5])
    #ax.set_title('{name}, {date}, {fore_sel}'.format(name=X['Name'].iloc[0],date=X['DATE'].iloc[0],
                                                 #fore_sel=X['fore sel'],fontsize=40))
    #
    # xdiff = max(xlims) - min(xlims)
    # ydiff = max(ylims) - min(ylims)
    ax.set_extent([min(x_lims)-x_margin,max(x_lims)+x_margin,min(y_lims)-y_margin,max(y_lims)+y_margin])
    if fore_sel == 'erly':
        ttype_plt = 'early'
    else:
        ttype_plt = fore_sel
    ax.set_title('{name}, {date}, {fore_sel}'.format(name=Xi['Name'].iloc[0],date=Xi['DATE'].iloc[0],fore_sel=ttype_plt),fontsize=22)
    return ax
##
def make_track_plt_climo(ax,Xi,X_out,Xi_clim,fore_sel,cmax=np.round(2/3,3),show_all=False,alpha=0.4,use_gradient_color=True):
    if show_all:
        leadtimes = np.arange(12,121,12)
        X = Xi.set_index(['ttype']).xs(fore_sel)
        X_clim = Xi_clim.set_index(['ttype']).xs(fore_sel)
    else:
        leadtimes = np.arange(12,121,12)
        ind_keep = [12,24,36,48,60,72,96,120]
        Xis = Xi[Xi.reset_index().set_index(['FHOUR']).index.isin(ind_keep)]
        Xic = Xi_clim[Xi_clim.reset_index().set_index(['FHOUR']).index.isin(ind_keep)]
        X = Xis.reset_index().set_index(['ttype']).xs(fore_sel)
        X_clim = Xic.reset_index().set_index(['ttype']).xs(fore_sel)
    # Test red_orange colors
    
   # Plot climo ellipse
    x_lims,y_lims,details,ax0 = plot_probability_ellipses(
        X_clim,
        ax=ax,
        leadtimes=leadtimes,
        contours=(cmax,),
        extent=None,
        alpha=alpha,
        colors=None,
        # colors = None,
        vector=True,
        plot_nhc_cone=False,
        is_fill=False,
        linetype='--',
        linewt=2,
        no_label=True,
        is_climo=True,
        use_gradient_color=True)
    # Plot TCANE forecast ellipse
    x_lims2,y_lims2,details2,ax2 = plot_probability_ellipses(
        X,
        ax=ax,
        leadtimes=leadtimes,
        contours=(cmax,),
        extent=None,
        alpha=1-alpha,
        colors=None,
       # colors = None,
        vector=True,
        plot_nhc_cone=False,
        is_fill=False,
        linetype='-',
        linewt=4,
        use_gradient_color=True)
#
    plot0_lon = float(X_out.set_index(['FHOUR','TTYPE']).loc[(0,'late')]['LONN'])
    plot0_lat = float(X_out.set_index(['FHOUR','TTYPE']).loc[(0,'late')]['LATN'])
    #
    ax.plot(plot0_lon,plot0_lat,'x',color='k',markersize=7,transform=ct.crs.PlateCarree(central_longitude=0.))
    ax.text(plot0_lon+0.5,plot0_lat+0.25,'0',fontsize=10,transform=ct.crs.PlateCarree(central_longitude=0.))
    # 
    x_spread = round((max(x_lims) - min(x_lims))/5)*5
    y_spread = round((max(y_lims) - min(y_lims))/5)*5
    x_margin = x_spread/4
    y_margin = y_spread/4
    #
    x_spread2 = round((max(x_lims2) - min(x_lims2))/5)*5
    y_spread2 = round((max(y_lims2) - min(y_lims2))/5)*5
    x_margin2 = x_spread2/4
    y_margin2 = y_spread2/4
    x_ext_min = min(min(x_lims)-x_margin,min(x_lims2)-x_margin2)
    x_ext_max = max(max(x_lims)+x_margin,max(x_lims2)+x_margin2)
    y_ext_min = min(min(y_lims)-y_margin,min(y_lims2)-y_margin2)
    y_ext_max = max(max(y_lims)+y_margin,max(y_lims2)+y_margin2)
    ax.set_extent([x_ext_min,x_ext_max,y_ext_min,y_ext_max])
    #
    if fore_sel == 'erly':
        ttype_plt = 'early'
    else:
        ttype_plt = fore_sel
    ax.set_title('{name}, {date}, {fore_sel}'.format(name=Xi['Name'].iloc[0],date=Xi['DATE'].iloc[0],fore_sel=ttype_plt),fontsize=22)
    return ax