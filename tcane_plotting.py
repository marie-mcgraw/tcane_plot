import numpy as np
import pandas as pd
import os,glob,re
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import palettable
from scipy.stats import skewnorm, rv_histogram
import scipy.stats as sps
import tcane_data_funcs
######
### `make_boxplot(ax,Y_e,Y_l,whis)`
# 
# This function takes the simulated TCANE data and makes a boxplot comparing the `erly` and `late` forecasts.
# 
# <b>Inputs</b>:
# * `ax`: axes for figure
# * `Y_e`: early SHASH data [Dataframe]
# * `Y_l`: late SHASH data [Dataframe]
# * `df_in`: Input date for TCANE forecast (used to add NHC forecast for comparison) [Dataframe]
# * `whis`: location of whiskers [lower, upper]. Should be between 0 and 100. Default is two-tailed 95 pct confidence (2.5-97.5) [array]

# <b>Outputs</b>:
# * `ax`: return the figure handles for further editing
def make_boxplot(ax,Y_e,Y_l,df_in,whis=[2.5,97.5]):
    box_colors = {'xkcd:yellow orange','xkcd:cornflower'}
    flierprops = dict(marker='d', markerfacecolor='xkcd:gray', markersize=3,
                  linestyle='none', markeredgecolor='xkcd:gray')
    df_box = pd.concat([Y_e,Y_l])
    sns.boxplot(data=df_box,x='FHOUR',y='Y',hue='TTYPE',palette=sns.set_palette(box_colors),flierprops=flierprops,
               whis=whis,linewidth=2,ax=ax)
    #
    sns.scatterplot(data=df_in.set_index(['FHOUR']),x=np.arange(0.8,9.85,1),y='VMXC',s=100,marker='v',color='xkcd:navy',label='TCANE Consensus',ax=ax)
    sns.scatterplot(data=df_in.set_index(['FHOUR']),x=np.arange(1.2,10.25,1),y='VMAXN',s=75,marker='s',color='xkcd:raspberry',label='NHC Official',ax=ax)
    #
    ax.grid()
    ax.tick_params(axis='both',labelsize=16)
    ax.set_xlabel('Forecast Hour',fontsize=22)
    ax.set_ylabel('Wind Speed (kt)',fontsize=22)
    ax.legend(loc='upper left',fontsize=14)
    return ax
#########################
### `get_cat_probs(ax,df,df_clim,df_in,ttype)`
# 
# This function takes the input dataframes containing the probabilities of achieving {Cat1...Cat5} wind strength and compares them to the climatological probability of a storm having that wind strength. 
# 
# <b>Input</b>: 
# * `ax`: figure axis
# * `df`: Dataframe containing the TCANE predicted probabilities of Cat 1, etc [Dataframe]
# * `df_clim`: Dataframe containing the TCANE climatological probabilities of Cat 1, etc [Dataframe]
# * `df_in`: Dataframe containing input TCANE file (just needed for the labeling) [Dataframe]
# * `ttype`: time type, `erly` or `late` (just needed for labeling) [str]
# 
# <b>Output</b>:
# * `ax`: figure axis
def get_cat_probs(ax,df,df_clim,df_in,ttype):
    colors = ('#284E60', '#E1A730', '#D95980', '#C3B1E1', '#351F27', '#A9C961')
    for i in range(0,5):
        # climo
        sns.lineplot(data=df_clim.reset_index(),x=df_clim.index,y='PCT Cat{i}'.format(i=i+1),color=colors[i],
                    label=('Climo' if i == 0 else None),alpha=0.55,linewidth=4,ax=ax,linestyle=':')
        # actual
        sns.lineplot(data=df.reset_index(),x=df.index,y='PCT Cat{i}'.format(i=i+1),color=colors[i],
                 label=('{exdate}'.format(exdate=df_in.iloc[0]['NAME']) if i == 0 else None),ax=ax,linewidth=4)
    #
    ax.set_xticks(np.arange(0,df.index.max()+1,12))
    ax.legend(fontsize=14)
    ax.grid()
    ax.set_yticks(np.arange(0,101,10))
    ax.set_ylabel('%',fontsize=22)
    ax.set_xlabel('Forecast Hour',fontsize=22)
    ax.tick_params(axis='both',labelsize=16)
    ax.set_title('{ttype}'.format(ttype=ttype),fontsize=28)
    return ax
#############################
### `plot_RI(ax,df,edeck_all)`
# 
# This function calculates the probability of rapid intensification (RI) for the TCANE forecasts. There is an option to compare the TCANE forecasts to other forecasts within the edecks. 
# 
# <b>Input</b>: 
# * `ax`: Figure axis
# * `df`: dataframe containing the RI probabilities (should combine `erly` and `late`) [Dataframe]
# * `edeck_all`: data from the edecks (optional). Default option is to <b>not</b> plot the edecks. [Dataframe]
# 
# <b>Output</b>: 
# * `ax`: Figure axis
def plot_RI(ax,df,edeck_all=pd.DataFrame()):
    RI_thresh = {12:20,
             24:30,
             36:45,
             48:55,
             72:65}
    #
    colors = {'xkcd:yellow orange','xkcd:cornflower'}
    sns.pointplot(data=df,x=df.index,y='PCT RI',hue='TTYPE',ax=ax)
    #
    if not edeck_all.empty:
        edeck_trim = edeck_all[(edeck_all['TAU'].isin(list(RI_thresh.keys()))) & (edeck_all['d_I'].isin(list(RI_thresh.values())))]
        edeck_trim['FHOUR'] = edeck_trim['TAU']
        edeck_trim = edeck_trim.set_index(['FHOUR'])
        sns.pointplot(data=edeck_trim,x=edeck_trim.index,y='Prob(RI)',hue='Technique',palette=sns.color_palette('Greys_r'),ax=ax,label='edeck')
    #
    ax.legend(fontsize=14)
    ax.grid()
    ax.set_ylabel('Pr(RI) (%)',fontsize=22)
    ax.set_xlabel('Forecast Hour',fontsize=22)
    ax.tick_params(axis='both',labelsize=14)
    # ax.set_title('Prob. of RI for {name} ({exdate})'.format(name=df_in.iloc[0]['NAME'],exdate=ex_date),fontsize=28)
    return ax
#################################
### `make_pctile_plot(ax,df,ttype,df_out,df_in,bdeck)`
# 
# This function plots the median TCANE intensity forecast as well as the IQR (25-75th pctiles), realistic best/worst case (10-90th pctiles), and extreme case (1-99th pctiles). `erly` and `late` are plotted separately.
# 
# <b>Input</b>:
# * `ax`: figure axis handle
# * `df`: Dataframe containing percentiles for TCANE plots [Dataframe]
# * `ttype`: time type (`erly` or `late`) [str]
# * `df_out`: Dataframe containing TCANE forecast info (needed for date info) [Dataframe]
# * `df_in`: Dataframe containing TCANE input data (needed for comparisons to other NHC forecasts) [Dataframe]
# * `b_deck`: (optional) dataframe containing b-deck info. Used if optional b-deck plotting is turnd on [Dataframe]
# 
# <b>Output</b>:
# * `ax`: figure axis handle
def make_pctile_plot(ax,df,ttype,df_out,d_in,b_deck=pd.DataFrame()):
    # colors = ('#6449a6','#e58835','#eec91c','#6fa8dc','#ce4969')
    colors = ('xkcd:charcoal','xkcd:turquoise','xkcd:tangerine','xkcd:crimson')
    df_i = df.set_index(['TTYPE']).xs(ttype)
    # make shading
    ax.fill_between(df_i['FHOUR'],df_i['P0.05'],df_i['P0.95'],color=colors[0],alpha=0.2,label='1-99th pctile')
    ax.fill_between(df_i['FHOUR'],df_i['P0.1'],df_i['P0.9'],color=colors[1],alpha=0.3,label='10-90th pctile')
    ax.fill_between(df_i['FHOUR'],df_i['P0.25'],df_i['P0.75'],color=colors[2],alpha=0.45,label='25-75th pctile')
    sns.lineplot(data=df_i,x='FHOUR',y='P0.5',color=colors[3],linewidth=4,label='median',ax=ax)
    sns.scatterplot(data=df_i,x='FHOUR',y='P0.5',color=colors[3],s=150,label=None,ax=ax)
    # obs
    # Add info for t = 0
    d_in.loc[-1,:] = d_in.loc[0,:]
    d_in.loc[-1,'FHOUR'] = 0
    d_in.loc[-1,'VMAXN'] = d_in.loc[-1,'VMAX0']
    d_in.loc[-1,'VMXC'] = d_in.loc[-1,'VMAX0']
    if ttype == 'erly':
        yvar = 'VMXC'
        yvlab = 'TCANE consensus'
    elif ttype == 'late':
        yvar = 'VMAXN'
        yvlab = 'NHC Official'
    sns.lineplot(data=d_in,x='FHOUR',y=yvar,ax=ax,color='xkcd:navy',linewidth=3,label=None)
    sns.scatterplot(data=d_in,x='FHOUR',y=yvar,ax=ax,color='xkcd:navy',marker='v',s=200,label=yvlab)
    
    # (optional) bdecks
    df_out['Forecast Date'] = df_out['DATE'] + pd.to_timedelta(df_out['FHOUR'],'H')
    if not b_deck.empty:
        b_deck = b_deck[b_deck['DATE'].isin(df_out['Forecast Date'])]
        b_deck['FHOUR'] = [b_deck['DATE'].iloc[i] - b_deck['DATE'].iloc[0] for i in np.arange(0,11)]#/np.timedelta64(1,'h')
        b_deck['FHOUR'] = b_deck['FHOUR']/np.timedelta64(1,'h')
        #
        sns.lineplot(data=b_deck,x='FHOUR',y='VMAX',color='xkcd:magenta',linewidth=3,label=None,ax=ax)
        sns.scatterplot(data=b_deck,x='FHOUR',y='VMAX',color='xkcd:magenta',marker='*',s=250,label='bdeck',ax=ax)

    #
    ax.grid()
    ax.set_xticks(df_i['FHOUR'])
    ax.tick_params(axis='both',labelsize=14)
    ax.legend(fontsize=14,loc='lower left')
    ax.set_xlabel('Forecast Hour',fontsize=22)
    ax.set_ylabel('Wind Speed (kt)',fontsize=22)
    ax.set_ylim(bottom=-10)
    ax.set_title('{ttype}'.format(ttype=ttype),fontsize=26)
    return ax
################################# 
# ### `make_all_plotting_data(df_in,df_out,xmax,pvc)`
# This function is just a wrapper that calls the `tcane_data_funcs` functions to make SHASH distributions, PDFs, CDFs, RI estimates, and percentiles for the TCANE results. 
# 
# <b>Inputs</b>:
# * `df_in`: Dataframe containing TCANE inputs (necessary for comparisons with NHC official forecast, initial intensity and so on) [Dataframe]
# * `df_out`: Dataframe containing TCANE outputs [Dataframe]
# * `xmax`: Maxmimum wind speed (needed for PDFs/CDFs) [float]
# * `pvc`: Percentiles we want to estimate (default: [.01,.05,.1,.25,.5,.75,.9,.95,.99]) [array]
# 
# <b>Outputs</b>:
# * `Yshash_erly`: SHASH-distributed data for all forecast times for `erly` forecast, following paramters in `df_out` [Dataframe]
# * `Yshash_late`: same as above but for `late` forecast [Dataframe]
# * `pdf_cdf_erly`: PDF and CDF for TCANE model for all forecast times for `erly` forecast [Dataframe]
# * `pdf_cdf_late`: same as above but for `late` forecast [Dataframe]
# * `Ypct_erly`: desired percentiles of TCANE model for all forecast times for `erly` forecast [Dataframae]
# * `Ypct_late`: same as above but for `late` forecast [Dataframe]
# * `RI_erly`: RI probability estimates for all forecast times for `erly` forecast [Dataframe]
# * `RI_late`: same as above but for `late` forecast [Dataframe]
# * `TC_thresh_erly`: probability estimates of reaching Cat1, Cat2, etc strength for all forecast times for `erly` forecast [Dataframe]
# * `TC_thresh_late`: same as above but for `late` forecast [Dataframe]
def make_all_plotting_data(df_in,df_out,xmax,pvc=None):
    if not pvc:
        pvc = [.01,.05,.1,.25,.5,.75,.9,.95,.99]
    # Make distribution parameters for ERLY and LATE forecasts and the climatological ERLY/LATE forecasts
    tcane_dist_ERLY = tcane_data_funcs.get_TCANE_distribution(df_out,'erly',df_in)
    tcane_dist_LATE = tcane_data_funcs.get_TCANE_distribution(df_out,'late',df_in)
    # Now get actual data following distributions
    V0 = df_in['VMAX0'].iloc[0] # initial intensity
    N = 100000
    Yshash_erly = tcane_data_funcs.make_SHASH(N,tcane_dist_ERLY,'erly',V0)
    Yshash_late = tcane_data_funcs.make_SHASH(N,tcane_dist_LATE,'late',V0)
    # Get PDF/CDF
    pdf_cdf_erly = tcane_data_funcs.get_PDF_CDF(xmax,Yshash_erly,'erly')
    pdf_cdf_late = tcane_data_funcs.get_PDF_CDF(xmax,Yshash_late,'late')
    # Estimate percentiles
    Ypct_erly = tcane_data_funcs.calc_pctiles(Yshash_erly,pvc)
    Ypct_late = tcane_data_funcs.calc_pctiles(Yshash_late,pvc)
    # Get RI/TC thresholds
    RI_erly,TC_thresh_erly = tcane_data_funcs.get_RI_info(pdf_cdf_erly,df_in,'erly')
    RI_late,TC_thresh_late = tcane_data_funcs.get_RI_info(pdf_cdf_late,df_in,'late')
    # 
    Yshash_erly = Yshash_erly.mask(Yshash_erly['Y'] < 0)
    Yshash_late = Yshash_late.mask(Yshash_late['Y'] < 0)
    return Yshash_erly,Yshash_late,pdf_cdf_erly,pdf_cdf_late,Ypct_erly,Ypct_late,RI_erly,TC_thresh_erly,RI_late,TC_thresh_late