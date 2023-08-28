#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import os,glob,re,sys
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import palettable
from scipy.stats import skewnorm, rv_histogram
import scipy.stats as sps
import tcane_data_funcs
import tcane_plotting
import bdeck_edeck_funcs


# In[2]:





# In[5]:


def call_TCANE_plotting(date,in_dir,out_dir,clim_dir,bdeck_dir):
    old_stdout = sys.stdout
    log_file = open("log_files/{fdate}.log".format(fdate=date),"w")
    sys.stdout = log_file
    bas_ab = date[0:2].lower()
    # Get dataframes
    df_climo = tcane_data_funcs.climo_to_df(clim_dir+'tcane_climo_format_{ba}.dat'.format(ba=bas_ab))
    df_out = tcane_data_funcs.read_in_TCANE(out_dir+'{ex_date}_tcane_output.dat'.format(ex_date=date))
    df_in = tcane_data_funcs.read_in_TCANE(in_dir+'{ex_date}_tcane_input.dat'.format(ex_date=date))    
    # Mask and convert str to float
    df_out.iloc[:,6:-1] = df_out.iloc[:,6:-1].astype('float')
    df_in.iloc[:,5:-1] = df_in.iloc[:,5:-1].astype('float')
    df_out = df_out.mask(df_out == -9999.0)
    df_in = df_in.mask(df_in == -9999.0)
    df_in = df_in[df_in['FHOUR'].isin(df_out['FHOUR'])]
    #
    df_out['Forecast Date'] = df_out['DATE'] + pd.to_timedelta(df_out['FHOUR'],'H')
    # Get all plotting data
    pvc = [.01,.05,.1,.25,.5,.75,.9,.95,.99]
    vmax = df_out['VMAXN'].max()
    Y_e,Y_l,pdf_e,pdf_l,pct_e,pct_l,RI_e,TC_e,RI_l,TC_l = tcane_plotting.make_all_plotting_data(df_in,df_out,vmax,pvc)
    #
    c_Y_e,c_Y_l,c_pdf_e,c_pdf_l,c_pct_e,c_pct_l,c_RI_e,c_TC_e,c_RI_l,c_TC_l = tcane_plotting.make_all_plotting_data(df_in,df_climo,vmax,pvc)
    ## Make plots
    fig1,ax1 = plt.subplots(1,1,figsize=(12,7))
    ax1 = tcane_plotting.make_boxplot(ax1,Y_e,Y_l,df_in,[1,99])
    ax1.set_title('TCANE Forecasts, {name}, {ex_date}'.format(name=df_in.iloc[0]['NAME'],ex_date=date),fontsize=26)
    fig1.savefig('Figures/boxplot_{name}_{exdate}.pdf'.format(name=df_in.iloc[0]['NAME'],exdate=date),format='pdf',bbox_inches='tight')
    fig1.savefig('Figures/boxplot_{name}_{exdate}.png'.format(name=df_in.iloc[0]['NAME'],exdate=date),format='png',dpi=400,bbox_inches='tight')
    plt.close()
    ###
    fig20,(ax20,ax20b) = plt.subplots(1,2,figsize=(15,6))
    ax20 = tcane_plotting.get_cat_probs(ax20,TC_e,c_TC_e,df_in,'erly')
    ax20b = tcane_plotting.get_cat_probs(ax20b,TC_l,c_TC_l,df_in,'late')
    fig20.suptitle('{name}, {exdate}'.format(name=df_in.iloc[0]['NAME'],exdate=date),fontsize=35,y=1.05)
    fig20.tight_layout()
    fig20.savefig('Figures/pr_cat_{name}_{exdate}.pdf'.format(name=df_in.iloc[0]['NAME'],exdate=date),format='pdf',bbox_inches='tight')
    fig20.savefig('Figures/pr_cat_{name}_{exdate}.png'.format(name=df_in.iloc[0]['NAME'],exdate=date),format='png',dpi=400,bbox_inches='tight')
    plt.close()
    ###
    fig2,ax2 = plt.subplots(1,1,figsize=(10,6))
    RI_all = pd.concat([RI_e,RI_l])
    edeck_all = bdeck_edeck_funcs.get_edeck_probs(bdeck_dir,date,bas_ab,['RIOC','RIOD','DTOP'])
    ax2 = tcane_plotting.plot_RI(ax2,RI_all,edeck_all)
    ax2.set_title('Prob. of RI for {name} ({exdate})'.format(name=df_in.iloc[0]['NAME'],exdate=date),fontsize=28)
    fig2.savefig('Figures/pr_RI_{name}_{exdate}_with_edeck.pdf'.format(name=df_in.iloc[0]['NAME'],exdate=date),format='pdf',bbox_inches='tight')
    fig2.savefig('Figures/pr_RI_{name}_{exdate}_with_edeck.png'.format(name=df_in.iloc[0]['NAME'],exdate=date),format='png',dpi=400,bbox_inches='tight')
    plt.close()
    ###
    fig22,ax22 = plt.subplots(1,1,figsize=(10,6))
    ax22 = tcane_plotting.plot_RI(ax22,RI_all)
    ax22.set_title('Prob. of RI for {name} ({exdate})'.format(name=df_in.iloc[0]['NAME'],exdate=date),fontsize=28)
    fig22.savefig('Figures/pr_RI_{name}_{exdate}.pdf'.format(name=df_in.iloc[0]['NAME'],exdate=date),format='pdf',bbox_inches='tight')
    fig22.savefig('Figures/pr_RI_{name}_{exdate}.png'.format(name=df_in.iloc[0]['NAME'],exdate=date),format='png',dpi=400,bbox_inches='tight')
    plt.close()
    ### Get bdecks
    b_deck_ALL,b_deck_trim = bdeck_edeck_funcs.get_bdecks(date[4:8],date[2:4],bas_ab,bdeck_dir)
    ##
    fig3,(ax3,ax3b) = plt.subplots(1,2,figsize=(15,8))
    pct_all = pd.concat([pct_e,pct_l])
    ax3 = tcane_plotting.make_pctile_plot(ax3,pct_all,'erly',df_out,df_in,b_deck_trim)
    ax3b = tcane_plotting.make_pctile_plot(ax3b,pct_all,'late',df_out,df_in,b_deck_trim)
    fig3.suptitle('TCANE Forecasts, {name}, {ex_date}'.format(name=df_in.iloc[0]['NAME'],ex_date=date),fontsize=36,y=1.02)
    fig3.tight_layout()
    fig3.savefig('Figures/p1-99_{name}_{exdate}_with_BTR.pdf'.format(name=df_in.iloc[0]['NAME'],exdate=date),format='pdf',bbox_inches='tight')
    fig3.savefig('Figures/p1-99_{name}_{exdate}_with_BTR.png'.format(name=df_in.iloc[0]['NAME'],exdate=date),format='png',dpi=400,bbox_inches='tight')
    plt.close()
    ###
    fig33,(ax33,ax33b) = plt.subplots(1,2,figsize=(15,8))
    ax33 = tcane_plotting.make_pctile_plot(ax33,pct_all,'erly',df_out,df_in)
    ax33b = tcane_plotting.make_pctile_plot(ax33b,pct_all,'late',df_out,df_in)
    fig33.suptitle('TCANE Forecasts, {name}, {ex_date}'.format(name=df_in.iloc[0]['NAME'],ex_date=date),fontsize=36,y=1.02)
    fig33.tight_layout()
    fig33.savefig('Figures/p1-99_{name}_{exdate}.pdf'.format(name=df_in.iloc[0]['NAME'],exdate=date),format='pdf',bbox_inches='tight')
    fig33.savefig('Figures/p1-99_{name}_{exdate}.png'.format(name=df_in.iloc[0]['NAME'],exdate=date),format='png',dpi=400,bbox_inches='tight')
    plt.close()
    #
    sys.stdout = old_stdout
    log_file.close()
    return 


# ##### * `sys.argv[1]`: forecast date (BBNNYYYY_MMDDHH)
# ##### * `sys.argv[2]`: input directory (location of TCANE input files)
# ##### * `sys.argv[3]`: output directory (location of TCANE output files)
# ##### * `sys.argv[4]`: bdeck directory (location of bdeck files)
# 

# In[ ]:


# output_dir = '/mnt/ssd-data1/galina/tcane/data/test_output/'
climo_dir = '/mnt/ssd-data1/galina/tcane/data/climo/'
# input_dir = '/mnt/ssd-data1/galina/tcane/data/test_input/'
# bdeck_dir = '/home/mcgraw/tcane_RT_output/best_tracks/'

# ex_date = 'EP052022_071012'
# ex_date = 'AL032023_062018'
# ex_date = 'AL072022_091612'
#


# In[6]:

if __name__=="__main__":
    call_TCANE_plotting(str(sys.argv[1]),str(sys.argv[2]),str(sys.argv[3]),climo_dir,str(sys.argv[4]))


# In[ ]:




