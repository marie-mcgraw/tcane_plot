import numpy as np
import pandas as pd
import os,glob,re
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import palettable
from scipy.stats import skewnorm, rv_histogram
import scipy.stats as sps

pd.options.mode.chained_assignment = None

# ### 1. `climo_to_df(file)`
# This script reads in the TCANE climatology files and saves the output to a `Pandas` DataFrame. 
# 
# <b>Inputs</b>:
# * `file`: name of file to read into Pandas dataframe [str]
# 
# <b>Outputs</b>: 
# * `df_clim`: contents of `file` converted to `Pandas` DataFrame [DataFrame]

def climo_to_df(file):
    with open(file) as f:
        col_headers = f.readline().split()
    #
    with open(file, 'r') as fp:
        for count, line in enumerate(fp):
            pass
    #
    df_clim = pd.DataFrame(columns=col_headers,index=np.arange(0,count))
    #
    with open(file,'r') as fi:
        fen = fi.readlines()
        for i in np.arange(1,count+1):
            fline = fen[i].split()
            df_clim.loc[i-1,:] = fline
    #
    df_clim.iloc[:,3:-1] = df_clim.iloc[:,3:-1].astype('float')
    df_clim = df_clim.mask(df_clim == -9999.0)
    df_clim['FHOUR'] = df_clim['FHOUR'].astype('int')
    return df_clim
############
# ### 2. `read_in_TCANE(filename)`
# This script reads in TCANE output files from the `.dat` format and saves the contents to a `Pandas` DataFrame. 
# 
# <b>Inputs</b>:
# * `file`: name of file to read into Dataframe [str]
# 
# <b>Outputs</b>: 
# * `df`: contents of `file` converted to `Pandas` DataFrame [DataFrame]

def read_in_TCANE(filename):
    # Get column headers
    with open(filename) as f:
        col_headers = f.readline().split()
    # Get number of lines in file
    with open(filename,'r') as fp:
        for count,line in enumerate(fp):
            pass
    # Make dataframe
    df = pd.DataFrame(columns=col_headers,index=np.arange(0,count))
    with open(filename,'r') as fi:
        fen = fi.readlines()
        for i in np.arange(1,count+1):
            fline = fen[i].split()
            df.loc[i-1,:] = fline
    # Make a date column
    df['FHOUR'] = df['FHOUR'].astype('int')
    df['DATE'] = pd.to_datetime(df['YEAR']+df['MMDDHH'],format='%Y%m%d%H')
    return df
################
### `get_TCANE_distribution_intensity (df,ttype_sel,df_in)`
# This script reads in TCANE output files from the `.dat` format and saves the contents to a `Pandas` DataFrame. 
# 
# <b>Inputs</b>:
# * `file`: name of file to read into Dataframe [str]
# 
# <b>Outputs</b>: 
# * `df_tcdist`: `Pandas` DataFrame that contains the `mu`, `sigma`, `gamma`, and `tau` variables from the TCANE files [DataFrame]
def get_TCANE_distribution(df,ttype_sel,df_in):
    dfx = df.set_index(['TTYPE','FHOUR']).xs(ttype_sel)
    dfi = df_in.set_index(['FHOUR'])
    df_tcdist = pd.DataFrame(index=dfx.index,columns=['DIST','TTYPE'])
    for ihr in dfx.index:
        if ihr not in dfi.index:
            continue
        dist_tcane = [float(dfx.loc[ihr]['MU']) + float(dfi.loc[ihr]['VMAXN']), float(dfx.loc[ihr]['SIGMA']), float(dfx.loc[ihr]['GAMMA']), float(dfx.loc[ihr]['TAU'])]
        df_tcdist.loc[ihr]['DIST'] = dist_tcane
        df_tcdist.loc[ihr]['TTYPE'] = ttype_sel
    return df_tcdist
#####################
### `make_SHASH(N,tcane_dist,ttype,V0)`
# This function takes the distribution parameters from the TCANE output files and recreates a SHASH distribution based on the data. This will help us estimate rapid intensification thresholds, probability of reach certain wind speeds, and uncertainty around our predictions. 
# <b>Inputs</b>: 
# * `N`: Number of datapoints in the distribution [int]
# * `tcane_dist`: distribution parameters [`mu`, `sigma`, `gamma`, `tau`] from TCANE output files; one set of parameters per forecast time [Dataframe]
# * `ttype`: time type; `erly` or `late` [str]
# * `V0`: initial wind speed [int]
# 
# <b>Outputs</b>:
# * `pdf_ALL`: DataFrame containing data simulated by SHASH distribution; there is one distribution per forecast time [DataFrame]
def make_SHASH(N,tcane_dist,ttype,V0):
    for ihr in tcane_dist.index:
        dist = tcane_dist.loc[ihr]
        df_pdf = pd.DataFrame(columns=['Y','FHOUR','TTYPE'],index=np.arange(0,N))
        if ((np.size(dist['DIST']) == 1)):
            if (np.isnan(dist['DIST'])):
                df_pdf['FHOUR'] = ihr
                df_pdf['Y'] = V0
                df_pdf['TTYPE'] = ttype
                # df_pdf['CDF'] = np.nan
        else:
            # 
            Z = sps.norm.rvs(size=N)
            Y = dist['DIST'][0] + dist['DIST'][1] * np.sinh((np.arcsinh(Z) + dist['DIST'][2]) / dist['DIST'][3])
            df_pdf['FHOUR'] = ihr
            df_pdf['Y'] = Y
            df_pdf['TTYPE'] = ttype
           # df_pdf['CDF'] = cdf_tcane
            
        if ihr == tcane_dist.index[0]:
            pdf_ALL = df_pdf.copy()
        else:
            pdf_ALL = pd.concat([pdf_ALL,df_pdf])
    return pdf_ALL
###########
### `calc_pctiles(Y,pctiles)`
# This function estimates the percentiles specified in `pctiles` from the estimated data distribution.
# 
# <b>Inputs</b>:
# * `Y`: input SHASH dataframe [Dataframe]
# * `pctiles`: array containing percentiles desired for estimation. Should be between 0 and 1. If nothing entered, use default set of [.01,.05,.1,.25,.5,.75,.9,.95,.99] [np array]
# 
# <b>Outputs</b>:
# * `Y`: input dataframe with additional columns for percentile estimates [Dataframe]
def calc_pctiles(Y,pctiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]):
    # Check to make sure percentiles are between 0 and 1; if not, divide by 100
    check_pct = True in (ip > 1 for ip in pctiles)
    Y['Y'] = Y['Y'].astype(float)
    if check_pct:
        pctiles = np.divide(pctiles,100)
    # Calculate pctiles
    for i in pctiles:
        Y['P{i}'.format(i=i)] = Y.groupby(['FHOUR'])['Y'].transform(lambda x: x.quantile(i))
    # Simplify dataset since transform has made it huge
    Y = Y.groupby(['TTYPE','FHOUR']).mean().reset_index()
    return Y
#############
### `get_PDF_CDF(xmax,Y,ttype)`
# 
# This function calculates probability density functions (PDFs) and cumulative density functions (CDFs) from the data. 
# 
# <b>Input</b>:
# * `xmax`: maximum value of winds [float]
# * `Y`: Dataframe containing modeled TCANE distribution [Dataframe]
# * `ttype`: time type (`erly` or `late`) [str]
# 
# <b>Output</b>:
# * `pdf_cdf_ALL`: Dataframe containing PDF and CDF data. Different PDF/CDF for each forecast time [Dataframe]
def get_PDF_CDF(xmax,Y,ttype):
    maxval = max(xmax*2,140)
    xbins = np.arange(0,maxval,1)
    pdf_cdf_ALL = pd.DataFrame()
    Yind = Y.set_index(['FHOUR'])
    for ihr in Yind.index.unique():
        Y_i = Yind.xs(ihr)
        count,bins_count = np.histogram(Y_i['Y'].astype(float),bins=xbins)
        pdf = count/np.sum(count)
        cdf = np.cumsum(pdf)
        #
        i_pdf = pd.DataFrame(columns=['FHOUR','Ycount','Ybins','PDF','CDF','TTYPE'])
        i_pdf['Ycount'] = count
        i_pdf['Ybins'] = bins_count[1:]
        i_pdf['PDF'] = pdf
        i_pdf['CDF'] = np.round(cdf,3)
        i_pdf['FHOUR'] = ihr
        i_pdf['TTYPE'] = ttype
        #
        pdf_cdf_ALL = pd.concat([pdf_cdf_ALL,i_pdf])
    return pdf_cdf_ALL
#######################
### `get_RI_info(df,x_in,ttype)`
# 
# This function uses the estimated SHASH PDFs to estimate the probability of rapid intensification (`RI_all`) and the probability of achieveing (Cat1, Cat2, etc) intensity (`df_thresh`).
# 
# <b>Inputs</b>:
# * `df`: Dataframe containing the estimated PDFs. Each forecast time has a different PDF [Dataframe]
# * `x_in`: Dataframe containing TCANE input data (used to get the initial wind speed needed for RI estimation) [Dataframe]
# * `ttype`: time type (`erly` or `late`) [str]
# 
# <b>Outputs</b>: 
# * `RI_all`: Dataframe containing probability of RI at each RI threshold for each forecast time [Dataframe]
# * `df_thresh`: Dataframe containing probability of achieving each category on the Saffir-Simpson scale for each forecast time [Dataframe]
def get_RI_info(df,x_in,ttype):
    RI_thresh = {12:20,
             24:30,
             36:45,
             48:55,
             72:65}
    TC_thresh = {1:64,
             2:83,
             3:96,
             4:113,
             5:137}
    ##
    df = df[df['FHOUR']<=x_in['FHOUR'].max()]

    for i in RI_thresh.keys():
        RI_prob = pd.DataFrame(columns=['TTYPE','PCT RI','RI THRESH'],index=[i])
        #
        foo = df[df['FHOUR']==i]
        x_in['VMAX0'] = x_in['VMAX0'].astype(float)
        if i not in x_in['FHOUR'].values:
            RI_prob['TTYPE'] = ttype
            RI_prob['PCT RI'].loc[i] = np.nan
            RI_prob['RI THRESH'].loc[i] = np.nan
        else:
            thresh = x_in.set_index(['FHOUR']).xs(i)['VMAX0']+RI_thresh[i]
            pct_RI = (1-foo.loc[thresh]['CDF'])*100
            RI_prob['TTYPE'].loc[i] = ttype
            RI_prob['PCT RI'].loc[i] = pct_RI
            RI_prob['RI THRESH'].loc[i] = thresh
        #
        if i == list(RI_thresh.keys())[0]:
            RI_all = RI_prob
        else:
            RI_all = pd.concat([RI_all,RI_prob])
    # Put into cats
    hrs = df['FHOUR'].unique().tolist()
    df_thresh = pd.DataFrame(columns=['TTYPE','PCT Cat1','PCT Cat2','PCT Cat3','PCT Cat4','PCT Cat5'],index=hrs)
    for ihr in hrs:
        ihfoo = df[df['FHOUR']==ihr]
        thresh = [TC_thresh[i] for i in TC_thresh.keys()]
        pct_RI = (1-ihfoo.loc[[y for y in thresh]]['CDF'])*100
        df_thresh['TTYPE'] = ttype
        df_thresh.loc[ihr,['PCT Cat1','PCT Cat2','PCT Cat3','PCT Cat4','PCT Cat5']] = pct_RI.values
    return RI_all,df_thresh
#############################
