import numpy as np
import pandas as pd
import os,glob,re
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import palettable
from scipy.stats import skewnorm, rv_histogram
import scipy.stats as sps

### `get_bdecks(year,stormno,basin,bdeck_dir)`
# 
# This function gets the best track files for a given storm, identified by `year`, `stormno`, and `basin`. 
# 
# <b>Inputs</b>:
# * `year`: year of desired storm
# * `stormno`: number of desired storm (01-49, starts over again every year) [str]
# * `basin`: basin where storm is located (2 characters) [str]
# * `bdeck_dir`: directory where bdeck files are located [str]
# 
# <b>Outputs</b>:
# * `b_deck_ALL`: Dataframe with all best track information [Dataframe]
# * `b_deck_trim`: same as above, but trimmed to only identifying information and `VMAX` and `MSLP` [Dataframe]
def get_bdecks(year,stormno,basin,bdeck_dir):
    # Create appropriate filename
    if (basin == 'AL')|(basin == 'EP'):
        bas_dir = 'NHC'
    else:
        bas_dir = 'DOD'
    bd_dir_full = bdeck_dir#+'{bas_dir}/'.format(bas_dir=bas_dir)
    fpath_full = bd_dir_full+'b{basin}{stormno}{yr}.dat'.format(basin=basin.lower(),stormno=stormno,yr=year)
    if not os.path.exists(fpath_full):
        b_deck_ALL = pd.DataFrame()
        bdeck_trim = pd.DataFrame()
    else:
        # Set up empty dataframe
        b_deck_ALL = pd.DataFrame()
        fnames_all = [fpath_full]
        # 
        i_bdeck = pd.DataFrame()
        # For each file, read in best-track information from text file
        for i_line in np.arange(0,len(fnames_all)):
            # print('reading ',fnames_all[i_line])
            lines = open(fnames_all[i_line]).readlines()
            b_deck = pd.DataFrame(columns=['BASIN','CYCLONE NO','DATE','TECHNUM','TECH','TAU','LAT','LON','VMAX','MSLP','TYPE',
                                      'RAD','WINDCODE','RAD1','RAD2','RAD3','RAD4','P Outer','R Outer','RMW','GUSTS','EYE',
                                      'SUBREGION','MAXSEAS','INITIALS','DIR','SPEED','NAME','DEPTH','SEAS','SEASCODE',
                                      'SEAS1','SEAS2','SEAS3','SEAS4'],
                             index = np.arange(0,len(lines)))
            for i_sub in np.arange(0,len(lines)):
                #
                i_sel = lines[i_sub].split()
                max_len = min(len(i_sel),35)
                b_deck.iloc[i_sub,0:max_len] = i_sel[0:max_len]
            i_bdeck = pd.concat([i_bdeck,b_deck],ignore_index=True)
        # Remove superfluous commas
        i_bdeck = i_bdeck.reset_index().replace(",","",regex=True)
        # Put date in datetime format and get ATCF ID
        i_bdeck['DATE'] = pd.to_datetime(i_bdeck['DATE'].astype(str),format='%Y%m%d%H')
        #
        ATCFID = i_bdeck['BASIN']+i_bdeck['CYCLONE NO']+str(year)
        i_bdeck['ATCF ID'] = ATCFID
        i_bdeck['TIME'] = i_bdeck['DATE'].dt.hour
        i_bdeck = i_bdeck.drop(columns='index')
        #
        b_deck_ALL = pd.concat([b_deck_ALL,i_bdeck])
        # Trim to only necessary columns
        col_keep = ['BASIN','CYCLONE NO','DATE','VMAX','MSLP','TYPE','RAD']
        bdeck_trim = b_deck_ALL[col_keep]
        col_to_int = ['VMAX','MSLP','RAD']
        bdeck_trim[col_to_int] = bdeck_trim[col_to_int].astype(int)
        bdeck_trim = bdeck_trim.where((bdeck_trim['RAD']==34)|(bdeck_trim['RAD']==0)).dropna(how='all',ignore_index=True)
        # Reformat Lat and Lon (for calculating storm relative motion)
        b_deck_ALL['TLAT'] = b_deck_ALL['LAT'].str.replace('\D', '', regex=True).astype(int)*0.1
        b_deck_ALL['TLON'] = b_deck_ALL['LAT'].str.replace('\D', '', regex=True).astype(int)*0.1
        b_deck_ALL[b_deck_ALL['LON'].str.contains('E')]['TLON'] = -1*b_deck_ALL[b_deck_ALL['LON'].str.contains('E')]['TLON']
        # Multiply W longitudes by -1
        xloc = b_deck_ALL['TLON'].where(b_deck_ALL['LON'].str.contains('W')).index
        b_deck_ALL['TLON'].loc[xloc] = b_deck_ALL['TLON'].loc[xloc]*-1

    return b_deck_ALL, bdeck_trim
#######
### `get_edeck_probs(edeck_dir,seldate,basin_abb)`
# 
# This function reads in the corresponding edeck (probabilistic) file for the desired storm. 
# 
# <b>Inputs</b>:
# * `edeck_dir`: directory where edeck files are located [str]
# * `seldate`: TCANE filename and date (`BBNNYYYY_MMDDFF`, where `BB` is basin, and `NN` is storm number) [str]
# * `basin_abb`: 2-character basin abbreviation [str]
# 
# <b>Outputs</b>:
# * `edeck_out`: Dataframe containing output of correspoding edeck file [str]
def get_edeck_probs(edeck_dir,seldate,basin_abb,tech_sel=['RIOC','RIOD']):
    # Get regular lsdiag file
    #fname = '{fdate}{BA}'.format(fdate=fdate,BA=basin_abb)
    # files = predict_test.get_all_lsdiag(fpath,fname)
    edeck_out = pd.DataFrame()
    # Get edeck identifying info
    nn = seldate[2:4]
    yy = seldate[6:8]
    fdate = '{yy}{mmddhh}'.format(yy=yy,mmddhh=seldate[9:])
    edeck_ID = '{BA}{nn}20{yy}'.format(BA=basin_abb,nn=nn,yy=yy)
    #
    edeck_tot = edeck_dir+'e{edeck}.dat'.format(edeck=edeck_ID)
    print(edeck_tot)
    # Get RI probabilities from edeck
    num_save = pd.Series()
    with open(edeck_tot) as f:
        for num, line in enumerate(f, 1):
            re_arnold = re.search(r',\s*([^,]*?{fdate}[^,]*?)\s*,'.format(fdate=fdate), line)

            if re_arnold:
                # print('{} {}'.format(re_arnold.group(1), num))
                num_save = np.append(num_save,num)
    #
    #
    with open(edeck_tot) as f:
        fen = f.readlines()
        
    column_names = ['BASIN','Cyclone No','Date_full','Prob Item','Technique','TAU','LAT','LON',
                                  'Prob(RI)','d_I','V_final','Forecaster ID','RI start time','RI end time']
    RI_prob_df = pd.DataFrame(columns=['BASIN','Cyclone No','Date_full','Prob Item','Technique','TAU','LAT','LON',
                          'Prob(RI)','d_I','V_final','Forecaster ID','RI start time','RI end time'])
    
    RI_prob_df = RI_prob_df.reindex(columns=column_names,index=num_save.astype(int))
    #
    for i in num_save.astype(int):
        #print(i)
        bloop = fen[i].split(',')
        # print(bloop)
        if bloop[3] == ' RI':
            RI_prob_df.loc[i,:] = bloop[0:14]
    #
    RI_prob_df = RI_prob_df.replace(',','',regex=True)
    RI_prob_df = RI_prob_df.replace(' ','',regex=True)
    #
    RI_prob_df['ATCF ID'] = RI_prob_df['BASIN']+RI_prob_df['Cyclone No']+RI_prob_df['Date_full'].str[:4]
    RI_prob_df['DATE'] = pd.to_datetime(RI_prob_df['Date_full'],format='%Y%m%d%H')
    RI_prob_df = RI_prob_df.dropna(how='all')
    #
    RI_prob_df[['TAU','Prob(RI)','d_I','V_final','RI start time','RI end time']] = RI_prob_df[['TAU','Prob(RI)','d_I','V_final','RI start time','RI end time']].astype(int)
    #
    RI_thresh = RI_prob_df#.where((RI_prob_df['d_I'] == 30)& (RI_prob_df['RI end time'] == 24)).dropna(how='all')
    RI_sel = RI_thresh[RI_thresh['Technique'].isin(tech_sel)]
    #RI_sel = RI_sel[['BASIN','Cyclone No','Date_full','Technique','Prob(RI)','d_I','V_final']]
    edeck_out = pd.concat([edeck_out,RI_sel])
    # print('{fdate} Consensus RI forecast: '.format(fdate=fdate+nn),RI_sel.set_index(['Technique']).xs('RIOC')['Prob(RI)'],'%')
    # print('{fdate} SHIPS-RII forecast: '.format(fdate=fdate+nn),RI_sel.set_index(['Technique']).xs('RIOD')['Prob(RI)'],'%')
    return edeck_out
