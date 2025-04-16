import matplotlib.pyplot as plt
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
import matplotlib.patheffects as pe

import importlib as imp
imp.reload(mahalanobis)

COLOR_DEFAULT = palettable.colorbrewer.diverging.RdYlBu_9_r.mpl_colors
COLORMAP_DEFAULT = palettable.colorbrewer.diverging.RdYlBu_9_r.get_mpl_colormap()

DATA_CRS = ct.crs.PlateCarree()
KM_TO_DEG = 1.0 / 111.
NMI_TO_DEG = 1.0 / 60.


def set_plot_rc():
    # for white background...
    plt.rc("text", usetex=True)
    plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Avant Garde"]})
    plt.rc("savefig", facecolor="white")
    plt.rc("figure", facecolor="white")
    plt.rc("axes", facecolor="white")
    plt.rc("axes", labelcolor="dimgrey")
    plt.rc("axes", labelcolor="dimgrey")
    plt.rc("xtick", color="dimgrey")
    plt.rc("ytick", color="dimgrey")
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "white"


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", 5))
        else:
            spine.set_color("none")
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        ax.yaxis.set_ticks([])
    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.xaxis.set_ticks([])


def format_spines(ax):
    adjust_spines(ax, ["left", "bottom"])
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("dimgrey")
    ax.spines["bottom"].set_color("dimgrey")
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params("both", length=4, width=2, which="major", color="dimgrey")


#     ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)


def draw_coastlines(ax):
    # ADD COASTLINES
    # ax.set_global()
    land_feature = cfeature.NaturalEarthFeature(
        category="physical",
        name="land",
        scale="50m",
        facecolor=(0.9, 0.9, 0.9),
        edgecolor="None",
        linewidth=0.0,
        zorder=0,
        alpha=1.,
        )
    land_feature_lines = cfeature.NaturalEarthFeature(
        category="physical",
        name="land",
        scale="50m",
        facecolor='None',
        edgecolor="gray",
        linewidth=0.25,
        zorder=100,
        alpha=1.,
    )
    ax.add_feature(land_feature)
    ax.add_feature(land_feature_lines)


def plot_probability_ellipses(
        df,
        ax,
        leadtimes,
        contours,
        extent=None,
        alpha=0.4,
        annotate_leadtimes=True,
        colors=None,
        vector=True,
        plot_nhc_cone=False,
        is_fill=True,
        linetype='-',
        linewt = 4,
        no_label=False,
        extra_labels=None,
        is_climo=False,
        use_gradient_color=False,
):

    if extent is None:
        extent = [195, 358, 5, 35]
    if (colors is None) & (use_gradient_color == False):
        colors = cmr.take_cmap_colors(
            "cmr.pride", len(contours), cmap_range=(0.2, 0.8), return_fmt="hex"
        )
    elif (colors is None) & (use_gradient_color == True):
        #colors = cmr.take_cmap_colors(
            #"cmr.cosmic", len(np.arange(12,121,12)), cmap_range=(0.1, 0.99), return_fmt="hex")
        colors = cmr.take_cmap_colors(
            "cmr.amber",len(np.arange(12,121,12)), cmap_range=(0.1,0.99), return_fmt="hex")

    if vector:
        # print(is_fill)
        ellipse_lims = plot_probability_ellipses_vector(
            df,
            leadtimes,
            contours,
            alpha=alpha,
            annotate_leadtimes=annotate_leadtimes,
            is_fill=is_fill,
            colors=colors,
            linetype=linetype,
            linewt=linewt,
            no_label=no_label,
            is_climo=is_climo,
            use_gradient_color=use_gradient_color,
        )
    else:
        raise NotImplementedError("no such vector plotting code")

    # plot NHC cone of uncertainty
    if plot_nhc_cone:
        for index, row in df.iterrows():
            if index == 0:
                label = "NHC Cone"
            elif no_label:
                label = None
            else:
                label = None
            circle = plt.Circle((row["LONN"], row["LATN"]), NMI_TO_DEG*row["nhc_cone_radius"],
                                color=colors[1], alpha=.075, label=label, transform=DATA_CRS,)
            ax.add_patch(circle)

    # get uninterpolated leadtimes
    df_nonan = df.dropna(subset="OFDX").copy()
    # plot forecast center
    plt.plot(
        df_nonan["LONN"].values,
        df_nonan["LATN"].values,
        marker=".",
        markersize=1,
        alpha=.5,
        linestyle="None",
        color="k",
        label="NHC Forecast" if not no_label else None,
        transform=DATA_CRS,
    )
    forecast_x = df_nonan["LONN"].values
    forecast_y = df_nonan["LATN"].values
    # print('forecast lon ranges from ',min(forecast_x),max(forecast_x))
    # print('forecast lat ranges from ',min(forecast_y),max(forecast_y))
    # plot bestrack centers
    besttrack_u = df_nonan["LONN"] + KM_TO_DEG * df_nonan["OFDX"]
    besttrack_v = df_nonan["LATN"] + KM_TO_DEG * df_nonan["OFDY"]
    plt.plot(
        besttrack_u.values,
        besttrack_v.values,
        "-x",
        color="k",
        markersize=7,
        linewidth=.5,
        label="BestTrack" if not no_label else None,
        transform=DATA_CRS,
    )
    btrack_x = besttrack_u.values
    # print('best track lon ranges from ',min(btrack_x),max(btrack_x))
    btrack_y = besttrack_v.values
    # print('best track lat ranges from ',min(btrack_y),max(btrack_y))
    # Get plot extents
    forecast_minx = min(min(forecast_x),min(btrack_x))
    forecast_maxx = max(max(forecast_x),max(btrack_x))
    uq_minx = min(ellipse_lims['xmin'])
    uq_maxx = max(ellipse_lims['xmax'])
    # print('uq ranges, lon:',uq_minx,uq_maxx)
    #
    forecast_miny = min(min(forecast_y),min(btrack_y))
    forecast_maxy = max(max(forecast_y),max(btrack_y))
    uq_miny = min(ellipse_lims['ymin'])
    uq_maxy = max(ellipse_lims['ymax'])
    # print('uq ranges, lat:',uq_miny,uq_maxy)
    x_min = min(uq_minx,forecast_minx)
    x_max = max(uq_maxx,forecast_maxx)
    xlims = [x_min,x_max]
    y_min = min(uq_miny,forecast_miny)
    y_max = max(uq_maxy,forecast_maxy)
    ylims = [y_min,y_max]
    # format plot
    format_spines(ax)

    # set grid lines
    gl = ax.gridlines(
        crs=ct.crs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.left_labels = False

    # setup legend
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    # print(extra_labels)
    if extra_labels:
        plt.gca().legend(handles, extra_labels, loc=2, frameon=True, fontsize=6)
    else:
        plt.gca().legend(handles, labels, loc=2, frameon=True, fontsize=6)
    

    # set storm title
    details = data_info.get_storm_details(df, 0)
    details = details[: details.rfind(" @")]
    plt.title(details)

    draw_coastlines(ax)

    ax.set_extent(extent, crs=ct.crs.PlateCarree())

    return xlims,ylims,details,ax


def plot_probability_ellipses_vector(
        df,
        leadtimes,
        contours,
        alpha=0.4,
        annotate_leadtimes=True,
        is_fill=True,
        colors=None,
        linetype='-',
        linewt=6,
        no_label=False,
        is_climo=False,
        use_gradient_color=False
):
    # print(is_fill)
    ellipse_lims = pd.DataFrame(columns=['xmin','xmax','ymin','ymax'],index=leadtimes)
    # Get colormap
    #color_amber = cmr.take_cmap_colors( "cmr.tropical", len(np.arange(12,121,12)), cmap_range=(0.2, 0.95), return_fmt="hex")
    color_amber = cmr.take_cmap_colors("magma", len(np.arange(12,121,12)), cmap_range=(0.2,0.8), return_fmt="hex")
    keys = np.arange(12,121,12)
    color_dict = {}
    for ikey in np.arange(0,len(keys)):
        color_dict[keys[ikey]] = color_amber[ikey]
    # for lead_time in leadtimes:
    for lead_time in leadtimes:
        if use_gradient_color:
            # print('using gradient color')
            #color_cosmic = {12:'#150C1C',
            #    24:'#2F1747',
             #   36:'#4A187F',
             #   48:'#5F04C5',
             #   60:'#573CE7',
              #  72:'#406AE1',
              #  84:'#2B8BDA',

             #   96:'#1CA7D7',
             #   108:'#0CC3D7',
             #   120:'#08E0D5'}
            # 
            
            #
                            
            colors = [color_dict[lead_time]]
            # print('color is ',colors)
        df_plot = df.loc[(df["ftime(hr)"] == lead_time)]
        line_widths = {12:2,
                           24:2,
                           36:3,
                           48:3,
                           60:4,
                           72:4,
                           84:5,
                           96:5,
                           108:6,
                           120:6}
        if df_plot.empty:
            # print(str(lead_time) + " dataframe is empty")
            continue

        # plot forecast
        if lead_time == leadtimes[0]:
            if not is_climo:
                label = contours
            else:
                label =['Tclimo '+str(i) for i in contours]
        else:
            label = None

        xmal,ymal = mahalanobis.plot_cdf(
            df_plot["LONN"].values + KM_TO_DEG * df_plot["mu_u"].values,
            df_plot["LATN"].values + KM_TO_DEG * df_plot["mu_v"].values,
            KM_TO_DEG * df_plot["sigma_u"].values,
            KM_TO_DEG * df_plot["sigma_v"].values,
            df_plot["rho"].values,
            contours=contours,
            data_crs=DATA_CRS,
            alpha=alpha,
            colors=colors,
            # label=label,
            is_fill=is_fill,
            label=label,
            linetype=linetype,
            linewt=line_widths[lead_time]
            )
        ellipse_lims['xmin'].loc[lead_time] = xmal.min()
        ellipse_lims['xmax'].loc[lead_time] = xmal.max()
        ellipse_lims['ymin'].loc[lead_time] = ymal.min()
        ellipse_lims['ymax'].loc[lead_time] = ymal.max()
        #print('mahalanobis x',min(xmal),max(xmal),'; mahalanobis y',min(ymal),max(ymal))
        if annotate_leadtimes:
            df_nonan = df_plot.dropna(subset="OFDX").copy()
            if(len(df_nonan) > 0):
                #for i in np.arange(0,len(df_nonan)):
                plt.text(
                    df_nonan["LONN"].values,
                    df_nonan["LATN"].values,
                    int(df_nonan["ftime(hr)"].values[0]),
                    # color="k",
                    color = colors[0],
                    path_effects=[pe.Stroke(linewidth=2, foreground='k',alpha=0.4), pe.Normal()],
                    fontsize=12,
                    horizontalalignment="left",
                    verticalalignment="bottom",
                    transform=DATA_CRS,
                    zorder=200,
                )
    #ellipse_x = [min(xmins),max(xmaxs)]
    #ellipse_y = [min(ymins),max(ymaxs)]
    return ellipse_lims


def plot_banana_of_uncertainty(ax, df_storm, extent, vector=True, colors=None, alpha=1., plot_nhc_cone=False):

    if colors is None:
        colors = ("gold",)

    df_storm = df_storm.sort_values("ftime(hr)").reset_index(drop=True)
    leadtimes = df_storm["ftime(hr)"].values
    x_interp = np.arange(leadtimes[0], leadtimes[-1]+1, 1)

    # interpolate things
    mu_u_interp = compute_predictions.interpolate_leadtimes(leadtimes, df_storm["mu_u"].values, x_interp)
    mu_v_interp = compute_predictions.interpolate_leadtimes(leadtimes, df_storm["mu_v"].values, x_interp)
    sigma_u_interp = compute_predictions.interpolate_leadtimes(leadtimes, df_storm["sigma_u"].values, x_interp)
    sigma_v_interp = compute_predictions.interpolate_leadtimes(leadtimes, df_storm["sigma_v"].values, x_interp)
    rho_interp = compute_predictions.interpolate_leadtimes(leadtimes, df_storm["rho"].values, x_interp)

    lonn_interp = compute_predictions.interpolate_leadtimes(leadtimes, df_storm["LONN"].values, x_interp)
    latn_interp = compute_predictions.interpolate_leadtimes(leadtimes, df_storm["LATN"].values, x_interp)

    ofdx_interp = compute_predictions.interpolate_leadtimes(leadtimes, df_storm["OFDX"].values, x_interp)
    ofdy_interp = compute_predictions.interpolate_leadtimes(leadtimes, df_storm["OFDY"].values, x_interp)

    nhc_cone_radius_interp = compute_predictions.interpolate_leadtimes(leadtimes, df_storm["nhc_cone_radius"].values, x_interp)

    # clip rho to be between (-1,1)
    rho_interp = np.clip(rho_interp, -1., 1.)

    # clip sigmas to be postive
    sigma_u_interp = np.clip(sigma_u_interp, 0., None)
    sigma_v_interp = np.clip(sigma_v_interp, 0., None)

    d_interp = {
        "ftime(hr)": x_interp,
        "mu_u": mu_u_interp,
        "mu_v": mu_v_interp,
        "sigma_u": sigma_u_interp,
        "sigma_v": sigma_v_interp,
        "rho": rho_interp,
        "LONN": lonn_interp,
        "LATN": latn_interp,
        "OFDX": ofdx_interp,
        "OFDY": ofdy_interp,
        "nhc_cone_radius": nhc_cone_radius_interp,

    }

    df_storm_interp = pd.DataFrame(data=d_interp)
    df_storm_interp[["Name", "year", "time"]] = df_storm[["Name", "year", "time"]]
    df_storm_interp.loc[~df_storm_interp["ftime(hr)"].isin(leadtimes), ["OFDX", "OFDY"]] = np.nan

    details = plot_probability_ellipses(
        df_storm_interp,
        ax=ax,
        leadtimes=df_storm_interp["ftime(hr)"].unique(),
        contours=(.6667, ),
        extent=extent,
        annotate_leadtimes=True,
        alpha=alpha,
        colors=colors,
        vector=vector,
        plot_nhc_cone=plot_nhc_cone,
    )

    return details

