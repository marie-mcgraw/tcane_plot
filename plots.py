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
        no_label=False,
        extra_labels=None,
        is_climo=False,
):

    if extent is None:
        extent = [195, 358, 5, 35]
    if colors is None:
        colors = cmr.take_cmap_colors(
            "cmr.pride", len(contours), cmap_range=(0.2, 0.8), return_fmt="hex"
        )

    if vector:
        print(is_fill)
        plot_probability_ellipses_vector(
            df,
            leadtimes,
            contours,
            alpha=alpha,
            annotate_leadtimes=annotate_leadtimes,
            is_fill=is_fill,
            colors=colors,
            linetype=linetype,
            no_label=no_label,
            is_climo=is_climo,
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

    # plot bestrack centers
    besttrack_u = df_nonan["LONN"] + KM_TO_DEG * df_nonan["OFDX"]
    besttrack_v = df_nonan["LATN"] + KM_TO_DEG * df_nonan["OFDY"]
    plt.plot(
        besttrack_u.values,
        besttrack_v.values,
        "-x",
        color="k",
        markersize=4,
        linewidth=.5,
        label="BestTrack" if not no_label else None,
        transform=DATA_CRS,
    )

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

    return details,ax


def plot_probability_ellipses_vector(
        df,
        leadtimes,
        contours,
        alpha=0.4,
        annotate_leadtimes=True,
        is_fill=True,
        colors=None,
        linetype='-',
        no_label=False,
        is_climo=False,
):
    print(is_fill)
    for lead_time in leadtimes:
        df_plot = df.loc[(df["ftime(hr)"] == lead_time)]
        if df_plot.empty:
            # print(str(lead_time) + " dataframe is empty")
            continue

        # plot forecast
        if lead_time == leadtimes[0]:
            if not is_climo:
                label = contours
            else:
                label =['climo '+str(i) for i in contours]
        else:
            label = None

        mahalanobis.plot_cdf(
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
            )
        if annotate_leadtimes:
            df_nonan = df_plot.dropna(subset="OFDX").copy()
            if(len(df_nonan) > 0):
                plt.text(
                    df_nonan["LONN"].values,
                    df_nonan["LATN"].values,
                    int(df_nonan["ftime(hr)"].values[0]),
                    color="k",
                    fontsize=5,
                    horizontalalignment="left",
                    verticalalignment="bottom",
                    transform=DATA_CRS,
                    zorder=200,
                )

    return None


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

