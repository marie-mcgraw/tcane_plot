"""Tools for the Mahalanobis cdf using a bivariate normal.

Functions
---------
plot_cdf(mu_u, mu_v, sigma_u, sigma_v, rho)
compute_cdf(mu_u, mu_v, sigma_u, sigma_v, rho, y)

Notes
-----
* The Mahalanobis cdf is defined in the "Cumulative distribution
    function" section of [1].

* The Mahalanobis cdf is the probability that a sample lies inside
    the ellipse determined by its Mahalanobis distance.

References
----------
[1] Wikipedia contributors. (2022, July 20). Multivariate normal
    distribution. In Wikipedia, The Free Encyclopedia. Retrieved
    19:53, August 5, 2022.

[2] Wikipedia contributors. (2022, June 21). Mahalanobis distance.
    In Wikipedia, The Free Encyclopedia. Retrieved 19:55, August 5, 2022.

[3] Michael Bensimhoun. (2009, June). N-dimension Cumulative Function
    and Other Useful Facts About Gaussians and Normal Densities.
    https://upload.wikimedia.org/wikipedia/commons/a/a2/Cumulative_function_n_dimensional_Gaussians_12.2013.pdf

"""
import matplotlib.pyplot as plt
import numpy as np
import palettable
import cartopy as ct

import silence_tensorflow.auto
import tensorflow as tf


__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "14 December 2022"

COLOR_DEFAULT = palettable.colorbrewer.diverging.RdYlBu_9_r.mpl_colors
COLORMAP_DEFAULT = palettable.colorbrewer.diverging.RdYlBu_9_r.get_mpl_colormap()
THETA = np.linspace(0, 2 * np.pi, 1000)


def get_pixel_array(
        mu_u,
        mu_v,
        sigma_u,
        sigma_v,
        rho,
        contours=np.arange(9, 0, -1) / 10.0,
):
    # Create the blank pixel array
    umin = 150.
    umax = 360.0

    vmin = 0.0
    vmax = 40.0

    ncol = 2_500
    nrow = 2_500

    # Define the blank pixel array
    u, v = np.meshgrid(np.linspace(umin, umax, ncol), np.linspace(vmin, vmax, nrow))
    z_full = np.zeros((nrow, ncol), dtype=int)
    for i, p in enumerate(contours):
        # Define the blank pixel array
        z = np.zeros((nrow, ncol), dtype=int)
        # Define the threshold on the rsqr.
        threshold = -2.0 * (np.log(1 - p))
        # Identify the pixel array nodes that fall inside the ellipse.
        U = (u - mu_u) / sigma_u
        V = (v - mu_v) / sigma_v
        rsqr = 1.0 / (1.0 - rho * rho) * (U * U - 2 * rho * U * V + V * V)

        z[rsqr <= threshold] = 1.0
        z_full = z_full + z
    # z_full = z_full.astype("float")
    # z_full[z_full == 0] = np.nan

    return np.unique(u), np.unique(v), z_full


def plot_cdf(
        mu_u,
        mu_v,
        sigma_u,
        sigma_v,
        rho,
        colors=COLOR_DEFAULT,
        contours=np.arange(9, 0, -1) / 10.0,
        data_crs=ct.crs.PlateCarree(),
        alpha=0.4,
        is_fill=True,
        label=None,
        linetype='-',
        ):
    """Plot the Mahalanobis cdf.

    Plot the elliptical contours of the Mahalanobis cdf for a bivariate
    normal distribution. The nine contour levels are the distribution's
    deciles: probability of capture = {0.10, 0.20, ..., 0.90}.

    Plot the Consensus track as a circle at (0, 0). If given, plot the
    Besttrack as a square.

    Arguments
    ---------
    mu_u : float
        mean of u.

    mu_v : float
        mean of v.

    sigma_u : float, u > 0.
        standard deviation of u.

    sigma_v : float, v > 0.
        standard deviation of v.

    rho : float, -1 < rho < 1.
        correlation between u and v.

    Returns
    -------
    None

    Notes
    -----
    * The equations for x and y come from the bottom of Page 2 in [3].

    """

    contours = np.sort(contours)[::-1]
    for i, p in enumerate(contours):
        r = np.sqrt(-2.0 * (np.log(1 - p)))
        x = r * sigma_u * np.cos(THETA) + mu_u
        y = (
                r * sigma_v * (rho * np.cos(THETA) + np.sqrt(1 - rho * rho) * np.sin(THETA))
                + mu_v
        )
        if label:
            if is_fill:
                plt.fill(x, y, color=colors[i], alpha=alpha, label=label[i], transform=data_crs, )
            else:
                plt.plot(x, y, color=colors[i], alpha=alpha,linestyle=linetype, label=label[i],transform=data_crs,)
        else:
            if is_fill:
                plt.fill(x, y, color=colors[i], alpha=alpha, label=None, transform=data_crs, )
            else:
                plt.plot(x, y, color=colors[i], alpha=alpha, linestyle=linetype, label=None, transform=data_crs, )


def compute_cdf(mu_u, mu_v, sigma_u, sigma_v, rho, u, v):
    """Compute the Mahalanobis cdf for [u, v] using a bivariate normal
    distributions.

    Arguments
    ---------
    mu_u : numpy.ndarray
        mean of u.
        shape = [n_data,]

    mu_v : numpy.ndarray
        mean of v.
        shape = [n_data,]

    sigma_u : numpy.ndarray, u > 0.
        standard deviation of u.
        shape = [n_data,]

    sigma_v : numpy.ndarray, v > 0.
        standard deviation of v.
        shape = [n_data,]

    rho : numpy.ndarray, -1 < rho < 1.
        correlation between u and v.
        shape = [n_data,]

    u : numpy.ndarray
        u values for evaluation of cdf.
        shape = [n_data,]

    v : numpy.ndarray
        v values for evaluation of cdf.
        shape = [n_data,]

    Returns
    -------
    rsqr : numpy.ndarray, dtype = float
        Mahalanobis cdf values of (u, v).
        shape = [n_data,]

    Notes
    -----
    * The equations for the Mahalanobis distance, rsqr, comes from
        the bottom of Page 2 in [3].

    * The Mahalanobis distance, rsqr, follows the chi-squared distribution
        with 2 degrees of freedom.

    * The equation for the returned cdf comes for the "Normal Distribution"
        section of [2], and the middle of Page 4 of [3].

    """
    rsqr = compute_rsqr(mu_u, mu_v, sigma_u, sigma_v, rho, u, v)
    return 1.0 - tf.math.exp(-rsqr / 2.0)


def compute_rsqr(mu_u, mu_v, sigma_u, sigma_v, rho, u, v):
    """Compute the Mahalanobis rsqr for [u, v] using a bivariate normal
    distribution.

    Arguments
    ---------
    mu_u : numpy.ndarray, dtype = float
        mean of u.
        shape = [n_data,]

    mu_v : numpy.ndarray, dtype = float
        mean of v.
        shape = [n_data,]

    sigma_u : numpy.ndarray, dtype = float, u > 0.
        standard deviation of u.
        shape = [n_data,]

    sigma_v : numpy.ndarray, dtype = float, v > 0.
        standard deviation of v.
        shape = [n_data,]

    rho : numpy.ndarray, dtype = float, -1 < rho < 1.
        correlation between u and v.
        shape = [n_data,]

    u : numpy.ndarray, dtype = float
        u values for evaluation of cdf.
        shape = [n_data,]

    v : numpy.ndarray, dtype = float
        v values for evaluation of cdf.
        shape = [n_data,]

    Returns
    -------
    rsqr : numpy.ndarray, dtype = float
        Mahalanobis rsqr values of (u, v).
        shape = [n_data,]

    Notes
    -----
    * The equations for the Mahalanobis distance, rsqr, comes from
        the bottom of Page 2 in [3].

    * The Mahalanobis distance, rsqr, follows the chi-squared distribution
        with 2 degrees of freedom.

    """
    U = (u - mu_u) / sigma_u
    V = (v - mu_v) / sigma_v
    rsqr = 1.0 / (1.0 - rho * rho) * (U * U - 2 * rho * U * V + V * V)
    return rsqr
