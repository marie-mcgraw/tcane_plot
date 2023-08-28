"""Collate and save the model predictions."""

import numpy as np
import scipy
import scipy.interpolate
import pandas as pd

__author__ = "Elizabeth A. Barnes and Randal J Barnes"
__version__ = "13 December 2022"


def save_predictions(model, settings, predictions_filename, df_data, x_data, label_data):
    """Make the predictions and compute the associated prediction metrics."""
    y_pred = model.predict(x_data)

    df_predictions = df_data.copy()
    df_predictions["mu_u"] = y_pred[:, 0]
    df_predictions["mu_v"] = y_pred[:, 1]
    df_predictions["sigma_u"] = y_pred[:, 2]
    df_predictions["sigma_v"] = y_pred[:, 3]
    df_predictions["rho"] = y_pred[:, 4]

    df_predictions["euclidian_error"] = np.hypot(
        y_pred[:, 0] - label_data[:, 0],
        y_pred[:, 1] - label_data[:, 1],
    )

    df_predictions.to_csv(predictions_filename)

    return None


def interpolate_leadtimes(leadtimes, y, x_interp=None):
    if x_interp is None:
        x_interp = np.arange(leadtimes[0], leadtimes[-1]+1)

    f_interp = scipy.interpolate.interp1d(leadtimes, y , kind="cubic")
    y_interp = f_interp(x_interp)
    return y_interp


def add_lead_zero(df_storm):

    row = df_storm[df_storm["ftime(hr)"] == df_storm["ftime(hr)"].min()].copy()

    row["ftime(hr)"] = 0.
    row["mu_u"] = 0.
    row["mu_v"] = 0.
    row["sigma_u"] = 15.
    row["sigma_v"] = 15.
    row["OFDX"] = 0.
    row["OFDY"] = 0.
    row["LATN"] = row["LAT0"]
    row["LONN"] = row["LON0"]

    return pd.concat([row,df_storm], ignore_index=True)

#%%
