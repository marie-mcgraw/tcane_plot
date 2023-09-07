import pandas as pd

def get_storm_details(df, isample):
    storm = df.iloc[isample]
    storm_name = storm["Name"]
    storm_ftime = int(storm["ftime(hr)"])
    # storm_month = str(int(storm["time"]))[:-4]
    storm_month = pd.to_datetime(storm['DATE']).month
    # storm_day = str(int(storm["time"]))[-4:-2]
    storm_day = pd.to_datetime(storm['DATE']).day
    # storm_hour = str(int(storm["time"]))[-2:]
    storm_hour = pd.to_datetime(storm['DATE']).hour
    # storm_year = int(storm["year"])
    storm_year = pd.to_datetime(storm['DATE']).year

    details = (
        storm_name
        + " "
        + str(storm_year)
        + "-"
        + str(storm_month)
        + "-"
        + str(storm_day)
        + " "
        + str(storm_hour)
        + "00 @"
        + str(storm_ftime)
        + "hr"
    )

    return details
