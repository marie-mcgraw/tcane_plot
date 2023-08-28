def get_storm_details(df, isample):
    storm = df.iloc[isample]
    storm_name = storm["Name"]
    storm_ftime = int(storm["ftime(hr)"])
    storm_month = str(int(storm["time"]))[:-4]
    storm_day = str(int(storm["time"]))[-4:-2]
    storm_hour = str(int(storm["time"]))[-2:]
    storm_year = int(storm["year"])

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
