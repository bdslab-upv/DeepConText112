"""
DESCRIPTION: functions to prepare data for covariate shift exploration.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
from pandas import DataFrame


# ONE-HOT ENCODING YEARS
def one_hot_encode_years(data: DataFrame) -> DataFrame:
    # Years
    years = (2009, 2010, 2011, 2012, 2014, 2015, 2016, 2017, 2018, 2019)

    # Year series extraction
    year_series = data['YEAR']

    # One-hot encoding of each year
    for year in years:
        data['YEAR_' + str(year)] = year_series.apply(lambda year_i: 1 if year_i == year else 0)

    # Dropping of year column
    data = data.drop(columns=['YEAR'])

    # Output
    return data
