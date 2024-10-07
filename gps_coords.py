from pyproj import Proj, Transformer
import pandas as pd

wgs84 = Proj(proj='latlong', datum='WGS84')
utm = Proj(proj='utm', zone=17, datum='WGS84')
transformer = Transformer.from_proj(wgs84, utm)


def latlon_to_utm(lat, lon):
    easting, northing = transformer.transform(lon, lat)

    return easting, northing


def convert_to_utm(row):
    easting, northing = latlon_to_utm(row['Latitude (degrees)'], row['Longitude (degrees)'])

    return pd.Series([easting, northing], index=['easting', 'northing'])


df = pd.read_csv('State.csv')

df[['easting', 'northing']] = df.apply(convert_to_utm, axis=1)

result_df = df[['Unix Time', 'Microseconds', 'easting', 'northing', 'Heading (degrees)']]

result_df.to_csv('Cleaned_state.csv')
