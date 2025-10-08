import cdsapi

dataset = "derived-near-surface-meteorological-variables"
request = {
    "variable": [
        "grid_point_altitude",
        "near_surface_wind_speed",
        "near_surface_air_temperature",
        "near_surface_specific_humidity",
        "surface_downwelling_shortwave_radiation",
        "surface_downwelling_longwave_radiation",
        "rainfall_flux"
    ],
    "reference_dataset": "cru",
    "year": [
        "1998", "1999", "2000",
        "2001", "2002", "2003",
        "2004", "2005", "2006",
        "2007", "2008"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "version": ["2_1"]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
