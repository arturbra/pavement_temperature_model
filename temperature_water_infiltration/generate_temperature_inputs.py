import pandas as pd

def generate_input_file(pavement, meteo_data, surface_temperature, rainfall, water_temperature, bottom_temperature):
    surface_temperature = surface_temperature[['date', f'{pavement}_surface_temperature']]
    merged_df = meteo_data.merge(surface_temperature, on='date', how='inner')
    merged_df = merged_df.merge(rainfall, on='date', how='left')

    if pavement != "CP":
        water_temperature = water_temperature[['date', f'{pavement}_water_temperature']]
        water_temperature.columns = ['date', 'WaterTemperature']
        bottom_temperature = bottom_temperature[['date', f'{pavement}_bottom_temperature']]
        bottom_temperature.columns = ['date', 'BottomTemperature']
        merged_df = merged_df.merge(bottom_temperature, on='date', how='left')
        merged_df = merged_df.merge(water_temperature, on='date', how='left')

    merged_df = merged_df.rename(columns={f'{pavement}_surface_temperature': 'PavementTemperature'})

    return merged_df

surface_temperature = pd.read_csv(r'input_data\surface_temperature.csv', parse_dates=[0])
rainfall = pd.read_csv(r'input_data\rainfall.csv', parse_dates=[0])
water_temperature = pd.read_csv(r'input_data\water_temperature.csv', parse_dates=[0])
meteo_data = pd.read_csv(r'input_data\meteo_data.csv', parse_dates=[0])
bottom_temperature = pd.read_csv(r'input_data\bottom_temperature.csv', parse_dates=[0])

# Define pavement types and date range
pavements = ['CP', 'PICP', 'PGr', 'PA', 'PC']


# Run the input generation for each pavement type
for pavement in pavements:
    print(f"Processing {pavement}...")

    # Generate input file for this pavement
    input_df = generate_input_file(
        pavement,
        meteo_data,
        surface_temperature,
        rainfall,
        water_temperature,
        bottom_temperature
    )

    input_df.to_csv(rf"C:\Users\Artur\PycharmProjects\pavement_temperature_model\temperature_water_infiltration\input_data\{pavement}\input_data_{pavement}.csv", index=False)