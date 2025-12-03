import pandas as pd
import temperature_model
import temperature_evaluate

# Run climate change scenario
# pavements = ["PA", "PICP", "PC", "PGr", "CP"]
pavements = ["PGr"]
# PA: Permeable Asphalt
# PICP: Permeable Interlocking Concrete Paver
# PC: Permeable Concrete
# PGr: Permeable Grid filled with gravel
# CP: Conventional Pavement
scenarios = ["ssp126", "ssp585", "historical"]

for pavement in pavements:
    for scenario in scenarios:
        input_df = pd.read_csv(rf"C:\Users\Artur\PycharmProjects\pavement_temperature_model\climate_change\UKESM1-0-LL\climate_change_{scenario}.csv")
        parameters_file = rf"C:\Users\Artur\PycharmProjects\pavement_temperature_model\input_data\1h\parameters_{pavement}.ini"
        climate_change = True

        model_results = temperature_model.model_pavement_temperature(input_df, parameters_file, climate_change)

        model_results.to_csv(rf"C:\Users\Artur\PycharmProjects\pavement_temperature_model\climate_change\output\{pavement}_scenario{scenario}.csv", index=False)