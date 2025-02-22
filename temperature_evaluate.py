import temperature_model
import pandas as pd

input_file = r"C:\Users\Artur\PycharmProjects\pavement_temperature_model\input_data\permeable_asphalt.csv"
sim_df = pd.read_csv(input_file)
params = [0.23, 0.31, 0.1]
surface_temp = temperature_model.model_pavement_temperature(sim_df, params)