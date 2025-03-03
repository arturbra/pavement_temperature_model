import pandas as pd
import numpy as np
import random
import configparser
import temperature_model
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms


# =============================================================================
# 1. Load Parameters Dynamically
# =============================================================================

def load_calibration_parameters(filename="input_data/parameters.ini"):
    """ Load calibration parameters and their bounds dynamically from parameters.ini """
    config = configparser.ConfigParser()
    config.read(filename)

    params = config["calibration"]
    param_names = []
    param_bounds = {}

    for key in params.keys():
        if not key.endswith("_min") and not key.endswith("_max"):
            param_names.append(key)
            # Default values in case bounds are missing
            default_value = float(params[key])
            lower_bound = float(params.get(f"{key}_min", 0.1 * default_value))
            upper_bound = float(params.get(f"{key}_max", 4.0 * default_value))
            param_bounds[key] = (lower_bound, upper_bound)

    return param_names, param_bounds


param_names, param_bounds = load_calibration_parameters()

def update_ini_file(filename, params_dict):
    """ Update the calibration section of the parameters.ini file """
    config = configparser.ConfigParser()
    config.read(filename)

    if "calibration" not in config:
        config.add_section("calibration")

    for key, value in params_dict.items():
        config.set("calibration", key, str(value))

    with open(filename, "w") as configfile:
        config.write(configfile)



# =============================================================================
# 2. Read and Split the Data
# =============================================================================

df = pd.read_csv(r'input_data/input_data_PICP.csv')

# Split into calibration (first 40%) and validation (remaining 60%)
calib_size = int(0.4 * len(df))
calib_df = df.iloc[:calib_size].reset_index(drop=True)
val_df = df.iloc[calib_size:].reset_index(drop=True)

# =============================================================================
# 3. Define the Calibration Setup with DEAP
# =============================================================================

# We wish to maximize the Nash–Sutcliffe Efficiency (NSE)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Create attributes dynamically based on the extracted parameters
for param_name, (low, high) in param_bounds.items():
    toolbox.register(f"attr_{param_name}", random.uniform, low, high)

# Initialize an individual dynamically based on the extracted parameters
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    tuple(getattr(toolbox, f"attr_{name}") for name in param_names),
    n=1
)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the genetic operators.
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


# --- Enforce bounds dynamically ---
def checkBounds(min_vals, max_vals):
    """
    Decorator to enforce bounds on an individual after a genetic operation.
    `min_vals` and `max_vals` should be lists of the same length as the individual.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                for i in range(len(child)):
                    child[i] = max(min(child[i], max_vals[i]), min_vals[i])
            return offspring

        return wrapper

    return decorator


BOUND_LOW = [bounds[0] for bounds in param_bounds.values()]
BOUND_UP = [bounds[1] for bounds in param_bounds.values()]

toolbox.decorate("mate", checkBounds(BOUND_LOW, BOUND_UP))
toolbox.decorate("mutate", checkBounds(BOUND_LOW, BOUND_UP))


# =============================================================================
# 4. Define the Evaluation Function
# =============================================================================

def evaluate(individual):
    """
    Evaluates the GA fitness function using RMSE.
    Returns negative RMSE so that a lower error corresponds to a higher fitness.
    """
    # Map individual values to parameter names
    params_dict = dict(zip(param_names, individual))

    # Update the parameters.ini file with new calibration values
    parameters_file = "input_data/parameters.ini"
    update_ini_file(parameters_file, params_dict)

    # Run the pavement temperature model with the updated parameters
    model_results = temperature_model.model_pavement_temperature(calib_df, parameters_file)

    # Check for invalid model outputs
    if np.any(np.isnan(model_results['surface_temp'])) or np.any(np.isinf(model_results['surface_temp'])):
        return (-1e6,)  # Large negative penalty for invalid outputs

    # Compute RMSE between observed and simulated surface temperature
    # Assuming calib_df['surface_temp'] holds the observations.
    rmse = np.sqrt(np.mean((calib_df['PavementTemperature'] - model_results['surface_temp']) ** 2))

    # Return negative RMSE as fitness (lower RMSE is better)
    return (-rmse,)


toolbox.register("evaluate", evaluate)

# =============================================================================
# 5. Run the Genetic Algorithm
# =============================================================================

random.seed(42)
pop = toolbox.population(n=100)
NGEN = 100
CXPB = 0.5  # crossover probability
MUTPB = 0.2  # mutation probability

print("Starting calibration...")

result, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB,
                                  ngen=NGEN, verbose=True)

# Retrieve and print the best individual
best_ind = tools.selBest(pop, 1)[0]
best_params = dict(zip(param_names, best_ind))

print("\nBest individual:")
for name, value in best_params.items():
    print(f"{name}: {value:.5f}")

print("Best NSE:", best_ind.fitness.values[0])

# =============================================================================
# 6. Validate the Calibrated Model
# =============================================================================
params_dict = dict(zip(param_names, best_ind))
parameters_file = "input_data/parameters.ini"
update_ini_file(parameters_file, params_dict)

modeled_calibration = temperature_model.model_pavement_temperature(calib_df, parameters_file)[25:]
modeled_validation = temperature_model.model_pavement_temperature(val_df, parameters_file)[25:]
nse_val = temperature_model.NSE(val_df, modeled_validation)

print("\nValidation NSE:", nse_val)

# =============================================================================
# 7. Plot Calibration and Validation Results
# =============================================================================

# Calibration period plot
plt.figure(figsize=(10, 4))
time_calib = calib_df.index
plt.plot(time_calib[25:], calib_df['PavementTemperature'][25:], label="Observed (Calibration)")
plt.plot(time_calib[25:], modeled_calibration['surface_temp'], label="Modeled (Calibration)")
plt.xlabel("Time Step")
plt.ylabel("Pavement Temperature (°C)")
plt.title("Calibration Period")
plt.legend()
plt.tight_layout()
plt.show()

# Validation period plot
time_val = val_df.index
plt.figure(figsize=(10, 4))
plt.plot(time_val[25:], val_df['PavementTemperature'][25:], label="Observed (Validation)")
plt.plot(time_val[25:], modeled_validation['surface_temp'], label="Modeled (Validation)")
plt.xlabel("Time Step")
plt.ylabel("Pavement Temperature (°C)")
plt.title("Validation Period")
plt.legend()
plt.tight_layout()
plt.show()
