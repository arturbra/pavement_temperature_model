import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import temperature_model

from deap import base, creator, tools, algorithms

# =============================================================================
# 1. Read and Split the Data
# =============================================================================

df = pd.read_csv(r'input_data/input_data_PA.csv')

# Split into calibration (first 40%) and validation (remaining 60%)
calib_size = int(0.4 * len(df))
calib_df = df.iloc[:calib_size].reset_index(drop=True)
val_df = df.iloc[calib_size:].reset_index(drop=True)


# =============================================================================
# 2. Define the Calibration Setup with DEAP (3 parameters)
# =============================================================================

# We wish to maximize the Nash–Sutcliffe Efficiency (NSE)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# -- Calibrated dynamic parameters --
toolbox.register("attr_reflectivity", random.uniform, 0.1, 1.0)
toolbox.register("attr_emissivity", random.uniform, 0.1, 1.0)

# -- Material parameter for Layer 1 (f_lam_layer1) --
toolbox.register("attr_f_lam_layer1", random.uniform, 0.1, 4.0)

# The individual is composed of 4 parameters in order:
# [reflectivity, emissivity, ER, f_lam_layer1]
toolbox.register("individual", tools.initCycle, creator.Individual, (
    toolbox.attr_reflectivity,
    toolbox.attr_emissivity,
    toolbox.attr_f_lam_layer1,
), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the genetic operators.
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


# --- Enforce bounds after genetic operations ---
def checkBounds(min_vals, max_vals):
    """
    Decorator to enforce bounds on an individual after a genetic operation.
    `min_vals` and `max_vals` should be lists of the same length as the individual.
    """

    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] < min_vals[i]:
                        child[i] = min_vals[i]
                    elif child[i] > max_vals[i]:
                        child[i] = max_vals[i]
            return offspring

        return wrapper

    return decorator


BOUND_LOW = [
    0.1,  # reflectivity lower bound
    0.1,  # emissivity lower bound
    0.1  # f_lam_layer1 lower bound
]

BOUND_UP = [
    1.0,  # reflectivity upper bound
    1.0,  # emissivity upper bound
    4.0  # f_lam_layer1 upper bound
]

toolbox.decorate("mate", checkBounds(BOUND_LOW, BOUND_UP))
toolbox.decorate("mutate", checkBounds(BOUND_LOW, BOUND_UP))


def evaluate(individual):
    params = individual
    model_results = temperature_model.model_pavement_temperature(calib_df, params)

    if np.any(np.isnan(model_results['surface_temp'])) or np.any(np.isinf(model_results['surface_temp'])):
        return (-1e6,)

    nse = temperature_model.NSE(calib_df, model_results)

    return (nse,)


toolbox.register("evaluate", evaluate)

# =============================================================================
# 3. Run the Evolutionary Algorithm
# =============================================================================

random.seed(41)
pop = toolbox.population(n=50)
NGEN = 5
CXPB = 0.5  # crossover probability
MUTPB = 0.2  # mutation probability

print("Starting calibration...")

result, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB,
                                  ngen=NGEN, verbose=True)

# Retrieve and print the best individual
best_ind = tools.selBest(pop, 1)[0]
print("\nBest individual:")
print("Parameters (in order):")
print("reflectivity, emissivity, f_lam_layer1")
print(best_ind)
print("Best NSE:", best_ind.fitness.values[0])


# =============================================================================
# 4. Validate the Calibrated Model
# =============================================================================

modeled_calibration = temperature_model.model_pavement_temperature(calib_df, best_ind)[25:]
modeled_validation = temperature_model.model_pavement_temperature(val_df, best_ind)[25:]
nse_val = temperature_model.NSE(val_df, modeled_validation)

print("\nValidation NSE:", nse_val)

# =============================================================================
# 5. Plot Calibration and Validation Results
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
