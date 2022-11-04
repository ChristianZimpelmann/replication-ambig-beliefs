import numpy as np
import pandas as pd


def get_model_parameters(model_parameters_init):
    # Create list of ModelParameter instances
    model_parameters = []
    for mp_init in model_parameters_init:
        if mp_init["varies_with_pref_type"]:
            for name in sorted(mp_init["start_value"].keys()):
                model_parameters.append(ModelParameter(mp_init, name))
        else:
            model_parameters.append(ModelParameter(mp_init))

    return model_parameters


def get_start_values_bounds(model_parameters):
    # Create list of (transformed) parameters that are optimised over
    start_values = []
    lower_bounds = []
    upper_bounds = []
    for p in model_parameters.values():
        if type(p) == dict:
            for p_by_wave in p.values():
                p_by_wave.set_parameter_location_start_values_bounds(
                    start_values, lower_bounds, upper_bounds
                )

        else:
            p.set_parameter_location_start_values_bounds(
                start_values, lower_bounds, upper_bounds
            )

    start_values = np.array(start_values, dtype=np.double)
    lower_bounds = np.array(
        [lb if lb is not None else -np.inf for lb in lower_bounds], dtype=np.double
    )
    upper_bounds = np.array(
        [ub if ub is not None else np.inf for ub in upper_bounds], dtype=np.double
    )

    return start_values, lower_bounds, upper_bounds
