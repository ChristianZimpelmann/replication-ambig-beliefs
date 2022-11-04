"""Basic mappings that should never change throughout one model
(neither during the estimation nor in any policy experiment).

"""


class _Parameter:
    def set_parameter_location_start_values_bounds(
        self, start_values, lower_bounds, upper_bounds
    ):
        """Store the location of this parameter in the vector
        of start values internally.

        Modify the lists with *start_values*, *lower_bounds* and
        *upper_bounds* in-place.

        """

        if self.fixed:
            self._location_in_parameter_vector = None
        else:
            self._location_in_parameter_vector = len(start_values)
            start_values.append(self.start_value)
            lower_bounds.append(self.lower_bound)
            upper_bounds.append(self.upper_bound)

    def value(self, coeff_vec):
        """Return the parameter's value from *coeff_vec* if the
        parameter can vary; return the start value if the parameter
        value is fixed.

        """

        if self.fixed:
            return self.fixed_value
        else:
            return coeff_vec[self._location_in_parameter_vector]

    def store_opt_value(self, coeff_vec):
        self.opt_value = self.value(coeff_vec)


class ModelParameter(_Parameter):

    """A parameter object for parameters of the actual model.
    Typical usage:

        #. Initialise, set all meta information
        #. Construct vectors with start values and bounds
        #. Extract the current value from a vector

    """

    def __init__(self, init_dict, pref_type_name=None):

        # The name of the model parameter, used in the code
        self.name = init_dict["name"]

        # The description of the model parameter.
        self.description = init_dict["description"]

        # The (Greek) symbol of the model parameter, used in the paper.
        self.symbol = init_dict["symbol"]

        # Whether the model parameter varies by wave.
        self.heterogeneous_over_waves = init_dict["heterogeneous_over_waves"]

        # Whether the model parameter is fixed at its starting value. If so, the
        # attributes *lower_bound* and *upper_bound* will be ignored.
        self.fixed = init_dict["fixed"]

        # The lower bound for the parameter. Set to -inf if not required.
        self.lower_bound = init_dict["lower_bound"]

        # The upper bound for the parameter. Set to *None* if not required.
        self.upper_bound = init_dict["upper_bound"]

        # The start value for the parameter.
        self.start_value = init_dict["start_value"]
        self.fixed_value = self.start_value
