import json
import logging
import random
from shutil import copyfile

import mystic
import numpy as np
import pandas as pd
import pytask
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.optimize import Bounds
from scipy.optimize import minimize

from ambig_beliefs.analysis.utils_analysis import calculate_matching_probabilites
from ambig_beliefs.model_code.agent import Agent
from ambig_beliefs.model_code.load_data import get_start_values_bounds
from ambig_beliefs.model_code.parameter import ModelParameter
from config import ESTIMATE
from config import IN_MODEL_CODE
from config import IN_MODEL_SPECS
from config import MODEL_NAMES_ESTIMATION
from config import MODES
from config import OPT_SETTINGS_NAMES
from config import OUT_ANALYSIS
from config import OUT_DATA
from config import OUT_DATA_LISS
from config import REDUCED_N_AGENTS
from config import ROOT

# from estimagic.differentiation.derivatives import first_derivative
# from estimagic.inference.likelihood_covs import cov_sandwich
# from estimagic.inference.likelihood_covs import se_from_cov

# from multiprocessing import Pool
# from run_mgmt.management_tries import update_model_spec, store_estimation_run_ssc

# from estimagic.differentiation.derivatives import first_derivative
# from estimagic.inference.likelihood_covs import cov_sandwich
# from estimagic.inference.likelihood_covs import se_from_cov

# from multiprocessing import Pool
# from run_mgmt.management_tries import update_model_spec, store_estimation_run_ssc


class ModelWithLikelihood:
    def __init__(self, model_name, opt_settings, reduced_n_agents, use_sim_data):
        self.model_name = model_name
        self.opt_settings_name = opt_settings
        self._set_parameters()
        self.use_sim_data = use_sim_data
        self.reduced_n_agents = reduced_n_agents
        self.load_choices()
        self.agents = self.get_agents()

    def _set_parameters(self):
        """
        Load model characteristics and parameters from the specification file.
        """
        with open(
            IN_MODEL_SPECS / (self.model_name + ".json"), encoding="utf-8"
        ) as file:
            m = json.load(file)
        with open(
            IN_MODEL_SPECS / (self.opt_settings_name + ".json"), encoding="utf-8"
        ) as file:
            self.opt_settings = json.load(file)

        # Load model characteristics
        self.least_squares_estimation = m["model_general"]["least_squares_estimation"]
        self.nested_estimation = m["model_general"]["nested_estimation"]

        self.pooling = m["model_general"]["pooling"]
        if self.pooling == "k_means":
            self.pooling_spec = m["model_general"]["pooling_spec"]
        self.use_event_0 = m["model_general"]["use_event_0"]
        self.error_event_level = m["model_general"]["error_event_level"]
        self.unrestricted_above_sigma = m["model_general"]["unrestricted_above_sigma"]
        self.het_subj_probs = m["model_general"]["het_subj_probs"]
        self.used_waves = m["sample"]["wave"]
        self.model_specs = m
        if "n_selected_waves" in m["sample"]:
            self.n_selected_waves = m["sample"]["n_selected_waves"]
        else:
            self.n_selected_waves = None

        if "restriction_ind_waves" in m["sample"]:
            self.restriction_ind_waves = m["sample"]["restriction_ind_waves"]
        else:
            self.restriction_ind_waves = "~(quickest_15perc & has_rec_pattern)"
        # Load model parameters
        self.model_parameters, self.parameter_names = self.get_model_parameters(
            m["model_parameters"]
        )
        (
            self.start_values,
            self.lower_bounds,
            self.upper_bounds,
        ) = get_start_values_bounds(self.model_parameters)

    def get_model_parameters(self, model_parameters_init):
        """
        Create list of ModelParameter
        """
        model_parameters = {}
        parameter_names = []
        for mp_init in model_parameters_init:
            if mp_init["heterogeneous_over_waves"]:
                het_parameters = {}
                for wave in self.used_waves:
                    adjusted_mp_init = mp_init.copy()
                    adjusted_mp_init["name"] += "_w" + str(wave)
                    het_parameters[wave] = ModelParameter(adjusted_mp_init)
                    parameter_names.append(ModelParameter(adjusted_mp_init).name)
                model_parameters[mp_init["name"]] = het_parameters
            else:
                model_parameters[mp_init["name"]] = ModelParameter(mp_init)
                parameter_names.append(ModelParameter(mp_init).name)
        return model_parameters, parameter_names

    def load_choices(self):
        """
        Load choices, matching probabilities etc.
        """
        # Load selection of valid waves
        valid_waves = pd.read_pickle(OUT_DATA / "pat_rec_and_dur_restrictions.pickle")

        # Load choice properties.
        self.choice_properties = pd.read_pickle(
            OUT_DATA / "choice_prop_prepared.pickle"
        ).xs(self.used_waves[0], level="wave")

        # Load choice data and matching probs from simulated data or true data
        if self.use_sim_data:
            self.choices = pd.read_pickle(
                OUT_ANALYSIS
                / self.model_name
                / self.opt_settings_name
                / "sim_choices.pickle"
            )
            self.choices["choice"] = self.choices["sim_choice"]
            self.matching_probs = pd.read_pickle(
                OUT_ANALYSIS
                / self.model_name
                / self.opt_settings_name
                / "sim_matching_probs.pickle"
            )
        else:
            self.choices = pd.read_pickle(OUT_DATA / "choices_prepared.pickle")

            self.matching_probs = pd.read_pickle(
                OUT_DATA_LISS / "ambiguous_beliefs" / "baseline_matching_probs.pickle"
            )

        # Load event properties and join to matching_probs
        event_properties = pd.read_pickle(
            OUT_DATA_LISS / "ambiguous_beliefs" / "event_properties.pickle"
        )

        self.matching_probs = self.matching_probs.join(
            event_properties.swaplevel().sort_index()[
                ["aex_1", "aex_pi_0", "aex_pi_1", "aex_pi_2"]
            ],
            on=["wave", "aex_event"],
        )

        # Select columns needed for estimation
        self.matching_probs = self.matching_probs[
            [
                "baseline_matching_prob_midp",
                "baseline_matching_prob_interval",
                "aex_1",
                "aex_pi_0",
                "aex_pi_1",
                "aex_pi_2",
            ]
        ]
        self.matching_probs["baseline_matching_prob_midp"] /= 100
        self.matching_probs["baseline_matching_prob_interval"] = self.matching_probs[
            "baseline_matching_prob_interval"
        ].apply(lambda x: x / 100)

        if not self.use_sim_data:
            # Select valid waves
            self.choices = self.choices.reset_index().set_index(["personal_id", "wave"])
            self.choices = self.choices.loc[
                self.choices.index.intersection(
                    valid_waves.query(self.restriction_ind_waves).index
                ),
                :,
            ]
            self.choices = self.choices.reset_index().set_index(
                ["personal_id", "wave", "choice_num"]
            )
            self.matching_probs = self.matching_probs.reset_index().set_index(
                ["personal_id", "wave"]
            )
            self.matching_probs = self.matching_probs.loc[
                self.matching_probs.index.intersection(
                    valid_waves.query(self.restriction_ind_waves).index
                ),
                :,
            ]

            # Select complete waves
            # (for choice probs this is already ensured in data_management)
            # ToDo: do this in data management
            n_events_per_wave = 7 if self.use_event_0 else 6
            self.matching_probs = self.matching_probs.loc[
                self.matching_probs.groupby(["personal_id", "wave"])[
                    "baseline_matching_prob_interval"
                ].count()
                == n_events_per_wave
            ]

            self.matching_probs = self.matching_probs.reset_index().set_index(
                ["personal_id", "wave", "aex_event"]
            )
            ix = pd.IndexSlice
            self.matching_probs = self.matching_probs.loc[ix[:, self.used_waves, :], :]
            self.choices = self.choices.loc[ix[:, self.used_waves, :], :]

        self.matching_probs = self.matching_probs.reset_index().set_index("personal_id")
        self.choices = self.choices.reset_index().set_index("personal_id").sort_index()

        # Find out if subjective probabilities are heterogeneous
        if self.het_subj_probs:
            paras_het_over_waves = [
                p
                for p in self.model_parameters
                if type(self.model_parameters[p]) == dict
            ]

            # Make sure that no variables other than subjective probabilities are heterogeneous
            for p in paras_het_over_waves:
                if "pi" not in p:
                    raise NotImplementedError(
                        "Only subjective probabilities are implemented to be",
                        "able to vary over waves.",
                        f"{p} is not a subjective probability.",
                    )

        # Remove event 0 if it should not be used
        if not self.use_event_0:
            self.choices = self.choices.loc[self.choices["aex_event"] != "0"]
            self.matching_probs = self.matching_probs.loc[
                self.matching_probs["aex_event"] != "0"
            ]

    def get_agents(self):
        """
        Initialize one agent for each observed participant.
        """

        # Initialize all agents
        agents = {}
        agents_to_opt = self.choices.index.get_level_values(level=0).unique()
        if self.reduced_n_agents:
            agents_to_opt = agents_to_opt[:10]
        for index in agents_to_opt:
            if self.use_sim_data:
                # ToDo: Calculate matching probs during simulation and load it from there
                matching_probs_ind = self.matching_probs.loc[index]
            else:
                matching_probs_ind = self.matching_probs.loc[index]

            used_waves_agent = [
                i
                for i in self.choices.loc[index]["wave"].unique()
                if i in self.used_waves
            ]

            # Reduce used waves if only a subset should be used
            if self.n_selected_waves and self.n_selected_waves < len(used_waves_agent):
                used_waves_agent = sorted(
                    random.sample(used_waves_agent, self.n_selected_waves)
                )

            agents[index] = Agent(
                choices=self.choices.loc[index],
                choice_properties=self.choice_properties,
                matching_probs=matching_probs_ind,
                error_event_level=self.error_event_level,
                personal_id=index,
                het_subj_probs=self.het_subj_probs,
                used_waves=used_waves_agent,
            )

        return agents

    def _unpack_para_vec(self, opt_para, model_parameters):
        """Split up the vector of input parameters.

        :param opt_para: parameter vector of all parameters which are optimized over

        """
        paras = {}
        for mp_name, mp in model_parameters.items():
            if type(mp) == dict:
                para_list = [mp_by_wave.value(opt_para) for mp_by_wave in mp.values()]
                paras[mp_name] = para_list
            else:
                paras[mp_name] = mp.value(opt_para)
        return paras

    def _unpack_para_vec_flattened(self, opt_para, model_parameters):
        """Split up the vector of input parameters. Return flat dictionary

        :param opt_para: parameter vector of all parameters which are optimized over

        """
        paras = {}
        for mp_name, mp in model_parameters.items():
            if type(mp) == dict:
                for mp_by_wave in mp.values():
                    paras[mp_by_wave.name] = mp_by_wave.value(opt_para)
            else:
                paras[mp_name] = mp.value(opt_para)
        return paras

    def get_individual_model_parameters(self, used_waves):
        """
        Create list of model parameters and names for one agent

        :param used_waves: a list of waves the agent participated
        """

        model_parameters = {}
        parameter_names = []
        for para_name, para in self.model_parameters.items():
            if isinstance(para, dict):
                model_parameters[para_name] = {i: para[i] for i in used_waves}
                parameter_names += [para_name + "_w" + str(i) for i in used_waves]
            else:
                model_parameters[para_name] = para
                parameter_names.append(para_name)
        return model_parameters, parameter_names

    def neg_log_likelihood(self, x, agents, model_parameters):
        """
        Returns the negative log likelihood for a dict of agents with the same parameter values.
        """
        paras = self._unpack_para_vec(x, model_parameters)
        likeli = 0
        for _agent_id, agent in agents.items():
            likeli += agent.eval_likelihood(paras)
        return -likeli

    def shrink_bounds(self, bounds):
        border = self.opt_settings["options"]["shrink_border"]
        bounds[bounds == np.inf] = border
        bounds[bounds == -np.inf] = -border
        return bounds

    def build_constraints(self, ndim, used_waves):
        """
        Build mystic constraints.
        """

        # Build list of constraints
        restrictions = []

        # Set number of subjective probability parameters (== n_waves for het_prob)
        n_subj_prob = len(used_waves) if self.het_subj_probs else 1
        for i in range(n_subj_prob):

            # p_0 > p_1 in each wave
            restrictions.append("x" + str(i) + " >= x" + str(n_subj_prob + i) + " + 0")

            # p_0 + p_2 <= 1 in each wave
            restrictions.append(
                "x" + str(2 * n_subj_prob + i) + " <= - x" + str(i) + " + 1"
            )

        # tau + sigma >= 0
        if self.unrestricted_above_sigma:
            restrictions.append(
                "x"
                + str(3 * n_subj_prob)
                + " >= - x"
                + str(3 * n_subj_prob + 1)
                + " + 0"
            )

        # tau + sigma <= 1
        else:
            restrictions.append(
                "x"
                + str(3 * n_subj_prob)
                + " <= - x"
                + str(3 * n_subj_prob + 1)
                + " + 1"
            )

        # Reformat constaints and build mystic object
        restrictions_conc = ".\n".join(restrictions) + "."
        # print(restrictions_conc)
        solv = mystic.symbolic.generate_solvers(restrictions_conc, nvars=ndim)
        return mystic.symbolic.generate_constraint(solv)

    def _max_ind(
        self, agents, model_parameters, ndim, constraints, lower_bounds, upper_bounds
    ):
        """
        Maximize likelihood for a group of agents with the same parameter values.
        """
        # mystic.tools.random_seed(999)
        # print(model_parameters.keys())

        # Select solvers
        if self.opt_settings["method"] == "lattice":

            # Set solver
            solver = mystic.solvers.LatticeSolver(
                ndim, nbins=self.opt_settings["options"]["nbins"]
            )
            solver.SetNestedSolver(mystic.solvers.PowellDirectionalSolver)

            # Set constraints
            solver.SetStrictRanges(lower_bounds, upper_bounds)
            solver.SetConstraints(constraints)
            # Optimize
            solver.Solve(
                self.neg_log_likelihood,
                ExtraArgs=[agents, model_parameters],
                disp=False,
            )

        elif self.opt_settings["method"] == "buckshot":

            # Set solver
            solver = mystic.solvers.BuckshotSolver(
                ndim, npts=self.opt_settings["options"]["npts"]
            )
            solver.SetNestedSolver(mystic.solvers.PowellDirectionalSolver)

            # Set constraints
            solver.SetStrictRanges(lower_bounds, upper_bounds)
            solver.SetConstraints(constraints)

            # Optimize
            solver.Solve(
                self.neg_log_likelihood,
                ExtraArgs=[agents, model_parameters],
                disp=False,
            )

        elif self.opt_settings["method"] == "diff_evolution":

            # Set solver
            solver = mystic.solvers.DifferentialEvolutionSolver(
                ndim, NP=self.opt_settings["options"]["NP"]
            )

            # Set constraints
            solver.SetStrictRanges(lower_bounds, upper_bounds)
            solver.SetRandomInitialPoints(lower_bounds, upper_bounds)
            solver.SetConstraints(constraints)

            # Set Termination
            if (
                self.opt_settings["options"]["termination_name"]
                == "VTRChangeOverGeneration"
            ):
                termination_options = self.opt_settings["options"][
                    "termination_options"
                ]
                solver.SetTermination(
                    mystic.termination.VTRChangeOverGeneration(
                        ftol=termination_options["ftol"],
                        gtol=termination_options["gtol"],
                        generations=termination_options["generations"],
                        target=termination_options["target"],
                    )
                )

            # Optimize
            solver.Solve(
                self.neg_log_likelihood,
                ExtraArgs=[agents, model_parameters],
                disp=False,
                strategy=mystic.strategy.Best1Exp,
                CrossProbability=self.opt_settings["options"]["CrossProbability"],
                ScalingFactor=self.opt_settings["options"]["ScalingFactor"],
            )
        else:
            raise ValueError("optimizer not implemented")
        return solver

    def estimate(self):
        """
        Estimate parameter values.
        """

        if self.nested_estimation:
            self.run_nested_estimation()
        else:
            if self.least_squares_estimation:
                results = self.estimate_least_squares()
            else:
                results = self.ml_estimation()
            # Store results
            self.store_results(results)

    def ml_estimation(self):
        """
        Run maximum likelihood estimation.
        """

        def save_opt_paras_one_agent(
            agent_id, agent, opt_solver, opt_paras, results, model_parameters
        ):
            res = {
                "fun": self.neg_log_likelihood(
                    opt_solver.bestSolution,
                    agents={agent_id: agent},
                    model_parameters=model_parameters,
                ),
                "terminated": opt_solver.Terminated(),
                "used_waves": agent.used_waves,
            }

            # Write opt_paras for this agent
            results.loc[agent_id] = pd.Series({**opt_paras, **res})

            return results

        # Initialize results DataFrame
        results = pd.DataFrame(
            columns=self.parameter_names + ["fun", "terminated", "used_waves"]
        )

        # Select which groups should be pooled.
        if self.pooling == "individual":

            # Run Optimization for each individual
            for agent_id, agent in self.agents.items():
                # Generate parameters for this agent
                (
                    ind_model_parameters,
                    ind_para_names,
                ) = self.get_individual_model_parameters(agent.used_waves)
                sv, lower_bounds, upper_bounds = get_start_values_bounds(
                    ind_model_parameters
                )
                lower_bounds = self.shrink_bounds(lower_bounds)
                upper_bounds = self.shrink_bounds(upper_bounds)
                ndim = len(lower_bounds)
                constraints = self.build_constraints(ndim, agent.used_waves)

                # Run individual estimation
                opt_solver = self._max_ind(
                    agents={agent_id: agent},
                    model_parameters=ind_model_parameters,
                    ndim=ndim,
                    constraints=constraints,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                )

                opt_paras = self._unpack_para_vec_flattened(
                    opt_solver.bestSolution, ind_model_parameters
                )

                # Build dict of results that should be saved with the optimal parameters.
                results = save_opt_paras_one_agent(
                    agent_id,
                    agent,
                    opt_solver,
                    opt_paras,
                    results,
                    ind_model_parameters,
                )
        else:
            raise NotImplementedError
        return results

    def run_nested_estimation(self):
        """
        Run several estimations varying one parameter that is then fixed for a particular run.
        """
        # Make sure needed parameters are specified
        if "nested_parameter" not in self.model_specs["model_general"]:
            raise ValueError("Specify nested_parameter for nested estimation.")
        if "nested_fixed_values" not in self.model_specs["model_general"]:
            raise ValueError("Specify nested_fixed_values for nested estimation.")
        if "nested_n_cores" not in self.model_specs["model_general"]:
            raise ValueError("Specify nested_n_cores for nested estimation.")
        # Check other requirements
        if self.model_specs["model_general"]["nested_parameter"] != "theta":
            raise NotImplementedError(
                "Nested estimation currently only implemented for theta."
            )
        assert self.model_parameters[
            "theta"
        ].fixed, "Parameter used for nested estimation needs to be fixed."

        if self.least_squares_estimation:
            raise NotImplementedError(
                "Nested Estimation for least-squares not implemented, yet."
            )
        else:
            fixed_values = self.model_specs["model_general"]["nested_fixed_values"]
            results_for_values = []

            def run_one_estimation(value):
                self.model_parameters["theta"].fixed_value = value
                results = self.ml_estimation()

                # Store intermediate results
                self.store_results(results, nested_estimation_value=value)
                return results

            # Loop over values and run model for these values
            n_cores = self.model_specs["model_general"]["nested_n_cores"]
            pool = Pool(processes=n_cores)
            results_for_values = pool.map(run_one_estimation, fixed_values)

            # Select best run and save it
            mean_log_likelis = pd.Series(
                [r["fun"].mean() for r in results_for_values], index=fixed_values
            )
            best_run = np.nanargmin(mean_log_likelis)

            # Use former results if available
            try:
                former_likelis = pd.from_csv(
                    OUT_ANALYSIS
                    / self.model_name
                    / self.opt_settings_name
                    / ("mean_log_likelis" + ".csv"),
                    sep=";",
                    header=True,
                )
                mean_log_likelis = pd.concat([former_likelis, mean_log_likelis])
            except Exception:
                pass

            mean_log_likelis.to_csv(
                OUT_ANALYSIS
                / self.model_name
                / self.opt_settings_name
                / ("mean_log_likelis" + ".csv"),
                sep=";",
                header=True,
            )
            self.store_results(results_for_values[best_run])

    def build_scipy_constraints(self, ndim, used_waves):
        """
        Build scipy constraints.
        """

        def const_p0_p1(i):
            jac = np.zeros(ndim)
            jac[i] = 1
            jac[n_subj_prob + i] = -1

            return {
                "type": "ineq",
                "fun": lambda x: np.array([x[i] - x[n_subj_prob + i]]),
                "jac": lambda x: jac,
            }

        def const_p0_p2(i):
            jac = np.zeros(ndim)
            jac[i] = -1
            jac[2 * n_subj_prob + i] = -1

            return {
                "type": "ineq",
                "fun": lambda x: np.array([1 - x[i] - x[2 * n_subj_prob + i]]),
                "jac": lambda x: jac,
            }

        def const_tau_sigma(i):
            jac = np.zeros(ndim)
            jac[3 * n_subj_prob] = -1
            jac[3 * n_subj_prob + 1] = -1

            return {
                "type": "ineq",
                "fun": lambda x: np.array(
                    [1 - x[3 * n_subj_prob] - x[3 * n_subj_prob + 1]]
                ),
                "jac": lambda x: jac,
            }

        # Build list of constraints
        restrictions = []

        # Set number of subjective probability parameters (== n_waves for het_prob)
        if self.het_subj_probs:
            n_subj_prob = len(used_waves)
        else:
            n_subj_prob = 1

        for i in range(n_subj_prob):

            # p_0 > p_1 in each wave
            restrictions.append(const_p0_p1(i))

            # p_0 + p_2 <= 1 in each wave
            restrictions.append(const_p0_p2(i))

        if not self.unrestricted_above_sigma:

            # tau + sigma >= 0
            restrictions.append(const_tau_sigma(i))
        return restrictions

    def estimate_least_squares(self):
        """
        Estimate least-squares parameter estimates using the midpoints
        of the matching probability intervals.
        """

        def least_squares_objective(x, agent, model_parameters):
            paras = self._unpack_para_vec(x, model_parameters)
            return agent.least_squares_objective(paras)

        # Initialize results DataFrame
        results = pd.DataFrame(columns=self.parameter_names + ["sq_dev", "std_error"])

        for agent_id, agent in self.agents.items():

            # Calculate individual model parameters
            ind_model_parameters, ind_para_names = self.get_individual_model_parameters(
                agent.used_waves
            )

            # Calculate individual bounds and constraints
            sv, lower_bounds, upper_bounds = get_start_values_bounds(
                ind_model_parameters
            )
            n_dim = len(lower_bounds)
            constraints = self.build_scipy_constraints(n_dim, agent.used_waves)

            # Minimize
            solution = minimize(
                least_squares_objective,
                sv,
                args=(agent, ind_model_parameters),
                method="SLSQP",
                constraints=constraints,
                bounds=Bounds(lower_bounds, upper_bounds),
                options={"maxiter": 1e4},
            )

            res = {
                "sq_dev": solution.fun,
                "std_error": np.sqrt(max(0, solution.fun) / agent.n_error_draws),
            }
            opt_paras = self._unpack_para_vec_flattened(
                solution.x, ind_model_parameters
            )
            results.loc[agent_id] = pd.Series({**opt_paras, **res})

        return results

    def simulate(self, n_agents, paras):
        """
        Simulate all choices for given starting_values
        """
        sim_choices = {}

        # Select one (the first) agent
        agent = self.agents[list(self.agents.keys())[0]]
        agent.used_waves = self.used_waves

        # Simulate the choices n_agents times
        for i in range(n_agents):
            sim_choices[i] = agent.sim_choices(paras)
        sim_choices = pd.concat(
            sim_choices, axis=0, names=["personal_id", "wave", "choice_num"]
        )
        sim_choices = sim_choices.join(self.choice_properties, on="choice_num")

        # Store the simulated data
        sim_choices.to_pickle(
            OUT_ANALYSIS
            / self.model_name
            / self.opt_settings_name
            / "sim_choices.pickle"
        )
        sim_choices["choice_final"] = False
        sim_choices["choice"] = sim_choices["sim_choice"]
        matching_probs = calculate_matching_probabilites(sim_choices)
        matching_probs.to_pickle(
            OUT_ANALYSIS
            / self.model_name
            / self.opt_settings_name
            / "sim_matching_probs.pickle"
        )

    def store_results(self, results, nested_estimation_value=None):
        """
        Store optimal values
        """

        # NB: pandas to_pickle cannot save to subdirs that dont
        # alredy exist, so need to first make them
        import os

        subdir_path = OUT_ANALYSIS / self.model_name / self.opt_settings_name
        os.makedirs(subdir_path, exist_ok=True)
        if self.use_sim_data:
            results.to_pickle(
                OUT_ANALYSIS
                / self.model_name
                / self.opt_settings_name
                / "results_on_sim.pickle"
            )
        else:
            results.to_pickle(
                OUT_ANALYSIS
                / self.model_name
                / self.opt_settings_name
                / ("results" + str(nested_estimation_value or "") + ".pickle")
            )


# a function to define dependecies and products
def estimation_parametrization(model_names, modes, opt_settings_names):
    parametrization = {}
    parametrization_install = {}
    for model_name in model_names:
        for mode in modes:
            for opt_settings_name in opt_settings_names:
                id = f"{model_name}:{mode}:{opt_settings_name}"
                parametrization[id] = {
                    "model_name": model_name,
                    "mode": mode,
                    "opt_settings_name": opt_settings_name,
                    "depends_on": [
                        IN_MODEL_SPECS / (model_name + ".json"),
                        IN_MODEL_SPECS / (opt_settings_name + ".json"),
                        IN_MODEL_CODE / "agent.py",
                        IN_MODEL_CODE / "parameter.py",
                        IN_MODEL_CODE / "load_data.py",
                        OUT_DATA / "choice_prop_prepared.pickle",
                        OUT_DATA / "pat_rec_and_dur_restrictions.pickle",
                        OUT_DATA / "choices_prepared.pickle",
                        OUT_DATA_LISS / "ambiguous_beliefs" / "event_properties.pickle",
                        OUT_DATA_LISS
                        / "ambiguous_beliefs"
                        / "baseline_matching_probs.pickle",
                    ],
                }
                parametrization_install[id] = {
                    "depends_on": [
                        OUT_ANALYSIS / model_name / opt_settings_name / "results.pickle"
                    ]
                }
                if mode == "simulate":
                    parametrization[id]["produces"] = [
                        OUT_ANALYSIS
                        / model_name
                        / opt_settings_name
                        / "sim_choices.pickle"
                    ]
                    parametrization_install[id]["produces"] = [
                        ROOT
                        / "out_under_git"
                        / model_name
                        / opt_settings_name
                        / "sim_choices.pickle"
                    ]
                elif mode == "monte_carlo":
                    parametrization[id]["produces"] = [
                        OUT_ANALYSIS
                        / model_name
                        / opt_settings_name
                        / "results_on_sim.pickle",
                        OUT_ANALYSIS
                        / model_name
                        / opt_settings_name
                        / "sim_choices.pickle",
                    ]
                    parametrization_install[id]["produces"] = [
                        ROOT
                        / "out_under_git"
                        / model_name
                        / opt_settings_name
                        / "results_on_sim.pickle",
                        ROOT
                        / "out_under_git"
                        / model_name
                        / opt_settings_name
                        / "sim_choices.pickle",
                    ]
                else:
                    parametrization[id]["produces"] = [
                        OUT_ANALYSIS / model_name / opt_settings_name / "results.pickle"
                    ]
                    parametrization_install[id]["produces"] = [
                        ROOT
                        / "out_under_git"
                        / model_name
                        / opt_settings_name
                        / "results.pickle"
                    ]

    return parametrization, parametrization_install


def run_estimation(model_name, mode, opt_settings_name, reduced_n_agents):

    logging.basicConfig(
        filename=OUT_ANALYSIS / model_name / "main.log",
        filemode="w",
        level=logging.INFO,
    )
    np.random.seed(2981326)

    if mode == "estimate":
        m = ModelWithLikelihood(
            model_name, opt_settings_name, reduced_n_agents, use_sim_data=False
        )
        m.estimate()
    elif mode == "simulate":
        m = ModelWithLikelihood(
            model_name, opt_settings_name, reduced_n_agents=True, use_sim_data=False
        )
        m.simulate(n_agents=500)
    elif mode == "monte_carlo":
        m = ModelWithLikelihood(
            model_name, opt_settings_name, reduced_n_agents, use_sim_data=False
        )
        m.simulate()
        m = ModelWithLikelihood(
            model_name, opt_settings_name, reduced_n_agents, use_sim_data=True
        )
        m.estimate()
    else:
        raise ValueError("mode not implemented")


PARAMETRIZATION, PARAMETRIZATION_INSTALL = estimation_parametrization(
    model_names=MODEL_NAMES_ESTIMATION,
    modes=MODES,
    opt_settings_names=OPT_SETTINGS_NAMES,
)

for id_, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=id_)
    @pytask.mark.skipif(not ESTIMATE, reason="Skip estimation, do only final analysis")
    def task_main(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        mode=kwargs["mode"],
        model_name=kwargs["model_name"],
        opt_settings_name=kwargs["opt_settings_name"],
        reduced_n_agents=REDUCED_N_AGENTS,
    ):
        run_estimation(model_name, mode, opt_settings_name, reduced_n_agents)


skip_install = (not ESTIMATE) or REDUCED_N_AGENTS

if not skip_install:
    for id_, kwargs in PARAMETRIZATION_INSTALL.items():

        @pytask.mark.task(id=id_)
        def task_install_files(
            depends_on=kwargs["depends_on"],
            produces=kwargs["produces"],
        ):
            for i in range(len(depends_on)):
                copyfile(depends_on[i], produces[i])
