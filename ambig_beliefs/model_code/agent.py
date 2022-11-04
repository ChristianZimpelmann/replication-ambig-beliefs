import warnings
from math import erf

import numpy as np
import pandas as pd
from numba import njit
from numba.core.errors import NumbaDeprecationWarning
from numba.core.errors import NumbaPendingDeprecationWarning
from scipy import stats

# from numba.typed import List


class Agent:
    """One individual participating in the experiment"""

    def __init__(
        self,
        choices,
        choice_properties,
        matching_probs,
        error_event_level,
        personal_id,
        het_subj_probs,
        used_waves,
    ):
        """
        One individual participating in the experiment
        :param: choices: DataFrame of choices
        :param: choice_properties: choice properties of all choice situations
                (needed for simulations)
        :param: error_event_level: if additive error is drawn on event level

        """
        warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
        warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

        self.used_waves = used_waves
        self.choice_lens = []

        self.choices = choices.query("wave == @used_waves").sort_values(
            ["wave", "choice_num"]
        )
        self.choice_properties = choice_properties
        self.matching_probs = matching_probs.query("wave == @used_waves").sort_values(
            ["wave", "aex_event"]
        )
        self.matching_probs_spec = self.matching_probs[
            ["aex_1", "aex_pi_0", "aex_pi_1", "aex_pi_2"]
        ].values.astype(float)

        # Calc some more variables if heterogeneous subjective probabilities
        # are used
        self.het_subj_probs = het_subj_probs
        if self.het_subj_probs:

            # Make list of choices (one for each wave)
            self.aex_prob_chunks = []
            for wave in self.used_waves:  # noqa: B007
                self.aex_prob_chunks.append(
                    choices.query("wave == @wave")[
                        ["aex_1", "aex_pi_0", "aex_pi_1", "aex_pi_2"]
                    ].values
                )

            # Make list of choices (one for each wave)
            self.aex_prob_chunks_mp = []
            for wave in self.used_waves:  # noqa: B007
                self.aex_prob_chunks_mp.append(
                    self.matching_probs.query("wave == @wave")[
                        ["aex_1", "aex_pi_0", "aex_pi_1", "aex_pi_2"]
                    ].values
                )

        # ToDo: Test/check if choices is correctly set
        self.mintervals = np.array(
            [
                np.array([i.left, i.right])
                for i in self.matching_probs["baseline_matching_prob_interval"]
            ]
        )
        self.matching_prob_midp = self.matching_probs[
            "baseline_matching_prob_midp"
        ].values
        self.personal_id = personal_id
        self.error_event_level = error_event_level

    def eval_likelihood(self, paras):
        """
        Calculate the individual log_likelihood for all choices of one agent.
        :param: paras: dict of parameters

        Expects for each parameter a scalar if it is homogeneous over waves and
        a list with length equals the number of waves if it is heterogeneous over waves.
        """
        # Calculate all subjective probabilites (pi)
        if "pi_0" not in paras:
            paras["pi_0"] = 0
        if self.error_event_level:
            # Set number of error draws
            self.n_error_draws = self.matching_probs_spec.shape[0]

            # Calculate ambiguity neutral probabilities for each event
            pis = self.calc_subjective_probs_for_all_events(
                paras, error_event_level=True
            )

            likelihood = likeli_error_event_level(
                pis=pis,
                tau=paras["tau"],
                sigma=paras["sigma"],
                omega=paras["omega"],
                theta=paras["theta"],
                mintervals=self.mintervals,
            )
        else:
            # Set number of error draws
            self.n_error_draws = self.choices.shape[0]

            # Calculate ambiguity neutral probabilities for each event
            pis = self.calc_subjective_probs_for_all_events(
                paras, error_event_level=False
            )

            likelihood = likeli_all_choices(
                pis=pis,
                tau=paras["tau"],
                sigma=paras["sigma"],
                omega=paras["omega"],
                theta=paras["theta"],
                ps=self.choices["p"].values,
                choices=self.choices["choice"].values.astype("bool"),
            ).sum()

        return likelihood

    def eval_likelihood_ind(self, paras):
        """
        Calculate the individual log_likelihood for all choices of one agent and return as list.
        :param: paras: dict of parameters

        Expects for each parameter a scalar if it is homogeneous over waves and
        a list with length equals the number of waves if it is heterogeneous over waves.
        """
        # Calculate all subjective probabilites (pi)

        if "pi_0" not in paras:
            paras["pi_0"] = 0
        if self.error_event_level:
            raise NotImplementedError(
                "List of individual likelihodds not yet implemented for error_event_level"
            )
        else:
            # Set number of error draws
            self.n_error_draws = self.choices.shape[0]

            # Calculate ambiguity neutral probabilities for each event
            pis = self.calc_subjective_probs_for_all_events(
                paras, error_event_level=False
            )

            likelihood_list = likeli_all_choices(
                pis=pis,
                tau=paras["tau"],
                sigma=paras["sigma"],
                omega=paras["omega"],
                theta=paras["theta"],
                ps=self.choices["p"].values,
                choices=self.choices["choice"].values.astype("bool"),
            )

        return likelihood_list

    def least_squares_objective(self, paras):
        """
        Calculate objective for least squares estimation.
        """
        if "pi_0" not in paras:
            paras["pi_0"] = 0

        # Calculate ambiguity neutral probabilities for each event
        pis = self.calc_subjective_probs_for_all_events(paras, error_event_level=True)

        # Set number of error draws
        self.n_error_draws = self.matching_probs_spec.shape[0]
        # print(len(pis), self.matching_prob_midp.shape)
        result = least_squares_objective(
            pis=pis,
            tau=paras["tau"],
            sigma=paras["sigma"],
            mprobs=self.matching_prob_midp,
        )
        return result

    def calc_subjective_probs_for_all_events(self, paras, error_event_level):
        """
        Calculate subjective probabilities for all 6 or 7 events (or all choices)
        :param paras: dict of parameters
        :return: array of subjective probabilities
        """
        if error_event_level:

            # Differentiate if subj are homogeneous or heterogeneous (over waves)
            if self.het_subj_probs:

                assert len(self.aex_prob_chunks_mp) == len(paras["pi_0"])
                assert len(self.aex_prob_chunks_mp) == len(paras["pi_1"])
                assert len(self.aex_prob_chunks_mp) == len(paras["pi_2"])

                # Create para_mat
                pi_chunks = []
                for i in range(len(self.aex_prob_chunks_mp)):
                    pi_chunks.append(
                        self.aex_prob_chunks_mp[i]
                        @ np.array(
                            [1, paras["pi_0"][i], paras["pi_1"][i], paras["pi_2"][i]]
                        )
                    )
                pis = np.concatenate(pi_chunks)
            else:
                pis = self.matching_probs_spec @ np.array(
                    [1, paras["pi_0"], paras["pi_1"], paras["pi_2"]]
                )

        else:

            # Differentiate if subj are homogeneous or heterogeneous (over waves)
            if self.het_subj_probs:
                assert len(self.aex_prob_chunks) == len(paras["pi_0"])
                assert len(self.aex_prob_chunks) == len(paras["pi_1"])
                assert len(self.aex_prob_chunks) == len(paras["pi_2"])

                # Create para_mat
                pi_chunks = []
                for i in range(len(self.aex_prob_chunks)):
                    pi_chunks.append(
                        self.aex_prob_chunks[i]
                        @ np.array(
                            [1, paras["pi_0"][i], paras["pi_1"][i], paras["pi_2"][i]]
                        )
                    )
                pis = np.concatenate(pi_chunks)
            else:
                pis = self.choices[
                    ["aex_1", "aex_pi_0", "aex_pi_1", "aex_pi_2"]
                ].values @ np.array([1, paras["pi_0"], paras["pi_1"], paras["pi_2"]])
        # typed_pis = List()
        # [typed_pis.append(x) for x in pis]

        typed_pis = pis
        return typed_pis

    def calc_implied_matching_probs(self, paras):
        pis = self.calc_subjective_probs_for_all_events(paras, error_event_level=True)
        m_hat = np.empty(len(pis))
        for i in range(len(pis)):
            m_hat[i] = calc_decision_weight(
                pi=pis[i], tau=paras["tau"], sigma=paras["sigma"]
            )

        return m_hat

    def least_squares_objective_ind(self, paras):
        """
        Calculate objective for least squares estimation.
        """
        if "pi_0" not in paras:
            paras["pi_0"] = 0

        # Calculate ambiguity neutral probabilities for each event
        pis = self.calc_subjective_probs_for_all_events(paras, error_event_level=True)

        # Set number of error draws
        self.n_error_draws = self.matching_probs_spec.shape[0]

        return least_squares_all_events(
            pis=pis,
            tau=paras["tau"],
            sigma=paras["sigma"],
            std_dev=paras["theta"],
            mprobs=self.matching_prob_midp,
        )

    def sim_choices(self, paras):
        """
        Calc choices for all events.
        :param: paras: dict of parameters

        """
        choices_index = pd.MultiIndex.from_arrays(
            [[], []], names=["wave", "choice_num"]
        )
        choices = pd.DataFrame(index=choices_index, columns=["sim_choice", "random"])
        choice_prop_one_wave = self.choice_properties.groupby("choice_num").first()
        for wave in self.used_waves:
            choice_num = 1
            while choice_num < 92:
                choice, random = sim_one_choice(
                    paras, choice_prop_one_wave.loc[choice_num]
                )
                choices.loc[(wave, choice_num), :] = choice, random
                if choice:
                    choice_num = choice_prop_one_wave.loc[
                        choice_num, "next_choice_after_aex"
                    ]
                else:
                    choice_num = choice_prop_one_wave.loc[
                        choice_num, "next_choice_after_lot"
                    ]

        return choices


@njit
def fast_normal_cdf(x, scale=1):
    sqrt_2 = 1.41421356237309504880168872420969807856967187537694807317667973
    return 0.5 * (1 + erf(x / (scale * sqrt_2)))


@njit
def logistic_cdf(x, scale=1):
    return 1 / (1 + np.exp(-x / scale))


@njit
def calc_decision_weight(pi, tau, sigma):
    """
    Calculate decision weight.
    """

    decision_weight = tau + sigma * pi
    # return min(max(decision_weight, 0), 1)
    return decision_weight


@njit
def prelec(pi, sigma, tau):
    """
    Calculate alternative decision weight based on prelec formula
    """
    return (np.exp(-((-np.log(pi)) ** sigma))) ** tau


@njit
def choice_without_trembling(pi, tau, sigma, theta, p):
    """
    Calculate likelihood of choice for one choice situation without trembling hand error.
    :param p: winning probability of lottery
    :return: likelihood
    """

    decision_weight = calc_decision_weight(pi=pi, tau=tau, sigma=sigma)
    # decision_weight = prelec(pi=pi, tau=tau, sigma=sigma)
    # print(pi, tau, sigma, decision_weight)
    # if decision_weight > 1 or decision_weight < 0:
    #     return np.nan
    # else:
    #     if theta == 0:
    #         if decision_weight == p:
    #             return 0.5
    #         else:
    #             return decision_weight > p
    #     else:
    #         return fast_normal_cdf(decision_weight - p, scale=theta)
    if theta == 0:
        if decision_weight == p:
            return 0.5
        else:
            return decision_weight > p
    else:
        return fast_normal_cdf(decision_weight - p, scale=theta)
        # return logistic_cdf(decision_weight - p, scale=theta)


@njit
def likeli_one_choice(pi, tau, sigma, omega, theta, p, choice):
    """
    Calculate likelihood of choice for one choice situation.
    :param p: winning probability of lottery
    :return: likelihood
    """

    det_choice = choice_without_trembling(pi=pi, tau=tau, sigma=sigma, theta=theta, p=p)

    likelihood_aex = omega / 2 + (1 - omega) * det_choice
    if choice:
        return likelihood_aex
    else:
        return 1 - likelihood_aex


@njit
def likeli_all_choices(pis, tau, sigma, omega, theta, ps, choices):
    """
    Calculate likelihood of choice for all choices.
    :param pis: list of pis
    :param tau: level of ambiguity aversion
    :param sigma: a-sensitivity
    :param omega: trembling hand error
    :param theta: std of Fechner error
    :param ps: list of ps
    :param choices: list of choices
    :return: likelihood
    """
    # print(pis, tau, sigma, omega, theta, ps, choices)
    log_likeli_choices = np.empty(len(choices))
    for i in range(len(choices)):
        log_likeli_choices[i] = np.log(
            likeli_one_choice(
                pi=pis[i],
                tau=tau,
                sigma=sigma,
                theta=theta,
                omega=omega,
                p=ps[i],
                choice=choices[i],
            )
        )
    return log_likeli_choices


@njit
def likeli_error_event_level(pis, tau, sigma, omega, theta, mintervals):
    """
    Calculate likelihood on event level if the additive error (Fechner) is drawn
    for each event (not for each choice).

    # ToDo: implement omega
    # -> then remove the following two lines in main:
    if not self.error_event_level:
        paras_random['omega'] = 1

    :param pis: list of pis
    :param tau: level of ambiguity aversion
    :param sigma: a-sensitvoteivity
    :param omega: trembling hand error
    :param theta: std of Fechner error
    :param mintervals: list of matching probabilities (intervals)

    :return: likelihood
    """
    # trembling hand error not yet implemented
    assert omega == 0
    # print(pis)
    log_likeli_events = 0
    for i in range(len(mintervals)):
        decision_weight = calc_decision_weight(pi=pis[i], tau=tau, sigma=sigma)
        error_interval = mintervals[i] - decision_weight

        # Calculate the probability that the additive error falls in the particular range
        likeli = fast_normal_cdf(error_interval[1], scale=theta) - fast_normal_cdf(
            error_interval[0], scale=theta
        )
        # error = mintervals[i].mean() - decision_weight
        # likeli = stats.norm.pdf(error, scale=theta)
        log_likeli_events += np.log(likeli)
    return log_likeli_events


@njit
def least_squares_objective(pis, tau, sigma, mprobs):
    """
    Calculate squared deviation for least squares estimation.
    """

    m_hat = np.empty(len(pis))
    for i in range(len(pis)):
        m_hat[i] = calc_decision_weight(pi=pis[i], tau=tau, sigma=sigma)
    result = mprobs.dot(mprobs) + m_hat.dot(m_hat) - 2 * mprobs.dot(m_hat)
    return result


def least_squares_all_events(pis, tau, sigma, std_dev, mprobs):
    m_hat = np.empty(len(pis))
    log_likeli = np.empty(len(pis))
    for i in range(len(pis)):
        m_hat[i] = calc_decision_weight(pi=pis[i], tau=tau, sigma=sigma)
        log_likeli[i] = np.log(
            stats.norm.pdf(x=(mprobs[i] - m_hat[i]), loc=0, scale=std_dev)
        )
    return log_likeli


def get_subj_prob(pi_0, pi_1, pi_2, event):
    """
    Calculate subjective probability based on AEX-event.
    """
    subj_prob_dict = {
        "0": pi_0,
        "1": pi_1,
        "2": pi_2,
        "3": 1 - pi_1 - pi_2,
        "1c": 1 - pi_1,
        "2c": 1 - pi_2,
        "3c": pi_1 + pi_2,
    }

    return subj_prob_dict[event]


def sim_one_choice(paras, choice_prop):
    """
    Calculate likelihood of AEX-choice for one choice situation.

    :return: likelihood
    """
    # With probablity omega the choice_prop is random with equal probability
    if np.random.uniform(0, 1) < paras["omega"]:
        random = 1
        choice_prob = 0.5

    # With probability 1 - omega, the one with the deterministic choice_prop is made
    else:
        random = 0
        pi = get_subj_prob(
            pi_0=paras["pi_0"],
            pi_1=paras["pi_1"],
            pi_2=paras["pi_2"],
            event=choice_prop["aex_event"],
        )
        choice_prob = choice_without_trembling(
            pi=pi,
            tau=paras["tau"],
            sigma=paras["sigma"],
            theta=paras["theta"],
            p=choice_prop["p"],
        )
    choice = np.random.uniform(0, 1) < choice_prob

    return choice, random
