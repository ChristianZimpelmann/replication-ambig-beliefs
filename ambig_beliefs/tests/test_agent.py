"""Test the ssc_agent

"""
import numpy as np
import pandas as pd
import pytest

from ambig_beliefs.model_code.agent import Agent
from ambig_beliefs.model_code.agent import calc_decision_weight
from ambig_beliefs.model_code.agent import choice_without_trembling
from ambig_beliefs.model_code.agent import get_subj_prob
from ambig_beliefs.model_code.agent import likeli_error_event_level
from ambig_beliefs.model_code.agent import likeli_one_choice
from ambig_beliefs.model_code.agent import sim_one_choice
from config import OUT_DATA


@pytest.fixture()
def default_para_values():
    default_para_values = {}
    default_para_values["pi_0"] = 0.2
    default_para_values["pi_1"] = 0.1
    default_para_values["pi_2"] = 0.6
    default_para_values["omega"] = 0.2
    default_para_values["tau"] = 0.2
    default_para_values["sigma"] = 0.5
    default_para_values["theta"] = 0
    return default_para_values


@pytest.fixture()
def default_choice():
    default_choice = {}
    default_choice["aex_event"] = "3"
    default_choice["p"] = 0.2
    default_choice["choice"] = True
    return default_choice


@pytest.fixture()
def choice_properties():
    choice_properties = pd.read_pickle(OUT_DATA / "choice_prop_prepared.pickle")

    choice_properties = choice_properties[
        ["aex_event", "p", "next_choice_after_aex", "next_choice_after_lot"]
    ]
    return choice_properties


@pytest.fixture()
def matching_probs():
    matching_probs = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0, 0.005, pd.Interval(0.00, 0.01, closed="both"), 1],
                [0, 0, 1, 0, 0.005, pd.Interval(0.00, 0.01, closed="both"), 1],
                [1, 0, -1, 0, 0.075, pd.Interval(0.05, 0.10, closed="both"), 1],
                [0, 0, 0, 1, 0.075, pd.Interval(0.05, 0.10, closed="both"), 1],
                [1, 0, 0, -1, 0.075, pd.Interval(0.05, 0.10, closed="both"), 1],
                [1, 0, -1, -1, 0.075, pd.Interval(0.05, 0.10, closed="both"), 1],
                [0, 0, 1, 1, 0.075, pd.Interval(0.05, 0.10, closed="both"), 1],
            ]
        ),
        columns=[
            "aex_1",
            "aex_pi_0",
            "aex_pi_1",
            "aex_pi_2",
            "baseline_matching_prob_midp",
            "baseline_matching_prob_interval",
            "wave",
        ],
    )
    return matching_probs


@pytest.fixture()
def choices():
    choices = pd.DataFrame(
        np.array(
            [
                [0, 0, 1, 0, 0, 0.5, False, 1],
                [1, 0, 1, 0, 0, 0.1, False, 1],
                [2, 0, 1, 0, 0, 0.05, False, 1],
                [3, 0, 1, 0, 0, 0.01, False, 1],
                [4, 0, 0, 1, 0, 0.5, False, 1],
                [5, 0, 0, 1, 0, 0.1, False, 1],
                [6, 0, 0, 1, 0, 0.05, False, 1],
                [7, 0, 0, 1, 0, 0.01, False, 1],
                [8, 0, 0, 0, 1, 0.5, False, 1],
                [9, 0, 0, 0, 1, 0.1, False, 1],
                [10, 0, 0, 0, 1, 0.05, True, 1],
                [11, 1, 0, -1, -1, 0.5, False, 1],
                [12, 1, 0, -1, -1, 0.1, False, 1],
                [13, 1, 0, -1, -1, 0.05, True, 1],
                [14, 1, 0, -1, 0, 0.5, False, 2],
                [15, 1, 0, -1, 0, 0.1, False, 2],
                [16, 1, 0, -1, 0, 0.05, True, 2],
                [17, 1, 0, 0, -1, 0.5, False, 2],
                [18, 1, 0, 0, -1, 0.1, False, 2],
                [19, 1, 0, 0, -1, 0.05, True, 2],
                [20, 0, 0, 1, 1, 0.5, False, 2],
                [21, 0, 0, 1, 1, 0.1, False, 2],
                [22, 0, 0, 1, 1, 0.05, True, 2],
            ]
        ),
        columns=[
            "choice_num",
            "aex_1",
            "aex_pi_0",
            "aex_pi_1",
            "aex_pi_2",
            "p",
            "choice",
            "wave",
        ],
    )
    return choices


@pytest.fixture()
def agent(choices, choice_properties, matching_probs):
    return Agent(
        choices=choices,
        choice_properties=choice_properties,
        matching_probs=matching_probs,
        error_event_level=False,
        personal_id=np.nan,
        het_subj_probs=False,
    )


@pytest.fixture()
def agent_het_probs(choices, choice_properties, matching_probs):
    return Agent(
        choices=choices,
        choice_properties=choice_properties,
        matching_probs=matching_probs,
        error_event_level=False,
        personal_id=np.nan,
        het_subj_probs=True,
    )


def test_get_subj_prob():
    """calc_subjective_probs_for_all_events
    Test get_subj_prob.
    """

    result = get_subj_prob(pi_0=0.5, pi_1=0.1, pi_2=0.6, event="3")
    expected_result = 0.3
    pytest.approx(result, expected_result)


def test_calc_subj_prob(agent, default_para_values):
    """
    Test get_subj_prob.
    """
    result = agent.calc_subjective_probs_for_all_events(
        default_para_values, error_event_level=False
    )
    pytest.approx(result[0], 0.2)
    pytest.approx(result[-1], 0.7)


def test_calc_subj_prob_heterogeneous(default_para_values, agent_het_probs):
    """
    Test calculation of heterogeneous sujective probabilities.
    """
    paras = default_para_values.copy()
    paras["pi_0"] = [0.2, 0.2]
    paras["pi_1"] = [0.1, 0.1]
    paras["pi_2"] = [0.6, 0.6]
    result = agent_het_probs.calc_subjective_probs_for_all_events(
        paras, error_event_level=False
    )
    pytest.approx(result[0], 0.2)
    pytest.approx(result[-1], 0.7)


def test_calc_subj_prob_heterogeneous_2(default_para_values, agent_het_probs):
    """
    Test calculation of heterogeneous sujective probabilities.
    """
    paras = default_para_values.copy()
    paras["pi_0"] = [0.2, 0.5]
    paras["pi_1"] = [0.1, 0.3]
    paras["pi_2"] = [0.6, 0.3]
    result = agent_het_probs.calc_subjective_probs_for_all_events(
        paras, error_event_level=False
    )
    pytest.approx(result[0], 0.2)
    pytest.approx(result[-1], 0.6)


def test_calc_decision_weight_trivial():
    """
    Test calc_decision_weight for trivial case.
    """
    result = calc_decision_weight(pi=0.6, tau=0, sigma=1)
    expected_result = 0.6
    pytest.approx(result, expected_result)


def test_calc_decision_weight():
    """
    Test calc_decision_weight.
    """
    result = calc_decision_weight(pi=0.6, tau=0.2, sigma=0.5)
    expected_result = 0.5
    pytest.approx(result, expected_result)


def test_choice_without_trembling(default_choice):
    """
    Test choice_without_trembling.
    """
    result = choice_without_trembling(
        pi=0.3, tau=0.2, sigma=0.5, theta=0, p=default_choice["p"]
    )
    expected_result = True
    pytest.approx(result, expected_result)

    # Change parameter such that excepted choice is False
    result = choice_without_trembling(
        pi=0.3, tau=0.01, sigma=0.5, theta=0, p=default_choice["p"]
    )
    expected_result = False
    pytest.approx(result, expected_result)


def test_choice_without_trembling_equality():
    """
    Test choice_without_trembling for decisionweight equal to p.
    """
    result = choice_without_trembling(pi=0.3, tau=0.2, sigma=0.5, theta=0, p=0.35)
    expected_result = 0.5
    pytest.approx(result, expected_result)

    result = choice_without_trembling(pi=0.3, tau=0.2, sigma=0.5, theta=1, p=0.35)
    expected_result = 0.5
    pytest.approx(result, expected_result)


def test_likeli_one_choice_trivial(default_choice):
    """
    Test likeli_one_choice for omega=1.
    """
    result = likeli_one_choice(
        pi=0.3,
        tau=0.2,
        sigma=0.5,
        theta=0,
        omega=1,
        p=default_choice["p"],
        choice=default_choice["choice"],
    )
    expected_result = 0.5
    pytest.approx(result, expected_result)


def test_likeli_one_choice_omega_0(default_choice):
    """
    Test likeli_one_choice for omega=0.
    """
    result_stochastic = likeli_one_choice(
        pi=0.3,
        tau=0.2,
        sigma=0.5,
        theta=0,
        omega=0,
        p=default_choice["p"],
        choice=default_choice["choice"],
    )
    result_deterministic = choice_without_trembling(
        pi=0.3, tau=0.2, sigma=0.5, theta=0, p=default_choice["p"]
    )
    pytest.approx(result_stochastic, result_deterministic)

    # Change parameter such that excepted choice is False
    result_stochastic = likeli_one_choice(
        pi=0.3,
        tau=0.01,
        sigma=0.5,
        theta=0,
        omega=0,
        p=default_choice["p"],
        choice=default_choice["choice"],
    )
    result_deterministic = choice_without_trembling(
        pi=0.3, tau=0.01, sigma=0.5, theta=0, p=default_choice["p"]
    )
    pytest.approx(result_stochastic, result_deterministic)


def test_likeli_one_choice(default_choice):
    """
    Test likeli_one_choice.
    """
    result = likeli_one_choice(
        pi=0.3,
        tau=0.2,
        sigma=0.5,
        theta=0,
        omega=0.2,
        p=default_choice["p"],
        choice=default_choice["choice"],
    )
    expected_result = 0.9
    pytest.approx(result, expected_result)


def test_likeli_one_choice_reg(default_choice):
    """
    Test likeli_one_choice. (regression)
    """
    result = likeli_one_choice(
        pi=0.3,
        tau=0.2,
        sigma=0.5,
        theta=0.5,
        omega=0.2,
        p=default_choice["p"],
        choice=default_choice["choice"],
    )
    expected_result = 0.594329137751162
    pytest.approx(result, expected_result)


def test_likeli_one_choice_high_theta(default_choice):
    """
    Test likeli_one_choice for high theta.
    """
    result = likeli_one_choice(
        pi=0.3,
        tau=0.2,
        sigma=0.5,
        theta=1000000,
        omega=0.2,
        p=default_choice["p"],
        choice=default_choice["choice"],
    )
    expected_result = 0.5
    pytest.approx(result, expected_result)


def test_sim_one_choice_omega_0(default_para_values, default_choice):
    """
    Test sim_one_choice for omega=0.
    """
    para = default_para_values.copy()
    para["omega"] = 0
    para["theta"] = 0

    result_stochastic = sim_one_choice(para, default_choice)[0]
    result_deterministic = choice_without_trembling(
        pi=0.3, tau=0.2, sigma=0.5, theta=0, p=default_choice["p"]
    )
    pytest.approx(result_stochastic, result_deterministic)


def test_likeli_error_event_level_simple():
    """
    Test likeli_error_event_level for simple values.
    """
    res = likeli_error_event_level(
        pis=np.array([0]),
        tau=0,
        sigma=1,
        omega=0,
        theta=1,
        mintervals=np.array([[0, 1000]]),
    )
    expected_result = np.log(0.5)
    pytest.approx(res, expected_result)


def test_likeli_error_event_level_reg():
    """
    Test likeli_error_event_level (regression).
    """
    res = likeli_error_event_level(
        pis=np.array([0.5]),
        tau=0,
        sigma=1,
        omega=0,
        theta=1,
        mintervals=np.array([[0.2, 0.3]]),
    )
    expected_result = -3.253164190574811
    pytest.approx(res, expected_result)


def test_sim_one_agent(agent, default_para_values):
    """
    Check that sim_choices runs through and returned object has correct shape.
    """

    sim_choices = agent.sim_choices(paras=default_para_values)
    assert sim_choices.shape[0] > 40
    assert sim_choices.shape[0] < 58


def test_likelihood_event_level_reg(agent):

    para = {}
    para["pi_0"] = 0
    para["pi_1"] = 0
    para["pi_2"] = 0.5
    para["omega"] = 0
    para["tau"] = 0.01
    para["sigma"] = 0.15
    para["theta"] = 0.5
    agent.error_event_level = True
    result = agent.eval_likelihood(para)

    expected_result = np.log(6.321941092056277e-12)
    pytest.approx(result, expected_result)


# def test_likelihood_choice_level_reg(agent):
#     para = {}
#     para["pi_0"] = 0
#     para["pi_1"] = 0
#     para["pi_2"] = 0.5
#     para["omega"] = 0
#     para["tau"] = 0.01
#     para["sigma"] = 0.15
#     para["theta"] = 0.5
#     agent.error_event_level = False
#     result = agent.eval_likelihood(para)

#     expected_result = np.log(6.9331538675175545e-06)
#     pytest.approx(result, expected_result)


# def test_likelihood_trembling_reg(agent):
#     para = {}
#     para["pi_0"] = 0
#     para["pi_1"] = 0
#     para["pi_2"] = 0.5
#     para["omega"] = 0.2
#     para["tau"] = 0.01
#     para["sigma"] = 0.15
#     para["theta"] = 0
#     agent.error_event_level = False
#     result = agent.eval_likelihood(para)

#     expected_result = np.log(0.0030394163647642332)
#     pytest.approx(result, expected_result)


# def test_het_prob_equals_hom_prob(agent):
#     para = {}
#     para["pi_0"] = 0
#     para["pi_1"] = 0
#     para["pi_2"] = 0.5
#     para["omega"] = 0
#     para["tau"] = 0.01
#     para["sigma"] = 0.15
#     para["theta"] = 0.5
#     agent.error_event_level = False
#     result_hom_prob = agent.eval_likelihood(para)

#     for pi in ["pi_0", "pi_1", "pi_2"]:
#         para[pi] = [para[pi]] * 2
#     result_het_prob = agent_het_probs.eval_likelihood(para)

#     pytest.approx(result_hom_prob, result_het_prob)


# def test_het_prob_reg(agent):
#     para = {}
#     para["pi_0"] = [0, 0.1]
#     para["pi_1"] = [0, 0.3]
#     para["pi_2"] = [0.5, 0.3]
#     para["omega"] = 0
#     para["tau"] = 0.01
#     para["sigma"] = 0.15
#     para["theta"] = 0.5
#     agent.error_event_level = False
#     result = agent_het_probs.eval_likelihood(para)
#     pytest.approx(result, np.log(7.0180408828340524e-06))
