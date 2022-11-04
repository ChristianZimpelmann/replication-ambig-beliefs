# """Test the ssc_agent
# """
# import pandas as pd
# import numpy as np
# import scipy as sp
# import unittest
# import pickle
# import json
# from unittest import TestCase
# from nose.tools import (
#     assert_almost_equal,
#     assert_equal,
#     assert_true,
#     assert_false,
#     assert_not_almost_equal,
# )
# from bld.project_paths import project_paths_join as ppj
# from ambig_beliefs.model_code.agent import (
#     Agent,
#     get_subj_prob,
#     calc_decision_weight,
#     choice_without_trembling,
#     likeli_one_choice,
#     sim_one_choice,
#     likeli_error_event_level,
# )
# from scipy.stats import logistic
# from mock import MagicMock
# class TestAgent(TestCase):
#     def setUp(self):
#         # self.model_name = "ssc_w2_1"
#         self.default_para_values = {}
#         self.default_para_values["pi_0"] = 0.2
#         self.default_para_values["pi_1"] = 0.1
#         self.default_para_values["pi_2"] = 0.6
#         self.default_para_values["omega"] = 0.2
#         self.default_para_values["tau"] = 0.2
#         self.default_para_values["sigma"] = 0.5
#         self.default_para_values["theta"] = 0
#         self.default_choice = {}
#         self.default_choice["aex_event"] = "3"
#         self.default_choice["p"] = 0.2
#         self.default_choice["choice"] = True
#         choice_properties = pd.read_pickle(ppj("OUT_DATA", "choice_properties.pickle"))
#         choice_properties = choice_properties.loc[choice_properties["aex_event"] != "0"]
#         choice_properties["p"] = choice_properties["lottery_p_win"] / 100
#         choice_properties = choice_properties[
#             ["aex_event", "p", "next_choice_after_aex", "next_choice_after_lot"]
#         ]
#         matching_probs = pd.DataFrame(
#             np.array(
#                 [
#                     [0, 1, 0, 0, 0.005, pd.Interval(0.00, 0.01, closed="both"), 1],
#                     [0, 0, 1, 0, 0.005, pd.Interval(0.00, 0.01, closed="both"), 1],
#                     [1, 0, -1, 0, 0.075, pd.Interval(0.05, 0.10, closed="both"), 1],
#                     [0, 0, 0, 1, 0.075, pd.Interval(0.05, 0.10, closed="both"), 1],
#                     [1, 0, 0, -1, 0.075, pd.Interval(0.05, 0.10, closed="both"), 1],
#                     [1, 0, -1, -1, 0.075, pd.Interval(0.05, 0.10, closed="both"), 1],
#                     [0, 0, 1, 1, 0.075, pd.Interval(0.05, 0.10, closed="both"), 1],
#                 ]
#             ),
#             columns=[
#                 "aex_1",
#                 "aex_pi_0",
#                 "aex_pi_1",
#                 "aex_pi_2",
#                 "baseline_matching_prob_midp",
#                 "baseline_matching_prob_interval",
#                 "wave",
#             ],
#         )
#         choices = pd.DataFrame(
#             np.array(
#                 [
#                     [0, 1, 0, 0, 0.5, False, 1],
#                     [0, 1, 0, 0, 0.1, False, 1],
#                     [0, 1, 0, 0, 0.05, False, 1],
#                     [0, 1, 0, 0, 0.01, False, 1],
#                     [0, 0, 1, 0, 0.5, False, 1],
#                     [0, 0, 1, 0, 0.1, False, 1],
#                     [0, 0, 1, 0, 0.05, False, 1],
#                     [0, 0, 1, 0, 0.01, False, 1],
#                     [0, 0, 0, 1, 0.5, False, 1],
#                     [0, 0, 0, 1, 0.1, False, 1],
#                     [0, 0, 0, 1, 0.05, True, 1],
#                     [1, 0, -1, -1, 0.5, False, 1],
#                     [1, 0, -1, -1, 0.1, False, 1],
#                     [1, 0, -1, -1, 0.05, True, 1],
#                     [1, 0, -1, 0, 0.5, False, 1],
#                     [1, 0, -1, 0, 0.1, False, 1],
#                     [1, 0, -1, 0, 0.05, True, 1],
#                     [1, 0, 0, -1, 0.5, False, 1],
#                     [1, 0, 0, -1, 0.1, False, 1],
#                     [1, 0, 0, -1, 0.05, True, 1],
#                     [0, 0, 1, 1, 0.5, False, 1],
#                     [0, 0, 1, 1, 0.1, False, 1],
#                     [0, 0, 1, 1, 0.05, True, 1],
#                 ]
#             ),
#             columns=[
#                 "aex_1",
#                 "aex_pi_0",
#                 "aex_pi_1",
#                 "aex_pi_2",
#                 "p",
#                 "choice",
#                 "wave",
#             ],
#         )
#         self.agent = Agent(
#             choices=choices,
#             choice_properties=choice_properties,
#             matching_probs=matching_probs,
#             error_event_level=False,
#             personal_id=np.nan,
#             het_subj_probs=False,
#         )
#         choices_het_probs = pd.DataFrame(
#             np.array(
#                 [
#                     [0, 1, 0, 0, 0.5, False, 1],
#                     [0, 1, 0, 0, 0.1, False, 1],
#                     [0, 1, 0, 0, 0.05, False, 1],
#                     [0, 1, 0, 0, 0.01, False, 1],
#                     [0, 0, 1, 1, 0.5, False, 2],
#                     [0, 0, 1, 1, 0.1, False, 2],
#                     [0, 0, 1, 1, 0.05, True, 2],
#                 ]
#             ),
#             columns=[
#                 "aex_1",
#                 "aex_pi_0",
#                 "aex_pi_1",
#                 "aex_pi_2",
#                 "p",
#                 "choice",
#                 "wave",
#             ],
#         )
#         self.agent_het_probs = Agent(
#             choices=choices_het_probs,
#             choice_properties=choice_properties,
#             matching_probs=matching_probs,
#             error_event_level=False,
#             personal_id=np.nan,
#             het_subj_probs=True,
#         )
#     def test_get_subj_prob(self):
#         """calc_subjective_probs_for_all_events
#         Test get_subj_prob.
#         """
#         result = get_subj_prob(pi_1=0.1, pi_2=0.6, event="3")
#         expected_result = 0.3
#         assert_almost_equal(result, expected_result)
#     def test_calc_subj_prob(self):
#         """
#         Test get_subj_prob.
#         """
#         result = self.agent.calc_subjective_probs_for_all_events(
#             self.default_para_values, error_event_level=False
#         )
#         assert_almost_equal(result[0], 0.2)
#         assert_almost_equal(result[-1], 0.7)
#     def test_calc_subj_prob_heterogeneous(self):
#         """
#         Test calculation of heterogeneous sujective probabilities.
#         """
#         paras = self.default_para_values.copy()
#         paras["pi_0"] = [0.2, 0.2]
#         paras["pi_1"] = [0.1, 0.1]
#         paras["pi_2"] = [0.6, 0.6]
#         result = self.agent_het_probs.calc_subjective_probs_for_all_events(
#             paras, error_event_level=False
#         )
#         assert_almost_equal(result[0], 0.2)
#         assert_almost_equal(result[-1], 0.7)
#     def test_calc_subj_prob_heterogeneous_2(self):
#         """
#         Test calculation of heterogeneous sujective probabilities.
#         """
#         paras = self.default_para_values.copy()
#         paras["pi_0"] = [0.2, 0.5]
#         paras["pi_1"] = [0.1, 0.3]
#         paras["pi_2"] = [0.6, 0.3]
#         result = self.agent_het_probs.calc_subjective_probs_for_all_events(
#             paras, error_event_level=False
#         )
#         assert_almost_equal(result[0], 0.2)
#         assert_almost_equal(result[-1], 0.6)
#     def test_calc_decision_weight_trivial(self):
#         """
#         Test calc_decision_weight for trivial case.
#         """
#         result = calc_decision_weight(pi=0.6, tau=0, sigma=1)
#         expected_result = 0.6
#         assert_almost_equal(result, expected_result)
#     def test_calc_decision_weight(self):
#         """
#         Test calc_decision_weight.
#         """
#         result = calc_decision_weight(pi=0.6, tau=0.2, sigma=0.5)
#         expected_result = 0.5
#         assert_almost_equal(result, expected_result)
#     def test_choice_without_trembling(self):
#         """
#         Test choice_without_trembling.
#         """
#         result = choice_without_trembling(
#             pi=0.3, tau=0.2, sigma=0.5, theta=0, p=self.default_choice["p"]
#         )
#         expected_result = True
#         assert_almost_equal(result, expected_result)
#         # Change parameter such that excepted choice is False
#         result = choice_without_trembling(
#             pi=0.3, tau=0.01, sigma=0.5, theta=0, p=self.default_choice["p"]
#         )
#         expected_result = False
#         assert_almost_equal(result, expected_result)
#     def test_choice_without_trembling_equality(self):
#         """
#         Test choice_without_trembling for decisionweight equal to p.
#         """
#         result = choice_without_trembling(pi=0.3, tau=0.2, sigma=0.5, theta=0, p=0.35)
#         expected_result = 0.5
#         assert_almost_equal(result, expected_result)
#         result = choice_without_trembling(pi=0.3, tau=0.2, sigma=0.5, theta=1, p=0.35)
#         expected_result = 0.5
#         assert_almost_equal(result, expected_result)
#     def test_likeli_one_choice_trivial(self):
#         """
#         Test likeli_one_choice for omega=1.
#         """
#         result = likeli_one_choice(
#             pi=0.3,
#             tau=0.2,
#             sigma=0.5,
#             theta=0,
#             omega=1,
#             p=self.default_choice["p"],
#             choice=self.default_choice["choice"],
#         )
#         expected_result = 0.5
#         assert_almost_equal(result, expected_result)
#     def test_likeli_one_choice_omega_0(self):
#         """
#         Test likeli_one_choice for omega=0.
#         """
#         result_stochastic = likeli_one_choice(
#             pi=0.3,
#             tau=0.2,
#             sigma=0.5,
#             theta=0,
#             omega=0,
#             p=self.default_choice["p"],
#             choice=self.default_choice["choice"],
#         )
#         result_deterministic = choice_without_trembling(
#             pi=0.3, tau=0.2, sigma=0.5, theta=0, p=self.default_choice["p"]
#         )
#         assert_almost_equal(result_stochastic, result_deterministic)
#         # Change parameter such that excepted choice is False
#         result_stochastic = likeli_one_choice(
#             pi=0.3,
#             tau=0.01,
#             sigma=0.5,
#             theta=0,
#             omega=0,
#             p=self.default_choice["p"],
#             choice=self.default_choice["choice"],
#         )
#         result_deterministic = choice_without_trembling(
#             pi=0.3, tau=0.01, sigma=0.5, theta=0, p=self.default_choice["p"]
#         )
#         assert_almost_equal(result_stochastic, result_deterministic)
#     def test_likeli_one_choice(self):
#         """
#         Test likeli_one_choice.
#         """
#         result = likeli_one_choice(
#             pi=0.3,
#             tau=0.2,
#             sigma=0.5,
#             theta=0,
#             omega=0.2,
#             p=self.default_choice["p"],
#             choice=self.default_choice["choice"],
#         )
#         expected_result = 0.9
#         assert_almost_equal(result, expected_result)
#     def test_likeli_one_choice_reg(self):
#         """
#         Test likeli_one_choice. (regression)
#         """
#         result = likeli_one_choice(
#             pi=0.3,
#             tau=0.2,
#             sigma=0.5,
#             theta=0.5,
#             omega=0.2,
#             p=self.default_choice["p"],
#             choice=self.default_choice["choice"],
#         )
#         expected_result = 0.594329137751162
#         assert_almost_equal(result, expected_result)
#     def test_likeli_one_choice_high_theta(self):
#         """
#         Test likeli_one_choice for high theta.
#         """
#         result = likeli_one_choice(
#             pi=0.3,
#             tau=0.2,
#             sigma=0.5,
#             theta=1000000,
#             omega=0.2,
#             p=self.default_choice["p"],
#             choice=self.default_choice["choice"],
#         )
#         expected_result = 0.5
#         assert_almost_equal(result, expected_result)
#     def test_sim_one_choice_omega_0(self):
#         """
#         Test sim_one_choice for omega=0.
#         """
#         para = self.default_para_values.copy()
#         para["omega"] = 0
#         para["theta"] = 0.1
#         result_stochastic = sim_one_choice(para, self.default_choice)[0]
#         result_deterministic = choice_without_trembling(
#             pi=0.3, tau=0.2, sigma=0.5, theta=0, p=self.default_choice["p"]
#         )
#         assert_almost_equal(result_stochastic, result_deterministic)
#     def test_likeli_error_event_level_simple(self):
#         """
#         Test likeli_error_event_level for simple values.
#         """
#         res = likeli_error_event_level(
#             pis=[0], tau=0, sigma=1, omega=0, theta=1, mintervals=np.array([[0, 1000]])
#         )
#         expected_result = 0.5
#         assert_almost_equal(res, expected_result)
#     def test_likeli_error_event_level_reg(self):
#         """
#         Test likeli_error_event_level (regression).
#         """
#         res = likeli_error_event_level(
#             pis=[0.5],
#             tau=0,
#             sigma=1,
#             omega=0,
#             theta=1,
#             mintervals=np.array([[0.2, 0.3]]),
#         )
#         expected_result = 0.0386517
#         assert_almost_equal(res, expected_result)
#     def test_sim_one_agent(self):
#         """
#         Check that sim_choices runs through and returned object has correct shape.
#         """
#         sim_choices = self.agent.sim_choices(paras=self.default_para_values)
#         assert sim_choices.shape[0] > 20
#         assert sim_choices.shape[0] < 29
#     def test_likelihood_event_level_reg(self):
#         para = {}
#         para["pi_0"] = 0
#         para["pi_1"] = 0
#         para["pi_2"] = 0.5
#         para["omega"] = 0
#         para["tau"] = 0.01
#         para["sigma"] = 0.15
#         para["theta"] = 0.5
#         self.agent.error_event_level = True
#         result = self.agent.eval_likelihood(para)
#         expected_result = 6.321941092056277e-12
#         assert_almost_equal(result, expected_result)
#     def test_likelihood_choice_level_reg(self):
#         para = {}
#         para["pi_0"] = 0
#         para["pi_1"] = 0
#         para["pi_2"] = 0.5
#         para["omega"] = 0
#         para["tau"] = 0.01
#         para["sigma"] = 0.15
#         para["theta"] = 0.5
#         self.agent.error_event_level = False
#         result = self.agent.eval_likelihood(para)
#         expected_result = 6.9331538675175545e-06
#         assert_almost_equal(result, expected_result)
#     def test_likelihood_trembling_reg(self):
#         para = {}
#         para["pi_0"] = 0
#         para["pi_1"] = 0
#         para["pi_2"] = 0.5
#         para["omega"] = 0.2
#         para["tau"] = 0.01
#         para["sigma"] = 0.15
#         para["theta"] = 0
#         self.agent.error_event_level = False
#         result = self.agent.eval_likelihood(para)
#         expected_result = 0.0030394163647642332
#         assert_almost_equal(result, expected_result)
# if __name__ == "__main__":
#     unittest.main(verbosity=2)
