"""
Collect some functions for the regressions.
"""
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from pandas.api.types import is_numeric_dtype
from statsmodels.compat.python import lrange
from statsmodels.iolib.summary2 import _col_info
from statsmodels.iolib.summary2 import _col_params
from statsmodels.iolib.summary2 import _make_unique
from statsmodels.iolib.summary2 import Summary

reg_order = [
    "Intercept",
    "k4[T.1]",
    "k4[T.2]",
    "k4[T.3]",
    "k8[T.1]",
    "k8[T.2]",
    "k8[T.3]",
    "k8[T.4]",
    "k8[T.5]",
    "k8[T.6]",
    "k8[T.7]",
    "C(k4, Treatment(reference=3))[T.0]",
    "C(k4, Treatment(reference=3))[T.1]",
    "C(k4, Treatment(reference=3))[T.2]",
    "C(k8, Treatment(reference=7))[T.0]",
    "C(k8, Treatment(reference=7))[T.1]",
    "C(k8, Treatment(reference=7))[T.2]",
    "C(k8, Treatment(reference=7))[T.3]",
    "C(k8, Treatment(reference=7))[T.4]",
    "C(k8, Treatment(reference=7))[T.5]",
    "C(k8, Treatment(reference=7))[T.6]",
    "C(wave)[2]",
    "C(wave)[3]",
    "C(wave)[4]",
    "C(wave)[5]",
    "C(wave)[6]",
    "C(wave)[7]",
    "C(wave)[T.2]",
    "C(wave)[T.3]",
    "C(wave)[T.4]",
    "C(wave)[T.5]",
    "C(wave)[T.6]",
    "C(wave)[T.7]",
    "C(wave)[T.2.0]",
    "C(wave)[T.3.0]",
    "C(wave)[T.4.0]",
    "C(wave)[T.5.0]",
    "C(wave)[T.6.0]",
    "C(wave)[T.7.0]",
    "climate_wave[T.True]",
    "ambig_av_last",
    "ll_insen_last",
    "theta_last",
    "ambig_av_indep_var",
    "ll_insen_indep_var",
    "theta_indep_var",
    "female[T.True]",
    "age_groups[T.B2]",
    "age_groups[T.B3]",
    "age_groups[T.B4]",
    "net_income_groups[T.Q2]",
    "net_income_groups[T.Q3]",
    "net_income_groups[T.Q4]",
    "edu[T.upper_secondary]",
    "edu[T.tertiary]",
    "total_financial_assets_groups[T.Q2]",
    "total_financial_assets_groups[T.Q3]",
    "total_financial_assets_groups[T.Q4]",
    "log_wealth",
    "risk_aversion_index",
    "prob_numeracy",
    "fin_numeracy",
    "basic_numeracy",
    "hist_perf_has_additivity_vio[T.True]",
    "hist_perf_has_set_mono_vio[T.True]",
    "jf_less_hf_avg_abs_dev",
]

demographics_to_proper_names = {
    "age_groups_(0, 35]": "Age: $\\leq 35$",
    "age_groups_(35, 50]": "Age: $\\in (35,50]$",
    "age_groups_(50, 65]": "Age: $\\in (50,65]$",
    "age_groups_(65, 105]": "Age: $ \\geq 65$",
    "age": "Age",
    "age_groups[T.B2]": "Age: $\\in (35,50]$",
    "age_groups[T.B3]": "Age: $\\in (50,65]$",
    "age_groups[T.B4]": "Age: $ \\geq 65$",
    "age_groups_last[T.B2]": "Age: $\\in (35,50]$",
    "age_groups_last[T.B3]": "Age: $\\in (50,65]$",
    "age_groups_last[T.B4]": "Age: $ \\geq 65$",
    "age_groups[T.Q2]": "Age Quartile 2",
    "age_groups[T.Q3]": "Age Quartile 3",
    "age_groups[T.Q4]": "Age Quartile 4",
    "female": "Female",
    "female[T.True]": "Female",
    "female_last": "Female",
    "female_last[T.True]": "Female",
    "edu_lower_secondary_and_lower": "Education: Lower secondary and below",
    "edu_upper_secondary": "Education: Upper secondary",
    "edu_tertiary": "Education: Tertiary",
    "edu[T.upper_secondary]": "Education: Upper secondary",
    "edu[T.tertiary]": "Education: Tertiary",
    "edu_last[T.upper_secondary]": "Education: Upper secondary",
    "edu_last[T.tertiary]": "Education: Tertiary",
    "net_income": "Monthly hh net income (equiv., thousands)",
    "net_income_groups[T.Q2]": "Income: $\\in (1.1, 1.6]$",
    "net_income_groups[T.Q3]": "Income: $\\in (1.6, 2.2]$",
    "net_income_groups[T.Q4]": "Income: $\\geq 2.2$",
    "has_rfa": "Owns risky financial assets",
    "frac_of_tfa_in_rfa": "Share risky financial assets",
    "frac_of_tfa_in_rfa_cond_any": "Share risky financial assets (if any)",
    "wealth": "Wealth (thousands)",
    "log_wealth": "log wealth",
    "total_financial_assets": "Total hh financial assets (equiv., thousands)",
    "total_financial_assets_groups[T.Q2]": "Financial assets: $\\in (1.8, 11.2]$",
    "total_financial_assets_groups[T.Q3]": "Financial assets: $\\in (11.2, 32]$",
    "total_financial_assets_groups[T.Q4]": "Financial assets: $\\geq 32$",
}
parameters_to_proper_names = {
    "ambig_av_w1": "Ambiguity aversion, wave 1",
    "ambig_av_w2": "Ambiguity aversion, wave 2",
    "ll_insen_w1": "Perc. level of ambiguity, wave 1",
    "ll_insen_w2": "Perc. level of ambiguity, wave 2",
    "pi_0_w1": "Subj. Prob. AEX > 0, wave 1",
    "pi_0_w2": "Subj. Prob. AEX > 0, wave 2",
    "theta_w1": "Std of error, wave 1",
    "theta_w2": "Std of error, wave 2",
    "ambig_av": "$\\alpha$",
    "ll_insen": "$\\ell$",
    "theta": "$\\sigma$",
    "ambig_av_index": r"$\alpha^{AEX}_\text{BBLW-Index}$",
    "ll_insen_index": r"$\ell^{AEX}_\text{BBLW-Index}$",
    "ambig_av_last": "$\\alpha$ last wave",
    "ll_insen_last": "$\\ell$ last wave",
    "theta_last": "$\\sigma$ last wave",
    "mp_e0": r"$E^{AEX}_0: Y_{t+6} \in (1000, \infty)$",
    "mp_e1": r"$E^{AEX}_1: Y_{t+6} \in (1100, \infty]$",
    "mp_e2": r"$E^{AEX}_2: Y_{t+6} \in (-\infty, 950)$",
    "mp_e3": r"$E^{AEX}_3: Y_{t+6} \in [950, 1100]$",
    "mp_e1c": r"$E^{AEX}_{1, C}: Y_{t+6} \in (-\infty, 1100]$",
    "mp_e2c": r"$E^{AEX}_{2, C}: Y_{t+6} \in [950, \infty)$",
    "mp_e3c": r"$E^{AEX}_{3, C}: Y_{t+6} \in (-\infty, 950) \cup (1100, \infty)$",
    "mp_e0_temp": r"$E_0: \Delta T > 0^\circ C$",
    "mp_e1_temp": r"$E_1: \Delta T > 1^\circ C$",
    "mp_e1c_temp": r"$E^C_1: \Delta T \leq 1^\circ C$",
    "mp_e2_temp": r"$E_2: \Delta T < -0.5^\circ C$",
    "mp_e2c_temp": r"$E^C_2: \Delta T \geq -0.5^\circ C$",
    "mp_e3_temp": r"$E_3: -0.5^\circ C \leq \Delta T \leq 1^\circ C$",
    "mp_e3c_temp": r"$E^C_3: (\Delta T < -0.5^\circ C) \cup (\Delta T > 1^\circ C)$",
    "w4": "AEX param wave 4",
    "w3": "AEX param wave 3",
    # "ll_insen": "AEX param",
    # "theta": "AEX param",
}


judged_freq_to_proper_names = {
    "hist_perf_has_additivity_vio": "Judged hist. freqs. nonadditive",
    "hist_perf_has_set_mono_vio": "Judged hist. freqs. have set mono. viol.",
    "jf_less_hf_avg_abs_dev": "Judged hist. freqs: mean absolute deviation",
    "hist_perf_e0_nocheck": "Judged hist. freq: positive return",
    "hist_perf_e1_combined": "Judged hist. freq: $r > 10\\,\\%$",
    "hist_perf_e2_combined": "Judged hist. freq: $r < -5\\,\\%$",
    "hist_perf_response_error": "Judged hist. freq: any response error",
}

special_attributes_to_proper_names = {
    "risk_aversion_index": "Risk aversion index",
    "general_risk_q": "Risk aversion (qualitative)",
    "quantitative_risk_q": "Risk aversion (quantitative)",
    "risk_aversion_quant": "Risk aversion (quantitative)",
    "risk_aversion_qual": "Risk aversion (qualitative)",
    "basic_numeracy": "Basic numeracy",
    "prob_numeracy": "Probability numeracy",
    "fin_numeracy": "Financial numeracy",
    "numeracy_index": "Numeracy index",
    "optimism_pessimsm": "Optimism",
    "understands_climate_change": "Understands climate change",
    "threatened_by_climate_change": "Threatened by climate change",
}
other_to_proper_names = {
    "p_won_20_eur": "Average money Won",
    "duration_in_m": "Mean duration (minutes)",
    "C(wave)[2]": "Nov. 2018",
    "C(wave)[3]": "May 2019",
    "C(wave)[4]": "Nov. 2019",
    "C(wave)[5]": "May 2020",
    "C(wave)[6]": "Nov. 2020",
    "C(wave)[7]": "May 2021",
    "C(wave)[T.2]": "2018-11",
    "C(wave)[T.3]": "2019-05",
    "C(wave)[T.4]": "2019-11",
    "C(wave)[T.5]": "2020-05",
    "C(wave)[T.6]": "2020-11",
    "C(wave)[T.7]": "2021-05",
    "C(wave)[T.2.0]": "2018-11",
    "C(wave)[T.3.0]": "2019-05",
    "C(wave)[T.4.0]": "2019-11",
    "C(wave)[T.5.0]": "2020-05",
    "C(wave)[T.6.0]": "2020-11",
    "C(wave)[T.7.0]": "2021-05",
    "climate_wave[T.True]": "Climate wave",
}

variable_to_proper_name = {
    **demographics_to_proper_names,
    **parameters_to_proper_names,
    **judged_freq_to_proper_names,
    **special_attributes_to_proper_names,
    **other_to_proper_names,
}

col_name_to_proper_name = {
    # "ambig_av": r"$\Delta$ Ambiguity aversion ($\alpha$)",
    # "ll_insen": r"$\Delta$ Perc. level of ambiguity ($\ell$)",
    # "pi_0": "Subj. Prob. AEX > 0",
    # "theta": r" $\Delta$ Model error ($\sigma$)",
    "participated I": "Participated in a wave",
    "participated II": "Participated in a wave",
    "ambig_av": r"$\alpha$",
    "ambig_av I": r"$\alpha$",
    "ambig_av II": r"$\alpha$",
    "ambig_av III": r"$\alpha$",
    "ll_insen": r"$\ell$",
    "ll_insen I": r"$\ell$",
    "ll_insen II": r"$\ell$",
    "ll_insen III": r"$\ell$",
    "ambig_av_index": r"$\alpha^{AEX}_\text{BBLW-Index}$",
    "ll_insen_index": r"$\ell^{AEX}_\text{BBLW-Index}$",
    "pi_0": "Subj. Prob. AEX > 0",
    "theta": r"$\sigma$",
    "theta I": r"$\sigma$",
    "theta II": r"$\sigma$",
    "theta III": r"$\sigma$",
    "risk_aversion_index": "Risk aversion index",
    "numeracy_index": "Numeracy index",
    "prob_numeracy": "Probability Numeracy",
    "fin_numeracy": "Financial Numeracy",
    "basic_numeracy": "Basic Numeracy",
    "general_risk_q": "Risk aversion (qualitative)",
    "quantitative_risk_q": "Risk aversion (quantitative)",
    "has_rfa": "Owns risky financial assets",
    "has_rfa I": "Owns risky financial assets",
    "has_rfa II": "Owns risky financial assets",
    "has_rfa III": "Owns risky financial assets",
    "frac_of_tfa_in_rfa": "Share risky financial assets",
    "frac_of_tfa_in_rfa I": "Share risky financial assets",
    "frac_of_tfa_in_rfa II": "Share risky financial assets",
    "frac_of_tfa_in_rfa III": "Share risky financial assets",
    "share_rfa_of_tfa": "Share risky financial assets",
    "share_rfa_of_tfa I": "Share risky financial assets",
    "share_rfa_of_tfa II": "Share risky financial assets",
    "share_rfa_of_tfa III": "Share risky financial assets",
}

# Specifications of ambiguity types
near_seu_spec = {"name": "Near SEU", "color": "#1f77b4", "marker": "D"}
ambig_av_spec = {"name": "Ambiguity averse", "color": "#d62728", "marker": "^"}
ambig_seek_spec = {"name": "Ambiguity seeking", "color": "#ff7f0e", "marker": "o"}
high_noise_spec = {"name": "High noise", "color": "#636363", "marker": "X"}

standard_ambig_spec = [
    dict(near_seu_spec, **{"orig_order": 3}),
    dict(ambig_av_spec, **{"orig_order": 0}),
    dict(ambig_seek_spec, **{"orig_order": 1}),
    dict(high_noise_spec, **{"orig_order": 2}),
]

ambig_groups_spec = {
    "event_level_het_prob_2_7": {
        "k3": [
            {
                "name": "Ambiguity seeking / near SEU",
                "color": "#2ca02c",
                "marker": "s",
                "orig_order": 2,
            },
            dict(ambig_av_spec, **{"orig_order": 0}),
            dict(high_noise_spec, **{"orig_order": 1}),
        ],
        "k4": standard_ambig_spec,
        "k5": [
            dict(near_seu_spec, **{"orig_order": 4}),
            dict(ambig_av_spec, **{"orig_order": 0}),
            dict(ambig_seek_spec, **{"orig_order": 1}),
            {
                "name": "Ambiguity averse / high noise",
                "color": "#8c564b",
                "marker": "v",
                "orig_order": 2,
            },
            dict(high_noise_spec, **{"orig_order": 3}),
        ],
        "k8": [
            dict(near_seu_spec, **{"orig_order": 7}),
            {
                "name": "Near SEU / ambiguity averse",
                "color": "#9467bd",
                "marker": "P",
                "orig_order": 5,
            },
            {
                "name": "Near SEU / ambiguity seeking",
                "color": "#2ca02c",
                "marker": "s",
                "orig_order": 4,
            },
            dict(ambig_av_spec, **{"orig_order": 0}),
            {
                "name": "Somewhat ambiguity averse",
                "color": "#e377c2",
                "marker": "h",
                "orig_order": 1,
            },
            dict(ambig_seek_spec, **{"orig_order": 2}),
            {
                "name": "Ambiguity averse / high noise",
                "color": "#8c564b",
                "marker": "v",
                "orig_order": 3,
            },
            dict(high_noise_spec, **{"orig_order": 6}),
        ],
    },
    "event_level_het_prob_2_7_unrestricted_above_sigma": {
        "k4": standard_ambig_spec,
    },
    "event_level_het_prob_2_7_all_obs": {
        "k4": standard_ambig_spec,
    },
    "event_level_het_prob_2_7_balanced_panel": {
        "k4": standard_ambig_spec,
    },
    "indices_mean": {
        "k3": [
            dict(near_seu_spec, **{"orig_order": 2}),
            dict(ambig_av_spec, **{"orig_order": 1}),
            dict(ambig_seek_spec, **{"orig_order": 0}),
        ],
        "k4": [
            dict(near_seu_spec, **{"orig_order": 3}),
            dict(ambig_av_spec, **{"orig_order": 2}),
            dict(ambig_seek_spec, **{"orig_order": 1}),
            {
                "name": "Monotonicity violating",
                "color": "#636363",
                "marker": "X",
                "orig_order": 0,
            },
        ],
        "k5": [
            dict(near_seu_spec, **{"orig_order": 4}),
            {
                "name": "Somewhat ambiguity averse",
                "color": "#e377c2",
                "marker": "h",
                "orig_order": 3,
            },
            dict(ambig_av_spec, **{"orig_order": 1}),
            dict(ambig_seek_spec, **{"orig_order": 2}),
            {
                "name": "Monotonicity violating",
                "color": "#636363",
                "marker": "X",
                "orig_order": 0,
            },
        ],
    },
    "indices_single_waves": {
        "k3": [
            dict(near_seu_spec, **{"orig_order": 2}),
            dict(ambig_av_spec, **{"orig_order": 1}),
            dict(ambig_seek_spec, **{"orig_order": 0}),
        ],
        "k4": [
            dict(near_seu_spec, **{"orig_order": 3}),
            dict(ambig_av_spec, **{"orig_order": 1}),
            dict(ambig_seek_spec, **{"orig_order": 2}),
            {
                "name": "Monotonicity violating",
                "color": "#636363",
                "marker": "X",
                "orig_order": 0,
            },
        ],
        "k5": [
            dict(near_seu_spec, **{"orig_order": 4}),
            {
                "name": "Somewhat ambiguity averse",
                "color": "#e377c2",
                "marker": "h",
                "orig_order": 2,
            },
            dict(ambig_av_spec, **{"orig_order": 1}),
            dict(ambig_seek_spec, **{"orig_order": 3}),
            {
                "name": "Monotonicity violating",
                "color": "#636363",
                "marker": "X",
                "orig_order": 0,
            },
        ],
    },
}


def std_normalise(x):
    return (x - x.mean()) / x.std()


def ambig_to_rgb(ambig_av, ll_insen):
    """
    Map ambiguity profile to an rgb colour profile with each value in [0,1] s.t.

    More aversion = more red
    More seeking = more green
    More rationality = more blue

    """
    av = ambig_av + 0.5
    # 1 = identity mapping. larger values give S shape with steeper middle section
    # = sharper colour contrasts between ambiguity profiles in middle of triangle
    delta = 3

    def s_shaped_function(x):
        return 1 / (1 + (x / (1 - x)) ** (-delta))

    r = s_shaped_function(av)
    g = s_shaped_function(1 - av)
    b = s_shaped_function(1 - ll_insen)
    return (r, g, b)


def make_fancy_scatterplot(
    predictor,
    outcome,
    df,
    controls,
    scatter_col="black",
    regline_col="blue",
    lowess=False,
    ax=None,
):
    """
    Makes a scatter plot with least squares line.
    Optional: Partialling out control variables, adding lowess line.
    """
    predictor_vals = df[predictor]
    outcome_vals = df[outcome]
    if len(controls) > 0:
        outcome_vals = smf.ols(f"{outcome} ~ {'+'.join(controls)}", data=df).fit().resid
        predictor_vals = (
            smf.ols(f"{predictor} ~ {'+'.join(controls)}", data=df).fit().resid
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    sns.regplot(
        x=predictor_vals,
        y=outcome_vals,
        fit_reg=False,
        scatter_kws={"s": 1, "color": scatter_col, "marker": ".", "alpha": 0.3},
        ax=ax,
    )
    sns.regplot(
        x=predictor_vals,
        y=outcome_vals,
        scatter=False,
        line_kws={"lw": 2, "color": regline_col},
        ax=ax,
    )
    if lowess:
        sns.regplot(
            x=predictor_vals,
            y=outcome_vals,
            scatter=False,
            lowess=True,
            line_kws={"lw": 2, "ls": "--", "color": regline_col},
            ax=ax,
        )
    ax.set_xlabel(predictor)
    ax.set_ylabel(outcome)


def calc_upwards_sloping_interval(tau, sigma):
    """
    Finds the interval over which m(p) rises in the [0, 1] x [0, 1] quadrant
    """
    if sigma > 0:
        p_l = min(max(-(tau / sigma), 0), 1)
        p_u = min(max((1 - tau) / sigma, 0), 1)
    else:
        p_l = 0
        p_u = 1
    return p_l, p_u


def calc_generalised_ambig_av(tau, sigma):
    """
    Ambiguity aversion = Integrate/Average (p - m(p)) in the [0,1] x [0,1] quadrant.
    = area under the 45 deg line (0.5) less area under the m(p) curve inside the quadrant.
    """
    p_l, p_u = calc_upwards_sloping_interval(tau, sigma)

    def M(x):
        return tau * x + sigma * x**2 / 2  # antiderivative of m(p) = tau + sigma * p

    rectangle = (
        1 - p_u
    )  # if m(p) = 1 before p=1, the rectangle until p=1 is part of the area under m(p)

    return 0.5 - (M(p_u) - M(p_l) + rectangle)


def calc_generalised_ll_insen(tau, sigma, interval=(0.05, 0.95)):
    """
    Liklihood insensitivity = 1 - average increase over (l, u)
    = 1 -  (m(u) - m(l)) / (u - l)
    Baillon et al p.11 footnote suggests l=0.05, u=0.95
    """
    lower = interval[0]
    upper = interval[1]

    def source_func(x):
        return min(max(tau + sigma * x, 0), 1)

    return 1 - (source_func(upper) - source_func(lower)) / (upper - lower)


def summary_col(
    results,
    float_format="%.4f",
    model_names=(),
    stars=False,
    info_dict=None,
    regressor_order=(),
    drop_omitted=False,
):
    """
    Summarize multiple results instances side-by-side (coefs and SEs)

    Parameters
    ----------
    results : statsmodels results instance or list of result instances
    float_format : str, optional
        float format for coefficients and standard errors
        Default : '%.4f'
    model_names : list[str], optional
        Must have same length as the number of results. If the names are not
        unique, a roman number will be appended to all model names
    stars : bool
        print significance stars
    info_dict : dict
        dict of functions to be applied to results instances to retrieve
        model info. To use specific information for different models, add a
        (nested) info_dict with model name as the key.
        Example: `info_dict = {"N":lambda x:(x.nobs), "R2": ..., "OLS":{
        "R2":...}}` would only show `R2` for OLS regression models, but
        additionally `N` for all other results.
        Default : None (use the info_dict specified in
        result.default_model_infos, if this property exists)
    regressor_order : list[str], optional
        list of names of the regressors in the desired order. All regressors
        not specified will be appended to the end of the list.
    drop_omitted : bool, optional
        Includes regressors that are not specified in regressor_order. If
        False, regressors not specified will be appended to end of the list.
        If True, only regressors in regressor_order will be included.
    """

    if not isinstance(results, list):
        results = [results]

    cols = [_col_params(x, stars=stars, float_format=float_format) for x in results]

    # New: drop R-squared and R-squared adjusted
    def drop_r_squared(c):
        if "R-squared" in c.index.get_level_values(0):
            c = c.drop(["R-squared", "R-squared Adj."], level=0)
        return c

    cols = [drop_r_squared(c) for c in cols]

    # Unique column names (pandas has problems merging otherwise)
    if model_names:
        colnames = _make_unique(model_names)
    else:
        colnames = _make_unique([x.columns[0] for x in cols])
    for i in range(len(cols)):
        cols[i].columns = [colnames[i]]

    def merg(x, y):
        return x.merge(y, how="outer", right_index=True, left_index=True)

    summ = reduce(merg, cols)
    if regressor_order:
        varnames = summ.index.get_level_values(0).tolist()
        ordered = [x for x in regressor_order if x in varnames]
        unordered = [x for x in varnames if x not in regressor_order + [""]]
        order = ordered
        if drop_omitted and len(unordered) > 0:
            print("the following columns have been dropped", np.unique(unordered))
        else:
            order += list(np.unique(unordered))

        def f(idx):
            return sum([[x + "coef", x + "stde"] for x in idx], [])

        summ.index = f(pd.unique(varnames))
        summ = summ.reindex(f(order))
        summ.index = [x[:-4] for x in summ.index]

    idx = pd.Series(lrange(summ.shape[0])) % 2 == 1
    summ.index = np.where(idx, "", summ.index.get_level_values(0))

    # add infos about the models.
    if info_dict:
        # print(info_dict)
        # print(results)
        # x = results[0]
        # print(getattr(x, "default_model_infos", None))
        # print(x.model.__class__.__name__)
        cols = [
            _col_info(x, info_dict.get(x.model.__class__.__name__, info_dict))
            for x in results
        ]
    else:
        cols = [_col_info(x, getattr(x, "default_model_infos", None)) for x in results]

    # use unique column names, otherwise the merge will not succeed
    for df, name in zip(cols, _make_unique([df.columns[0] for df in cols])):
        df.columns = [name]

    # def merg(x, y):
    #     return x.merge(y, how="outer", right_index=True, left_index=True)

    info = reduce(merg, cols)
    dat = pd.DataFrame(np.vstack([summ, info]))  # pd.concat better, but error
    dat.columns = summ.columns
    dat.index = pd.Index(summ.index.tolist() + info.index.tolist())
    summ = dat

    summ = summ.fillna("")

    smry = Summary()
    smry._merge_latex = True
    smry.add_df(summ, header=True, align="l")
    smry.add_text("Standard errors in parentheses.")
    if stars:
        smry.add_text("* p<.1, ** p<.05, ***p<.01")

    return smry


def wave_from_string(model_name):
    """Return short wave from model name. Either integer, temp or pooled"""
    if "_w" in model_name:
        val = model_name.split("_w")[-1].strip("w")[0]
        return int(val)
    elif "temp" in model_name:
        return "temp"
    else:
        return "pooled"


def merge_model_results(models, in_path_dict):
    """
    Collects model result columns in a dataframe with index ["personal_id", "wave"]
    """

    dfs = []
    for m in models:
        path = in_path_dict[m]
        df = pd.read_pickle(path)
        df["wave"] = wave_from_string(m)
        dfs.append(df)

    results = pd.concat(dfs, axis=0, sort=False)
    results.index.name = "personal_id"
    results = results.reset_index().set_index(["personal_id", "wave"])
    results.sort_index(inplace=True)
    if {"pi_1", "pi_2"} in set(results.columns):
        results["pi_3"] = 1 - results["pi_1"] - results["pi_2"]

    results["ll_insen_old"] = 1 - results["sigma"]
    results["ambig_av_old"] = 1 - results["sigma"] - 2 * results["tau"]

    # NB We scale ambiguity aversion to lie in -0.5 to 0.5 instead of [-1, 1]
    results["ambig_av_old"] *= 1 / 2

    # generalised
    temp_df = results[["tau", "sigma"]]
    results["ll_insen"] = temp_df.apply(
        lambda x: calc_generalised_ll_insen(x.iloc[0], x.iloc[1]), axis=1
    )
    results["ambig_av"] = temp_df.apply(
        lambda x: calc_generalised_ambig_av(x.iloc[0], x.iloc[1]), axis=1
    )
    results["rise_interval_midp"] = temp_df.apply(
        lambda x: np.array(calc_upwards_sloping_interval(x.iloc[0], x.iloc[1])).mean(),
        axis=1,
    )

    # rename standard error to theta if necessary
    if "std_error" in results.columns:
        results.rename(columns={"std_error": "theta"}, inplace=True)

    return results


def regression_table_wrapper(
    models,
    regressor_order=None,
    reg_to_table=None,
    depvar_to_table=None,
    prec=2,
    drop_omitted=False,
):
    """
    A wrapper for statsmodels summary_cols
    """

    basic_table = summary_col(
        results=models,
        regressor_order=regressor_order,
        float_format=f"%.{prec}f",
        stars=True,
        info_dict={
            "N": lambda x: "{:d}".format(int(x.nobs)),
            "$R^2$": lambda x: f"{x.rsquared:.3f}",
        },
        drop_omitted=drop_omitted,
    ).tables[0]
    if reg_to_table:
        basic_table.rename(index=reg_to_table, inplace=True)
    if depvar_to_table:
        basic_table.rename(columns=depvar_to_table, inplace=True)
    return basic_table


def put_reg_sample_together(
    in_path_dict,
    asset_calc,
    restrictions,
    models,
    indices=False,
    indices_mean=False,
    var_standardize=None,
    var_normalize=None,
    var_to_qu=None,
    var_to_bins=None,
):
    """
    Put together a sample for regressions.
    :para: models: List of model names (if indices=False) or list of waves (if indices=True)
    :para: indices_mean: Whether the mean over all indices estimates should be returned
    :para: var_standardize: List of variables to be standardized
    :para: var_normalize: List of variables to be standardnormalized
    :para: var_to_qu: Dictionary with keys variables and values number of quantiles
    :para: var_to_bins: Dictionary with keys variables and values list of thresholds for binning
    :
    """
    if not var_standardize:
        var_standardize = []
    if not var_normalize:
        var_normalize = [
            "risk_aversion_index",
            "general_risk_q",
            "quantitative_risk_q",
            "prob_numeracy",
            "fin_numeracy",
            "basic_numeracy",
            "numeracy_index",
            "optimism_pessimsm",
        ]
    if not var_to_qu:
        var_to_qu = {"wealth": 4, "total_financial_assets": 4, "net_income": 4}
    if not var_to_bins:
        var_to_bins = {"age": [0, 35, 50, 65, 150]}
    individual = pd.read_pickle(in_path_dict["individual"])

    # Choose the assets variables of selected asset_calc specification
    individual = choose_asset_calc(individual, asset_calc)

    if indices:
        indices = pd.read_pickle(in_path_dict["indices"])
        results = indices.loc[pd.IndexSlice[:, models], :]
        if indices_mean:
            results = results.groupby("personal_id").mean()
            results["wave"] = "pooled"
    else:
        results = (
            merge_model_results(in_path_dict=in_path_dict, models=models)
            .reset_index()
            .set_index("personal_id")
        )
    data = results.join(individual, how="outer")
    data.index.name = "personal_id"

    # implement sample restrictions
    sample_restrictions = pd.read_pickle(in_path_dict["sample_restrictions"])
    data = data.join(sample_restrictions, how="right")
    data = data.query(restrictions).copy()
    data = data.reset_index().set_index(["personal_id", "wave"])

    data = standardize_variables(
        var_standardize=[v for v in var_standardize if v not in var_normalize],
        var_normalize=var_normalize,
        data=data,
    )

    for v, qu in var_to_qu.items():
        grouped = data[v].groupby("personal_id").mean()
        grouped_var = pd.qcut(
            data[v],
            q=qu,
            labels=[f"Q{i + 1}" for i in range(qu)],  # , duplicates="drop"
        )
        grouped_var.name = v + "_groups"
        data = data.join(pd.DataFrame(grouped_var))

    for v, bins in var_to_bins.items():
        grouped = data[v].groupby("personal_id").mean()
        grouped_var = pd.cut(
            grouped,
            bins=bins,
            labels=[f"B{i + 1}" for i in range(len(bins) - 1)],  # , duplicates="drop"
        )
        grouped_var.name = v + "_groups"
        data = data.join(pd.DataFrame(grouped_var))

    data = take_logs(data)

    return data


def take_logs(data):
    """
    Take logs of some financial variables.
    """
    log_variables = ["total_financial_assets", "wealth", "net_income"]
    for v in log_variables:
        data["log_" + v] = np.log(data[v].where(data[v] > 0))
    return data


def standardize_variables(var_standardize, var_normalize, data):
    """
    Normalize and standardize some variables.

    """
    data_copy = data.copy()
    individual_level_data = data_copy.groupby("personal_id").mean()

    for v in var_normalize:
        variable = individual_level_data[v].dropna()
        data_copy[v] = (data_copy[v] - variable.mean()) / variable.std()

    for v in var_standardize:
        variable = individual_level_data[v].dropna()
        data_copy[v] = (data_copy[v]) / variable.std()

    return data_copy


def add_midrules_to_latex(out, rows, midrule_text=r"\midrule"):
    # Add midrules
    latex_list = out.splitlines()
    for row in rows:
        latex_list.insert(row, midrule_text)

    # join split lines to get the modified latex output string
    out = "\n".join(latex_list)
    return out


def select_group_label(pooled_model_name, n_groups, group_n):
    if (
        pooled_model_name in ambig_groups_spec
        and n_groups in ambig_groups_spec[pooled_model_name]
    ):
        group_label = ambig_groups_spec[pooled_model_name][n_groups][group_n]["name"]
    else:
        group_label = f"Group {group_n + 1}"

    return group_label


def select_group_colors(pooled_model_name, n_groups_str):
    if (
        pooled_model_name in ambig_groups_spec
        and n_groups_str in ambig_groups_spec[pooled_model_name]
    ):
        out = {
            i: group["color"]
            for i, group in enumerate(
                ambig_groups_spec[pooled_model_name][n_groups_str]
            )
        }
    else:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        n_groups = int(n_groups_str[1:])
        out = {g: colors[g] for g in range(n_groups)}
    return out


def select_manual_group_order(pooled_model_name, n_groups_str):
    if (
        pooled_model_name in ambig_groups_spec
        and n_groups_str in ambig_groups_spec[pooled_model_name]
    ):
        out = {
            i: group["orig_order"]
            for i, group in enumerate(
                ambig_groups_spec[pooled_model_name][n_groups_str]
            )
        }
    else:
        n_groups = int(n_groups_str[1:])
        out = {g: g for g in range(n_groups)}

    return out


def select_group_marker(pooled_model_name, n_groups_str):
    if (
        pooled_model_name in ambig_groups_spec
        and n_groups_str in ambig_groups_spec[pooled_model_name]
    ):
        out = {
            i: group["marker"]
            for i, group in enumerate(
                ambig_groups_spec[pooled_model_name][n_groups_str]
            )
        }
    else:
        n_groups = int(n_groups_str[1:])
        out = {g: "D" for g in range(n_groups)}

    return out


def choose_asset_calc(df, asset_calc):
    """
    Choose investment variables based on selected asset_calc
    """
    assert asset_calc in [
        "_ind_first",
        "_ind_last",
        "_ind_mean",
        "_com_first",
        "_com_last",
        "_com_mean",
        "_comnoadj_last",
        "_com_mean_inc_ind_mean_assets",
    ]

    asset_vars = [
        "has_rfa_wide_def",
        "has_rfa",
        "risky_financial_assets",
        "total_financial_assets",
        "wealth",
        "frac_of_tfa_in_rfa",
        "frac_of_tfa_in_rfa_cond_any",
    ]
    income_vars = [
        "net_income",
    ]

    if asset_calc == "_com_mean_inc_ind_mean_assets":
        for c in asset_vars:
            df[c] = df[c + "_indadj_mean"]
        for c in income_vars:
            df[c] = df[c + "_com_mean"]
    else:
        for c in asset_vars + income_vars:
            df[c] = df[c + asset_calc]

    return df


def make_diagnosis_plots(person, combined_results):
    fig, ax = plt.subplots(1, 4, figsize=(40, 10))
    for w in [1, 2, 3, 4]:
        try:
            individual_data = combined_results.loc[person].loc[w]
            make_diagnosis_plot(
                individual_data,
                variants=["idx", "res", "unres", "runres_ls"],
                ax=ax[w - 1],
            )
            ax[w - 1].set_title(f"Wave {w}", fontsize=14)
        # Not sure if correct Error type
        except ValueError:
            print(f"Oh dear looks like wave {w} isn't available for person {person}")


def make_diagnosis_plot(
    individual_data,
    variants=None,
    ax=None,
):
    if not variants:
        variants = ["idx", "res_het", "unres_het", "runres_ls", "unres_het_ls"]

    def matching_prob_fun(x, tau, sigma):
        return max(min(tau + sigma * x, 1), 0)

    matching_prob_fun = np.vectorize(matching_prob_fun, excluded=["tau", "sigma"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    p = np.linspace(0, 1, 100)
    ax.plot(p, matching_prob_fun(p, 0, 1), color="black")

    mprobs = ["mp_e1", "mp_e1c", "mp_e2", "mp_e2c", "mp_e3", "mp_e3c"]
    for i, mp in enumerate(mprobs):
        matching_prob = individual_data.loc[mp]
        # ax.axhline(matching_prob, color="gray", alpha=0.5, lw=4, ls="--")
        # ax.annotate(s=f"{mp}", xy=(0, matching_prob + 0.01))
        ax.annotate(s=f"{mp}    {matching_prob:.2f}", xy=(0, 1 - 3 * i / 100))

    for j, vari in enumerate(variants):
        tau, sigma = individual_data.loc[[f"tau_{vari}", f"sigma_{vari}"]]
        ax.plot(p, matching_prob_fun(p, tau, sigma), lw=4, label=vari, alpha=0.7)
        ax.annotate(s=f"{vari}", xy=(0.2 + j * 0.15, 1))
        for k in [1, 2]:
            p_var = f"pi_{k}_{vari}" if vari != "idx" else f"subj_p_e{k}"
            prob = individual_data.loc[p_var]

            ax.annotate(s=f"p_e{k}  {prob:.2f}", xy=(0.2 + j * 0.15, 1 - 3 * k / 100))
        if vari != "idx":
            ax.annotate(
                s=r"$\sigma$" + f"      {individual_data.loc[f'theta_{vari}']:.2f}",
                xy=(0.2 + j * 0.15, 1 - 3 * 3 / 100),
            )
    ax.legend(loc="right")
    ax.set_xlabel("Subjective probability", fontsize=15)
    ax.set_ylabel("Matching probability", fontsize=15)
    ax.tick_params(labelsize=13)


def apply_number_format_to_series(series, number_format):
    """Apply string format to a pandas Series."""
    formatted = series.copy(deep=True).astype("float")
    for formatter in number_format[:-1]:
        formatted = formatted.apply(formatter.format).astype("float")
    formatted = formatted.astype("float").apply(number_format[-1].format)
    return formatted


def _add_multicolumn_left_format_to_column(column):
    """Align oservation numbers at the center of model column."""
    out = column.replace(
        {i: f"\\multicolumn{{1}}{{r}}{{{i}}}" for i in column.unique()}
    )
    return out


def apply_custom_number_format(data, int_cols, number_format):
    out = data.copy()
    for c in int_cols:
        out[c] = out[c].apply(lambda x: f"{x:.0f}")
        out[c] = _add_multicolumn_left_format_to_column(out[c])

    for c in out:
        if c not in int_cols and is_numeric_dtype(data[c]):
            out[c] = apply_number_format_to_series(out[c], number_format)

    out = out.replace({"nan": ""})
    return out
