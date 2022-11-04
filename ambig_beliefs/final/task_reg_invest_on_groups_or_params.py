"""
Runs regressions of model results on individual characteristics
"""
import json

import estimagic.visualization.estimation_table as et
import numpy as np
import pandas as pd
import pytask
import statsmodels.formula.api as smf

from ambig_beliefs.final.utils_final import add_midrules_to_latex
from ambig_beliefs.final.utils_final import col_name_to_proper_name
from ambig_beliefs.final.utils_final import put_reg_sample_together
from ambig_beliefs.final.utils_final import select_group_label
from ambig_beliefs.final.utils_final import select_manual_group_order
from ambig_beliefs.final.utils_final import variable_to_proper_name
from config import BASIC_CONTROLS
from config import IN_DATA
from config import MODEL_SPECS
from config import NAMES_INDICES_SPEC
from config import NAMES_MAIN_SPEC
from config import NAMES_ROBUSTNESS_SPEC
from config import OUT_ANALYSIS
from config import OUT_DATA
from config import OUT_DATA_LISS
from config import OUT_TABLES
from config import OUT_UNDER_GIT
from config import ROOT


# def _extract_params_from_sm_margin(model):
#     """Convert statsmodels DiscreteMargins like estimation result to estimagic like
#     params dataframe."""

#     params_df = (
#         model.summary_frame()
#         .rename(
#             columns={
#                 "dy/dx": "value",
#                 "Std. Err.": "standard_error",
#                 "Pr(>|z|)": "p_value",
#                 "Conf. Int. Low": "ci_lower",
#                 "Cont. Int. Hi.": "ci_upper",
#             }
#         )
#         .drop("z", axis=1)
#     )

#     return params_df


# def _extract_info_from_sm_margin(model):
#     """Process statsmodels DiscreteMargins estimation result to retrieve summary
#     statistics as dict."""
#     info = {}
#     key_values = [
#         "prsquared",
#         "df_model",
#         "df_resid",
#     ]
#     for kv in key_values:
#         info[kv] = getattr(model.results, kv)
#     info["name"] = model.results.model.endog_names
#     info["n_obs"] = model.results.df_model + model.results.df_resid + 1
#     return info


# def _extract_params_from_tobit_r_margin(model):
#     """Convert R censReg margeff like estimation result to estimagic like params dataframe."""

#     # Adjust dataframe as expected by estimagic
#     col_renaming = {
#         "estimate": "value",
#         "std.error": "standard_error",
#         # "statistic": "t",
#         "p.value": "p_value",
#         # "conf.low": "ci_lower",
#         # "conf.high": "ci_upper",
#     }
#     model = model.rename(columns=col_renaming)
#     out = model[[col_renaming.values()]]
#     return out


# def _extract_info_from_tobit_r_margin(model):
#     """Process R censReg margeff estimation result to retrieve summary statistics as dict."""
#     info = pandas2ri.conversion.rpy2py(model).iloc[0]
#     info = info.rename({"nObs": "n_obs", "pseudo_R2": "prsquared"})

#     return info.to_dict()


def _extract_params_from_r_marginaleffects(model):
    """Convert R marginaleffects like estimation result to estimagic like params dataframe."""

    # Adjust dataframe as expected by estimagic
    col_renaming = {
        "estimate": "value",
        "std.error": "standard_error",
        "statistic": "t",
        "p.value": "p_value",
        "conf.low": "ci_lower",
        "conf.high": "ci_upper",
    }
    model = model.rename(columns=col_renaming)
    out = model[col_renaming.values()]
    return out


def reg_invest_on_groups(data, ga, controls, produces, m, asset_calc):
    k = len(data[ga].unique())
    ga_string = "C(type_man_sort, Treatment(reference=0))"

    group_label_dict = {
        f"{ga_string}[T.{g}]": f"{select_group_label(m, ga, g)} type"
        for g in range(0, k)
    }
    group_label_dict[
        "Intercept"
    ] = f"Intercept (left-out type: {select_group_label(m, ga, 0)})"

    models = []
    temp = data.copy()
    dependent_vars = ["has_rfa", "frac_of_tfa_in_rfa"]
    for dep_var in dependent_vars:
        models.append(
            smf.ols(formula=f"{dep_var} ~ {ga_string}", data=temp).fit(cov_type="HC3")
        )
        models.append(
            smf.ols(
                formula=f"{dep_var} ~ {ga_string} + " + "+".join(controls), data=temp
            ).fit(cov_type="HC3")
        )
    models_prep = [
        {
            "params": et._extract_params_from_sm(m),
            "info": et._extract_info_from_sm(m),
            "name": et._extract_info_from_sm(m).pop("name"),
        }
        for m in models
    ]
    for model_p in models_prep:
        model_p["info"]["mean_dep_var"] = temp[model_p["info"]["name"]].mean()

    out = et.estimation_table(
        models_prep,
        return_type="render_inputs",
        add_trailing_zeros=False,
        siunitx_warning=False,
        custom_param_names={**variable_to_proper_name, **group_label_dict},
        custom_col_groups=col_name_to_proper_name,
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
        stats_options={
            "mean_dep_var": "Mean dependent variable",
            "n_obs": "Observations",
            "rsquared": "R$^2$",
            "rsquared_adj": "Adj. R$^2$",
            "show_dof": None,
        },
    )

    out_latex = et.render_latex(
        out["body"],
        out["footer"],
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    path_out = produces[f"invest_on_groups_{ga}{asset_calc}"]
    with open(path_out, "w") as my_table:
        my_table.write(out_latex)

    # Without controls
    new_row = pd.DataFrame(
        [["No", "Yes"] * 2],
        columns=out["footer"].columns,
        index=pd.MultiIndex.from_arrays([["Controls"]]),
    )
    out["footer"] = pd.concat([new_row, out["footer"]])

    out_no_controls = out["body"].iloc[: k * 2, :].copy()

    # Don't show intercept if controls are used
    out_no_controls.iloc[[0, 1], [1, 3]] = ""

    out_latex = et.render_latex(
        out_no_controls,
        out["footer"],
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    path_out = produces[f"invest_on_groups_{ga}{asset_calc}_without_cont"]
    with open(path_out, "w") as my_table:
        my_table.write(out_latex)

    return out


def reg_invest_on_groups_probit_tobit(k, ga, depends_on, produces, m, asset_calc):

    ga_string = "C(type_man_sort, Treatment(reference=0))"
    group_label_dict = {
        f"{ga_string}[T.{g}]": f"{select_group_label(m, ga, g)} type"
        for g in range(0, k)
    }
    group_label_dict[
        "Intercept"
    ] = f"Intercept (left-out type: {select_group_label(m, ga, 0)})"

    models_prep = []
    for mod_name in [
        "probit_short",
        "probit_controls",
        "tobit_short",
        "tobit_controls",
    ]:
        res = pd.read_parquet(depends_on[f"results_{mod_name}"])
        info = json.load(open(depends_on[f"results_info_{mod_name}"]))

        # Values are saved in one-element lists by R. Get those values directly.
        info = {k: v[0] for k, v in info.items()}

        # Rename index such that equal to statsmodels result
        res["index"] = res["term"] + ":" + res["contrast"]
        res["index"] = res["index"].replace(
            {
                **{
                    "age_groups:B2 - B1": "age_groups[T.B2]",
                    "age_groups:B3 - B1": "age_groups[T.B3]",
                    "age_groups:B4 - B1": "age_groups[T.B4]",
                    "female:TRUE - FALSE": "female",
                    "edu:upper_secondary - lower_secondary_and_lower": "edu[T.upper_secondary]",
                    "edu:tertiary - lower_secondary_and_lower": "edu[T.tertiary]",
                    "net_income_groups:Q2 - Q1": "net_income_groups[T.Q2]",
                    "net_income_groups:Q3 - Q1": "net_income_groups[T.Q3]",
                    "net_income_groups:Q4 - Q1": "net_income_groups[T.Q4]",
                    "total_financial_assets_groups:"
                    "Q2 - Q1": "total_financial_assets_groups[T.Q2]",
                    "total_financial_assets_groups:"
                    "Q3 - Q1": "total_financial_assets_groups[T.Q3]",
                    "total_financial_assets_groups:"
                    "Q4 - Q1": "total_financial_assets_groups[T.Q4]",
                    "risk_aversion_index:dY/dX": "risk_aversion_index",
                    "numeracy_index:dY/dX": "numeracy_index",
                },
                **{
                    f"type_man_sort:T{g} - T0": f"{ga_string}[T.{g}]"
                    for g in range(1, k)
                },
            }
        )
        res = res.set_index("index")

        # Rename columns
        res = _extract_params_from_r_marginaleffects(res)

        models_prep.append(
            {
                "params": res,
                "info": info,
            }
        )

    # for model_p in models_prep:
    #     model_p["info"]["mean_dep_var"] = temp[model_p["info"]["name"]].mean()

    # pairwise significant differences
    group_diff_label_dict = {
        f"Type{g_1}-Type{g_2}": f"{select_group_label(m, ga, g_1)}, "
        f"{select_group_label(m, ga, g_2)}"
        for g_2 in range(2, k)
        for g_1 in range(1, g_2)
    }

    out = et.estimation_table(
        models_prep,
        return_type="render_inputs",
        add_trailing_zeros=False,
        siunitx_warning=False,
        custom_param_names={**variable_to_proper_name, **group_label_dict},
        custom_col_groups=[
            "Owns risky assets (Probit)",
            "Owns risky assets (Probit)",
            "Share risky assets (Tobit)",
            "Share risky assets (Tobit)",
        ],
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
        stats_options={
            **{
                # "mean_dep_var": "Mean dependent variable",
                "n_obs": "Observations",
                "pseudo_r2": "Pseudo R$^2$",
                # "rsquared_adj": "Adj. R$^2$",
                # "resid_std_err": "Residual Std. Error",
                # "fvalue": "F Statistic",
                "show_dof": None,
            },
            **group_diff_label_dict,
        },
    )

    # Add title for comparisons
    n_comp = int((k - 1) * (k - 2) / 2)
    title_row = pd.DataFrame(
        [[""] * len(out["footer"].columns)],
        columns=out["footer"].columns,
        index=pd.MultiIndex.from_arrays([[r"$p$-values for differences between"]]),
    )
    out["footer"] = pd.concat(
        [out["footer"].iloc[:-n_comp, :], title_row, out["footer"].iloc[-n_comp:, :]]
    )

    # With controls
    out_latex = et.render_latex(
        out["body"],
        out["footer"],
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    out_latex = add_midrules_to_latex(out_latex, [39 + (k - 1) * 2])

    path_out = produces[f"invest_on_groups_{ga}{asset_calc}_probit_tobit"]
    with open(path_out, "w") as my_table:
        my_table.write(out_latex)

    # Without controls
    new_row = pd.DataFrame(
        [["No", "Yes"] * 2],
        columns=out["footer"].columns,
        index=pd.MultiIndex.from_arrays([["Controls"]]),
    )
    out["footer"] = pd.concat([new_row, out["footer"]])

    out_no_controls = out["body"].iloc[: (k - 1) * 2, :].copy()

    out_latex = et.render_latex(
        out_no_controls,
        out["footer"],
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    out_latex = add_midrules_to_latex(out_latex, [12 + (k - 1) * 2])

    path_out = produces[f"invest_on_groups_{ga}{asset_calc}_probit_tobit_without_cont"]
    with open(path_out, "w") as my_table:
        my_table.write(out_latex)

    return out


def reg_invest_on_params_probit_tobit(depends_on, produces, indices_params=False):
    models_prep = []
    for mod_name in [
        "probit_short",
        "probit_controls",
        "tobit_short",
        "tobit_controls",
    ]:
        res = pd.read_parquet(depends_on[f"results_{mod_name}"])
        info = json.load(open(depends_on[f"results_info_{mod_name}"]))

        # Values are saved in one-element lists by R. Get those values directly.
        info = {k: v[0] for k, v in info.items()}

        # Rename index such that equal to statsmodels result
        if "contrast" in res:
            res["index"] = res["term"] + ":" + res["contrast"]
            res["index"] = res["index"].replace(
                {
                    **{
                        "age_groups:B2 - B1": "age_groups[T.B2]",
                        "age_groups:B3 - B1": "age_groups[T.B3]",
                        "age_groups:B4 - B1": "age_groups[T.B4]",
                        "female:TRUE - FALSE": "female",
                        "edu:upper_secondary - "
                        "lower_secondary_and_lower": "edu[T.upper_secondary]",
                        "edu:tertiary - lower_secondary_and_lower": "edu[T.tertiary]",
                        "net_income_groups:Q2 - Q1": "net_income_groups[T.Q2]",
                        "net_income_groups:Q3 - Q1": "net_income_groups[T.Q3]",
                        "net_income_groups:Q4 - Q1": "net_income_groups[T.Q4]",
                        "total_financial_assets_groups:"
                        "Q2 - Q1": "total_financial_assets_groups[T.Q2]",
                        "total_financial_assets_groups:"
                        "Q3 - Q1": "total_financial_assets_groups[T.Q3]",
                        "total_financial_assets_groups:"
                        "Q4 - Q1": "total_financial_assets_groups[T.Q4]",
                        "risk_aversion_index:dY/dX": "risk_aversion_index",
                        "numeracy_index:dY/dX": "numeracy_index",
                    },
                    **{
                        f"{param}:dY/dX": param
                        for param in [
                            "ambig_av",
                            "ll_insen",
                            "theta",
                            # "ambig_av_index",
                            # "ll_insen_index",
                        ]
                    },
                }
            )
            res = res.set_index("index")
        else:
            res = res.set_index("term")

        # Rename columns
        res = _extract_params_from_r_marginaleffects(res)

        models_prep.append({"params": res, "info": info})

    out = et.estimation_table(
        models_prep,
        return_type="render_inputs",
        add_trailing_zeros=False,
        siunitx_warning=False,
        custom_param_names={**variable_to_proper_name},
        custom_col_groups=[
            "Owns risky assets (Probit)",
            "Owns risky assets (Probit)",
            "Share risky assets (Tobit)",
            "Share risky assets (Tobit)",
        ],
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
        stats_options={
            **{
                "n_obs": "Observations",
                "pseudo_r2": "Pseudo R$^2$",
                "show_dof": None,
            }
        },
    )
    # With controls
    out_latex = et.render_latex(
        out["body"],
        out["footer"],
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )

    path_out = produces["invest_on_params_probit_tobit"]
    with open(path_out, "w") as my_table:
        my_table.write(out_latex)

    # Without controls
    new_row = pd.DataFrame(
        [["No", "Yes"] * 2],
        columns=out["footer"].columns,
        index=pd.MultiIndex.from_arrays([["Controls"]]),
    )
    out["footer"] = pd.concat([new_row, out["footer"]])

    out_no_controls = out["body"].iloc[: (6 if indices_params else 8), :].copy()

    out_latex = et.render_latex(
        out_no_controls,
        out["footer"],
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    path_out = produces["invest_on_params_probit_tobit_without_cont"]
    with open(path_out, "w") as my_table:
        my_table.write(out_latex)

    models_prep = []
    for sel_var in [
        "",
        # "_all_indices_valid",
        # "_at_least_2_waves_with_valid_choice_and_index",
    ]:
        for mod_name in [
            "probit_short",
            "probit_controls",
            "tobit_short",
            "tobit_controls",
        ]:
            res = pd.read_parquet(depends_on[f"results_{mod_name}{sel_var}"])
            info = json.load(open(depends_on[f"results_info_{mod_name}{sel_var}"]))

            # Values are saved in one-element lists by R. Get those values directly.
            info = {k: v[0] for k, v in info.items()}

            # Rename index such that equal to statsmodels result
            if "contrast" in res:
                res["index"] = res["term"] + ":" + res["contrast"]
                res["index"] = res["index"].replace(
                    {
                        **{
                            "age_groups:B2 - B1": "age_groups[T.B2]",
                            "age_groups:B3 - B1": "age_groups[T.B3]",
                            "age_groups:B4 - B1": "age_groups[T.B4]",
                            "female:TRUE - FALSE": "female",
                            "edu:upper_secondary - "
                            "lower_secondary_and_lower": "edu[T.upper_secondary]",
                            "edu:tertiary - lower_secondary_and_lower": "edu[T.tertiary]",
                            "net_income_groups:Q2 - Q1": "net_income_groups[T.Q2]",
                            "net_income_groups:Q3 - Q1": "net_income_groups[T.Q3]",
                            "net_income_groups:Q4 - Q1": "net_income_groups[T.Q4]",
                            "total_financial_assets_groups:"
                            "Q2 - Q1": "total_financial_assets_groups[T.Q2]",
                            "total_financial_assets_groups:"
                            "Q3 - Q1": "total_financial_assets_groups[T.Q3]",
                            "total_financial_assets_groups:"
                            "Q4 - Q1": "total_financial_assets_groups[T.Q4]",
                            "risk_aversion_index:dY/dX": "risk_aversion_index",
                            "numeracy_index:dY/dX": "numeracy_index",
                        },
                        **{
                            f"{param}:dY/dX": param
                            for param in [
                                "ambig_av",
                                "ll_insen",
                                "theta",
                            ]
                        },
                    }
                )
                res = res.set_index("index")
            else:
                res = res.set_index("term")

            # Rename columns
            res = _extract_params_from_r_marginaleffects(res)

            models_prep.append({"params": res, "info": info})

    out = et.estimation_table(
        models_prep,
        return_type="render_inputs",
        add_trailing_zeros=False,
        siunitx_warning=False,
        custom_param_names={**variable_to_proper_name},
        # custom_col_groups=["Full Sample"] * 4
        # + ["All BBLW-indices valid"] * 4
        # + ["At least two waves with valid BBLW-indices"] * 4,
        custom_col_groups=[
            "Owns risky assets (Probit)",
            "Owns risky assets (Probit)",
            "Share risky assets (Tobit)",
            "Share risky assets (Tobit)",
        ],
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
        stats_options={
            **{
                "n_obs": "Observations",
                "pseudo_r2": "Pseudo R$^2$",
                "show_dof": None,
            }
        },
    )
    # # Add second level to columns
    # second_level_columns = [
    #     "Owns risky assets (Probit)",
    #     "Owns risky assets (Probit)",
    #     "Share risky assets (Tobit)",
    #     "Share risky assets (Tobit)",
    # ] * 3
    # out["body"].columns = pd.MultiIndex.from_tuples(
    #     (col_tuple[0], second_level_columns[i], col_tuple[1])
    #     for i, col_tuple in enumerate(out["body"].columns)
    # )
    # out["footer"].columns = pd.MultiIndex.from_tuples(
    #     (col_tuple[0], second_level_columns[i], col_tuple[1])
    #     for i, col_tuple in enumerate(out["footer"].columns)
    # )

    # With controls
    out_latex = et.render_latex(
        out["body"],
        out["footer"],
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )

    path_out = produces["invest_on_params_probit_tobit_sel_samples"]
    with open(path_out, "w") as my_table:
        my_table.write(out_latex)

    # Without controls
    new_row = pd.DataFrame(
        [["No", "Yes"] * 2],
        columns=out["footer"].columns,
        index=pd.MultiIndex.from_arrays([["Controls"]]),
    )
    out["footer"] = pd.concat([new_row, out["footer"]])

    out_no_controls = out["body"].iloc[: (6 if indices_params else 8), :].copy()

    out_latex = et.render_latex(
        out_no_controls,
        out["footer"],
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    path_out = produces["invest_on_params_probit_tobit_sel_samples_without_cont"]
    with open(path_out, "w") as my_table:
        my_table.write(out_latex)

    return out


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC + NAMES_INDICES_SPEC:
    asset_calc = MODEL_SPECS[m]["asset_calc"]
    for ga in MODEL_SPECS[m]["k_groups"]:
        id = f"{m}:{ga}"
        if MODEL_SPECS[m]["indices_params"]:
            depends_on = {
                "individual": OUT_DATA / "individual.pickle",
                "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
                "indices": OUT_DATA_LISS / "ambiguous_beliefs" / "indices.pickle",
                "utils_final": "utils_final.py",
                "group_assignments": OUT_ANALYSIS / f"group_assignments_{m}.pickle",
            }
        else:
            depends_on = {
                "individual": OUT_DATA / "individual.pickle",
                "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
                MODEL_SPECS[m]["est_model_name"]: OUT_UNDER_GIT
                / MODEL_SPECS[m]["est_model_name"]
                / "opt_diff_evolution"
                / "results.pickle",
                "utils_final": "utils_final.py",
                "group_assignments": OUT_ANALYSIS / f"group_assignments_{m}.pickle",
            }
        for probit_tobit in ["probit", "tobit"]:
            depends_on.update(
                {
                    f"results_{probit_tobit}_short": OUT_ANALYSIS
                    / m
                    / f"results_{probit_tobit}_short_{ga}{asset_calc}.parquet",
                    f"results_{probit_tobit}_controls": OUT_ANALYSIS
                    / m
                    / f"results_{probit_tobit}_controls_{ga}{asset_calc}.parquet",
                    f"results_info_{probit_tobit}_short": OUT_ANALYSIS
                    / m
                    / f"results_info_{probit_tobit}_short_{ga}{asset_calc}.json",
                    f"results_info_{probit_tobit}_controls": OUT_ANALYSIS
                    / m
                    / f"results_info_{probit_tobit}_controls_{ga}{asset_calc}.json",
                }
            )
        produces = {
            f"invest_on_groups_{ga}{asset_calc}": OUT_TABLES
            / m
            / (f"invest_on_groups_{ga}{asset_calc}.tex"),
            f"invest_on_groups_{ga}{asset_calc}_without_cont": OUT_TABLES
            / m
            / (f"invest_on_groups_{ga}{asset_calc}_without_cont.tex"),
            f"invest_on_groups_{ga}{asset_calc}_probit_tobit": OUT_TABLES
            / m
            / (f"invest_on_groups_{ga}{asset_calc}_probit_tobit.tex"),
            f"invest_on_groups_{ga}{asset_calc}_probit_tobit_without_cont": OUT_TABLES
            / m
            / (f"invest_on_groups_{ga}{asset_calc}_probit_tobit_without_cont.tex"),
        }
        PARAMETRIZATION[id] = {
            "m": m,
            "ga": ga,
            "depends_on": depends_on,
            "produces": produces,
            "model_spec": MODEL_SPECS[m],
        }


for id_, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=id_)
    def task_reg_invest_on_groups(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        model_spec=kwargs["model_spec"],
        m=kwargs["m"],
        ga=kwargs["ga"],
    ):
        models = (
            model_spec["wbw_models"]
            if model_spec["indices_params"]
            else [model_spec["est_model_name"]]
        )
        df = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=model_spec["asset_calc"],
            restrictions=model_spec["restrictions"],
            models=models,
            indices=model_spec["indices_params"],
            indices_mean=model_spec.get("indices_mean"),
        )

        group_assignments = pd.read_pickle(depends_on["group_assignments"])
        data = df.join(group_assignments[ga])
        data[ga] = pd.Categorical(data[ga])

        # Sort mapping of manual sorting of groups
        g_man_to_g = select_manual_group_order(m, ga)
        g_to_g_man = {j: i for i, j in g_man_to_g.items()}
        data["type_man_sort"] = data[ga].map(g_to_g_man)
        data = data.dropna(subset=["type_man_sort"])

        reg_invest_on_groups(data, ga, BASIC_CONTROLS, produces, m, asset_calc)

        k = len(data[ga].unique())
        reg_invest_on_groups_probit_tobit(k, ga, depends_on, produces, m, asset_calc)


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_INDICES_SPEC:
    depends_on = {
        "utils_final": ROOT / "ambig_beliefs" / "final" / "utils_final.py",
    }
    for probit_tobit in ["probit", "tobit"]:
        for sel_var in [
            "",
            # "_all_indices_valid",
            # "_at_least_2_waves_with_valid_choice_and_index",
        ]:
            depends_on.update(
                {
                    f"results_{probit_tobit}_short{sel_var}": OUT_ANALYSIS
                    / m
                    / f"results_params_{probit_tobit}_short{asset_calc}{sel_var}.parquet",
                    f"results_{probit_tobit}_controls{sel_var}": OUT_ANALYSIS
                    / m
                    / f"results_params_{probit_tobit}_controls{asset_calc}{sel_var}.parquet",
                    f"results_info_{probit_tobit}_short{sel_var}": OUT_ANALYSIS
                    / m
                    / f"results_params_info_{probit_tobit}_short{asset_calc}{sel_var}.json",
                    f"results_info_{probit_tobit}_controls{sel_var}": OUT_ANALYSIS
                    / m
                    / f"results_params_info_{probit_tobit}_controls{asset_calc}{sel_var}.json",
                }
            )
    produces = {
        "invest_on_params_probit_tobit": OUT_TABLES
        / m
        / (f"invest_on_params{asset_calc}_probit_tobit.tex"),
        "invest_on_params_probit_tobit_without_cont": OUT_TABLES
        / m
        / (f"invest_on_params{asset_calc}_probit_tobit_without_cont.tex"),
        "invest_on_params_probit_tobit_sel_samples": OUT_TABLES
        / m
        / (f"invest_on_params{asset_calc}_probit_tobit_sel_samples.tex"),
        "invest_on_params_probit_tobit_sel_samples_without_cont": OUT_TABLES
        / m
        / (f"invest_on_params{asset_calc}_probit_tobit_sel_samples_without_cont.tex"),
    }
    PARAMETRIZATION[m] = {
        "m": m,
        "depends_on": depends_on,
        "produces": produces,
        "model_spec": MODEL_SPECS[m],
    }


for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=m)
    def task_reg_invest_on_params(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        model_spec=kwargs["model_spec"],
    ):
        reg_invest_on_params_probit_tobit(
            depends_on, produces, indices_params=model_spec["indices_params"]
        )


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC:
    k_groups_final_sel = ["k4"]
    for ga in k_groups_final_sel:
        depends_on = {
            "output_220805": IN_DATA / "cbs_output" / "output_220805.xlsx",
            "utils_final": ROOT / "ambig_beliefs" / "final" / "utils_final.py",
        }
        produces = {
            f"invest_on_groups_{ga}_cbs": OUT_TABLES
            / m
            / (f"invest_on_groups_{ga}_cbs.tex"),
            f"invest_on_groups_{ga}_cbs_without_cont": OUT_TABLES
            / m
            / (f"invest_on_groups_{ga}_cbs_without_cont.tex"),
        }
        PARAMETRIZATION[m] = {
            "m": m,
            "depends_on": depends_on,
            "produces": produces,
        }


for id_, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=id_)
    def task_reg_invest_on_groups_cbs(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        m=kwargs["m"],
    ):
        # Load output
        cbs_results = pd.read_excel(
            depends_on["output_220805"], sheet_name=None, index_col=0
        )

        ga = "k4"

        # Select sheet and columns
        out = cbs_results["ambig_pf_choice"].iloc[:, [0, 2, 3, 5]]
        out = out.rename(
            index={
                "C(ambig_type, Treatment"
                "(reference='Near SEU'))[T.Ambiguity averse]": "Ambiguity averse",
                "C(ambig_type, Treatment"
                "(reference='Near SEU'))[T.Ambiguity seeking]": "Ambiguity seeking",
                "C(ambig_type, Treatment"
                "(reference='Near SEU'))[T.High noise]": "High noise",
                "female": "Female",
                "age_groups_ambig[T.between 36 and 50]": "Age: $\\in (35,50]$",
                "age_groups_ambig[T.between 51 and 65]": "Age: $\\in (50,65]$",
                "age_groups_ambig[T.> 65]": "Age: $ \\geq 65$",
                "b_edu[T.upper_secondary]": "Education: Upper secondary",
                "b_edu[T.tertiary]": "Education: Tertiary",
                "net_income_hh_simple_groups[T.Q2]": "Income: Quartile 2",
                "net_income_hh_simple_groups[T.Q3]": "Income: Quartile 3",
                "net_income_hh_simple_groups[T.Q4]": "Income: Quartile 4",
                "total_financial_assets_groups_ambig"
                "[T.Q2]": "Financial assets: Quartile 2",
                "total_financial_assets_groups_ambig"
                "[T.Q3]": "Financial assets: Quartile 3",
                "total_financial_assets_groups_ambig"
                "[T.Q4]": "Financial assets: Quartile 4",
                "numeracy_index_stdzd": "Numeracy index",
                "risk_aversion_index_stdzd": "Risk aversion index",
                "N": "Observations",
                "R^2": "R$^2$",
            },
            columns=col_name_to_proper_name,
        )

        # Replace nan
        out = out.reset_index()
        out = out.replace({np.nan: ""})
        out = out.set_index("index")
        out.index.name = ""

        # Columns to multiindex
        out.columns = pd.MultiIndex.from_arrays(
            [out.columns, [f"({i})" for i in range(len(out.columns))]]
        )

        body = out.iloc[:-3, :]
        body = pd.concat([body.iloc[:-4, :], body.iloc[-2:, :], body.iloc[-4:-2, :]])
        footer = out.iloc[-3:-1, :]

        # Save table
        out_latex = et.render_latex(
            body,
            footer,
            append_notes=False,
            show_footer=True,
            siunitx_warning=False,
            escape_special_characters=False,
        )

        path_out = produces[f"invest_on_groups_{ga}_cbs"]
        with open(path_out, "w") as my_table:
            my_table.write(out_latex)

        # Without controls
        new_row = pd.DataFrame(
            [["No", "Yes"] * 2],
            columns=footer.columns,
            index=pd.MultiIndex.from_arrays([["Controls"]]),
        )
        footer = pd.concat([new_row, footer])

        out_no_controls = body.iloc[: 4 * 2, :].copy()

        # Don't show intercept if controls are used
        out_no_controls.iloc[[0, 1], [1, 3]] = ""

        out_latex = et.render_latex(
            out_no_controls,
            footer,
            append_notes=False,
            show_footer=True,
            siunitx_warning=False,
            escape_special_characters=False,
        )
        path_out = produces[f"invest_on_groups_{ga}_cbs_without_cont"]
        with open(path_out, "w") as my_table:
            my_table.write(out_latex)
