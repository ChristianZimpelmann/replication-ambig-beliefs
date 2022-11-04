"""
Run regressions of climate related ambiguity parameters on aex related ambiguity parameters.
"""
import estimagic.visualization.estimation_table as et
import linearmodels as lm
import numpy as np
import pandas as pd
import pytask
import statsmodels.formula.api as smf

from ambig_beliefs.final.utils_final import add_midrules_to_latex
from ambig_beliefs.final.utils_final import put_reg_sample_together
from ambig_beliefs.final.utils_final import variable_to_proper_name
from config import BASIC_CONTROLS
from config import MODEL_SPECS
from config import NAMES_MAIN_SPEC
from config import NAMES_ROBUSTNESS_SPEC
from config import OUT_ANALYSIS
from config import OUT_DATA
from config import OUT_DATA_LISS
from config import OUT_TABLES
from config import OUT_UNDER_GIT

para_name_dict = {
    "ambig_av": r"\alpha",
    "ll_insen": r"\ell",
    "theta": r"\sigma",
}
wave_to_date = {
    1: "2018-05",
    2: "2018-11",
    3: "2019-05",
    4: "2019-11",
    5: "2020-05",
    6: "2020-11",
    7: "2021-05",
}


def _extract_params_from_linearmodels(model):
    """Convert linearmodels like estimation result to estimagic like params dataframe."""
    to_concat = []
    params_list = ["params", "pvalues", "std_errors"]
    for col in params_list:
        to_concat.append(getattr(model, col))
    to_concat.append(model.conf_int())
    params_df = pd.concat(to_concat, axis=1)
    params_df.columns = ["value", "p_value", "standard_error", "ci_lower", "ci_upper"]
    return params_df


def _extract_info_from_linearmodels(model):
    """Process linearmodels estimation result to retrieve summary statistics as dict."""
    info = {}
    key_values = {
        "df_model": "df_model",
        "df_resid": "df_resid",
    }
    for key, key_lm in key_values.items():
        info[key] = getattr(model, key_lm)

    info["rsquared"] = np.nan
    info["rsquared_adj"] = np.nan
    info["fvalue"] = model.f_statistic.stat
    info["fvalue"] = model.f_statistic.stat
    info["f_pvalue"] = model.f_statistic.pval
    info["fvalue_first_stage"] = int(model.first_stage.diagnostics["f.stat"].iloc[0])
    info["dependent_variable"] = model.model.dependent.cols[0]
    info["resid_std_err"] = np.sqrt(model.s2)
    info["n_obs"] = model.df_model + model.df_resid + 1
    return info


def make_lm_oriv_formula(y, X_exog, X_endog, Z):
    regs_exog = " + ".join(X_exog)
    if len(regs_exog) > 0:
        regs_exog = " + " + regs_exog
    regs_endog = " + ".join(X_endog)
    ivs = " + ".join(Z)
    return f"{y} ~ 1 {regs_exog} + [{regs_endog} ~ {ivs}]"


def para_name_dep_var(para, outcome_waves):
    if len(outcome_waves) > 1 and 7 in outcome_waves:
        out = f"${para_name_dict[para]}^{{AEX}}_\\text{{last {len(outcome_waves)} waves}}$"
    else:
        if "temp" in outcome_waves:
            out = f"${para_name_dict[para]}^{{climate}}_{{2019-11}}$"
        else:
            out = (
                f"${para_name_dict[para]}^{{AEX}}_{{{wave_to_date[outcome_waves[0]]}}}$"
            )

    return out


def para_name_indep_var(para, indep_waves):
    if len(indep_waves) > 1 and 2 in indep_waves:
        out = (
            f"${para_name_dict[para]}^{{AEX}}_\\text{{first {len(indep_waves)} waves}}$"
        )
    else:
        if "temp" in indep_waves:
            out = f"${para_name_dict[para]}^{{climate}}_{{2019-11}}$"
        else:
            out = f"${para_name_dict[para]}^{{AEX}}_{{{wave_to_date[indep_waves[0]]}}}$"
    return out


def create_stacked_data(
    waves, para_results_dict, outcome_waves, indep_waves, inst_waves
):
    stacked_data = {}
    for dep_var_w in waves:
        for indep_var_w in [w for w in waves if w != dep_var_w]:
            for inst_w in [w for w in waves if w not in [dep_var_w, indep_var_w]]:

                # depvar
                dep_var = para_results_dict[dep_var_w].copy()
                dep_var.columns = [c + "_dep_var" for c in dep_var]

                # indepvar
                indep_var = para_results_dict[indep_var_w].copy()
                indep_var.columns = [c + "_indep_var" for c in indep_var]

                # inst
                inst = para_results_dict[inst_w].copy()
                inst.columns = [c + "_inst" for c in inst]

                stacked_data[(dep_var_w, indep_var_w, inst_w)] = dep_var.join(
                    indep_var, how="outer"
                ).join(inst, how="outer")
    stacked_data = pd.concat(stacked_data).sort_index()
    stacked_data.index.names = [
        "wave_dep_var",
        "wave_indep_var",
        "wave_inst",
        "personal_id",
    ]

    # Create data sets
    stacked_data = (
        stacked_data.reset_index()
        .query("wave_dep_var == @outcome_waves")
        .query("wave_indep_var == @indep_waves")
        .query("wave_inst == @inst_waves")
    )

    stacked_data["stack"] = (
        stacked_data["wave_dep_var"].astype(str)
        + stacked_data["wave_indep_var"].astype(str)
        + stacked_data["wave_inst"].astype(str)
    )
    return stacked_data


def reg_para_wave_on_wave(
    df,
    parameters,
    controls,
    outcome_waves,
    indep_waves,
    inst_waves,
    std_parameters,
    path_dict,
    idx=False,
    save_table=True,
):

    para_results = df[parameters].copy()
    controls_df = df[controls].groupby("personal_id").first()
    waves = para_results.index.unique(level="wave")
    para_results_dict = {w: para_results.xs(w, level="wave") for w in waves}

    # Create stacked data set
    stacked_data = create_stacked_data(
        waves, para_results_dict, outcome_waves, indep_waves, inst_waves
    )

    # Run OLS regressions and ORIV
    spec_to_model = {}

    reg_data_ols = (
        stacked_data.groupby(["wave_dep_var", "wave_indep_var", "personal_id"])
        .first()
        .reset_index()
        .set_index("personal_id")
    ).join(controls_df)

    reg_data_oriv = stacked_data.set_index("personal_id").join(controls_df)
    for para in parameters:
        for controls_spec, cont in {
            "no_controls": [],
            "with_controls": controls,
        }.items():

            form = (
                f"{para}_dep_var ~ 1 + {para}_indep_var"
                if len(cont) == 0
                else f"{para}_dep_var ~ {para}_indep_var + {'+'.join(controls)}"
            )

            # Also drop inst_var here?
            temp_ols = reg_data_ols[
                [f"{para}_dep_var", f"{para}_indep_var", f"{para}_inst"] + cont
            ].dropna()

            # Run ols and save
            mod_ols = smf.ols(form, data=temp_ols).fit(
                cov_type="cluster",
                cov_kwds={"groups": temp_ols.reset_index()["personal_id"]},
            )
            spec_to_model[f"{para}_{controls_spec}_ols"] = {
                "params": et._extract_params_from_sm(mod_ols),
                "info": et._extract_info_from_sm(mod_ols),
                "name": "OLS",
            }

            # Sort params
            order = ["Intercept", f"{para}_indep_var"]
            spec_to_model[f"{para}_{controls_spec}_ols"]["params"] = spec_to_model[
                f"{para}_{controls_spec}_ols"
            ]["params"].reindex(
                order
                + [
                    i
                    for i in spec_to_model[f"{para}_{controls_spec}_ols"][
                        "params"
                    ].index
                    if i not in order
                ]
            )

            spec_to_model[f"{para}_{controls_spec}_ols"]["info"]["n_obs"] = len(
                temp_ols.reset_index()["personal_id"].unique()
            )

            # Run IV
            formula_ORIV = make_lm_oriv_formula(
                y=f"{para}_dep_var",
                X_exog=cont,
                X_endog=[f"{para}_indep_var"],
                Z=[f"{para}_inst"],
            )
            # Drop missing observations
            temp_iv = reg_data_oriv[
                [f"{para}_dep_var", f"{para}_indep_var", f"{para}_inst"] + cont
            ].dropna()

            m_oriv = lm.IV2SLS.from_formula(
                formula=formula_ORIV,
                data=temp_iv,
            ).fit(cov_type="clustered", clusters=temp_iv.reset_index()["personal_id"])

            spec_to_model[f"{para}_{controls_spec}_iv"] = {
                "params": _extract_params_from_linearmodels(m_oriv),
                "info": _extract_info_from_linearmodels(m_oriv),
                "name": "2SLS" if len(indep_waves) == 1 else "ORIV",
            }

            # Sort params
            order = ["Intercept", f"{para}_indep_var"]
            spec_to_model[f"{para}_{controls_spec}_iv"]["params"] = spec_to_model[
                f"{para}_{controls_spec}_iv"
            ]["params"].reindex(
                order
                + [
                    i
                    for i in spec_to_model[f"{para}_{controls_spec}_iv"]["params"].index
                    if i not in order
                ]
            )

            spec_to_model[f"{para}_{controls_spec}_iv"]["info"]["n_obs"] = len(
                temp_iv.reset_index()["personal_id"].unique()
            )

    # put tables together
    table_chunks = {}
    for para in parameters:
        out_one_para = et.estimation_table(
            [
                spec_to_model[mn]
                for mn in [
                    f"{para}_no_controls_ols",
                    f"{para}_no_controls_iv",
                    f"{para}_with_controls_iv",
                ]
            ],
            return_type="render_inputs",
            add_trailing_zeros=False,
            siunitx_warning=False,
            custom_param_names=variable_to_proper_name,
            number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
            stats_options={
                "n_obs": "N Subjects",
                "rsquared_adj": "Adj. R$^2$",
                "fvalue_first_stage": "1st st. F",
                "show_dof": None,
            },
        )
        out_one_para_fmt_2_digits = et.estimation_table(
            [
                spec_to_model[mn]
                for mn in [
                    f"{para}_no_controls_ols",
                    f"{para}_no_controls_iv",
                    f"{para}_with_controls_iv",
                ]
            ],
            return_type="render_inputs",
            add_trailing_zeros=False,
            siunitx_warning=False,
            custom_param_names=variable_to_proper_name,
            number_format=("{0:.2f}"),
            stats_options={
                "n_obs": "N Subjects",
                "rsquared_adj": "Adj. R$^2$",
                "fvalue_first_stage": "1st st. F",
                "show_dof": None,
            },
        )
        table_chunks[para] = pd.concat(
            [
                out_one_para["body"].iloc[:2, :],
                out_one_para_fmt_2_digits["body"].iloc[2:4, :],
                out_one_para["footer"].drop("N Subjects", level=0),
            ]
        )
        table_chunks[para].iloc[
            [
                0,
                1,
            ],
            [2],
        ] = ""

    body_final = pd.concat(table_chunks, names=["Dep. Variable", "Regressors"])

    # Rename parameters
    body_final = body_final.rename(
        index={para: para_name_dep_var(para, outcome_waves) for para in parameters},
        level=0,
    ).rename(
        index={
            f"{para}_indep_var": para_name_indep_var(para, indep_waves)
            for para in parameters
        },
        level=1,
    )
    body_final = body_final.rename(index=variable_to_proper_name)
    new_row = pd.DataFrame(
        [["No", "No", "Yes"]],
        columns=body_final.columns,
        index=pd.MultiIndex.from_arrays([["Controls"]]),
    )
    footer_final = pd.concat([new_row, out_one_para["footer"].loc[["N Subjects"]]])
    footer_final.index = pd.MultiIndex.from_tuples(
        [[i[0], ""] for i in footer_final.index]
    )

    out_latex = et.render_latex(
        body_final,
        footer_final,
        append_notes=False,
        render_options={},
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    out_latex = add_midrules_to_latex(out_latex, [14] if idx else [14, 21])
    # Do this backwards because 'insert' is modifying the line numbers in-place.
    out_latex = add_midrules_to_latex(
        out_latex,
        [20, 19, 13, 12] if idx else [27, 26, 20, 19, 13, 12],
        midrule_text=r"\cmidrule{2-5}",
    )

    def wave_str(waves):
        return (
            str(waves)
            .replace("[", "")
            .replace("]", "")
            .replace(", ", "_")
            .replace("'", "")
        )

    if save_table:
        path_out = path_dict[
            f"reg_stab_out_{wave_str(outcome_waves)}_indep_{wave_str(indep_waves)}_"
            f"inst_{wave_str(inst_waves)}_short" + ("_idx" if idx else "")
        ]
        with open(path_out, "w") as my_table:
            my_table.write(out_latex)

    # Save table only ambig
    body_final_only_ambig = body_final.loc[
        [i for i in body_final.index.unique(level=0) if "sigma" not in i]
    ]
    out_latex = et.render_latex(
        body_final_only_ambig,
        footer_final,
        append_notes=False,
        render_options={},
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    out_latex = add_midrules_to_latex(out_latex, [14])
    # Do this backwards because 'insert' is modifying the line numbers in-place.
    out_latex = add_midrules_to_latex(
        out_latex, [20, 19, 13, 12], midrule_text=r"\cmidrule{2-5}"
    )
    if save_table:
        path_out = path_dict[
            f"reg_stab_out_{wave_str(outcome_waves)}_indep_{wave_str(indep_waves)}_"
            f"inst_{wave_str(inst_waves)}_short_only_ambig" + ("_idx" if idx else "")
        ]
        with open(path_out, "w") as my_table:
            my_table.write(out_latex)

    # Full regression table
    table_chunks = {}
    table_footer_chunks = {}

    for para in parameters:
        out_one_para = et.estimation_table(
            [
                spec_to_model[mn]
                for mn in [
                    # f"{para}_with_controls_ols",
                    f"{para}_with_controls_iv",
                ]
            ],
            return_type="render_inputs",
            add_trailing_zeros=False,
            siunitx_warning=False,
            custom_param_names=variable_to_proper_name,
            number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
            stats_options={
                "n_obs": "N Subjects",
                # "rsquared_adj": "Adj. R$^2$",
                "fvalue_first_stage": "1st st. F",
                "show_dof": None,
            },
        )
        table_chunks[para] = out_one_para["body"].rename(
            {
                f"{para}_indep_var": f"AEX parameter first {len(indep_waves)} waves"
                if len(indep_waves) > 1
                else f"AEX parameter {wave_to_date[indep_waves[0]]}"
            }
        )
        table_footer_chunks[para] = out_one_para["footer"]

    body_final = pd.concat(table_chunks, axis=1)
    footer_final = pd.concat(table_footer_chunks, axis=1)

    body_final = body_final.rename(
        columns={para: para_name_dep_var(para, outcome_waves) for para in parameters}
    )

    out_latex = et.render_latex(
        body_final,
        footer_final,
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )

    if save_table:
        path_out = path_dict[
            f"reg_stab_out_{wave_str(outcome_waves)}_indep_{wave_str(indep_waves)}_"
            f"inst_{wave_str(inst_waves)}_long" + ("_idx" if idx else "")
        ]
        with open(path_out, "w") as my_table:
            my_table.write(out_latex)
    return out_latex


# def reg_para_wave_on_wave_by_type(
#     df,
#     parameters,
#     outcome_waves,
#     indep_waves,
#     std_parameters,
#     path_dict,
#     idx=False,
#     save_table=True,
# ):
#     para_results = df[parameters].copy()
#     controls_df = df[["type_man_sort"]].groupby("personal_id").first()
#     waves = para_results.index.unique(level="wave")
#     para_results_dict = {w: para_results.xs(w, level="wave") for w in waves}

#     # Create stacked data set
#     stacked_data = create_stacked_data(
#         waves, para_results_dict, outcome_waves, indep_waves, indep_waves
#     )

#     # Run OLS regressions and ORIV
#     spec_to_model = {}

#     reg_data_ols = (
#         stacked_data.groupby(["wave_dep_var", "wave_indep_var", "personal_id"])
#         .first()
#         .reset_index()
#         .set_index("personal_id")
#     ).join(controls_df)

#     for para in parameters:
#         for controls_spec, cont in {
#             "no_controls": [],
#             # "with_controls": controls,
#         }.items():
#             for ambig_type in df["type_man_sort"].cat.categories:
#                 form = (
#                     f"{para}_dep_var ~ 1 + {para}_indep_var"
#                     if len(cont) == 0
#                     else f"{para}_dep_var ~ {para}_indep_var"
#                 )

#                 # Also drop inst_var here?
#                 temp_ols = (
#                     reg_data_ols[
#                         [f"{para}_dep_var", f"{para}_indep_var", f"{para}_inst"]
#                         + cont
#                         + ["type_man_sort"]
#                     ]
#                     .dropna()
#                     .query(f"type_man_sort == '{ambig_type}'")
#                 )

#                 # Run ols and save
#                 mod_ols = smf.ols(form, data=temp_ols).fit(
#                     cov_type="cluster",
#                     cov_kwds={"groups": temp_ols.reset_index()["personal_id"]},
#                 )
#                 spec_to_model[f"{para}_{controls_spec}_ols_{ambig_type}"] = {
#                     "params": et._extract_params_from_sm(mod_ols),
#                     "info": et._extract_info_from_sm(mod_ols),
#                     "name": "OLS",
#                 }

#                 # Sort params
#                 order = ["Intercept", f"{para}_indep_var"]
#                 spec_to_model[f"{para}_{controls_spec}_ols_{ambig_type}"][
#                     "params"
#                 ] = spec_to_model[f"{para}_{controls_spec}_ols_{ambig_type}"][
#                     "params"
#                 ].reindex(
#                     order
#                     + [
#                         i
#                         for i in spec_to_model[
#                             f"{para}_{controls_spec}_ols_{ambig_type}"
#                         ]["params"].index
#                         if i not in order
#                     ]
#                 )

#                 spec_to_model[f"{para}_{controls_spec}_ols_{ambig_type}"]["info"][
#                     "n_obs"
#                 ] = len(temp_ols.reset_index()["personal_id"].unique())

#     # put tables together
#     table_chunks = {}
#     for para in parameters:
#         out_one_para = et.estimation_table(
#             [
#                 spec_to_model[f"{para}_no_controls_ols_{ambig_type}"]
#                 for ambig_type in df["type_man_sort"].cat.categories
#             ],
#             return_type="render_inputs",
#             add_trailing_zeros=False,
#             siunitx_warning=False,
#             custom_param_names=variable_to_proper_name,
#             custom_col_names=list(df["type_man_sort"].cat.categories),
#             number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
#             stats_options={
#                 "n_obs": "N Subjects",
#                 "rsquared_adj": "Adj. R$^2$",
#                 # "fvalue_first_stage": "1st st. F",
#                 "show_dof": None,
#             },
#         )
#         out_one_para_fmt_2_digits = et.estimation_table(
#             [
#                 spec_to_model[f"{para}_no_controls_ols_{ambig_type}"]
#                 for ambig_type in df["type_man_sort"].cat.categories
#             ],
#             return_type="render_inputs",
#             add_trailing_zeros=False,
#             siunitx_warning=False,
#             custom_param_names=variable_to_proper_name,
#             custom_col_names=list(df["type_man_sort"].cat.categories),
#             number_format=("{0:.2f}"),
#             stats_options={
#                 "n_obs": "N Subjects",
#                 "rsquared_adj": "Adj. R$^2$",
#                 # "fvalue_first_stage": "1st st. F",
#                 "show_dof": None,
#             },
#         )
#         table_chunks[para] = pd.concat(
#             [
#                 out_one_para["body"].iloc[:2, :],
#                 out_one_para_fmt_2_digits["body"].iloc[2:4, :],
#                 out_one_para["footer"].drop("N Subjects", level=0),
#             ]
#         )
#     body_final = pd.concat(table_chunks, names=["Dep. Variable", "Regressors"])

#     # Rename parameters
#     body_final = body_final.rename(
#         index={para: para_name_dep_var(para, outcome_waves) for para in parameters},
#         level=0,
#     ).rename(
#         index={
#             f"{para}_indep_var": para_name_indep_var(para, indep_waves)
#             for para in parameters
#         },
#         level=1,
#     )
#     body_final = body_final.rename(index=variable_to_proper_name)

#     new_row = pd.DataFrame(
#         [["No", "No", "No", "No"]],
#         columns=body_final.columns,
#         index=pd.MultiIndex.from_arrays([["Controls"]]),
#     )
#     footer_final = pd.concat([new_row, out_one_para["footer"].loc[["N Subjects"]]])
#     footer_final.index = pd.MultiIndex.from_tuples(
#         [[i[0], ""] for i in footer_final.index]
#     )

#     out_latex = et.render_latex(
#         body_final,
#         footer_final,
#         append_notes=False,
#         render_options={},
#         show_footer=True,
#         siunitx_warning=False,
#         escape_special_characters=False,
#     )
#     out_latex = add_midrules_to_latex(out_latex, [13] if idx else [12, 18])
#     # Do this backwards because 'insert' is modifying the line numbers in-place.
#     out_latex = add_midrules_to_latex(
#         out_latex,
#         [20, 13] if idx else [23, 17, 11],
#         midrule_text=r"\cmidrule{2-6}",
#     )

#     def wave_str(waves):
#         return (
#             str(waves)
#             .replace("[", "")
#             .replace("]", "")
#             .replace(", ", "_")
#             .replace("'", "")
#         )

#     if save_table:
#         path_out = path_dict[
#             f"reg_stab_out_{wave_str(outcome_waves)}_indep_{wave_str(indep_waves)}"
#             "_short_by_type"
#         ]
#         with open(path_out, "w") as my_table:
#             my_table.write(out_latex)

PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC:
    est_model_name = MODEL_SPECS[m]["est_model_name"]

    wbw_models = MODEL_SPECS[m]["wbw_models"]
    climate_model = MODEL_SPECS[m]["climate_model"]
    depends_on = {
        "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
        "individual": OUT_DATA / "individual.pickle",
        "indices": OUT_DATA_LISS / "ambiguous_beliefs" / "indices.pickle",
        est_model_name: OUT_UNDER_GIT
        / est_model_name
        / "opt_diff_evolution"
        / "results.pickle",
        "group_assignments": OUT_ANALYSIS / f"group_assignments_{m}.pickle",
    }

    depends_on.update(
        {
            mod: OUT_UNDER_GIT / mod / "opt_diff_evolution" / "results.pickle"
            for mod in wbw_models + [climate_model]
        }
    )
    produces = {
        name: OUT_TABLES / m / f"{name}.tex"
        for name in [
            "reg_stab_out_temp_indep_4_inst_2_3_5_6_7_short",
            "reg_stab_out_temp_indep_2_3_inst_2_3_short",
            "reg_stab_out_4_5_6_7_indep_2_3_inst_2_3_short",
            "reg_stab_out_5_6_7_indep_2_3_4_inst_2_3_4_short",
            "reg_stab_out_6_7_indep_2_3_4_5_inst_2_3_4_5_short",
            "reg_stab_out_7_indep_2_3_4_5_6_inst_2_3_4_5_6_short",
            "reg_stab_out_temp_indep_4_inst_2_3_5_6_7_long",
            "reg_stab_out_temp_indep_2_3_inst_2_3_long",
            "reg_stab_out_4_5_6_7_indep_2_3_inst_2_3_long",
            "reg_stab_out_5_6_7_indep_2_3_4_inst_2_3_4_long",
            "reg_stab_out_6_7_indep_2_3_4_5_inst_2_3_4_5_long",
            "reg_stab_out_7_indep_2_3_4_5_6_inst_2_3_4_5_6_long",
            "reg_stab_out_temp_indep_4_inst_2_3_5_6_7_short_only_ambig",
            "reg_stab_out_temp_indep_2_3_inst_2_3_short_only_ambig",
            "reg_stab_out_4_5_6_7_indep_2_3_inst_2_3_short_only_ambig",
            "reg_stab_out_5_6_7_indep_2_3_4_inst_2_3_4_short_only_ambig",
            "reg_stab_out_6_7_indep_2_3_4_5_inst_2_3_4_5_short_only_ambig",
            "reg_stab_out_7_indep_2_3_4_5_6_inst_2_3_4_5_6_short_only_ambig",
            # "reg_stab_out_5_6_7_indep_2_3_4_short_by_type",
        ]
    }
    produces.update(
        {
            name: OUT_TABLES / m / "idx" / f"{name}.tex"
            for name in [
                "reg_stab_out_temp_indep_4_inst_2_3_5_6_7_short_idx",
                "reg_stab_out_5_6_7_indep_2_3_4_inst_2_3_4_short_idx",
                "reg_stab_out_temp_indep_4_inst_2_3_5_6_7_long_idx",
                "reg_stab_out_5_6_7_indep_2_3_4_inst_2_3_4_long_idx",
                "reg_stab_out_temp_indep_4_inst_2_3_5_6_7_short_only_ambig_idx",
                "reg_stab_out_5_6_7_indep_2_3_4_inst_2_3_4_short_only_ambig_idx",
            ]
            # f"{spec}_long": OUT_TABLES / m /
            #  f"{spec}_long.tex",
            # f"{spec}_2slsonly": OUT_TABLES / m / f"{spec}_2slsonly.tex",
            # f"{spec}_olsonly": OUT_TABLES / m / f"{spec}_olsonly.tex",
        }
    )
    PARAMETRIZATION[m] = {
        "depends_on": depends_on,
        "produces": produces,
        "wbw_models": wbw_models,
        "climate_model": climate_model,
    }

for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=m)
    def task_reg_para_wave_on_wave(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        m=m,
        wbw_models=kwargs["wbw_models"],
        climate_model=kwargs["climate_model"],
    ):
        restrictions = "&".join(MODEL_SPECS[m]["restrictions"].split(","))

        models = wbw_models + [climate_model]
        parameters = ["ambig_av", "ll_insen", "theta"]
        params_idx = ["ambig_av", "ll_insen"]

        climate_controls = [
            "understands_climate_change",
            "threatened_by_climate_change",
        ]
        controls = BASIC_CONTROLS

        # Put sample together
        df = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=MODEL_SPECS[m]["asset_calc"],
            restrictions=restrictions,
            models=models,
        )
        # Load indices
        indices = pd.read_pickle(depends_on["indices"])
        indices = indices.reindex(df.index)
        indices = indices.join(df[[c for c in df if c not in indices]])

        waves = df[parameters].index.unique(level="wave")

        # Stability over time
        exclude_waves = ["temp"]
        for first_outcome in range(4, 7 + 1):
            outcome_waves = list(range(first_outcome, 7 + 1))
            indep_waves = sorted(
                [w for w in waves if w not in outcome_waves + exclude_waves]
            )
            reg_para_wave_on_wave(
                df,
                parameters,
                controls,
                outcome_waves=outcome_waves,
                indep_waves=indep_waves,
                inst_waves=indep_waves,
                std_parameters=False,
                path_dict=produces,
            )

        # Stability over time index
        outcome_waves = [5, 6, 7]
        exclude_waves = ["temp"]
        indep_waves = sorted(
            [w for w in waves if w not in outcome_waves + exclude_waves]
        )
        reg_para_wave_on_wave(
            indices,
            params_idx,
            controls,
            outcome_waves=outcome_waves,
            indep_waves=indep_waves,
            inst_waves=indep_waves,
            std_parameters=False,
            path_dict=produces,
            idx=True,
        )

        # Explain climate 2SLS
        outcome_waves = ["temp"]
        indep_waves = [4]
        inst_waves = sorted([w for w in waves if w not in outcome_waves + indep_waves])
        reg_para_wave_on_wave(
            df,
            parameters,
            controls + climate_controls,
            outcome_waves=outcome_waves,
            indep_waves=indep_waves,
            inst_waves=inst_waves,
            std_parameters=False,
            path_dict=produces,
        )

        # Explain climate 2SLS index
        reg_para_wave_on_wave(
            indices,
            params_idx,
            controls + climate_controls,
            outcome_waves=outcome_waves,
            indep_waves=indep_waves,
            inst_waves=inst_waves,
            std_parameters=False,
            path_dict=produces,
            idx=True,
        )

        # Explain climate with ORIV
        outcome_waves = ["temp"]
        indep_waves = [2, 3]
        reg_para_wave_on_wave(
            df,
            parameters,
            controls + climate_controls,
            outcome_waves=outcome_waves,
            indep_waves=indep_waves,
            inst_waves=indep_waves,
            std_parameters=False,
            path_dict=produces,
        )

        # # OLS by ambig type
        # ga = "k4"
        # group_assignments = pd.read_pickle(depends_on["group_assignments"])
        # g_man_to_g = select_manual_group_order(m, ga)
        # g_to_g_man = {j: i for i, j in g_man_to_g.items()}
        # group_assignments["type_man_sort"] = group_assignments[ga].map(g_to_g_man)

        # group_assignments["type_man_sort"] = pd.Categorical(
        #     group_assignments["type_man_sort"],
        #     ordered=True,
        # )

        # # Column names
        # n_groups = len(group_assignments["type_man_sort"].unique())
        # group_assignments["type_man_sort"] = group_assignments["type_man_sort"].replace(
        #     {g: f"{select_group_label(m, ga, g)}" for g in range(n_groups)}
        # )
        # data = df.join(group_assignments["type_man_sort"])

        # outcome_waves = [5, 6, 7]
        # indep_waves = [2, 3, 4]
        # reg_para_wave_on_wave_by_type(
        #     data,
        #     parameters,
        #     outcome_waves=outcome_waves,
        #     indep_waves=indep_waves,
        #     std_parameters=False,
        #     path_dict=produces,
        # )
