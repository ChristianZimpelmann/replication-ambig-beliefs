"""
Runs regressions of model results on individual characteristics
"""
import estimagic.visualization.estimation_table as et
import numpy as np
import pandas as pd
import pytask
import statsmodels.formula.api as smf

from ambig_beliefs.final.utils_final import add_midrules_to_latex
from ambig_beliefs.final.utils_final import apply_custom_number_format
from ambig_beliefs.final.utils_final import put_reg_sample_together
from ambig_beliefs.final.utils_final import select_group_label
from ambig_beliefs.final.utils_final import select_manual_group_order
from ambig_beliefs.final.utils_final import variable_to_proper_name
from config import BASIC_CONTROLS
from config import MODEL_SPECS
from config import NAMES_INDICES_SPEC
from config import NAMES_MAIN_SPEC
from config import NAMES_ROBUSTNESS_SPEC
from config import OUT_ANALYSIS
from config import OUT_DATA
from config import OUT_DATA_LISS
from config import OUT_TABLES
from config import OUT_UNDER_GIT


def make_formula(y, X):
    regs = " + ".join(X)
    return f"{y} ~  {regs}"


def pval_to_star(pval):
    if pval <= 0.01:
        star = "***"
    elif pval <= 0.05:
        star = "**"
    elif pval <= 0.1:
        star = "*"
    else:
        star = ""
    return star


def make_mnlogit_margeff_table(mod, col_name_to_proper_name, path_out):
    margeff_res = mod.get_margeff(at="overall")
    margeff = margeff_res.margeff
    margeff_bse = margeff_res.margeff_se
    pvalues = margeff_res.pvalues
    r_squared = mod.prsquared
    n_obs = mod.nobs
    k = mod.params.shape[1] + 1

    # Split each column in seperate list entry before feeding into estimation table
    split_col_info = []
    for i in range(k):
        params = pd.DataFrame(
            {
                "value": margeff[:, i],
                "standard_error": margeff_bse[:, i],
                "p_value": pvalues[:, i],
            },
            index=mod.params.index[1:],
        )
        info = {"n_obs": n_obs, "pseudo_r2": r_squared}
        split_col_info.append({"params": params, "info": info, "name": str(i)})

    out = et.estimation_table(
        split_col_info,
        return_type="render_inputs",
        add_trailing_zeros=False,
        siunitx_warning=False,
        custom_param_names=variable_to_proper_name,
        custom_col_names=col_name_to_proper_name,
        custom_col_groups=["Ambiguity types"] * k,
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
        stats_options={
            "n_obs": "Observations",
            "pseudo_r2": "Pseudo R$^2$",
            # "Adj. R$^2$": "rsquared_adj",
            # "Residual Std. Error": "resid_std_err",
            # "F Statistic": "fvalue",
            "show_dof": None,
        },
    )

    with open(path_out, "w") as my_table:
        out_latex = et.render_latex(
            out["body"],
            out["footer"],
            append_notes=False,
            show_footer=True,
            siunitx_warning=False,
            escape_special_characters=False,
        )
        # out = add_midrules_to_latex(
        #     out,
        #     [3],
        #     midrule_text=rf"\cmidrule{{2-{len(table.columns) + 1}}}",
        # )
        my_table.write(out_latex)


# def make_mnlogit_coef_table(mod):
#     params = mod.params
#     se = mod.bse
#     pvalues = mod.pvalues
#     r2 = mod.prsquared
#     n = mod.nobs
#     k = mod.params.shape[1] + 1

#     table = pd.DataFrame(
#         index=range(params.index.shape[0] * 2 + 2),
#         columns=["variable"] + [f"Group 0 v Group {g+1}" for g in range(k - 1)],
#     )
#     for i, v in enumerate(params.index):
#         i_coef = i * 2
#         i_se = i_coef + 1
#         table.iloc[i_coef, 0] = v
#         table.iloc[i_se, 0] = ""

#         for j in range(k - 1):
#             pvalue = pvalues.iloc[i, j]
#             star = pval_to_star(pvalue)
#             table.iloc[i_coef, j + 1] = f"{params.iloc[i, j]:.2f}" + star
#             table.iloc[i_se, j + 1] = f"({se.iloc[i, j]:.2f})"

#     for j in range(k - 1):
#         table.iloc[-1, j + 1] = f"{r2:.2f}"
#         table.iloc[-2, j + 1] = n

#     table.iloc[-1, 0] = "Pseudo $R^2$"
#     table.iloc[-2, 0] = "N"
#     table.set_index("variable", inplace=True)
#     table.index.name = ""
#     return table


def create_overview_table(
    produces, ga, asset_calc, df, controls, col_name_to_proper_name, model_spec
):
    # # Build overview table for groups
    # choices_pay_out_info = pd.read_pickle(depends_on["choices_pay_out_info"])

    # # Add data about probability won
    # choices_pay_out_info = choices_pay_out_info.query("wave.isin([1, 2, 3])").copy()
    # choices_pay_out_info.groupby("personal_id")["p_won_20_eur"].mean()
    # mean_won = choices_pay_out_info.groupby("personal_id")["p_won_20_eur"].mean() * 20
    # df = df.join(mean_won)

    # # Add data about durations
    # durations = pd.read_pickle(depends_on["durations"])
    # temp = durations.groupby(["personal_id", "wave"])["duration_in_s"].sum()
    # temp /= 60
    # temp.name = "duration_in_m"
    # df = df.join(temp.groupby("personal_id").mean())

    df["total_financial_assets"] /= 1000
    df["net_income"] /= 1000
    df["edu_lower_secondary_and_lower"] = (
        df["edu"] == "lower_secondary_and_lower"
    ).astype(float)
    df["edu_upper_secondary"] = (df["edu"] == "upper_secondary").astype(float)
    df["edu_tertiary"] = (df["edu"] == "tertiary").astype(float)
    df["female"] = df["female"].astype(float)

    # Select overview vars
    params = (
        ["ambig_av", "ll_insen"]
        if model_spec["indices_params"]
        else ["ambig_av", "ll_insen", "theta"]
    )
    overview_vars = []
    for var in (
        params
        + [
            "edu_lower_secondary_and_lower",
            "edu_upper_secondary",
            "edu_tertiary",
        ]
        + [c for c in controls if c != "edu"]
    ):
        if var.endswith("_groups"):
            overview_vars.append(var[:-7])
        else:
            overview_vars.append(var)

    overview_table = df.groupby("group")[overview_vars].mean()

    groupby = df[overview_vars + ["group"]].groupby("group")
    means = groupby.mean()
    # means["N"] = groupby.count().iloc[:, 0]

    se = groupby.std() / np.sqrt(groupby.count() - 1)
    se.columns = ["se_" + c for c in se]
    overview_table = pd.concat([means, se], axis=1).T
    overview_table = apply_custom_number_format(
        overview_table, int_cols=[], number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}")
    )

    # Change order of rows
    index = []
    for c in overview_vars:
        index.append(c)
        index.append("se_" + c)
        overview_table.loc["se_" + c] = overview_table.loc["se_" + c].apply(
            lambda x: f"({x})"
        )
        overview_table.loc[c] = overview_table.loc[c].apply(lambda x: f"{x}")

    # index.append("N")
    overview_table = overview_table.reindex(index)
    final_index = ["" if i.startswith("se_") else i for i in index]
    overview_table.index = final_index

    # # Take median instead of mean for income and financial assets
    # for c in ["total_financial_assets", "net_income"]:
    #     overview_table[c] = df.groupby("group")[c].median()

    # overview_table["SubsetViolations"] = group_stats.loc[ga]["mean_midp_dist_to_no_vio"]

    # add share
    overview_table = overview_table.T
    sum_groups = df["group"].value_counts()
    overview_table.insert(0, "Share", sum_groups / sum_groups.sum())
    overview_table["Share"] = overview_table["Share"].round(2).apply(lambda x: f"{x}")
    overview_table = overview_table.rename_axis(None)

    overview_table = overview_table.T.rename(
        index=variable_to_proper_name, columns=col_name_to_proper_name
    )
    overview_table = overview_table.rename(
        index={
            "$\\alpha$": r"$\alpha^{AEX}$",
            "$\\ell$": r"$\ell^{AEX}$",
            "$\\sigma$": r"$\sigma^{AEX}$",
        }
    )

    # Single-level column
    path_out = produces[f"group_on_chars_{ga}_overview_single_col" + asset_calc]
    with open(path_out, "w") as my_table:
        # formatting
        out = et.render_latex(
            overview_table,
            {},
            append_notes=False,
            render_options={},
            show_footer=False,
            siunitx_warning=False,
            escape_special_characters=False,
        )

        out = add_midrules_to_latex(out, [5, 9])
        my_table.write(out)

    overview_table.columns = pd.MultiIndex.from_product(
        [["Ambiguity types"], overview_table.columns]
    )

    path_out = produces[f"group_on_chars_{ga}_overview" + asset_calc]
    with open(path_out, "w") as my_table:
        # formatting
        # out = apply_custom_number_format(
        #     overview_table, int_cols=[], number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}")
        # )
        out = et.render_latex(
            overview_table,
            {},
            append_notes=False,
            render_options={},
            show_footer=False,
            siunitx_warning=False,
            escape_special_characters=False,
        )
        out = add_midrules_to_latex(
            out, [3], midrule_text=rf"\cmidrule{{2-{len(overview_table.columns) + 1}}}"
        )
        if model_spec["indices_params"]:
            out = add_midrules_to_latex(out, [9, 14])
        else:
            out = add_midrules_to_latex(out, [9, 16])
        my_table.write(out)


def std_single_estimates_by_type(df, params, col_name_to_proper_name, path_out):
    types = df.groupby("personal_id")["type_man_sort"].first()
    std_params = df[params].groupby("personal_id").std()
    mean = std_params.groupby(types).mean()
    mean.index.name = None

    mean = apply_custom_number_format(
        mean, int_cols=[], number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}")
    )
    mean = mean.rename(index=col_name_to_proper_name, columns=variable_to_proper_name)
    mean = mean.rename(
        columns={
            "$\\alpha$": r"$\alpha^{AEX}$",
            "$\\ell$": r"$\ell^{AEX}$",
            "$\\sigma$": r"$\sigma^{AEX}$",
        }
    )

    se = std_params.groupby(types).std() / np.sqrt(
        std_params.groupby(types).count() - 1
    )
    se = se.rename(index=col_name_to_proper_name, columns=variable_to_proper_name)
    se = se.rename(
        columns={
            "$\\alpha$": r"$\alpha^{AEX}$",
            "$\\ell$": r"$\ell^{AEX}$",
            "$\\sigma$": r"$\sigma^{AEX}$",
        }
    )
    se = apply_custom_number_format(
        se, int_cols=[], number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}")
    )
    se.index = ["se_" + str(i) for i in se.index]
    out = pd.concat([mean, se], axis=0)

    # Change order of rows
    new_index = []
    for c in mean.index:
        new_index.append(c)
        new_index.append("se_" + c)
        out.loc["se_" + c] = out.loc["se_" + c].apply(lambda x: f"({x})")
        out.loc[c] = out.loc[c].apply(lambda x: f"{x}")

    # new_index.append("N")
    out = out.reindex(new_index)
    final_index = ["" if i.startswith("se_") else i for i in new_index]
    out.index = final_index
    with open(path_out, "w") as my_table:
        out = et.render_latex(
            out,
            {},
            append_notes=False,
            render_options={},
            show_footer=False,
            siunitx_warning=False,
            escape_special_characters=False,
        )
        my_table.write(out)

    return out


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC + NAMES_INDICES_SPEC:

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
            depends_on.update(
                {
                    mod: OUT_UNDER_GIT / mod / "opt_diff_evolution" / "results.pickle"
                    for mod in MODEL_SPECS[m]["wbw_models"]
                    + [MODEL_SPECS[m]["climate_model"]]
                }
            )
        produces = {
            # f"group_on_chars_{ga}_coef"
            # + MODEL_SPECS[m]["asset_calc"]: OUT_TABLES
            # / m
            # / (f"group_on_chars_{ga}_coef" + MODEL_SPECS[m]["asset_calc"] + ".tex"),
            f"group_on_chars_{ga}_margeff"
            + MODEL_SPECS[m]["asset_calc"]: OUT_TABLES
            / m
            / (f"group_on_chars_{ga}_margeff" + MODEL_SPECS[m]["asset_calc"] + ".tex"),
            f"group_on_chars_{ga}_overview"
            + MODEL_SPECS[m]["asset_calc"]: OUT_TABLES
            / m
            / (f"group_on_chars_{ga}_overview" + MODEL_SPECS[m]["asset_calc"] + ".tex"),
            f"group_on_chars_{ga}_overview_single_col"
            + MODEL_SPECS[m]["asset_calc"]: OUT_TABLES
            / m
            / (
                f"group_on_chars_{ga}_overview_single_col"
                + MODEL_SPECS[m]["asset_calc"]
                + ".tex"
            ),
            "mean_std_by_type": OUT_TABLES / m / (f"mean_std_by_type_{ga}.tex"),
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
    def task_reg_groups_on_ind_chars(
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
        df_single_waves = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=model_spec["asset_calc"],
            restrictions=model_spec["restrictions"],
            models=model_spec["wbw_models"],
            indices=model_spec["indices_params"],
        )
        if not model_spec["indices_params"] or model_spec["indices_mean"]:
            df = df.droplevel(level="wave")

        group_assignments = pd.read_pickle(depends_on["group_assignments"])

        # Sort mapping of manual sorting of groups
        g_man_to_g = select_manual_group_order(m, ga)
        g_to_g_man = {j: i for i, j in g_man_to_g.items()}

        df["group_original"] = group_assignments[ga]
        df["group"] = df["group_original"].map(g_to_g_man)
        df_single_waves = df_single_waves.join(
            group_assignments[ga].map(g_to_g_man)
        ).rename(columns={ga: "type_man_sort"})
        # Column names
        n_groups = len(group_assignments[ga].unique())
        col_name_to_proper_name = {
            g: f"{select_group_label(m, ga, g)}" for g in range(n_groups)
        }

        # organise objects for regressions
        outcomes = ["group"]
        controls = BASIC_CONTROLS  # + judged_freqs + personality

        formula = make_formula(y=outcomes[0], X=controls)
        if model_spec["indices_params"] and not model_spec["indices_mean"]:
            sel = df.dropna(subset=[outcomes[0]] + controls)
            mod = smf.mnlogit(formula, data=sel).fit(
                cov_type="cluster",
                cov_kwds={"groups": sel.reset_index()["personal_id"]},
            )
        else:
            mod = smf.mnlogit(formula, data=df).fit(cov_type="HC3")

        # coef_table = make_mnlogit_coef_table(mod)
        # coef_table = coef_table.rename(index=variable_to_proper_name)
        # coef_table.columns = pd.MultiIndex.from_product(
        #     [["Ambiguity types"], coef_table.columns]
        # )
        # path_out = produces[f"group_on_chars_{ga}_coef" + asset_calc]
        # with open(path_out, "w") as my_table:
        #     fmt = "l" + ">{\\raggedleft\\arraybackslash}p{1.2cm}" * (k - 1)
        #     out = coef_table.to_latex(escape=False, column_format=fmt, multicolumn_format="c")
        #     out = add_midrules_to_latex(
        #         out, [3], midrule_text=rf"\cmidrule{{2-{len(coef_table.columns) + 1}}}"
        #     )
        #     my_table.write(out)

        make_mnlogit_margeff_table(
            mod,
            {str(g): f"{select_group_label(m, ga, g)}" for g in range(n_groups)},
            produces[f"group_on_chars_{ga}_margeff" + model_spec["asset_calc"]],
        )

        create_overview_table(
            produces,
            ga,
            model_spec["asset_calc"],
            df,
            controls,
            col_name_to_proper_name,
            model_spec,
        )
        params = (
            ["ambig_av", "ll_insen"]
            if model_spec["indices_params"]
            else ["ambig_av", "ll_insen", "theta"]
        )
        std_single_estimates_by_type(
            df_single_waves,
            params,
            col_name_to_proper_name,
            produces["mean_std_by_type"],
        )
