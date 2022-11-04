"""
Runs regressions of model results on individual characteristics
"""
import estimagic.visualization.estimation_table as et
import pandas as pd
import pytask
import statsmodels.formula.api as smf

from ambig_beliefs.final.utils_final import col_name_to_proper_name
from ambig_beliefs.final.utils_final import put_reg_sample_together
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


def make_mnlogit_table(
    mod,
):
    params = mod.params
    se = mod.bse
    k = mod.params.shape[1] + 1

    table = pd.DataFrame(
        index=range(params.index.shape[0] * 2 + 2),
        columns=["variable"] + [f"Group 0 v Group {g+1}" for g in range(k - 1)],
    )
    for i, v in enumerate(params.index):
        i_coef = i * 2
        i_se = i_coef + 1
        table.iloc[i_coef, 0] = v
        table.iloc[i_se, 0] = ""

        for j in range(k - 1):
            table.iloc[i_coef, j + 1] = f"{params.iloc[i, j]:.2f}"
            table.iloc[i_se, j + 1] = f"({se.iloc[i, j]:.2f})"

    for j in range(k - 1):
        table.iloc[-1, j + 1] = f"{mod.prsquared:.2f}"
        table.iloc[-2, j + 1] = mod.nobs

    table.iloc[-1, 0] = "Pseudo $R^2$"
    table.iloc[-2, 0] = "N"
    table.set_index("variable", inplace=True)
    table.index.name = ""
    return table


def reg_para_on_ind_chars(df, params, controls, cluster_var, path_out):
    df = df.copy().reset_index()
    col_name_to_proper_name = {
        "ambig_av": r"$\alpha^{AEX}$",
        "ambig_av_index": r"$\alpha^{AEX}_\text{BBLW-Index}$",
        "ll_insen": r"$\ell^{AEX}$",
        "ll_insen_index": r"$\ell^{AEX}_\text{BBLW-Index}$",
        "theta": r"$\sigma^{AEX}$",
    }

    # Make regression table that also includes columns on subset
    def run_one_reg(y, sel, models):
        f = make_formula(y=y, X=controls)
        if cluster_var:
            mod = smf.ols(formula=f, data=sel).fit(
                cov_type="cluster", cov_kwds={"groups": sel[cluster_var]}
            )
        else:
            mod = smf.ols(formula=f, data=sel).fit(cov_type="HC3")
        models.append(mod)
        return models

    models = []
    for para in params:
        models = run_one_reg(para, df, models)

    # for para in params:
    #     # duplicate dependent variable to avoid estimagic error: "If there are
    #     # repetitions in model_names, models with the same name need to be
    #     # adjacent."
    #     df[f"{para}_1"] = df[para]
    #     models = run_one_reg(f"{para}_1", df.query("all_indices_valid"), models)

    # for para in params:
    #     # duplicate dependent variable to avoid estimagic error: "If there are
    #     # repetitions in model_names, models with the same name need to be
    #     # adjacent."
    #     df[f"{para}_2"] = df[para]
    #     models = run_one_reg(
    #         f"{para}_2",
    #         df.query("at_least_2_waves_with_valid_choice_and_index"),
    #         models,
    #     )

    out = et.estimation_table(
        models,
        return_type="render_inputs",
        add_trailing_zeros=False,
        siunitx_warning=False,
        custom_param_names=variable_to_proper_name,
        custom_col_names={
            **col_name_to_proper_name,
            # **
        },
        # custom_col_groups=["Full Sample"] * len(params)
        # + ["All BBLW-indices valid"] * len(params)
        # + ["At least two waves with valid BBLW-indices"] * len(params),
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
        stats_options={
            "n_obs": "Observations",
            "rsquared_adj": "Adj. R$^2$",
            "show_dof": None,
        },
    )

    col_renaming = {
        **{f"{para}_1": col_name_to_proper_name[para] for para in params},
        **{f"{para}_2": col_name_to_proper_name[para] for para in params},
    }
    out["body"] = out["body"].rename(columns=col_renaming)
    out["footer"] = out["footer"].rename(columns=col_renaming)
    out_latex = et.render_latex(
        out["body"],
        out["footer"],
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    with open(path_out, "w") as my_table:
        my_table.write(out_latex)


def reg_risk_numeracy_on_controls(df, path_out):
    risk_num_controls = [c for c in BASIC_CONTROLS if "risk" in c or "numeracy" in c]
    non_risk_num_controls = [c for c in BASIC_CONTROLS if c not in risk_num_controls]
    models = []
    for y in risk_num_controls:
        f = make_formula(y=y, X=non_risk_num_controls)
        mod = smf.ols(formula=f, data=df).fit(cov_type="HC3")
        models.append(mod)

    out = et.estimation_table(
        models,
        return_type="latex",
        add_trailing_zeros=False,
        siunitx_warning=False,
        custom_param_names=variable_to_proper_name,
        custom_col_names=col_name_to_proper_name,
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
        stats_options={
            "n_obs": "Observations",
            "rsquared_adj": "Adj. R$^2$",
            "show_dof": None,
        },
        escape_special_characters=False,
    )
    with open(path_out, "w") as my_table:
        my_table.write(out)


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC + NAMES_INDICES_SPEC:
    depends_on = {
        "individual": OUT_DATA / "individual.pickle",
        "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
        "indices": OUT_DATA_LISS / "ambiguous_beliefs" / "indices.pickle",
        "utils_final": "utils_final.py",
        "group_assignments": OUT_ANALYSIS / f"group_assignments_{m}.pickle",
        "group_stats": OUT_ANALYSIS / f"group_stats_{m}.pickle",
        "pat_rec_and_dur_restrictions": OUT_DATA
        / "pat_rec_and_dur_restrictions.pickle",
    }
    if not MODEL_SPECS[m]["indices_params"]:
        depends_on[MODEL_SPECS[m]["est_model_name"]] = (
            OUT_UNDER_GIT
            / MODEL_SPECS[m]["est_model_name"]
            / "opt_diff_evolution"
            / "results.pickle"
        )
    produces = {
        "model_comp_on_chars": OUT_TABLES
        / m
        / ("model_comp_on_chars" + MODEL_SPECS[m]["asset_calc"] + ".tex"),
        "additional_vars_on_chars": OUT_TABLES
        / m
        / ("additional_vars_on_chars" + ".tex"),
    }
    PARAMETRIZATION[m] = {
        "depends_on": depends_on,
        "produces": produces,
        "model_spec": MODEL_SPECS[m],
    }

for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=m)
    def task_reg_model_comps_on_ind_chars(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        model_spec=kwargs["model_spec"],
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

        controls = BASIC_CONTROLS
        if model_spec["indices_params"]:
            params = ["ambig_av", "ll_insen"]
        else:
            params = ["ambig_av", "ll_insen", "theta"]

        df = df[params + controls].dropna()

        # Merge indicator whether indices are valid
        pat_rec_dur = pd.read_pickle(depends_on["pat_rec_and_dur_restrictions"])
        indices_single_waves = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=model_spec["asset_calc"],
            restrictions=model_spec["restrictions"],
            models=list(range(1, 7)),
            indices=True,
            indices_mean=False,
        )

        indices_single_waves["valid_indices"] = (
            (indices_single_waves["ll_insen"] <= 1)
            & (
                -indices_single_waves["ll_insen"]
                <= indices_single_waves["ambig_av"] * 2
            )
            & (indices_single_waves["ambig_av"] * 2 <= indices_single_waves["ll_insen"])
        )
        temp = indices_single_waves.groupby("personal_id")["valid_indices"].all()
        temp.name = "all_indices_valid"
        df = df.join(temp)

        indices_single_waves = indices_single_waves.join(pat_rec_dur["valid_choice"])
        indices_single_waves["valid_choice_and_index"] = (
            indices_single_waves["valid_choice"] & indices_single_waves["valid_indices"]
        )
        temp = (
            indices_single_waves.groupby("personal_id")["valid_choice_and_index"].sum()
            >= 2
        )
        temp.name = "at_least_2_waves_with_valid_choice_and_index"
        df = df.join(temp)
        cluster_var = (
            "personal_id"
            if (model_spec["indices_params"] and not model_spec["indices_mean"])
            else None
        )
        # controls = [c for c in controls if "numeracy" not in c]

        reg_para_on_ind_chars(
            df,
            params,
            controls,
            cluster_var,
            produces["model_comp_on_chars"],
        )

        # Also regress risk_aversion and numeracy on characteristics
        reg_risk_numeracy_on_controls(df, produces["additional_vars_on_chars"])
