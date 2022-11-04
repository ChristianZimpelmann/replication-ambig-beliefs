"""
Run regressions of climate related ambiguity parameters on aex related ambiguity parameters.
"""
import estimagic.visualization.estimation_table as et
import pandas as pd
import pytask
import statsmodels.formula.api as smf

from ambig_beliefs.final.utils_final import put_reg_sample_together
from ambig_beliefs.final.utils_final import variable_to_proper_name
from config import BASIC_CONTROLS
from config import MODEL_SPECS
from config import NAMES_MAIN_SPEC
from config import OUT_DATA
from config import OUT_TABLES
from config import OUT_UNDER_GIT
from config import ROOT


def reg_parameter_over_waves_on_chars(df, control_variables, path_out=None):
    params = ["ambig_av", "ll_insen", "theta"]
    df.dropna(subset=params, inplace=True)
    df = df.sort_index().reset_index()

    models = []
    for v in params:
        mod = smf.ols(f"{v} ~ C(wave)", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["personal_id"]}
        )
        models.append(mod)

        sel = df.dropna(subset=control_variables)
        mod = smf.ols(
            f"{v} ~ C(wave) +  {' + '.join(control_variables)}", data=sel
        ).fit(cov_type="cluster", cov_kwds={"groups": sel["personal_id"]})
        models.append(mod)

        sel = df.query("completed_at_least_6_waves_with_sensible_choices_excl1").dropna(
            subset=control_variables
        )
        mod = smf.ols(
            f"{v} ~ C(wave) +  {' + '.join(control_variables)}", data=sel
        ).fit(cov_type="cluster", cov_kwds={"groups": sel["personal_id"]})
        models.append(mod)

    out = et.estimation_table(
        models,
        return_type="render_inputs",
        add_trailing_zeros=False,
        siunitx_warning=False,
        custom_param_names={**variable_to_proper_name},
        custom_col_groups=[r"$\alpha$"] * 3 + [r"$\ell$"] * 3 + [r"$\sigma$"] * 3,
        #     custom_col_names=["full sample",
        # "full sample + controls", "participated in all waves"] * 3,
        #     custom_col_names={    "ambig_av": r"\alpha",
        # "ll_insen": r"\ell",
        # "theta": r"\sigma"},
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
        stats_options={
            "n_obs": "Observations",
            # "prsquared": "Pseudo R$^2$",
            "rsquared_adj": "Adj. R$^2$",
            "show_dof": None,
        },
    )
    new_row = pd.DataFrame(
        [["No", "No", "Yes"] * 3],
        columns=out["footer"].columns,
        index=pd.MultiIndex.from_arrays([["Balanced sample"]]),
    )
    out["footer"] = pd.concat([new_row, out["footer"]])

    out_latex = et.render_latex(
        out["body"],
        out["footer"],
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    if path_out:
        with open(path_out, "w") as my_table:
            my_table.write(out_latex)
    return out["body"]


def reg_parameter_climate_vs_AEX_on_chars(df, control_variables, path_out=None):
    params = ["ambig_av", "ll_insen", "theta"]
    df = df.reset_index().copy()
    df["climate_wave"] = df["wave"] == "temp"
    df = df.dropna(subset=params)
    df = df.sort_index()

    models = []
    for v in params:
        mod = smf.ols(f"{v} ~ climate_wave", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["personal_id"]}
        )
        models.append(mod)

        sel = df.dropna(subset=control_variables)
        mod = smf.ols(
            f"{v} ~ climate_wave +  {' + '.join(control_variables)}", data=sel
        ).fit(cov_type="cluster", cov_kwds={"groups": sel["personal_id"]})
        models.append(mod)

        sel = df.query("completed_at_least_6_waves_with_sensible_choices_excl1").dropna(
            subset=control_variables
        )
        mod = smf.ols(
            f"{v} ~ climate_wave +  {' + '.join(control_variables)}", data=sel
        ).fit(cov_type="cluster", cov_kwds={"groups": sel["personal_id"]})
        models.append(mod)

    out = et.estimation_table(
        models,
        return_type="render_inputs",
        add_trailing_zeros=False,
        siunitx_warning=False,
        custom_param_names={**variable_to_proper_name},
        custom_col_groups=[r"$\alpha$"] * 3 + [r"$\ell$"] * 3 + [r"$\sigma$"] * 3,
        #     custom_col_names=["full sample",
        # "full sample + controls", "participated in all waves"] * 3,
        #     custom_col_names={    "ambig_av": r"\alpha",
        # "ll_insen": r"\ell",
        # "theta": r"\sigma"},
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
        stats_options={
            "n_obs": "Observations",
            # "prsquared": "Pseudo R$^2$",
            "rsquared_adj": "Adj. R$^2$",
            "show_dof": None,
        },
    )
    new_row = pd.DataFrame(
        [["No", "No", "Yes"] * 3],
        columns=out["footer"].columns,
        index=pd.MultiIndex.from_arrays([["Balanced sample"]]),
    )
    out["footer"] = pd.concat([new_row, out["footer"]])

    out_latex = et.render_latex(
        out["body"],
        out["footer"],
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    if path_out:
        with open(path_out, "w") as my_table:
            my_table.write(out_latex)
    return out["body"]


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC:
    wbw_models = MODEL_SPECS[m]["wbw_models"]
    climate_model = MODEL_SPECS[m]["climate_model"]

    depends_on = {
        "utils_final": ROOT / "ambig_beliefs" / "final" / "utils_final.py",
        "pat_rec_and_dur_restrictions": OUT_DATA
        / "pat_rec_and_dur_restrictions.pickle",
        "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
        "individual": OUT_DATA / "individual.pickle",
        m: OUT_UNDER_GIT / m / "opt_diff_evolution" / "results.pickle",
    }
    depends_on.update(
        {
            mod: OUT_UNDER_GIT / mod / "opt_diff_evolution" / "results.pickle"
            for mod in wbw_models + [climate_model]
        }
    )
    produces = {
        "parameter_over_waves_on_chars": OUT_TABLES
        / m
        / "parameter_over_waves_on_chars.tex",
        "parameter_climate_vs_AEX_on_chars": OUT_TABLES
        / m
        / "parameter_climate_vs_AEX_on_chars.tex",
    }
    PARAMETRIZATION[m] = {
        "depends_on": depends_on,
        "produces": produces,
        "wbw_models": wbw_models,
        "climate_model": climate_model,
    }

for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=m)
    def task_reg_parameter_over_waves_on_chars(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        m=m,
        wbw_models=kwargs["wbw_models"],
        climate_model=kwargs["climate_model"],
        asset_calc=MODEL_SPECS[m]["asset_calc"],
        restrictions=MODEL_SPECS[m]["restrictions"],
    ):

        df = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=asset_calc,
            restrictions=restrictions,
            models=wbw_models + [climate_model],
        )

        # parameter changes
        reg_parameter_over_waves_on_chars(
            df.query("wave != 'temp'").copy(),
            control_variables=BASIC_CONTROLS,
            path_out=produces["parameter_over_waves_on_chars"],
        )

        # AEX vs climate
        reg_parameter_climate_vs_AEX_on_chars(
            df,
            control_variables=BASIC_CONTROLS,
            path_out=produces["parameter_climate_vs_AEX_on_chars"],
        )
