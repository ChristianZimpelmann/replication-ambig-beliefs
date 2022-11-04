"""
Regresses presence of set-monotonicity violation on the gap in judged historical
frequencies of the events involved in the error.
"""
import estimagic.visualization.estimation_table as et
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import scipy.stats
import seaborn as sns
import statsmodels.formula.api as smf

from ambig_beliefs.final.utils_final import add_midrules_to_latex
from ambig_beliefs.final.utils_final import apply_custom_number_format
from ambig_beliefs.final.utils_final import put_reg_sample_together
from config import MODEL_SPECS
from config import NAMES_MAIN_SPEC
from config import NAMES_ROBUSTNESS_SPEC
from config import OUT_DATA
from config import OUT_FIGURES
from config import OUT_TABLES
from config import OUT_UNDER_GIT
from config import ROOT


def make_set_mono_viol_fechner_evidence_plot(df, path_out):

    # aggregated = df.groupby("superset_subset").mean()

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=False)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    # sns.regplot(
    #     x="judged_freq_gap",
    #     y="min_dist_is_vio",
    #     data=aggregated,
    #     ax=ax[0],
    #     # color="black",
    #     ci=None,
    #     scatter=True,
    #     # x_bins=15,
    #     scatter_kws={"alpha": 1, "s": 60},
    # )

    sns.regplot(
        x="judged_freq_gap",
        y="min_dist_is_vio",
        data=df,
        ax=ax,
        # color="black",
        ci=None,
        scatter=True,
        fit_reg=False,
        x_bins=10,
        scatter_kws={"alpha": 1, "s": 40},
    )

    # calculate linear regression function
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        x=df["judged_freq_gap"], y=df["min_dist_is_vio"]
    )

    # plot the regression line
    reg_line_start = 0
    reg_line_end = 0.9
    new_x = np.arange(
        reg_line_start, reg_line_end, (reg_line_end - reg_line_start) / 250.0
    )
    ax.plot(new_x, intercept + slope * new_x, color="black", linestyle="-", alpha=0.8)

    # ax.set_ylim(0, 0.4)
    # ax[0].set_xlim(0.2, 0.5)
    # ax[0].set_xlabel("Jud. freq. superset - Jud. freq. subset")
    # ax[0].set_ylabel("Error frequency")
    # ax[0].set_ylim(0.04, 0.27)
    # ax[0].set_title("A) By superset-subset pair", pad=20)  # , y=-0.35)

    ax.set_xlim(-0.02, 1)
    ax.set_ylim(0.04, 0.2)
    ax.set_xlabel("Jud. freq. superset - Jud. freq. subset")
    ax.set_ylabel("Error frequency")
    # ax.set_title(
    #     r"B) By subject $\times$ superset-subset pair (binscatter)", pad=20
    # )  # , y=-0.35

    fig.tight_layout()
    fig.savefig(path_out, format="pdf", bbox_inches="tight")
    fig.clear()


def descriptive_set_monotonicity_violations(df_full, path_out, ss_pairs):
    outcome = "min_dist_is_vio"

    df_aex = df_full.query("wave != 'temp'").copy()
    df_climate = df_full.query("wave == 'temp'").copy()

    # out = df.groupby("superset_subset")[[outcome, "judged_freq_gap"]].mean()
    out_aex = (
        df_aex.groupby(["personal_id", "superset_subset"])
        .mean()
        .reset_index()
        .groupby("superset_subset")[[outcome]]
        .mean()
    )
    out_aex = out_aex.reindex(ss_pairs)

    out_climate = (
        df_climate.groupby("superset_subset")[[outcome]].mean().reindex(ss_pairs)
    )

    out = pd.concat({"AEX": out_aex, "climate": out_climate}, axis=1)

    # Make Multiindex in both index and columns
    out.index = pd.MultiIndex.from_tuples(
        out.index.str.split("_").tolist(), names=["Superset", "Subset"]
    )
    out.columns = pd.MultiIndex.from_tuples(
        [(outcome, "AEX"), (outcome, "climate")],
        names=["", ""],
    )

    # Add percentage of subjects that have any violation in a wave
    any_violation_aex = (
        df_aex.groupby(["personal_id", "wave"])[outcome].mean() > 0
    ).mean()
    any_violation_climate = (
        df_climate.groupby(["personal_id"])[outcome].mean() > 0
    ).mean()
    df_excl_0_aex = df_aex.loc[~df_aex["superset_subset"].isin(["0_1", "2c_0"])]
    any_violation_excl_0_aex = (
        df_excl_0_aex.groupby(["personal_id", "wave"])[outcome].mean() > 0
    ).mean()
    df_excl_0_climate = df_climate.loc[
        ~df_climate["superset_subset"].isin(["0_1", "2c_0"])
    ]
    any_violation_excl_0_climate = (
        df_excl_0_climate.groupby(["personal_id", "wave"])[outcome].mean() > 0
    ).mean()

    out.loc[
        ("Any violation", r"excluding $E^{S}_0$"), (outcome, "AEX")
    ] = any_violation_excl_0_aex
    out.loc[
        ("Any violation", r"including $E^{S}_0$"), (outcome, "AEX")
    ] = any_violation_aex
    out.loc[
        ("Any violation", r"excluding $E^{S}_0$"), (outcome, "climate")
    ] = any_violation_excl_0_climate
    out.loc[
        ("Any violation", r"including $E^{S}_0$"), (outcome, "climate")
    ] = any_violation_climate
    # col_name = (
    #     r"\makecell{Individuals with any\\ set-monotonicity violation \\ per wave}"
    # )
    # out.loc[("2c", "1"), (col_name, r"excluding $E^{AEX}_0$")] = any_violation_excl_0
    # out[(col_name, r"excluding $E^{AEX}_0$")] = out[
    #     (col_name, r"excluding $E^{AEX}_0$")
    # ]
    # col_name = (
    #     r"\makecell{Individuals with any\\ set-monotonicity violation \\ per wave}"
    # )
    # out.loc[("3c", "1"), (col_name, r"including $E^{AEX}_0$")] = any_violation
    # out[(col_name, r"including $E^{AEX}_0$")] = out[
    #     (col_name, r"including $E^{AEX}_0$")
    # ]

    # Rename columns and indices
    out = out.rename(
        index={
            "0": r"$E^{S}_0$",
            "1": r"$E^{S}_1$",
            "2": r"$E^{S}_2$",
            "3": r"$E^{S}_3$",
            "1c": r"$E^{S}_{1, C}$",
            "2c": r"$E^{S}_{2, C}$",
            "3c": r"$E^{S}_{3, C}$",
            # "0_1": r"$E^{AEX}_0 \supseteq E^{AEX}_1$",
            # "1c_2": r"$E^{AEX}_{1, C} \supseteq E^{AEX}_2$",
            # "1c_3": r"$E^{AEX}_{1, C} \supseteq E^{AEX}_3$",
            # "2c_0": r"$E^{AEX}_{2, C} \supseteq E^{AEX}_0$",
            # "2c_1": r"$E^{AEX}_{2, C} \supseteq E^{AEX}_1$",
            # "2c_3": r"$E^{AEX}_{2, C} \supseteq E^{AEX}_3$",
            # "3c_1": r"$E^{AEX}_{3, C} \supseteq E^{AEX}_1$",
            # "3c_2": r"$E^{AEX}_{3, C} \supseteq E^{AEX}_2$",
        },
        columns={
            outcome: r"\makecell{Rate of set-monotonicity violations}",
            # "to_replace_later": ""
            # "judged_freq_gap": r"$\Delta$ Judged frequencies",
        },
    )
    # any_violation_total = (
    #     df_full.groupby(["personal_id"])["min_dist_is_vio"].mean() > 0
    # ).mean()

    # out.loc[
    #     ("", r"\multicolumn{2}{l}{Mean set-monotonicity violation rate}"),
    #     col_name,
    # ] = out[col_name].mean()

    # out.loc[
    #     ("", r"\multicolumn{2}{l}{Any violation over all waves}"), col_name
    # ] = any_violation_total

    # formatting
    out = apply_custom_number_format(
        out, int_cols=[], number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}")
    )
    # Delete & for multiindex in index
    # out = out.rename("to_replace_later", "")

    out = et.render_latex(
        out,
        {},
        append_notes=False,
        render_options={},
        # padding=2,
        show_footer=False,
        siunitx_warning=False,
        escape_special_characters=False,
    )

    # Add midrules
    out = add_midrules_to_latex(out, [13, 16])

    with open(path_out, "w") as file:
        file.write(out)


def create_regression_table(df, produces):
    # fit models
    y = "min_dist_is_vio"
    only_intercept = smf.ols(formula=f"{y} ~ 1", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["personal_id"]}
    )
    pooled_ols = smf.ols(formula=f"{y} ~ judged_freq_gap", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["personal_id"]}
    )
    pair_fe = smf.ols(
        formula=f"{y} ~ judged_freq_gap + C(superset_subset)", data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df["personal_id"]})
    pair_ind_fe = smf.ols(
        formula=f"{y} ~ judged_freq_gap +  C(superset_subset) +  C(personal_id)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["personal_id"]})

    models = [only_intercept, pooled_ols, pair_fe, pair_ind_fe]

    out = et.estimation_table(
        models,
        return_type="render_inputs",
        # left_decimals=1,
        add_trailing_zeros=False,
        siunitx_warning=False,
        custom_param_names={
            "judged_freq_gap": "Judged frequencies (superset - subset)"
        },
        custom_col_groups=[
            "Dependent variable: Set-monotonicity violation",
            "Dependent variable: Set-monotonicity violation",
            "Dependent variable: Set-monotonicity violation",
            "Dependent variable: Set-monotonicity violation",
        ],
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
        stats_options={
            "n_obs": "Observations",
            # "prsquared": "Pseudo R$^2$",
            # "Adj. R$^2$": "rsquared_adj",
            # "Residual Std. Error": "resid_std_err",
            # "F Statistic": "fvalue",
            "show_dof": None,
        },
    )
    out_no_controls = out["body"].iloc[:4, :].copy()

    # Remove R squared for regression without regressors
    out_no_controls.iloc[-1, 0] = ""

    # Remove intercept for last two columns
    out_no_controls.iloc[[0, 1], [2, 3]] = ""
    new_row = pd.DataFrame(
        [["No", "No", "No", "Yes"]],
        columns=out["footer"].columns,
        index=pd.MultiIndex.from_arrays([["Individual fixed effects"]]),
    )
    out["footer"] = pd.concat([new_row, out["footer"]])
    new_row = pd.DataFrame(
        [["No", "No", "Yes", "Yes"]],
        columns=out["footer"].columns,
        index=pd.MultiIndex.from_arrays([["Superset-subset pair fixed effects"]]),
    )
    out["footer"] = pd.concat([new_row, out["footer"]])

    out_latex = et.render_latex(
        out_no_controls,
        out["footer"],
        append_notes=False,
        show_footer=True,
        siunitx_warning=False,
    )

    # table = summary_col(
    #     results=models,
    #     regressor_order=["Intercept", "judged_freq_gap"],
    #     drop_omitted=True,
    #     float_format="%.3f",
    #     stars=True,
    #     info_dict={
    #         "N": lambda x: "{:d}".format(int(x.nobs)),
    #         "$R^2$": lambda x: f"{x.rsquared:.2f}",
    #     },
    # )
    # a = table.tables[0].iloc[:4, :]
    # b = pd.DataFrame(
    #     [["No", "No", "Yes", "Yes"], ["No", "No", "No", "Yes"]],
    #     index=["pair_fe", "person_fe"],
    #     columns=a.columns,
    # )
    # c = table.tables[0].iloc[-2:, :]
    # table = pd.concat([a, b, c])
    # table = table.rename(
    #     index={
    #         "judged_freq_gap": "Judged frequencies (superset - subset)",
    #         "pair_fe": "Superset-subset pair fixed effects",
    #         "person_fe": "Individual fixed effects",
    #         "N": "N subjects $\times$ superset-subset pairs",
    #     }
    # )

    # col_idx = pd.MultiIndex.from_product(
    #     iterables=[
    #         ["Average set-monotonicity violations across waves"],
    #         [""] * len(table.columns),
    #     ],
    #     names=["", ""],
    # )
    # table.columns = col_idx

    path_out = produces["set_monotonicity_violations_on_judged_freqs"]
    with open(path_out, "w") as my_table:
        my_table.write(out_latex)


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC:
    est_model_name = MODEL_SPECS[m]["est_model_name"]
    wbw_models = MODEL_SPECS[m]["wbw_models"]
    climate_model = MODEL_SPECS[m]["climate_model"]

    produces = {}
    depends_on = {
        "individual": OUT_DATA / "individual.pickle",
        "set_monotonicity_violations": OUT_DATA / "set_monotonicity_violations.pickle",
        "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
        "utils_final": ROOT / "ambig_beliefs" / "final" / "utils_final.py",
        m: OUT_UNDER_GIT / est_model_name / "opt_diff_evolution" / "results.pickle",
    }
    depends_on.update(
        {
            mod: OUT_UNDER_GIT / mod / "opt_diff_evolution" / "results.pickle"
            for mod in wbw_models + [climate_model]
        }
    )
    produces = {
        "set_monotonicity_violations_on_judged_freqs": OUT_TABLES
        / m
        / "set_monotonicity_violations_on_judged_freqs.tex",
        "set_monotonicity_violations_fechner_evidence": OUT_FIGURES
        / m
        / "set_monotonicity_violations_fechner_evidence.pdf",
        "descriptive_stats_set_monotonicity_violation": OUT_TABLES
        / m
        / "descriptive_stats_set_monotonicity_violation.tex",
    }
    PARAMETRIZATION[m] = {
        "depends_on": depends_on,
        "produces": produces,
        "wbw_models": wbw_models,
        "climate_model": climate_model,
    }

for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=m)
    def task_reg_set_monotonicity_violations_on_judged_freqs(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        wbw_models=kwargs["wbw_models"],
        climate_model=kwargs["climate_model"],
        asset_calc=MODEL_SPECS[m]["asset_calc"],
        restrictions=MODEL_SPECS[m]["restrictions"],
    ):
        restrictions = "&".join(restrictions.split(","))
        individual = pd.read_pickle(depends_on["individual"])
        set_mono_viol = pd.read_pickle(depends_on["set_monotonicity_violations"])

        # Select valid responses and requested waves
        df = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=asset_calc,
            restrictions=restrictions,
            models=wbw_models + [climate_model],
        )
        set_mono_viol = set_mono_viol.reindex(df.index)

        ss_pairs = ["1c_2", "1c_3", "2c_1", "2c_3", "3c_1", "3c_2", "0_1", "2c_0"]
        events = ["0", "1", "2", "3", "1c", "2c", "3c"]
        # combined: uses additive values if necessary. nocheck: uses first values people
        # entered, possibly not additive.
        suffix = "combined"
        individual["0"] = individual["hist_perf_e0_nocheck"]
        individual["1"] = individual[f"hist_perf_e1_{suffix}"]
        individual["2"] = individual[f"hist_perf_e2_{suffix}"]
        individual["3"] = individual[f"hist_perf_e3_{suffix}"]
        individual["1c"] = 1 - individual[f"hist_perf_e1_{suffix}"]
        individual["2c"] = 1 - individual[f"hist_perf_e2_{suffix}"]
        individual["3c"] = 1 - individual[f"hist_perf_e3_{suffix}"]
        individual.dropna(subset=events, inplace=True)
        freqs_are_valid = (
            individual[events].dropna().apply(lambda x: x.between(0, 1)).all(axis=1)
        )
        judged_frequency_pair_gap = pd.DataFrame(
            index=individual[freqs_are_valid].index
        )
        for pair in ss_pairs:
            superset, subset = pair.split("_")
            gap = individual[superset] - individual[subset]
            judged_frequency_pair_gap[pair] = gap
        judged_frequency_pair_gap = judged_frequency_pair_gap.dropna()
        judged_frequency_pair_gap.columns.name = "superset_subset"
        judged_frequency_pair_gap = judged_frequency_pair_gap.stack().to_frame(
            name="judged_freq_gap"
        )

        # create dataframe with error dummies
        pair_level_errors = set_mono_viol.loc[
            slice(None), (ss_pairs, ["midp_dist_is_vio", "min_dist_is_vio"])
        ].stack("superset_subset")[["midp_dist_is_vio", "min_dist_is_vio"]]
        pair_level_errors.reset_index(inplace=True)

        # merge  errors and judged frequencies
        df = pd.merge(
            left=pair_level_errors,
            right=judged_frequency_pair_gap,
            how="inner",
            left_on=["personal_id", "superset_subset"],
            right_on=["personal_id", "superset_subset"],
        )

        df["midp_dist_is_vio"] = df["midp_dist_is_vio"].astype("float")
        df["min_dist_is_vio"] = df["min_dist_is_vio"].astype("float")

        # average across waves, making error rates for each individual and
        # superset-subset combination
        df_full = df.copy()
        df = (
            df.query("wave != 'temp'")
            .groupby(["personal_id", "superset_subset"])
            .mean()
            .reset_index()
        )

        descriptive_set_monotonicity_violations(
            df_full,
            path_out=produces["descriptive_stats_set_monotonicity_violation"],
            ss_pairs=ss_pairs,
        )

        create_regression_table(df, produces)

        # set-monotonicity violation plots
        make_set_mono_viol_fechner_evidence_plot(
            df=df,
            path_out=produces["set_monotonicity_violations_fechner_evidence"],
        )

        # # By waves
        # out = df_full.groupby(["wave", "superset_subset"])["min_dist_is_vio"].mean().unstack()
        # out["mean"] = out.mean(axis=1)
        # out.loc["mean"] = out.mean(axis=0)
        # out
