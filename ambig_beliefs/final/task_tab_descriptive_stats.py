"""
Make summary statistics table
"""
import estimagic.visualization.estimation_table as et
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask

from ambig_beliefs.data_management.task_make_individual_chars import (
    get_historical_returns,
)
from ambig_beliefs.final.utils_final import add_midrules_to_latex
from ambig_beliefs.final.utils_final import apply_custom_number_format
from ambig_beliefs.final.utils_final import put_reg_sample_together
from ambig_beliefs.final.utils_final import variable_to_proper_name
from ambig_beliefs.final.utils_final import wave_from_string
from config import BASIC_CONTROLS
from config import IN_DATA
from config import MODEL_SPECS
from config import NAMES_MAIN_SPEC
from config import NAMES_ROBUSTNESS_SPEC
from config import OUT_DATA
from config import OUT_DATA_LISS
from config import OUT_FIGURES
from config import OUT_TABLES
from config import OUT_UNDER_GIT
from config import ROOT


col_name_to_proper_name = {
    "count": "N subj.",
    "mean": "Mean",
    "std": "Std. dev.",
    "5%": "$q_{0.05}$",
    "10%": "$q_{0.1}$",
    "25%": "$q_{0.25}$",
    "90%": "$q_{0.9}$",
    "25%": "$q_{0.25}$",
    "50%": "$q_{0.5}$",
    "75%": "$q_{0.75}$",
    "95%": "$q_{0.95}$",
    1: "2018-05",
    2: "2018-11",
    3: "2019-05",
    4: "2019-11",
    5: "2020-05",
    6: "2020-11",
    7: "2021-05",
    "temp": "2019-11 (Climate Change)",
    "pooled": "Pooled",
}


def create_descriptive_stats(
    data, variables, dummy_vars, file_name, midrules, path_dict
):
    # making summary stats
    result = data[variables].groupby("personal_id").first()

    # scaling money variables
    scale_to_thousands = ["net_income", "total_financial_assets"]
    for v in scale_to_thousands:
        if v in result:
            result[v] *= 1 / 1000

    descriptive_stats = result.describe(percentiles=[0.25, 0.5, 0.75]).loc[
        ["count", "mean", "std", "25%", "50%", "75%"]
    ]
    descriptive_stats = descriptive_stats.T
    for v in dummy_vars:
        descriptive_stats.loc[v, ["std", "25%", "50%", "75%"]] = np.nan

    out = descriptive_stats.rename(
        index=variable_to_proper_name, columns=col_name_to_proper_name
    )

    # formatting
    out = apply_custom_number_format(
        out, int_cols=["N subj."], number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}")
    )
    # for v in ["risk_aversion_index", "numeracy_index", "optimism"]:
    #     if v in out.index:
    #         out.loc[
    #             v, "mean"
    #         ] = f'{np.abs(out.loc[v, "mean"]):.0f}'

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
    out = add_midrules_to_latex(out, midrules)
    out = out.replace("Std. dev.", r"\makecell{Std. \\ Dev.}")
    out = out.replace("N subj.", r"\makecell{N\\ Subj.}")

    with open(path_dict[file_name], "w") as my_table:
        my_table.write(out)


def create_descriptive_stats_mp(
    data, variables, judged_returns, path_dict, file_name=None
):
    # making summary stats
    result = data[variables]  # .groupby("personal_id").mean()
    descriptive_stats = result.describe(percentiles=[0.1, 0.9]).loc[
        ["mean", "std", "10%", "50%", "90%"]
    ]
    # descriptive_stats.loc["$q_{0.9} - q_{0.1}$"] = (
    #     descriptive_stats.loc["90%"] - descriptive_stats.loc["10%"]
    # )
    # descriptive_stats = descriptive_stats.drop(["10%", "90%"])

    # Add historical frequencies for matching probabilities
    ret = get_historical_returns()
    event_to_hf = {
        "0": ret["0"],
        "1": ret["1"],
        "1c": 1 - ret["1"],
        "2": ret["2"],
        "2c": 1 - ret["2"],
        "3": ret["3"],
        "3c": 1 - ret["3"],
    }
    descriptive_stats.loc[r"\makecell{Empir.\\ Freq.\\'99-'19}"] = np.array(
        list(event_to_hf.values())
    )

    # Add judged returns
    event_to_jf = {
        "0": judged_returns["hist_perf_e0_nocheck"].mean(),
        "1": judged_returns["hist_perf_e1_combined"].mean(),
        "1c": np.nan,
        "2": judged_returns["hist_perf_e2_combined"].mean(),
        "2c": np.nan,
        "3": judged_returns["hist_perf_e3_combined"].mean(),
        "3c": np.nan,
    }
    descriptive_stats.loc[r"\makecell{Judged\\ Freq.,\\'99-'19}"] = np.array(
        list(event_to_jf.values())
    )

    # formatting
    descriptive_stats = descriptive_stats.T  #

    # descriptive_stats["count"] = descriptive_stats["count"].astype(int)
    descriptive_stats.rename(
        index=variable_to_proper_name, columns=col_name_to_proper_name, inplace=True
    )

    # descriptive_stats = descriptive_stats.fillna("")
    # descriptive_stats["Observations"] = descriptive_stats["Observations"].astype("int")
    # formatting
    out = apply_custom_number_format(
        descriptive_stats, int_cols=[], number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}")
    )
    out_latex = et.render_latex(
        out,
        {},
        append_notes=False,
        render_options={},
        show_footer=False,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    out_latex = add_midrules_to_latex(out_latex, [5, 8, 11])
    out_latex = out_latex.replace("Std. dev.", r"\makecell{Std. \\ Dev.}")

    if file_name:
        with open(path_dict[file_name], "w") as file:
            file.write(out_latex)

    return out


def create_descriptive_stats_temp(data, file_name, path_dict):
    # Add historical frequencies for matching probabilities
    # Get input data:
    # https://www.knmi.nl/over-het-knmi/nieuws/centraal-nederland-temperatuur
    analysis_df = pd.read_fwf(
        IN_DATA / "temperature" / "temp_nl.txt", sep="   "
    ).set_index("YEAR,")

    # Create averages of winter
    analysis_df["average_winter"] = analysis_df[["Jan,", "Feb,", "Mar,"]].mean(axis=1)
    analysis_df["average_winter_5"] = (
        analysis_df["average_winter"].rolling(5).mean().shift(1)
    )
    analysis_df = analysis_df[["average_winter_5", "average_winter"]]

    # Get relevant size
    analysis_df = analysis_df.loc[1990:2019]
    analysis_df["diff_means"] = analysis_df.eval("average_winter - average_winter_5")

    # Get freqencies
    analysis_df["diff"] = pd.cut(
        analysis_df["diff_means"],
        [-np.inf, -0.5, 0, 1, np.inf],
        labels=["<-0.5", "-0.5_0", "0_1", ">1"],
    )
    frequencies = analysis_df["diff"].value_counts() / analysis_df["diff"].count()

    event_to_hf = {
        "0": frequencies["0_1"] + frequencies[">1"],
        "1": frequencies[">1"],
        "1c": np.nan,
        "2": frequencies["<-0.5"],
        "2c": np.nan,
        "3": frequencies["-0.5_0"] + frequencies["0_1"],
        "3c": np.nan,
    }

    # making summary stats
    descriptive_stats = (
        data.describe(percentiles=[0.1, 0.9])
        # .loc[["count", "mean", "std", "10%", "50%", "90%"]]
        .loc[["count", "mean", "10%", "50%", "90%"]]
    )
    # descriptive_stats.loc["$q_{0.9} - q_{0.1}$"] = (
    #     descriptive_stats.loc["90%"] - descriptive_stats.loc["10%"]
    # )
    # descriptive_stats = descriptive_stats.drop(["10%", "90%"])

    descriptive_stats.loc[r"\makecell{Empirical\\ Frequency,\\ 1999-2019}"] = np.array(
        list(event_to_hf.values())
    )

    # formatting
    descriptive_stats = descriptive_stats.T
    descriptive_stats.rename(
        index={
            "mp_e0": r"$E^{climate}_0: \Delta T \in (0 ^{\circ}C, \infty)$",
            "mp_e1": r"$E^{climate}_1: \Delta T \in (1 ^{\circ}C, \infty]$",
            "mp_e2": r"$E^{climate}_2: \Delta T \in (-\infty, -0.5 ^{\circ}C)$",
            "mp_e3": r"$E^{climate}_3: \Delta T \in [-0.5 ^{\circ}C, 1 ^{\circ}C]$",
            "mp_e1c": r"$E^{climate}_{1, C}: \Delta T \in (-\infty, 1 ^{\circ}C]$",
            "mp_e2c": r"$E^{climate}_{2, C}: \Delta T \in [-0.5 ^{\circ}C, \infty)$",
            "mp_e3c": r"$E^{climate}_{3, C}: \Delta T \in "
            r"(-\infty, -0.5 ^{\circ}C) \cup (1 ^{\circ}C, \infty)$",
        },
        columns=col_name_to_proper_name,
        inplace=True,
    )

    # formatting
    out = apply_custom_number_format(
        descriptive_stats,
        int_cols=["N subj."],
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
    )

    out = et.render_latex(
        out,
        {},
        append_notes=False,
        render_options={},
        show_footer=False,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    out = add_midrules_to_latex(out, [5, 8, 11])

    with open(path_dict[file_name], "w") as file:
        file.write(out)


def plot_distribution_mp(data, path_out):
    data = data.replace(
        {
            0.005: pd.Interval(0, 1, closed="both"),
            0.03: pd.Interval(1, 5, closed="both"),
            0.075: pd.Interval(5, 10, closed="both"),
            0.15: pd.Interval(10, 20, closed="both"),
            0.25: pd.Interval(20, 30, closed="both"),
            0.35: pd.Interval(30, 40, closed="both"),
            0.45: pd.Interval(40, 50, closed="both"),
            0.55: pd.Interval(50, 60, closed="both"),
            0.65: pd.Interval(60, 70, closed="both"),
            0.75: pd.Interval(70, 80, closed="both"),
            0.85: pd.Interval(80, 90, closed="both"),
            0.925: pd.Interval(90, 95, closed="both"),
            0.97: pd.Interval(95, 99, closed="both"),
            0.995: pd.Interval(99, 100, closed="both"),
        }
    )
    fig, ax = plt.subplots(4, 2, figsize=(10, 15))

    for i, event in enumerate(
        [c for c in data if "0" not in c] + [c for c in data if "0" in c]
    ):

        probs = data[event]
        (probs.value_counts() / probs.count()).sort_index().plot.bar(
            ax=ax[int(i / 2), i % 2]
        )
        ax[int(i / 2), i % 2].set_title(variable_to_proper_name[event], pad=10)
        ax[int(i / 2), i % 2].set_ylim((0, 0.25))

    ax[-1, -1].axis("off")
    fig.tight_layout()
    fig.savefig(path_out, format="pdf")


def descriptive_by_wave_mp(data, path_out):
    """
    Create table of mean value of all matching probabilities over all waves.
    """
    descriptive_stats = data.groupby("wave").mean()

    # Remove index name (wave)
    descriptive_stats.rename_axis(None)

    descriptive_stats.rename(
        index=col_name_to_proper_name, columns=variable_to_proper_name, inplace=True
    )
    descriptive_stats.index.name = None

    # descriptive_stats.loc["Pooled"] = data.mean().values
    # Add historic rates

    descriptive_stats = descriptive_stats.T
    # formatting
    out = apply_custom_number_format(
        descriptive_stats,
        int_cols=[],
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
    )
    out = et.render_latex(
        out,
        {},
        append_notes=False,
        render_options={},
        show_footer=False,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    out = add_midrules_to_latex(out, [5, 8, 11])

    with open(path_out, "w") as file:
        file.write(out)


def observations_table(depends_on, path_out, waves):
    collected_data = pd.read_pickle(depends_on["ambiguous_beliefs"])
    collected_data = collected_data.query(f"wave == {waves}")
    pat_rec_and_dur_restrictions = pd.read_pickle(
        depends_on["pat_rec_and_dur_restrictions"]
    )
    pat_rec_and_dur_restrictions = pat_rec_and_dur_restrictions.query(
        f"wave == {waves}"
    )

    sample_restrictions = pd.read_pickle(depends_on["sample_restrictions"])
    pat_rec_and_dur_restrictions = pat_rec_and_dur_restrictions.join(
        sample_restrictions
    )

    pat_rec_and_dur_restrictions["comp_2_valid_waves_and_valid"] = (
        pat_rec_and_dur_restrictions[
            "completed_at_least_2_waves_with_sensible_choices_excl1"
        ]
        & pat_rec_and_dur_restrictions["valid_choice"]
    )

    # Add number of valid responses
    pat_rec_and_dur_restrictions["has_no_rec_pattern"] = (
        pat_rec_and_dur_restrictions["completed_elicitation"]
        & ~pat_rec_and_dur_restrictions["has_rec_pattern"]
    )

    # Add observations for unique individuals row
    temp = pat_rec_and_dur_restrictions.groupby("personal_id").max()
    temp["wave"] = "unique_inds"
    temp = temp.reset_index().set_index(["personal_id", "wave"])
    pat_rec_and_dur_restrictions = pd.concat(
        [pat_rec_and_dur_restrictions, temp]
    ).sort_index()

    obs_table = pat_rec_and_dur_restrictions.groupby("wave")[
        [
            "completed_elicitation",
            # "has_no_rec_pattern",
            "valid_choice",
            "comp_2_valid_waves_and_valid",
        ]
    ].sum()

    # Calc number of individuals who participated in the respective questionnaire
    participated = collected_data.groupby("wave").count().max(axis=1)
    participated.loc["temp"] = participated.loc[4]
    participated.loc["unique_inds"] = len(
        collected_data.reset_index()["personal_id"].unique()
    )
    participated
    obs_table["Participated"] = participated

    # Rename columns and index
    renaming_cols = {
        "Participated": "Participated",
        "completed_elicitation": r"\makecell{Completed\\ elicitation}",
        # "has_no_rec_pattern": "No recurring pattern",
        "valid_choice": r"\makecell{Proper\\ response}",
        "comp_2_valid_waves_and_valid": r"\makecell{In final\\ data set}",
    }
    obs_table = obs_table[renaming_cols.keys()].rename(columns=renaming_cols)
    renaming_index = {
        2: "2018-11",
        3: "2019-05",
        4: "2019-11",
        "temp": "2019-11 (Climate Change)",
        5: "2020-05",
        6: "2020-11",
        7: "2021-05",
        "unique_inds": "Unique Subjects",
    }
    obs_table.index.name = None
    obs_table = obs_table.loc[renaming_index.keys()].rename(index=renaming_index)

    # formatting
    out = apply_custom_number_format(
        obs_table,
        int_cols=obs_table.columns,
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
    )
    out = et.render_latex(
        out,
        {},
        append_notes=False,
        render_options={},
        show_footer=False,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    out = add_midrules_to_latex(out, [11])

    with open(path_out, "w") as file:
        file.write(out)


def create_stats_ambig(df, df_pooled, params, path_out, path_out_comp):

    # Append df with all waves pooled
    df_all = df.query("wave != 'temp'").copy()
    df_all = df_all.reset_index()
    df_all["wave"] = "Observations from all AEX waves"
    df_all = df_all.set_index(df.index.names)

    # ambiguity parameters over all waves
    ambig_stats = (
        pd.concat([df, df_all])
        .groupby("wave")[params]
        .describe(percentiles=[0.05, 0.25, 0.75, 0.95])
        .drop(["count", "min", "max"], axis=1, level=1)
    )

    # Remove index name (wave)
    ambig_stats = ambig_stats.rename_axis(None)

    # ambig_stats = ambig_stats.astype("object").T
    ambig_stats = (
        ambig_stats.stack(level=0).swaplevel().sort_index().rename_axis([None, None])
    )[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]

    ambig_stats = ambig_stats.rename(
        index=col_name_to_proper_name, columns=variable_to_proper_name
    ).rename(columns=col_name_to_proper_name, index=variable_to_proper_name)

    # formatting
    out = apply_custom_number_format(
        ambig_stats, int_cols=[], number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}")
    )
    out = et.render_latex(
        out,
        {},
        append_notes=False,
        render_options={},
        show_footer=False,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    if len(params) == 3:
        out = add_midrules_to_latex(out, [12, 21])
        out = add_midrules_to_latex(
            out, [10, 12, 21, 23, 32, 34], midrule_text=r"\cmidrule{2-9}"
        )
    elif len(params) == 2:
        out = add_midrules_to_latex(out, [12])
        out = add_midrules_to_latex(
            out, [10, 12, 21, 23], midrule_text=r"\cmidrule{2-9}"
        )
    with open(path_out, "w") as my_table:
        pd.set_option("display.max_colwidth", None)
        my_table.write(out)

    # comparison table all single waves vs pooled estimate
    # ambiguity parameters
    df_pooled = df_pooled.rename(
        index={"pooled": "Pooled estimation over all AEX waves"}
    )
    ambig_stats = (
        pd.concat([df_all, df_pooled])
        .groupby("wave")[params]
        .describe(percentiles=[0.05, 0.25, 0.75, 0.95])
        .drop(["count", "min", "max"], axis=1, level=1)
    )

    # Remove index name (wave)
    ambig_stats = ambig_stats.rename_axis(None)

    # ambig_stats = ambig_stats.astype("object").T
    ambig_stats = (
        ambig_stats.stack(level=0).swaplevel().sort_index().rename_axis([None, None])
    )[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]

    ambig_stats = ambig_stats.rename(
        index=col_name_to_proper_name, columns=variable_to_proper_name
    ).rename(columns=col_name_to_proper_name, index=variable_to_proper_name)

    # formatting
    out = apply_custom_number_format(
        ambig_stats, int_cols=[], number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}")
    )
    out = et.render_latex(
        out,
        {},
        append_notes=False,
        render_options={},
        show_footer=False,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    if len(params) == 3:
        out = add_midrules_to_latex(out, [6, 9])
    elif len(params) == 2:
        out = add_midrules_to_latex(out, [6])
    with open(path_out_comp, "w") as my_table:
        pd.set_option("display.max_colwidth", None)
        my_table.write(out)


def create_stats_ambig_one_col_or_row_per_parameter(produces, df, transpose):
    # ambiguity parameters
    ambig_stats = (
        df.query("wave != 'temp'")[["ambig_av", "ll_insen", "theta"]]
        .describe(percentiles=[0.05, 0.25, 0.75, 0.95])
        .drop(["count", "min", "max"], axis=0)
    )

    ambig_stats = ambig_stats.rename(
        index=col_name_to_proper_name, columns=variable_to_proper_name
    ).rename(columns=col_name_to_proper_name, index=variable_to_proper_name)
    renames = {c: c.rstrip("$") + r"^{AEX}$" for c in ambig_stats.columns}
    ambig_stats = ambig_stats.rename(columns=renames)
    if transpose:
        ambig_stats = ambig_stats.T

    # formatting
    out = apply_custom_number_format(
        ambig_stats.round(3),
        int_cols=[],
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
    )
    out = et.render_latex(
        out,
        {},
        append_notes=False,
        render_options={},
        show_footer=False,
        siunitx_warning=False,
        escape_special_characters=False,
    )

    with open(produces, "w") as my_table:
        pd.set_option("display.max_colwidth", None)
        my_table.write(out)


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC:
    est_model_name = MODEL_SPECS[m]["est_model_name"]

    wbw_models = MODEL_SPECS[m]["wbw_models"]
    climate_model = MODEL_SPECS[m]["climate_model"]

    depends_on = {
        "ambiguous_beliefs": OUT_DATA_LISS / "ambiguous_beliefs.pickle",
        "individual": OUT_DATA / "individual.pickle",
        "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
        "utils_final": ROOT / "ambig_beliefs" / "final" / "utils_final.py",
        "indices": OUT_DATA_LISS / "ambiguous_beliefs" / "indices.pickle",
        "aex_returns": OUT_DATA / "aex_returns.pickle",
        "pat_rec_and_dur_restrictions": OUT_DATA
        / "pat_rec_and_dur_restrictions.pickle",
        est_model_name: OUT_UNDER_GIT
        / est_model_name
        / "opt_diff_evolution"
        / "results.pickle",
    }
    depends_on.update(
        {
            mod: OUT_UNDER_GIT / mod / "opt_diff_evolution" / "results.pickle"
            for mod in wbw_models + [climate_model]
        }
    )
    produces = {
        "descriptive_stats": OUT_TABLES / m / "descriptive_stats.tex",
        # "descriptive_stats_bins": OUT_TABLES / m / "descriptive_stats_bins.tex",
        "descriptive_stats_ambig": OUT_TABLES / m / "descriptive_stats_ambig.tex",
        "descriptive_stats_ambig_comp": OUT_TABLES
        / m
        / "descriptive_stats_ambig_comp.tex",
        "descriptive_stats_idx": OUT_TABLES / m / "idx" / "descriptive_stats_idx.tex",
        "descriptive_stats_idx_comp": OUT_TABLES
        / m
        / "idx"
        / "descriptive_stats_idx_comp.tex",
        "descriptive_stats_ambig_single_waves_only": OUT_TABLES
        / m
        / "descriptive_stats_ambig_single_waves_only.tex",
        "descriptive_stats_ambig_pooled_only": OUT_TABLES
        / m
        / "descriptive_stats_ambig_pooled_only.tex",
        "descriptive_stats_mp": OUT_TABLES / m / "descriptive_stats_mp.tex",
        "descriptive_stats_mp_temp": OUT_TABLES / m / "descriptive_stats_mp_temp.tex",
        "dist_mp": OUT_FIGURES / m / "dist_mp.pdf",
        "observations": OUT_TABLES / m / "observations.tex",
        "descriptive_stats_wave_mp": OUT_TABLES / m / "descriptive_stats_wave_mp.tex",
    }
    PARAMETRIZATION[m] = {
        "depends_on": depends_on,
        "produces": produces,
        "model_spec": MODEL_SPECS[m],
    }

for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=m)
    def task_tab_descriptive_stats(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        model_spec=kwargs["model_spec"],
    ):

        df = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=model_spec["asset_calc"],
            restrictions=model_spec["restrictions"],
            models=[model_spec["est_model_name"]],
        )
        df_single_waves = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=model_spec["asset_calc"],
            restrictions=model_spec["restrictions"],
            models=model_spec["wbw_models"] + [model_spec["climate_model"]],
        )
        indices_single_waves = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=model_spec["asset_calc"],
            restrictions=model_spec["restrictions"],
            models=list(range(1, 7)) + ["temp"],
            indices=True,
            indices_mean=False,
        )
        indices_mean = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=model_spec["asset_calc"],
            restrictions=model_spec["restrictions"],
            models=list(range(1, 7)),
            indices=True,
            indices_mean=True,
        )
        indices_single_waves = indices_single_waves.rename(
            columns={p: f"{p}_index" for p in ["ambig_av", "ll_insen"]}
        )
        indices_mean = indices_mean.rename(
            columns={p: f"{p}_index" for p in ["ambig_av", "ll_insen"]}
        )
        # Use select time-invariant variables
        # categoricals to dummies
        dummy_vars = []
        categoricals = [
            "edu",
            # "age_groups",
            # "net_income_groups",
            # "total_financial_assets_groups",
        ]
        for v in categoricals:
            dummies = pd.get_dummies(df[v])
            dummies.rename(
                columns={c: f"{v}_{c}" for c in dummies.columns}, inplace=True
            )

            df = df.join(dummies)

            dummy_vars += list(dummies.columns)

        # age categs
        age_labels = {
            "B1": "Age: $\\leq 35$",
            "B2": "Age: $\\in (35, 50]$",
            "B3": "Age: $\\in (50, 65]$",
            "B4": "Age: $ > 65$",
        }
        age_groups = df["age_groups"].groupby("personal_id").max()
        df = df.join(pd.get_dummies(age_groups.map(age_labels)))
        df["female"] = df["female"].astype(float)

        # Selecting and setting order of variables
        demographics = (
            # [v for v in dummy_vars if "age" in v]
            ["female"]
            + [v for v in dummy_vars if "edu" in v]
            # + list(age_labels.values())
            + ["age"]
        )
        risk_num_controls = [
            c for c in BASIC_CONTROLS if "risk" in c or "numeracy" in c
        ]

        # judged_freqs = [
        #     "hist_perf_e0_nocheck",
        #     # "hist_perf_e1_combined",
        #     # "hist_perf_e2_combined",
        #     # "hist_perf_has_additivity_vio",
        #     # "hist_perf_has_set_mono_vio",
        #     "jf_less_hf_avg_abs_dev",
        #     "hist_perf_response_error",
        # ]
        additional = [
            # "optimism_pessimsm",
            "understands_climate_change",
            "threatened_by_climate_change",
        ]
        income_wealth = ["net_income", "total_financial_assets"]
        rfa = ["has_rfa", "frac_of_tfa_in_rfa_cond_any"]

        demo_variables = (
            demographics
            + income_wealth
            + rfa
            # + judged_freqs
            + risk_num_controls
            + additional
        )
        create_descriptive_stats(
            data=df,
            path_dict=produces,
            variables=demo_variables,
            dummy_vars=dummy_vars
            + [
                # "hist_perf_response_error",
                "has_rfa",
                "female",
            ],  # + list(age_labels.values())
            midrules=[9, 12, 15, 18],
            file_name="descriptive_stats",
        )

        create_stats_ambig(
            df_single_waves,
            df,
            ["ambig_av", "ll_insen", "theta"],
            produces["descriptive_stats_ambig"],
            produces["descriptive_stats_ambig_comp"],
        )
        create_stats_ambig(
            indices_single_waves,
            indices_mean,
            ["ambig_av_index", "ll_insen_index"],
            produces["descriptive_stats_idx"],
            produces["descriptive_stats_idx_comp"],
        )
        create_stats_ambig_one_col_or_row_per_parameter(
            produces["descriptive_stats_ambig_single_waves_only"],
            df_single_waves,
            transpose=False,
        )
        create_stats_ambig_one_col_or_row_per_parameter(
            produces["descriptive_stats_ambig_pooled_only"], df, transpose=True
        )

        mp_events = ["mp_e0", "mp_e1", "mp_e1c", "mp_e2", "mp_e2c", "mp_e3", "mp_e3c"]

        # Load empirical and judged frequencies
        individual = pd.read_pickle(depends_on["individual"])
        judged_returns = individual[
            [
                "hist_perf_e0_nocheck",
                "hist_perf_e1_combined",
                "hist_perf_e2_combined",
                "hist_perf_e3_combined",
            ]
        ].dropna()

        # Calculate waves that should be used
        waves_excl_temp = [wave_from_string(w) for w in wbw_models]
        waves_incl_temp = [wave_from_string(w) for w in wbw_models + [climate_model]]

        create_descriptive_stats_mp(
            data=indices_single_waves.query(f"wave == {waves_excl_temp}")[
                mp_events
            ].copy(),
            variables=mp_events,
            path_dict=produces,
            judged_returns=judged_returns,
            file_name="descriptive_stats_mp",
        )

        create_descriptive_stats_temp(
            data=indices_single_waves.query("wave == 'temp'")[mp_events].copy(),
            path_dict=produces,
            file_name="descriptive_stats_mp_temp",
        )

        descriptive_by_wave_mp(
            indices_single_waves.query(f"wave == {waves_excl_temp}")[mp_events].copy(),
            path_out=produces["descriptive_stats_wave_mp"],
        )

        plot_distribution_mp(
            indices_single_waves.query(f"wave == {waves_excl_temp}")[mp_events].copy(),
            path_out=produces["dist_mp"],
        )

        observations_table(
            depends_on=depends_on,
            path_out=produces["observations"],
            waves=waves_incl_temp,
        )
