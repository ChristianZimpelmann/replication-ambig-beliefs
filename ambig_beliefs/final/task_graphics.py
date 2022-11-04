"""
Make plots using model output
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns

from ambig_beliefs.final.utils_final import make_fancy_scatterplot
from ambig_beliefs.final.utils_final import put_reg_sample_together
from config import MODEL_SPECS
from config import NAMES_MAIN_SPEC
from config import OUT_DATA
from config import OUT_DATA_LISS
from config import OUT_FIGURES
from config import OUT_UNDER_GIT


def make_formula(y, X):
    regs = " + ".join(X)
    return f"{y} ~  {regs}"


def plot_temp_aex_comparision(data, var_stocks, file_name, path_dict):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    rename_para = {"ambig_av": r"$\alpha$", "ll_insen": r"$\ell$"}
    for i, para in enumerate(["ambig_av", "ll_insen"]):
        props = {"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.7}
        r2 = data[[para + var_stocks, para + "_temp"]].corr().iloc[0, 1] ** 2
        ax[i].text(
            0.05,
            0.95,
            f"$R^2$ = {r2:.2f}",
            transform=ax[i].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )
        make_fancy_scatterplot(para + var_stocks, para + "_temp", df=data, ax=ax[i])
        ax[i].set_xlabel(rename_para[para] + " - Stocks")
        ax[i].set_ylabel(rename_para[para] + " - Climate")

    fig.tight_layout()
    fig.savefig(path_dict[file_name], format="pdf")


def plot_joint_distr_parameters(data, kind, wave, path_dict):
    data = data.query("wave == @wave").copy()
    # data["ambig_av"] = data["ambig_av"] / 2

    data[["ambig_av", "ll_insen"]].describe()
    g = sns.jointplot(x="ambig_av", y="ll_insen", kind=kind, data=data)
    # Make figure better looking  and save it
    # g.ax_joint.set_ylim(-0.7, 1.7)
    # plt.gca().set_aspect(1 / 2, adjustable="box")

    g.ax_joint.set_xlabel(r"$\alpha$", fontsize=20)
    g.ax_joint.set_ylabel(r"$\ell$", fontsize=20)
    g.ax_joint.tick_params(labelsize=13)
    plt.tight_layout()

    g.savefig(path_dict["joint_distr_parameters_" + kind], format="pdf")


def plot_joint_distr_indices(text_box_1, text_box_2, use_data, data, path_out):
    """
    Plot joint distribution of ambiguity aversion and ll insensitivity of the indices.
    """
    data = data.copy()
    # Create variable indicating if all restrictions are fullfilled
    data["valid"] = (
        (data["ll_insen"] <= 1)
        & (data["ll_insen"] >= 0)
        & (-data["ll_insen"] <= data["ambig_av"] * 2)
        & (data["ambig_av"] * 2 <= data["ll_insen"])
    )
    mean_correct = data["valid"].mean()
    data["valid"] = data["valid"].astype("category")
    data["valid"] = data["valid"].cat.reorder_categories([True, False], ordered=True)
    true_text = f"{(1 - mean_correct) * 100:2.0f}% of indices satisfy restrictions"
    false_text = f"{(1 - mean_correct) * 100:2.0f}% of indices violate restrictions"
    val_to_text = {True: true_text, False: false_text}
    data["valid"] = data["valid"].map(val_to_text)
    # data["ambig_av"] = data["ambig_av"] / 2

    fig, ax = plt.subplots(figsize=(5, 8))
    if use_data:
        sns.scatterplot(
            x="ambig_av",
            y="ll_insen",
            data=data,
            alpha=0.6,
            hue="valid",
            palette=["#ff7f0e", "#1f77b4"],
            s=30,
        )
    else:
        ax.fill([-0.5, 0, 0.5], [1, 0, 1], "#ff7f0e", alpha=0.5)
    # Add textbox explaining invalid observations
    if text_box_1 and use_data:
        ax.text(
            0.06,
            0.85,
            "Set-monotonicity violations / negative slope",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
    if text_box_2 and use_data:
        # Add textbox explaining invalid observations
        ax.text(
            0.35,
            0.2,
            '"Hypersensitive"',
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
    ax.set_ylim(-1.1, 2.1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1, 1.5, 2])
    # plt.gca().set_aspect(1 / 3, adjustable="box")
    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])

    ax.set_xlabel(r"$\alpha$", fontsize=20)
    ax.set_ylabel(r"$\ell$", fontsize=20)
    ax.tick_params(labelsize=13)
    handles, labels = ax.get_legend_handles_labels()
    if text_box_1 or text_box_2:
        ax.legend(
            loc="lower center",
            # bbox_to_anchor=(0.17, 0.92),
            # ncol=3,
            fancybox=True,
            handles=handles[1:],
            labels=labels[1:],
        )
    else:
        ax.legend().set_visible(False)
    fig.tight_layout()
    fig.savefig(path_out, format="pdf")


def make_duration_plot(path_in_dict, path_out_dict):
    """
    Boxplots of time taken per decision stage, pooled across events and waves.
    Separately for people who have and dont have recurring choice patterns.
    """
    durations = pd.read_pickle(path_in_dict["durations"])
    rec_pattern = pd.read_pickle(path_in_dict["dist_to_rec_patterns"]).query(
        "wave != 'temp'"
    )

    # summing durations within stage (people can go back to a previous stage
    # so there can be several rows corresponding to decision stage)
    temp = durations.groupby(["personal_id", "aex_event", "stage"])[
        ["duration_in_s"]
    ].sum()
    # excluding stage = 0, the event separation screen
    temp = temp.query("stage > 0")
    # excluding the top 1% of longest durations
    q = 0.01
    without_extremely_long_durs = temp["duration_in_s"].between(
        0, temp["duration_in_s"].quantile(1 - q)
    )
    temp = temp.loc[without_extremely_long_durs]
    # adding rec pattern measures
    temp = temp.join(rec_pattern)

    # adding
    temp["has_rec_pattern"] = (
        temp["dist_to_rec_patterns_always_lot_or_aex"].dropna() == 0
    )
    temp["choice"] = temp.index.get_level_values(level="stage").map(
        {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    )
    # only first choice
    temp = temp.query("choice == '1st'")
    # plot
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.violinplot(
        y="choice",
        x="duration_in_s",
        hue="has_rec_pattern",
        inner="quart",
        split=True,
        data=temp.reset_index(),
        palette=["lightblue", "lightcoral"],
        orient="h",
        fliersize=1,
        ax=ax,
    )
    ax.tick_params(axis="both", labelsize=13)
    legend = plt.legend(
        title="Always chooses AEX or LOT", fontsize=15, loc="upper right"
    )
    plt.setp(legend.get_title(), fontsize=15)
    ax.set_xlabel("Time taken for first choice per event (seconds)", fontsize=15)
    ax.set_ylabel("", fontsize=13)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="major", labelsize=15)
    fig.tight_layout()
    fig.savefig(path_out_dict["durations"], format="pdf")

    # median time taken per rep pattern type plot
    fig, ax = plt.subplots()
    rec_pattern_sum = (
        (rec_pattern["dist_to_rec_patterns_always_lot_or_aex"] == 0)
        .groupby("personal_id")
        .sum()
    )
    var_name = "Number of waves with rep. choice pattern"
    rec_pattern_sum.name = var_name
    median_plot_data = temp.join(rec_pattern_sum)
    median_plot_data[var_name] = median_plot_data[var_name].astype("int")
    median_plot_data = median_plot_data.groupby(["stage", var_name])[
        "duration_in_s"
    ].median()
    median_plot_data = median_plot_data.unstack(var_name)

    median_plot_data.plot(
        ax=ax,
        color=sns.diverging_palette(10, 180, sep=10, n=len(median_plot_data.columns))[
            ::-1
        ],
    )
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Choice")
    ax.set_ylabel("Median time taken in s")
    fig.tight_layout()
    fig.savefig(path_out_dict["median_durations"], format="pdf")


def make_set_mono_vio_heatmap(outcome, df, path_dict):
    pairs = ["0_1", "1c_2", "1c_3", "2c_0", "2c_1", "2c_3", "3c_1", "3c_2"]
    df = (
        df[pairs]
        .loc[slice(None), (slice(None), outcome)]
        .astype("float")
        .groupby("wave")
        .mean()
    ).copy()
    if "dist_to_no_vio" in outcome:
        df *= 1 / 100
        df = df.round(2)
    fig, ax = plt.subplots(figsize=(8, 3))
    pairs = [
        r"$E_0 \supseteq E_1$",
        r"$E_1^C \supseteq E_2$",
        r"$E_1^C \supseteq E_3$",
        r"$E_2^C \supseteq E_0$",
        r"$E_2^C \supseteq E_1$",
        r"$E_2^C \supseteq E_3$",
        r"$E_3^C \supseteq E_1$",
        r"$E_3^C \supseteq E_2$",
    ]
    sns.heatmap(
        data=df, cmap=sns.light_palette("indianred"), xticklabels=pairs, annot=True
    )
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("Superset-subset pair")

    fig.tight_layout()
    fig.savefig(
        path_dict[f"set_monotonicity_violations_pair_heatmap_{outcome}"], format="pdf"
    )


def create_stability_reg_over_waves_fig(df, param, ylim, path_out):
    fig, ax = plt.subplots(figsize=(6, 4))

    data = df.query("wave != 'temp'").reset_index()[[param, "wave"]].astype(float)

    sns.regplot(x="wave", y=param, data=data, x_estimator=np.mean, ax=ax)
    rename_para = {"ambig_av": r"$\alpha$", "ll_insen": r"$\ell$", "theta": r"$\sigma$"}
    ax.set_ylabel(rename_para[param])

    ax.set_ylim(ylim)

    ax.set_xticklabels(
        ["", "2018-11", "2019-05", "2019-11", "2020-05", "2020-11", "2021-05", ""]
    )
    plt.tight_layout()
    fig.savefig(path_out)
    plt.close("all")


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC:
    wbw_models = MODEL_SPECS[m]["wbw_models"]
    climate_model = MODEL_SPECS[m]["climate_model"]

    depends_on = {
        "individual": OUT_DATA / "individual.pickle",
        "ambiguity": OUT_DATA_LISS / "ambiguity.pickle",
        "durations": OUT_DATA / "durations.pickle",
        "dist_to_rec_patterns": OUT_DATA / "dist_to_rec_patterns.pickle",
        "aex_returns": OUT_DATA / "aex_returns.pickle",
        "indices": OUT_DATA_LISS / "ambiguous_beliefs" / "indices.pickle",
        "set_monotonicity_violations": OUT_DATA / "set_monotonicity_violations.pickle",
        "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
        m: OUT_UNDER_GIT / m / "opt_diff_evolution" / "results.pickle",
    }
    depends_on.update(
        {
            mod: OUT_UNDER_GIT / mod / "opt_diff_evolution" / "results.pickle"
            for mod in wbw_models + [climate_model]
        }
    )
    produces = {
        "stability_reg_over_waves_fig_ambig_av": OUT_FIGURES
        / m
        / "stability_reg_over_waves_fig_ambig_av.pdf",
        "stability_reg_over_waves_fig_ll_insen": OUT_FIGURES
        / m
        / "stability_reg_over_waves_fig_ll_insen.pdf",
        "stability_reg_over_waves_fig_sigma": OUT_FIGURES
        / m
        / "stability_reg_over_waves_fig_sigma.pdf",
        "stability_reg_over_waves_fig_ambig_av_partic_all_waves": OUT_FIGURES
        / m
        / "stability_reg_over_waves_fig_ambig_av_partic_all_waves.pdf",
        "stability_reg_over_waves_fig_ll_insen_partic_all_waves": OUT_FIGURES
        / m
        / "stability_reg_over_waves_fig_ll_insen_partic_all_waves.pdf",
        "stability_reg_over_waves_fig_sigma_partic_all_waves": OUT_FIGURES
        / m
        / "stability_reg_over_waves_fig_sigma_partic_all_waves.pdf",
        "joint_distr_indices_mean_no_data": OUT_FIGURES
        / m
        / "joint_distr_indices_mean_no_data.pdf",
        "joint_distr_indices_mean_no_boxes": OUT_FIGURES
        / m
        / "joint_distr_indices_mean_no_boxes.pdf",
        "joint_distr_indices_mean_b1": OUT_FIGURES
        / m
        / "joint_distr_indices_mean_b1.pdf",
        "joint_distr_indices_mean": OUT_FIGURES / m / "joint_distr_indices_mean.pdf",
        "joint_distr_indices_no_data": OUT_FIGURES
        / m
        / "joint_distr_indices_no_data.pdf",
        "joint_distr_indices_no_boxes": OUT_FIGURES
        / m
        / "joint_distr_indices_no_boxes.pdf",
        "joint_distr_indices_b1": OUT_FIGURES / m / "joint_distr_indices_b1.pdf",
        "joint_distr_indices": OUT_FIGURES / m / "joint_distr_indices.pdf",
        "durations": OUT_FIGURES / m / "durations.pdf",
        "median_durations": OUT_FIGURES / m / "median_durations.pdf",
        "set_monotonicity_violations_pair_heatmap_midp_dist_is_vio": OUT_FIGURES
        / m
        / "set_monotonicity_violations_pair_heatmap_midp_dist_is_vio.pdf",
    }

    PARAMETRIZATION[m] = {
        "depends_on": depends_on,
        "produces": produces,
        "wbw_models": wbw_models,
        "climate_model": climate_model,
    }


for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=m)
    def task_graphics(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        m=m,
        wbw_models=kwargs["wbw_models"],
        climate_model=kwargs["climate_model"],
        asset_calc=MODEL_SPECS[m]["asset_calc"],
        restrictions=MODEL_SPECS[m]["restrictions"],
    ):
        # restrictions
        restrictions = "&".join(restrictions.split(","))
        m = "event_level_het_prob_2_7"
        df_single_wave = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=asset_calc,
            restrictions=restrictions,
            models=[m],
        ).droplevel(level="wave")

        df = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=asset_calc,
            restrictions=restrictions,
            models=wbw_models + [climate_model],
        )

        # Load matching probs and set-monotonicity violations
        indices = pd.read_pickle(depends_on["indices"])
        set_monotonicity_violations = pd.read_pickle(
            depends_on["set_monotonicity_violations"]
        )
        individual = pd.read_pickle(depends_on["individual"])

        # Select valid responses
        indices = indices.reindex(df.index)
        set_monotonicity_violations = set_monotonicity_violations.reindex(df.index)
        individual = individual.reindex(df_single_wave.index)

        # Aggregate over all waves
        temp = indices.drop(["temp"], level="wave")[["ambig_av", "ll_insen"]]
        for mean_str, data in ("", temp), ("_mean", temp.groupby("personal_id").mean()):

            plot_joint_distr_indices(
                text_box_1=False,
                text_box_2=False,
                use_data=False,
                data=data,
                path_out=produces[f"joint_distr_indices{mean_str}_no_data"],
            )
            plot_joint_distr_indices(
                text_box_1=False,
                text_box_2=False,
                use_data=True,
                data=data,
                path_out=produces[f"joint_distr_indices{mean_str}_no_boxes"],
            )
            plot_joint_distr_indices(
                text_box_1=True,
                text_box_2=False,
                use_data=True,
                data=data,
                path_out=produces[f"joint_distr_indices{mean_str}_b1"],
            )
            plot_joint_distr_indices(
                text_box_1=True,
                text_box_2=True,
                use_data=True,
                data=data,
                path_out=produces[f"joint_distr_indices{mean_str}"],
            )

        # duration plot
        make_duration_plot(depends_on, produces)

        make_set_mono_vio_heatmap(
            outcome="midp_dist_is_vio",
            df=set_monotonicity_violations.query("wave != 'temp'"),
            path_dict=produces,
        )
        for param, ylim in [
            ("ambig_av", (-0.01, 0.06)),
            ("ll_insen", (0.55, 0.62)),
            ("theta", (0.085, 0.115)),
        ]:
            param_name = "sigma" if param == "theta" else param
            for df_sel, df_name in [
                (df, ""),
                (
                    df.query("completed_at_least_6_waves_with_sensible_choices_excl1"),
                    "_partic_all_waves",
                ),
            ]:
                create_stability_reg_over_waves_fig(
                    df_sel,
                    param,
                    ylim,
                    path_out=produces[
                        f"stability_reg_over_waves_fig_{param_name}{df_name}"
                    ],
                )
