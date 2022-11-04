"""
Makes a table with the pairwise correlations of measured parameters across waves.
"""
import re

import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from ambig_beliefs.final.utils_final import put_reg_sample_together
from config import MODEL_SPECS
from config import NAMES_MAIN_SPEC
from config import NAMES_ROBUSTNESS_SPEC
from config import OUT_DATA
from config import OUT_DATA_LISS
from config import OUT_FIGURES
from config import OUT_TABLES
from config import OUT_UNDER_GIT
from config import ROOT


param_to_name = {
    "ambig_av": r"$\alpha$",
    "ll_insen": r"$\ell$",
    "theta": r"$\sigma$",
}

param_to_name_idx = {
    "ambig_av": r"$\alpha_\mathrm{BBLW-Index}$",
    "ll_insen": r"$\ell_\mathrm{BBLW-Index}$",
}


def create_dist_plots(
    produces, param_to_x_specs, m, params, data, order_waves, idx=False
):
    # plot distributions for each wave
    fig, ax = plt.subplots(
        1, len(params), figsize=(4 * len(params), 4), tight_layout=True
    )

    for j, param in enumerate(params):
        sns.boxplot(
            data=data[param].reset_index(),
            x=param,
            y="wave",
            order=order_waves,
            orient="h",
            palette=sns.xkcd_palette(
                # ["windows blue", "amber", "greyish", "faded green"]
                ["windows blue"] * 6
                + ["greyish"]
            ),
            ax=ax[j],
            whis=[5, 95],
            showfliers=False,
        )
        ax[j].set_xlim(param_to_x_specs[param]["xlim"])
        if "xticks" in param_to_x_specs[param]:
            ax[j].set_xticks(param_to_x_specs[param]["xticks"])
        label = param_to_name_idx[param] if idx else param_to_name[param]
        ax[j].set_xlabel(label, fontsize=16)
        ax[j].set_ylabel("")
        if j != 0:
            ax[j].set_yticklabels([])
        else:
            ax[j].set_yticklabels(ax[j].get_yticklabels(), horizontalalignment="left")

            fig.canvas.draw()  # get_window_extent needs a renderer to work
            yax = ax[j].get_yaxis()
            # find the maximum width of the label on the major ticks
            pad = max(T.label1.get_window_extent().width for T in yax.majorTicks)

            yax.set_tick_params(pad=pad)

        ax[j].tick_params(labelsize=13)

    plt.tight_layout()
    path_out = produces["wbw_dists" + ("_idx" if idx else "")]
    fig.savefig(path_out)

    if not idx:
        # plot distributions for each wave -- only ambig
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        for j, param in enumerate(["ambig_av", "ll_insen"]):
            sns.boxplot(
                data=data[param].reset_index(),
                x=param,
                y="wave",
                order=order_waves,
                orient="h",
                palette=sns.xkcd_palette(
                    # ["windows blue", "amber", "greyish", "faded green"]
                    ["windows blue"]
                    * 5
                ),
                ax=ax[j],
            )
            ax[j].set_xlim(param_to_x_specs[param]["xlim"])
            if "xticks" in param_to_x_specs[param]:
                ax[j].set_xticks(param_to_x_specs[param]["xticks"])
            ax[j].set_xlabel(param_to_name[param], fontsize=16)
            ax[j].set_ylabel("")
            if j != 0:
                ax[j].set_yticklabels([])
            ax[j].tick_params(labelsize=13)

        plt.tight_layout()
        path_out = produces["wbw_dists_only_ambig"]
        fig.savefig(path_out)


def create_correlation_table(produces, m, params, data, idx=False):
    waves = sorted(data.index.unique(level="wave"))

    correlation_table = pd.DataFrame(
        index=params, columns=pd.MultiIndex.from_tuples([], names=[None, None])
    )
    for param in params:
        for i, w1 in enumerate(waves):
            for w2 in waves[i + 1 :]:
                measures = data[param].unstack()
                correlation_table.loc[param, (w1, w2)] = (
                    measures[[w1, w2]].corr().iloc[1, 0]
                )

    correlation_table["Average"] = correlation_table[
        [
            (c1, c2)
            for (c1, c2) in correlation_table
            if c1 != "2019-11 (climate)" and c2 != "2019-11 (climate)"
        ]
    ].mean(axis=1)

    correlation_table = correlation_table.applymap(lambda x: f"{x:.2f}")
    correlation_table = correlation_table.rename(index=param_to_name).T

    path_out = produces["correlation_table" + ("_idx" if idx else "")]
    pd.set_option("display.max_colwidth", None)
    ct_latex = correlation_table.style.to_latex(
        column_format="l" * 2 + "r" * correlation_table.shape[1]
    )
    # toprule
    loc = ct_latex.find(" &  & $\\alpha")
    assert loc >= 0, f"False assumption about table:\n\n\n{ct_latex}"
    ct_latex = f"{ct_latex[:loc]}\\toprule\n{ct_latex[loc:]}"
    # midrules -- always before multirow or a date
    lines = ct_latex.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"\\multirow|20[12][0-9]|Average", line):
            lines[i] = f"\\midrule\n{line}"
    ct_latex = "\n".join(lines)
    # bottomrule
    loc = ct_latex.find("\n\\end{tabular}")
    assert loc >= 0, f"False assumption about table:\n\n\n{ct_latex}"
    ct_latex = f"{ct_latex[:loc]}\n\\bottomrule{ct_latex[loc:]}"
    with open(path_out, "w") as f:
        f.write(ct_latex)


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC:
    est_model_name = MODEL_SPECS[m]["est_model_name"]

    wbw_models = MODEL_SPECS[m]["wbw_models"]
    climate_model = MODEL_SPECS[m]["climate_model"]
    depends_on = {
        "utils_final": ROOT / "ambig_beliefs" / "final" / "utils_final.py",
        "individual": OUT_DATA / "individual.pickle",
        "indices": OUT_DATA_LISS / "ambiguous_beliefs" / "indices.pickle",
        "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
    }
    depends_on.update(
        {
            mod: OUT_UNDER_GIT / mod / "opt_diff_evolution" / "results.pickle"
            for mod in wbw_models + [climate_model]
        }
    )

    produces = {
        "correlation_table": OUT_TABLES / m / "correlation_table.tex",
        "correlation_table_idx": OUT_TABLES / m / "idx" / "correlation_table_idx.tex",
        "wbw_dists": OUT_FIGURES / m / "wbw_dists.pdf",
        "wbw_dists_only_ambig": OUT_FIGURES / m / "wbw_dists_only_ambig.pdf",
        "wbw_dists_idx": OUT_FIGURES / m / "idx" / "wbw_dists_idx.pdf",
    }
    PARAMETRIZATION[m] = {
        "depends_on": depends_on,
        "produces": produces,
        "wbw_models": wbw_models,
        "climate_model": climate_model,
    }

for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=m)
    def task_temporal_stability(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        m=m,
        wbw_models=kwargs["wbw_models"],
        climate_model=kwargs["climate_model"],
    ):

        models = wbw_models + [climate_model]
        df = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=MODEL_SPECS[m]["asset_calc"],
            restrictions=MODEL_SPECS[m]["restrictions"],
            models=models,
        )
        params = ["ambig_av", "ll_insen", "theta"]
        data = df.dropna(subset=params)
        rename_dict_waves = {
            1: "2018-05",
            2: "2018-11",
            3: "2019-05",
            4: "2019-11",
            5: "2020-05",
            6: "2020-11",
            7: "2021-05",
            "temp": "2019-11 (climate)",
        }
        data = data.rename(index=rename_dict_waves)
        data_no_climate = data.query("wave != '2019-11 (climate)'").copy()

        # Load indices
        indices = pd.read_pickle(depends_on["indices"])
        indices = indices.reindex(df.index)
        params_idx = ["ambig_av", "ll_insen"]
        indices = indices.dropna(subset=params_idx)
        indices = indices.rename(index=rename_dict_waves)
        indices_no_climate = indices.query("wave != '2019-11 (climate)'").copy()

        create_correlation_table(produces, m, params, data_no_climate)
        create_correlation_table(produces, m, params_idx, indices_no_climate, idx=True)

        param_to_x_specs = {
            "ambig_av": {"xlim": (-0.55, 0.55), "xticks": [-0.5, -0.25, 0, 0.25, 0.5]},
            "ll_insen": {"xlim": (data["ll_insen"].min() - 0.05, 1.05)},
            "theta": {"xlim": (-0.05, 0.4)},
        }
        create_dist_plots(
            produces,
            param_to_x_specs,
            m,
            params,
            data,
            order_waves=[
                v
                for v in rename_dict_waves.values()
                if v in data.index.unique(level="wave")
            ],
        )
        param_to_x_specs = {
            "ambig_av": {"xlim": (-0.55, 0.55), "xticks": [-0.5, -0.25, 0, 0.25, 0.5]},
            "ll_insen": {"xlim": (-0.25, 2.05), "xticks": [0, 0.5, 1, 1.5, 2]},
        }
        create_dist_plots(
            produces,
            param_to_x_specs,
            m,
            params_idx,
            indices,
            order_waves=[
                v
                for v in rename_dict_waves.values()
                if v in indices.index.unique(level="wave")
            ],
            idx=True,
        )
