import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import scipy
import seaborn as sns

from ambig_beliefs.final.utils_final import put_reg_sample_together
from ambig_beliefs.final.utils_final import select_group_colors
from ambig_beliefs.final.utils_final import select_group_label
from ambig_beliefs.final.utils_final import select_group_marker
from ambig_beliefs.final.utils_final import select_manual_group_order
from config import MODEL_SPECS
from config import NAMES_INDICES_SPEC
from config import NAMES_MAIN_SPEC
from config import NAMES_ROBUSTNESS_SPEC
from config import OUT_ANALYSIS
from config import OUT_DATA
from config import OUT_DATA_LISS
from config import OUT_FIGURES
from config import OUT_UNDER_GIT


def make_triangular_plot(
    individual_data,
    means_data,
    produces,
    model_spec,
    m,
    k,
    ga,
    group_colors,
    group_markers,
    legend_right,
):
    if model_spec["indices_params"]:
        fig, ax = plt.subplots(figsize=(10, 14 + k / 2))
    else:
        fig, ax = plt.subplots(figsize=(10, 9 + k / 2))
    sns.scatterplot(
        x="ambig_av",
        y="ll_insen",
        hue=f"{ga}_man_sort",
        style=f"{ga}_man_sort",
        palette=group_colors,
        markers=group_markers,
        data=individual_data,
        ax=ax,
        legend=None,
        alpha=0.5,
    )
    if (individual_data["ll_insen"] < 0).any():
        # indicate where triangle would be if not all observations are in it
        ax.plot([0, 0.5], [0, 1], color="black", ls="--")
        ax.plot([0, -0.5], [0, 1], color="black", ls="--")
        ax.plot([-0.5, 0.5], [1, 1], color="black", ls="--")

    # Display group means
    for g_man in range(k):

        # Find out group label
        group_label = select_group_label(m, ga, g_man)
        label = (
            "$\\bf{"
            + group_label.replace(" ", r"\ ")
            + "}$: "
            + f"share = {means_data.loc[g_man, 'share']:.2f}"
            + r", $\bar{\alpha}^{AEX}$ = "
            + (" " if means_data.loc[g_man, "ambig_av"] > 0 else "")
            + f"{means_data.loc[g_man, 'ambig_av']:.2f}"
            + r", $\bar{\ell}^{AEX}$ = "
            + f"{means_data.loc[g_man, 'll_insen']:.2f}"
        )

        if not model_spec["indices_params"]:
            label += (
                r", $\bar{\sigma}^{AEX}$ = " + f"{means_data.loc[g_man, 'theta']:.2f}"
            )
        ax.scatter(
            means_data.loc[g_man, "ambig_av"],
            means_data.loc[g_man, "ll_insen"],
            color=group_colors[g_man],
            marker=group_markers[g_man],
            s=300,
            label=label,
        )
    max_group_length = max(len(select_group_label(m, ga, g)) for g in range(k))

    if legend_right:
        lgnd = plt.legend(
            bbox_to_anchor=(1.1, 0.75 if k <= 4 else 1),
            loc="upper left",
            title="Ambiguity types",
            fontsize=13,
            title_fontsize=16,
            labelspacing=3,
        )
        out_file = "ambig_group_triangle_legend_right"
    else:
        lgnd = plt.legend(
            bbox_to_anchor=(0.5, -0.1),
            loc="upper center",
            title="Ambiguity types",
            fontsize=13,
            title_fontsize=16,
        )
        for handle in lgnd.legendHandles:
            handle.set_sizes([50.0])

        hp = lgnd._legend_box.get_children()[1]
        for vp in hp.get_children():
            for row in vp.get_children():
                row.set_width(
                    510 + (max_group_length - 17) * 7
                )  # need to adapt this manually
                row.mode = "expand"
                row.align = "right"
        out_file = "ambig_group_triangle"

    ax.set_xlabel(r"$\alpha^{AEX}$", fontsize=20)
    ax.set_ylabel(r"$\ell^{AEX}$", fontsize=20)

    if model_spec["indices_params"]:
        ax.set_ylim(-1.1, 2.1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1, 1.5, 2])
    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])

    ax.tick_params(labelsize=17)
    plt.tight_layout()
    path_out = produces[out_file]
    fig.savefig(path_out)
    fig.clear()


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC + NAMES_INDICES_SPEC:
    if MODEL_SPECS[m]["indices_params"]:
        depends_on = {
            "individual": OUT_DATA / "individual.pickle",
            "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
            "indices": OUT_DATA_LISS / "ambiguous_beliefs" / "indices.pickle",
            "utils_final": "utils_final.py",
            "group_assignments": OUT_ANALYSIS / f"group_assignments_{m}.pickle",
            "group_stats": OUT_ANALYSIS / f"group_stats_{m}.pickle",
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
            "group_stats": OUT_ANALYSIS / f"group_stats_{m}.pickle",
        }

    for ga in MODEL_SPECS[m]["k_groups"]:
        produces = {
            "ambig_group_triangle": OUT_FIGURES / m / f"ambig_group_triangle_{ga}.pdf",
            "ambig_group_triangle_legend_right": OUT_FIGURES
            / m
            / f"ambig_group_triangle_legend_right_{ga}.pdf",
        }
        if not MODEL_SPECS[m]["indices_params"]:
            produces[f"ambig_group_prob_to_mprob_{ga}_g_0"] = (
                OUT_FIGURES / m / f"ambig_group_prob_to_mprob_{ga}_g_0.pdf",
            )
        PARAMETRIZATION[f"{m}:{ga}"] = {
            "m": m,
            "ga": ga,
            "depends_on": depends_on,
            "produces": produces,
            "model_spec": MODEL_SPECS[m],
        }


for id_, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=id_)
    def task_fig_ambiguity_groups(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        model_spec=kwargs["model_spec"],
        m=kwargs["m"],
        ga=kwargs["ga"],
    ):
        g_stats = pd.read_pickle(depends_on["group_stats"])
        g_assig = pd.read_pickle(depends_on["group_assignments"])

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

        k = len(g_assig[ga].unique())
        produces.update(
            {
                f"ambig_group_prob_to_mprob_{ga}_g_{g}": OUT_FIGURES
                / m
                / f"ambig_group_prob_to_mprob_{ga}_g_{g}.pdf"
                for g in range(k)
            }
        )
        # Sort mapping of manual sorting of groups
        g_man_to_g = select_manual_group_order(m, ga)
        g_to_g_man = {j: i for i, j in g_man_to_g.items()}

        means_data = g_stats.loc[ga].copy()
        means_data["type"] = means_data.index
        means_data["type_man_sort"] = means_data["type"].replace(g_to_g_man)
        means_data = means_data.set_index(["type_man_sort"]).sort_index()
        # overwrite estimated intercept and slope with values implied by ambig_av
        # and ll_insen which are "corrected" for hypersensitivity
        individual_data = g_assig.join(df)
        individual_data["sigma"] = 1 - individual_data["ll_insen"]
        individual_data["tau"] = (
            1 - individual_data["sigma"] - 2 * individual_data["ambig_av"]
        ) / 2
        individual_data[f"{ga}_man_sort"] = individual_data[ga].replace(g_to_g_man)

        group_colors = list(select_group_colors(m, ga).values())
        group_markers = list(select_group_marker(m, ga).values())

        # triangle plot
        make_triangular_plot(
            individual_data,
            means_data,
            produces,
            model_spec,
            m,
            k,
            ga,
            group_colors,
            group_markers,
            legend_right=False,
        )

        # Legend on the right
        make_triangular_plot(
            individual_data,
            means_data,
            produces,
            model_spec,
            m,
            k,
            ga,
            group_colors,
            group_markers,
            legend_right=True,
        )

        if not model_spec["indices_params"]:
            # Decision weight illustration: (prob, matching prob) plots
            for g_man in range(k):

                fig, ax = plt.subplots(figsize=(10, 10))
                p = np.linspace(0.01, 0.99, 100)
                ax.plot(p, p, color="gray", ls="solid")
                temp = individual_data.query(f"{ga}_man_sort == {g_man}")[
                    ["tau", "sigma", "theta"]
                ]
                tau_m, sig_m, theta_m = temp.mean()
                mprobhat = tau_m + sig_m * p

                # Find out group label
                group_label = select_group_label(m, ga, g_man)

                ax.plot(
                    p,
                    mprobhat,
                    color=group_colors[g_man],
                    lw=5,
                    label="",
                )
                ax.fill_between(
                    p,
                    np.maximum(
                        mprobhat - scipy.stats.norm.ppf(1 - 0.5 / 2) * theta_m, 0
                    ),
                    np.minimum(
                        mprobhat + scipy.stats.norm.ppf(1 - 0.5 / 2) * theta_m, 1
                    ),
                    alpha=0.3,
                    color=group_colors[g_man],
                )
                ax.fill_between(
                    p,
                    np.maximum(
                        mprobhat - scipy.stats.norm.ppf(1 - 0.25 / 2) * theta_m, 0
                    ),
                    np.minimum(
                        mprobhat + scipy.stats.norm.ppf(1 - 0.25 / 2) * theta_m, 1
                    ),
                    alpha=0.15,
                    color=group_colors[g_man],
                )
                ax.fill_between(
                    p,
                    np.maximum(
                        mprobhat - scipy.stats.norm.ppf(1 - 0.05 / 2) * theta_m, 0
                    ),
                    np.minimum(
                        mprobhat + scipy.stats.norm.ppf(1 - 0.05 / 2) * theta_m, 1
                    ),
                    alpha=0.05,
                    color=group_colors[g_man],
                )

                p_dots = np.array([0.25, 0.5, 0.75])
                m_dots = tau_m + sig_m * p_dots
                # End with vertical lines (so in foreground).
                for counter, p_dot in enumerate(p_dots):
                    m_dot = m_dots[counter]
                    shrink = np.abs(m_dot - p_dot) * 0.05
                    if m_dot > p_dot:
                        y_upper = m_dot - shrink
                        y_lower = p_dot + shrink
                    else:
                        y_upper = p_dot - shrink
                        y_lower = m_dot + shrink
                    ax.plot(
                        [p_dot, p_dot],
                        [y_lower, y_upper],
                        color="gray",
                        linestyle="dotted",
                    )

                ax.set_xlabel(r"$\Pr_\mathrm{subj}(E)$", fontsize=25)
                ax.set_ylabel("$m(E)$", fontsize=25)
                ax.set_xlim(-0.01, 1.01)
                ax.set_ylim(-0.01, 1.01)
                ax.set_title(
                    f"{group_label}",
                    # + r" ($\bar{\alpha}$: "
                    # + f"{means_data.loc[g_man, 'ambig_av']:.2f}"
                    # + r", $\bar{\ell}$: "
                    # + f"{means_data.loc[g_man, 'll_insen']:.2f}"
                    # + r", $\bar{\sigma}$: "
                    # + f"{means_data.loc[g_man, 'theta']:.2f})",
                    fontsize=30,
                )
                ax.tick_params(labelsize=17)
                # lgnd = plt.legend(loc="lower right", fontsize=13)
                plt.tight_layout()
                path_out = produces[f"ambig_group_prob_to_mprob_{ga}_g_{g_man}"]
                fig.savefig(path_out)
                fig.clear()
