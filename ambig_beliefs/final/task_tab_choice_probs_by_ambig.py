"""
Make table showing decision weights and choice probs depending on ambiguity parameters
"""
import estimagic.visualization.estimation_table as et
import pandas as pd
import pytask

from ambig_beliefs.final.utils_final import add_midrules_to_latex
from ambig_beliefs.final.utils_final import apply_custom_number_format
from ambig_beliefs.final.utils_final import put_reg_sample_together
from ambig_beliefs.final.utils_final import select_group_label
from ambig_beliefs.final.utils_final import select_manual_group_order
from ambig_beliefs.model_code.agent import fast_normal_cdf
from config import MODEL_SPECS
from config import NAMES_MAIN_SPEC
from config import NAMES_ROBUSTNESS_SPEC
from config import OUT_ANALYSIS
from config import OUT_DATA
from config import OUT_TABLES
from config import OUT_UNDER_GIT
from config import ROOT


def calc_choice_probs_for_full_table(out):
    for i in out.index:
        ambig_av, ll_insen, sigma = i
        tau_1 = 1 - ll_insen
        tau_0 = ll_insen / 2 - ambig_av
        for prob in out.columns.unique(level=0):
            decision_weight = tau_0 + prob * tau_1
            decision_weight_diff = decision_weight - prob
            out.loc[i, (prob, "decision_weight_diff")] = decision_weight_diff
            out.loc[i, (prob, "choice_prob_ambig")] = fast_normal_cdf(
                decision_weight_diff, scale=sigma
            )
    return out


def create_tab_choice_probs(produces, df, percentiles_used, file_name):

    # Calculate percentiles in pooled individual waves
    df_pooled = df.query("wave != 'temp'").copy()
    percentiles = (
        df_pooled[["ambig_av", "ll_insen", "theta"]]
        .describe(percentiles=percentiles_used)
        .drop(["count", "mean", "std", "min", "max"])
    )

    # Specify parameter values and values for the probability
    ambig_values = percentiles["ambig_av"]
    ll_insen_values = percentiles["ll_insen"]
    sigma_values = percentiles["theta"]
    probs = [0.25, 0.5, 0.75]

    # Initialize DataFrame
    out = pd.DataFrame(
        index=pd.MultiIndex.from_product([ambig_values, ll_insen_values, sigma_values]),
        columns=pd.MultiIndex.from_product(
            [probs, ["decision_weight_diff", "choice_prob_ambig"]]
        ),
        dtype=float,
    )
    out.index.names = ["ambig_av", "ll_insen", "sigma"]

    # Fill table
    out = calc_choice_probs_for_full_table(out)

    # Split up table such into one table for each sigma value
    for counter, sigma in enumerate(sigma_values):
        out_ind = out.xs(sigma, level="sigma").copy()
        out_ind = out_ind.round(2)

        out_ind = out_ind.rename(
            # index=variable_to_proper_name,
            columns={
                "decision_weight_diff": r"$W(E) - p$",
                "choice_prob_ambig": r"$\text{Pr}(\text{choice}={AEX})$",
                0.25: r"$\text{Pr}_\text{subj} = p = 0.25$",
                0.5: r"$\text{Pr}_\text{subj} = p = 0.5$",
                0.75: r"$\text{Pr}_\text{subj} = p = 0.75$",
            },
        )

        # Format and save table
        out_ind = out_ind.reset_index()
        out_ind = apply_custom_number_format(
            out_ind, int_cols=[], number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}")
        )
        # for c in ["ambig_av", "ll_insen"]:
        #     out_ind[c] = out_ind[c].round(2).apply(lambda x: str(x))
        out_ind = out_ind.set_index(["ambig_av", "ll_insen"])
        out_ind.index.names = [r"$\alpha$", r"$\ell$"]

        out_latex = et.render_latex(
            out_ind,
            {},
            append_notes=False,
            render_options={},
            show_footer=False,
            siunitx_warning=False,
            escape_special_characters=False,
            show_index_names=True,
        )

        # Add midrules
        len_perc = len(percentiles)
        midrules_loc = [9 + i * (len_perc + 1) for i in range(1, len_perc)]
        out_latex = add_midrules_to_latex(out_latex, midrules_loc)

        # out = add_midrules_to_latex(
        #     out, [10, 12, 21, 23, 32, 34], midrule_text=r"\cmidrule{2-9}"
        # )
        path_out = produces[f"{file_name}_sigma{counter + 1}"]
        with open(path_out, "w") as my_table:
            pd.set_option("display.max_colwidth", None)
            my_table.write(out_latex)


def create_tab_choice_probs_ambig_types(
    produces, group_stats, group_labels, file_name=None
):

    para_names = ["ambig_av", "ll_insen", "sigma"]
    group_stats = group_stats.rename({"theta": "sigma"}, axis=1)
    group_stats = group_stats.reset_index().set_index(para_names)

    probs = [0.25, 0.5, 0.75]
    # Initialize DataFrame
    out = pd.DataFrame(
        index=group_stats.index,
        columns=pd.MultiIndex.from_product(
            [probs, ["decision_weight_diff", "choice_prob_ambig"]]
        ),
        dtype=float,
    )
    # Fill table
    out = calc_choice_probs_for_full_table(out)

    out[("type_man_sort", "")] = group_stats["type_man_sort"].map(group_labels)
    # out = out.reset_index().set_index(["type_man_sort"] + para_names)

    out = out.rename(
        # index=variable_to_proper_name,
        columns={
            "decision_weight_diff": r"$W(E) - p$",
            "choice_prob_ambig": r"$\text{Pr}(\text{choice}={AEX})$",
            0.25: r"$\text{Pr}_\text{subj} = p = 0.25$",
            0.5: r"$\text{Pr}_\text{subj} = p = 0.5$",
            0.75: r"$\text{Pr}_\text{subj} = p = 0.75$",
        },
    )
    # Format and save table
    out = out.reset_index()
    out = apply_custom_number_format(
        out, int_cols=[], number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}")
    )
    # for c in ["ambig_av", "ll_insen"]:
    #     out[c] = out[c].round(2).apply(lambda x: str(x))
    out = out.set_index(["type_man_sort"] + para_names)
    out.index.names = ["Ambiguity type", r"$\alpha$", r"$\ell$", r"$\sigma$"]

    out_latex = et.render_latex(
        out,
        {},
        append_notes=False,
        render_options={},
        show_footer=False,
        siunitx_warning=False,
        escape_special_characters=False,
        show_index_names=True,
    )

    # # Add midrules
    # len_perc = len(percentiles)
    # midrules_loc = [9 + i * (len_perc + 1) for i in range(1, len_perc)]
    # out_latex = add_midrules_to_latex(out_latex, midrules_loc)

    # out = add_midrules_to_latex(
    #     out, [10, 12, 21, 23, 32, 34], midrule_text=r"\cmidrule{2-9}"
    # )
    if file_name:
        path_out = produces[file_name]
        with open(path_out, "w") as my_table:
            pd.set_option("display.max_colwidth", None)
            my_table.write(out_latex)
    return out


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC:
    wbw_models = MODEL_SPECS[m]["wbw_models"]

    depends_on = {
        "individual": OUT_DATA / "individual.pickle",
        "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
        "utils_final": ROOT / "ambig_beliefs" / "final" / "utils_final.py",
        "pat_rec_and_dur_restrictions": OUT_DATA
        / "pat_rec_and_dur_restrictions.pickle",
        m: OUT_UNDER_GIT / m / "opt_diff_evolution" / "results.pickle",
    }
    depends_on.update(
        {
            mod: OUT_UNDER_GIT / mod / "opt_diff_evolution" / "results.pickle"
            for mod in wbw_models
        }
    )
    produces = {
        "tab_choice_probs_sigma1": OUT_TABLES / m / "tab_choice_probs_sigma1.tex",
        "tab_choice_probs_sigma2": OUT_TABLES / m / "tab_choice_probs_sigma2.tex",
        "tab_choice_probs_sigma3": OUT_TABLES / m / "tab_choice_probs_sigma3.tex",
        "tab_choice_probs_q5_sigma1": OUT_TABLES / m / "tab_choice_probs_q5_sigma1.tex",
        "tab_choice_probs_q5_sigma2": OUT_TABLES / m / "tab_choice_probs_q5_sigma2.tex",
        "tab_choice_probs_q5_sigma3": OUT_TABLES / m / "tab_choice_probs_q5_sigma3.tex",
        "tab_choice_probs_q5_sigma4": OUT_TABLES / m / "tab_choice_probs_q5_sigma4.tex",
        "tab_choice_probs_q5_sigma5": OUT_TABLES / m / "tab_choice_probs_q5_sigma5.tex",
    }
    PARAMETRIZATION[m] = {
        "depends_on": depends_on,
        "produces": produces,
        "wbw_models": wbw_models,
    }

for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=m)
    def task_tab_choice_probs_by_ambig(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        wbw_models=kwargs["wbw_models"],
        asset_calc=MODEL_SPECS[m]["asset_calc"],
        restrictions=MODEL_SPECS[m]["restrictions"],
    ):

        # Load and restrict sample
        restrictions = "&".join(restrictions.split(","))
        df_single_waves = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=asset_calc,
            restrictions=restrictions,
            models=wbw_models,
        )

        create_tab_choice_probs(
            produces,
            df_single_waves,
            percentiles_used=[0.25, 0.5, 0.75],
            file_name="tab_choice_probs",
        )
        create_tab_choice_probs(
            produces,
            df_single_waves,
            percentiles_used=[0.05, 0.25, 0.5, 0.75, 0.95],
            file_name="tab_choice_probs_q5",
        )


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC:

    depends_on = {
        f"group_stats_{m}": OUT_ANALYSIS / f"group_stats_{m}.pickle",
        "utils_final": "utils_final.py",
    }
    for ga in MODEL_SPECS[m]["k_groups"]:
        id = f"{m}:{ga}"

        produces = {
            "tab_choice_probs_ambig_groups": OUT_TABLES
            / m
            / f"tab_choice_probs_ambig_groups_{ga}.tex",
        }

        PARAMETRIZATION[id] = {
            "m": m,
            "ga": ga,
            "depends_on": depends_on,
            "produces": produces,
        }

for id_, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=id_)
    def task_tab_choice_probs_by_ambig_types(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        m=kwargs["m"],
        ga=kwargs["ga"],
    ):
        group_stats = pd.read_pickle(depends_on[f"group_stats_{m}"])

        # Sort mapping of manual sorting of groups
        g_man_to_g = select_manual_group_order(m, ga)
        g_to_g_man = {j: i for i, j in g_man_to_g.items()}

        # Select correct number of groups and manually sort the groups
        group_stats = group_stats.loc[ga].copy()
        group_stats["type"] = group_stats.index
        group_stats["type_man_sort"] = group_stats["type"].replace(g_to_g_man)
        group_stats = group_stats.set_index(["type_man_sort"]).sort_index()

        # Load group names
        group_labels = {
            g: f"{select_group_label(m, ga, g)}" for g in range(len(g_man_to_g))
        }
        create_tab_choice_probs_ambig_types(
            produces,
            group_stats,
            group_labels,
            file_name="tab_choice_probs_ambig_groups",
        )
