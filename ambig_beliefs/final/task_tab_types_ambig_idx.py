"""
Runs regressions of model results on individual characteristics
"""
import estimagic.visualization.estimation_table as et
import pandas as pd
import pytask

from ambig_beliefs.final.utils_final import apply_custom_number_format
from ambig_beliefs.final.utils_final import put_reg_sample_together
from ambig_beliefs.final.utils_final import select_group_label
from ambig_beliefs.final.utils_final import select_manual_group_order
from config import MODEL_SPECS
from config import NAMES_INDICES_SPEC
from config import NAMES_MAIN_SPEC
from config import OUT_ANALYSIS
from config import OUT_DATA
from config import OUT_DATA_LISS
from config import OUT_TABLES
from config import OUT_UNDER_GIT


def create_crosstab(df, ga, path_out):
    out = pd.crosstab(df[ga], df[f"{ga}_idx"], normalize=True, margins=True).round(2)
    out.index = [f"Baseline: {i}" for i in out.index]
    out.columns.name = "Type based on BBLW-index"
    out = apply_custom_number_format(
        out,
        int_cols=[],
        number_format=("{0:.2g}", "{0:.4f}", "{0:.4g}"),
    )
    out_latex = et.render_latex(
        out,
        {},
        append_notes=False,
        show_footer=False,
        siunitx_warning=False,
        escape_special_characters=False,
    )
    with open(path_out, "w") as my_table:
        my_table.write(out_latex)


PARAMETRIZATION = {}
for m_estimated in NAMES_MAIN_SPEC:
    for m_idx in NAMES_INDICES_SPEC:
        for ga in MODEL_SPECS[m_idx]["k_groups"]:
            id = f"{m_estimated}:{m_idx}:{ga}"

            depends_on = {
                "individual": OUT_DATA / "individual.pickle",
                "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
                "indices": OUT_DATA_LISS / "ambiguous_beliefs" / "indices.pickle",
                "utils_final": "utils_final.py",
                "group_assignments_estimated": OUT_ANALYSIS
                / f"group_assignments_{m_estimated}.pickle",
                "group_assignments_idx": OUT_ANALYSIS
                / f"group_assignments_{m_idx}.pickle",
                "pat_rec_and_dur_restrictions": OUT_DATA
                / "pat_rec_and_dur_restrictions.pickle",
                MODEL_SPECS[m_estimated]["est_model_name"]: (
                    OUT_UNDER_GIT
                    / MODEL_SPECS[m_estimated]["est_model_name"]
                    / "opt_diff_evolution"
                    / "results.pickle"
                ),
            }
            produces = {
                "crosstab_assignments": OUT_TABLES
                / m_idx
                / f"crosstab_assignments_{ga}_{m_estimated}.tex",
            }
            PARAMETRIZATION[id] = {
                "depends_on": depends_on,
                "produces": produces,
                "model_spec": MODEL_SPECS[m_estimated],
                "model_spec_idx": MODEL_SPECS[m_idx],
                "m_estimated": m_estimated,
                "m_idx": m_idx,
                "ga": ga,
            }

for id, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=id)
    def task_crosstab_types_idx(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        model_spec=kwargs["model_spec"],
        model_spec_idx=kwargs["model_spec_idx"],
        ga=kwargs["ga"],
        m_estimated=kwargs["m_estimated"],
        m_idx=kwargs["m_idx"],
    ):
        group_assignments_estimated = pd.read_pickle(
            depends_on["group_assignments_estimated"]
        )
        group_assignments_idx = pd.read_pickle(depends_on["group_assignments_idx"])
        df = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=model_spec["asset_calc"],
            restrictions=model_spec["restrictions"],
            models=[model_spec["est_model_name"]],
        )
        group_assignments_estimated = group_assignments_estimated.reindex(
            df.droplevel(level="wave").index
        )

        # Prep group assignment estimated parameters
        g_man_to_g = select_manual_group_order(m_estimated, ga)
        g_to_g_man = {j: i for i, j in g_man_to_g.items()}

        group_assignments_estimated[ga] = group_assignments_estimated[ga].map(
            g_to_g_man
        )
        group_assignments_estimated[ga] = pd.Categorical(
            group_assignments_estimated[ga],
            ordered=True,
        )

        # Column names
        n_groups = len(group_assignments_estimated[ga].unique())
        group_assignments_estimated[ga] = group_assignments_estimated[ga].replace(
            {g: f"{select_group_label(m_estimated, ga, g)}" for g in range(n_groups)}
        )

        # Indices for wave-by-wave classification
        g_man_to_g = select_manual_group_order(m_idx, ga)
        g_to_g_man = {j: i for i, j in g_man_to_g.items()}
        group_assignments_idx[ga] = group_assignments_idx[ga].map(g_to_g_man)
        group_assignments_idx[ga] = pd.Categorical(
            group_assignments_idx[ga],
            ordered=True,
        )

        # Column names
        n_groups = len(group_assignments_idx[ga].unique())
        group_assignments_idx[ga] = group_assignments_idx[ga].replace(
            {g: f"{select_group_label(m_idx, ga, g)}" for g in range(n_groups)}
        )
        group_assignments_idx.columns = [f"{c}_idx" for c in group_assignments_idx]

        data = group_assignments_estimated[[ga]].join(
            group_assignments_idx[f"{ga}_idx"]
        )
        create_crosstab(
            data,
            ga,
            produces["crosstab_assignments"],
        )
