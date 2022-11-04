"""
Assigns individuals into groups based on their estimated
    ambiguity aversion
    likelihood insensitivity
    error
parameters using k-means.
"""
import numpy as np
import pandas as pd
import pytask
from sklearn.cluster import KMeans

from ambig_beliefs.final.utils_final import put_reg_sample_together
from config import K_MAX
from config import MODEL_SPECS
from config import NAMES_INDICES_SPEC
from config import NAMES_MAIN_SPEC
from config import NAMES_ROBUSTNESS_SPEC
from config import OUT_ANALYSIS
from config import OUT_DATA
from config import OUT_DATA_LISS
from config import OUT_UNDER_GIT


def run_kmeans(produces, k_max, sort_groups_by, sort_in_ascending_order, df, params):
    X_std = (df - df.mean()) / df.std()
    tss = (X_std**2).sum().sum()

    group_assignments = pd.DataFrame(index=df.index)
    group_stats = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [(str(k), g) for k in range(2, k_max + 1) for g in range(k)],
            names=["k", "g"],
        )
    )
    k_to_inertia = {}

    for k in range(2, k_max + 1):
        k_kmeans = k
        kmeans = KMeans(
            n_clusters=k_kmeans,
            init="k-means++",
            n_init=10,
            max_iter=300,
            tol=1e-8,
            verbose=0,
            random_state=1,
            copy_x=True,
            algorithm="auto",
        )
        ga = pd.Series(index=df.index, name="ga", dtype="float")
        ga.update(pd.Series(data=kmeans.fit_predict(X=X_std), index=X_std.index))
        ga = ga.fillna(k)
        k_to_inertia[k] = kmeans.inertia_

        # sorting groups
        ga_to_mean_sort_variab = (
            pd.concat([df, ga], axis=1).groupby("ga").mean()[sort_groups_by]
        )
        group_sorting_mapping = (
            (ga_to_mean_sort_variab.rank(ascending=sort_in_ascending_order) - 1)
            .astype("int")
            .to_dict()
        )
        ga = ga.map(group_sorting_mapping)

        # check it actually worked as intented
        ga_to_mean_sort_variab = (
            pd.concat([df, ga], axis=1).groupby("ga").mean()[sort_groups_by]
        )
        assert (
            ga_to_mean_sort_variab.rank(ascending=sort_in_ascending_order)
            .astype("int")
            .values
            == np.array([i + 1 for i in range(k)])
        ).all(), (
            f"Groups not sorted by {sort_groups_by} with ascending equal "
            f"to {sort_in_ascending_order} for k={k_kmeans}, m={m}"
        )

        group_assignments[f"k{k}"] = ga

    df = df.join(group_assignments)
    group_stats = {}
    for k in range(2, k_max + 1):
        within_group_ss = k_to_inertia[k]
        between_group_tss_share = 1 - within_group_ss / tss

        ga = f"k{k}"
        grouped = df.groupby(ga)
        counts = grouped["ll_insen"].count()
        group_shares = counts / counts.sum()
        means = grouped[params].mean()
        means["n"] = counts.values
        means["share"] = group_shares.values
        means["between_group_tss_share"] = between_group_tss_share
        group_stats[ga] = means

    group_stats = pd.concat(group_stats)

    group_assignments.to_pickle(produces["group_assignments"])
    group_stats.to_pickle(produces["group_stats"])


k_max = K_MAX[0]
PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC + NAMES_INDICES_SPEC:
    if MODEL_SPECS[m]["indices_params"]:
        depends_on = {
            "individual": OUT_DATA / "individual.pickle",
            "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
            "indices": OUT_DATA_LISS / "ambiguous_beliefs" / "indices.pickle",
        }
    else:
        depends_on = {
            "individual": OUT_DATA / "individual.pickle",
            "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
            MODEL_SPECS[m]["est_model_name"]: OUT_UNDER_GIT
            / MODEL_SPECS[m]["est_model_name"]
            / "opt_diff_evolution"
            / "results.pickle",
        }
    produces = {
        "group_assignments": OUT_ANALYSIS / f"group_assignments_{m}.pickle",
        "group_stats": OUT_ANALYSIS / f"group_stats_{m}.pickle",
    }
    PARAMETRIZATION[m] = {
        "depends_on": depends_on,
        "produces": produces,
        "model_spec": MODEL_SPECS[m],
    }

for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=m)
    def task_grouping_ambiguity_profiles(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        k_max=K_MAX[0],
        model_spec=kwargs["model_spec"],
    ):

        # group sorting
        sort_groups_by = "ll_insen"
        sort_in_ascending_order = False

        # Put sample together
        if model_spec["indices_params"]:
            params = ["ambig_av", "ll_insen"]
            df = put_reg_sample_together(
                in_path_dict=depends_on,
                asset_calc=model_spec["asset_calc"],
                restrictions=model_spec["restrictions"],
                models=model_spec["wbw_models"],
                indices=True,
                indices_mean=model_spec["indices_mean"],
            )
        else:
            params = ["ambig_av", "ll_insen", "theta"]
            df = put_reg_sample_together(
                in_path_dict=depends_on,
                asset_calc=model_spec["asset_calc"],
                restrictions=model_spec["restrictions"],
                models=[model_spec["est_model_name"]],
            )

        df = df[params]
        if model_spec["indices_params"] and not model_spec["indices_mean"]:
            df = df.dropna()
        else:
            df = df.droplevel(level="wave")

        run_kmeans(produces, k_max, sort_groups_by, sort_in_ascending_order, df, params)
