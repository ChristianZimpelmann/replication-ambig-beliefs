"""
Runs regressions of model results on individual characteristics
"""
import pandas as pd
import pytask

from ambig_beliefs.final.utils_final import put_reg_sample_together
from ambig_beliefs.final.utils_final import select_manual_group_order
from config import BASIC_CONTROLS
from config import MODEL_SPECS
from config import NAMES_INDICES_SPEC
from config import NAMES_MAIN_SPEC
from config import NAMES_ROBUSTNESS_SPEC
from config import OUT_ANALYSIS
from config import OUT_DATA
from config import OUT_DATA_LISS
from config import OUT_UNDER_GIT
from config import ROOT


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC + NAMES_INDICES_SPEC:

    for ga in MODEL_SPECS[m]["k_groups"]:
        id = f"{m}:{ga}"
        depends_on = {
            "individual": OUT_DATA / "individual.pickle",
            "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
            "indices": OUT_DATA_LISS / "ambiguous_beliefs" / "indices.pickle",
            "utils_final": "utils_final.py",
            "group_assignments": OUT_ANALYSIS / f"group_assignments_{m}.pickle",
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
            "reg_sample_r": OUT_ANALYSIS
            / m
            / f"reg_sample_r_{ga}{MODEL_SPECS[m]['asset_calc']}.parquet",
        }
        PARAMETRIZATION[id] = {
            "m": m,
            "ga": ga,
            "depends_on": depends_on,
            "produces": produces,
            "model_spec": MODEL_SPECS[m],
        }


for id_, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=id_)
    def task_prep_reg_sample_tobit(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        model_spec=kwargs["model_spec"],
        m=kwargs["m"],
        ga=kwargs["ga"],
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
        group_assignments = pd.read_pickle(depends_on["group_assignments"])
        data = df.join(group_assignments[ga])

        # Sort mapping of manual sorting of groups
        g_man_to_g = select_manual_group_order(m, ga)
        g_to_g_man = {j: i for i, j in g_man_to_g.items()}
        data["type_man_sort"] = data[ga].map(g_to_g_man)
        data[ga] = pd.Categorical(data[ga])

        controls = BASIC_CONTROLS

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
        data = data.join(temp)

        indices_single_waves = indices_single_waves.join(pat_rec_dur["valid_choice"])
        indices_single_waves["valid_choice_and_index"] = (
            indices_single_waves["valid_choice"] & indices_single_waves["valid_indices"]
        )
        temp = (
            indices_single_waves.groupby("personal_id")["valid_choice_and_index"].sum()
            >= 2
        )
        temp.name = "at_least_2_waves_with_valid_choice_and_index"
        data = data.join(temp)

        if model_spec["indices_params"]:
            # data.columns = [f"{c}_index" for c in data]
            params = ["ambig_av", "ll_insen"]
        else:
            params = ["ambig_av", "ll_insen", "theta"]

        # Standardize ambiguity params
        for para in params:
            data[para] = (data[para] - data[para].mean()) / data[para].std()

        data_r = (
            data[
                params
                + [
                    "has_rfa",
                    "frac_of_tfa_in_rfa",
                    "type_man_sort",
                    "all_indices_valid",
                    "at_least_2_waves_with_valid_choice_and_index",
                ]
                + controls
            ]
            # .dropna(subset=controls)
            .copy()
        )
        data_r = data_r.dropna(subset=params)
        data_r["type_man_sort"] = data_r["type_man_sort"].replace(
            {i: f"T{i}" for i in range(len(data[ga].unique()))}
        )
        data_r["female"] = data_r["female"].astype(float)
        data_r["age_groups"] = data_r["age_groups"].cat.as_unordered()
        data_r["edu"] = data_r["edu"].cat.as_unordered()
        data_r["net_income_groups"] = data_r["net_income_groups"].cat.as_unordered()
        data_r["total_financial_assets_groups"] = data_r[
            "total_financial_assets_groups"
        ].cat.as_unordered()

        path_out = produces["reg_sample_r"]
        data_r.to_parquet(path_out)


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC + NAMES_INDICES_SPEC:
    asset_calc = MODEL_SPECS[m]["asset_calc"]

    for ga in MODEL_SPECS[m]["k_groups"]:
        id = f"{m}:{ga}"
        depends_on = {
            "reg_sample_r": OUT_ANALYSIS / m / f"reg_sample_r_{ga}{asset_calc}.parquet",
            "run_tobit": ROOT / "ambig_beliefs" / "final" / "run_tobit.R",
        }
        produces = {}
        for probit_tobit in ["probit", "tobit"]:
            produces.update(
                {
                    f"results_{probit_tobit}_short": OUT_ANALYSIS
                    / m
                    / f"results_{probit_tobit}_short_{ga}{asset_calc}.parquet",
                    f"results_{probit_tobit}_controls": OUT_ANALYSIS
                    / m
                    / f"results_{probit_tobit}_controls_{ga}{asset_calc}.parquet",
                    f"results_info_{probit_tobit}_short": OUT_ANALYSIS
                    / m
                    / f"results_info_{probit_tobit}_short_{ga}{asset_calc}.json",
                    f"results_info_{probit_tobit}_controls": OUT_ANALYSIS
                    / m
                    / f"results_info_{probit_tobit}_controls_{ga}{asset_calc}.json",
                }
            )
        PARAMETRIZATION[id] = {
            "depends_on": depends_on,
            "produces": produces,
            "cluster_std": (
                MODEL_SPECS[m]["indices_params"] and not MODEL_SPECS[m]["indices_mean"]
            ),
        }

for id_, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(
        id=id_,
        kwargs={
            "formula_short": "relevel(type_man_sort, ref='T0')",
            "formula_controls": "relevel(type_man_sort, ref='T0') + "
            + "+".join(BASIC_CONTROLS),
            "ambig_types": True,
            "cluster_std": kwargs["cluster_std"],
        },
    )
    @pytask.mark.produces(kwargs["produces"])
    @pytask.mark.depends_on(kwargs["depends_on"])
    @pytask.mark.r(script="run_tobit.R")
    def task_tobit_probit_invest_on_groups():
        pass


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_INDICES_SPEC:
    asset_calc = MODEL_SPECS[m]["asset_calc"]
    formula_short = (
        "ambig_av + ll_insen"
        if MODEL_SPECS[m]["indices_params"]
        else "ambig_av + ll_insen + theta"
    )

    depends_on = {
        "reg_sample_r": OUT_ANALYSIS / m / f"reg_sample_r_{ga}{asset_calc}.parquet",
        "run_tobit": ROOT / "ambig_beliefs" / "final" / "run_tobit.R",
    }
    produces = {}
    for probit_tobit in ["probit", "tobit"]:
        for sel_var in [
            "",
            # "_all_indices_valid",
            # "_at_least_2_waves_with_valid_choice_and_index",
        ]:
            produces.update(
                {
                    f"results_{probit_tobit}_short{sel_var}": OUT_ANALYSIS
                    / m
                    / f"results_params_{probit_tobit}_short{asset_calc}{sel_var}.parquet",
                    f"results_{probit_tobit}_controls{sel_var}": OUT_ANALYSIS
                    / m
                    / f"results_params_{probit_tobit}_controls{asset_calc}{sel_var}.parquet",
                    f"results_info_{probit_tobit}_short{sel_var}": OUT_ANALYSIS
                    / m
                    / f"results_params_info_{probit_tobit}_short{asset_calc}{sel_var}.json",
                    f"results_info_{probit_tobit}_controls{sel_var}": OUT_ANALYSIS
                    / m
                    / f"results_params_info_{probit_tobit}_controls{asset_calc}{sel_var}.json",
                }
            )
    PARAMETRIZATION[m] = {
        "depends_on": depends_on,
        "produces": produces,
        "ambig_types": False,
        "formula_short": formula_short,
        "cluster_std": (
            MODEL_SPECS[m]["indices_params"] and not MODEL_SPECS[m]["indices_mean"]
        ),
    }

for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(
        id=m,
        kwargs={
            "formula_short": kwargs["formula_short"],
            "formula_controls": f"{kwargs['formula_short']} + "
            + "+".join(BASIC_CONTROLS),
            "ambig_types": kwargs["ambig_types"],
            "cluster_std": kwargs["cluster_std"],
        },
    )
    @pytask.mark.produces(kwargs["produces"])
    @pytask.mark.depends_on(kwargs["depends_on"])
    @pytask.mark.r(script="run_tobit.R")
    def task_tobit_probit_invest_on_params():
        pass
