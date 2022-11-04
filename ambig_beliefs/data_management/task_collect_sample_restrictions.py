"""
Makes dataframes with a range of sample selections
"""
import pandas as pd
import pytask

from config import OUT_DATA
from config import OUT_DATA_LISS

DEPENDS_ON = {
    "ambiguous_beliefs": OUT_DATA_LISS / "ambiguous_beliefs.pickle",
    "individual": OUT_DATA / "individual.pickle",
    "pat_rec_and_dur_restrictions": OUT_DATA / "pat_rec_and_dur_restrictions.pickle",
}
PRODUCES = OUT_DATA / "sample_restrictions.pickle"


@pytask.mark.depends_on(DEPENDS_ON)
@pytask.mark.produces(PRODUCES)
def task_collect_sample_restrictions(depends_on, produces):

    # data
    collected_data = pd.read_pickle(depends_on["ambiguous_beliefs"])
    individual = pd.read_pickle(depends_on["individual"])
    pat_rec_dur = pd.read_pickle(depends_on["pat_rec_and_dur_restrictions"])

    # intial index: people who took part in (but possibly did not complete) any of our waves
    sample_restrictions = pd.DataFrame(index=collected_data.index.levels[0])

    for asset_calc in [
        "_ind_first",
        "_ind_last",
        "_ind_mean",
        "_com_first",
        "_com_last",
        "_com_mean",
    ]:

        sample_restrictions["has_at_least_5k_tfa" + asset_calc] = (
            individual["total_financial_assets" + asset_calc] >= 5000
        )
        sample_restrictions["has_at_least_10k_tfa" + asset_calc] = (
            individual["total_financial_assets" + asset_calc] >= 10000
        )

    # Count number of completed waves
    sample_restrictions["n_completed"] = (
        pat_rec_dur.query("wave != ['temp']")["completed_elicitation"]
        .groupby("personal_id")
        .sum()
    )
    sample_restrictions["n_completed_excl1"] = (
        pat_rec_dur.query("wave != ['temp', 1]")["completed_elicitation"]
        .groupby("personal_id")
        .sum()
    ).fillna(0)
    n_waves = len(pat_rec_dur.query("wave != ['temp']").index.unique(level="wave"))
    n_waves_excl1 = len(
        pat_rec_dur.query("wave != ['temp', 1]").index.unique(level="wave")
    )
    sample_restrictions["completed_any_wave"] = sample_restrictions["n_completed"] >= 1
    sample_restrictions["completed_any_wave_excl1"] = (
        sample_restrictions["n_completed_excl1"] >= 1
    )
    sample_restrictions["completed_all_obs"] = (
        sample_restrictions["n_completed"] == n_waves
    )
    sample_restrictions["completed_all_obs_excl1"] = (
        sample_restrictions["n_completed_excl1"] == n_waves_excl1
    )

    # Count number of valid choices
    sample_restrictions["n_valid_choices"] = (
        pat_rec_dur.query("wave != ['temp']")["valid_choice"]
        .groupby("personal_id")
        .sum()
    )

    sample_restrictions["n_valid_choices_excl1"] = (
        pat_rec_dur.query("wave != ['temp', 1]")["valid_choice"]
        .groupby("personal_id")
        .sum()
    )

    sample_restrictions["n_valid_choices_excl1"] = sample_restrictions[
        "n_valid_choices_excl1"
    ].fillna(0)
    for i in range(1, int(sample_restrictions["n_valid_choices"].max()) + 1):
        sample_restrictions[f"completed_at_least_{i}_waves_with_sensible_choices"] = (
            sample_restrictions["n_valid_choices"] >= i
        )
    for i in range(1, int(sample_restrictions["n_valid_choices_excl1"].max()) + 1):
        sample_restrictions[
            f"completed_at_least_{i}_waves_with_sensible_choices_excl1"
        ] = (sample_restrictions["n_valid_choices_excl1"] >= i)
    sample_restrictions = sample_restrictions.fillna(False)
    sample_restrictions.to_pickle(produces)
