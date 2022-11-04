"""
Makes dataframes with a range of sample selections
"""
import pandas as pd
import pytask

from config import OUT_DATA
from config import OUT_DATA_LISS


DEPENDS_ON = {
    "durations": OUT_DATA / "durations.pickle",
    "dist_to_rec_patterns": OUT_DATA / "dist_to_rec_patterns.pickle",
    "matching_probs": OUT_DATA_LISS
    / "ambiguous_beliefs"
    / "baseline_matching_probs.pickle",
}
PRODUCES = OUT_DATA / "pat_rec_and_dur_restrictions.pickle"


@pytask.mark.depends_on(DEPENDS_ON)
@pytask.mark.produces(PRODUCES)
def task_make_pat_rec_and_dur_restrictions(depends_on, produces):
    df_rec_patt_dist = pd.read_pickle(depends_on["dist_to_rec_patterns"])
    durations = pd.read_pickle(depends_on["durations"])
    matching_probs = pd.read_pickle(depends_on["matching_probs"])
    durations = durations.groupby(["personal_id", "aex_event", "stage", "wave"])[
        ["duration_in_s"]
    ].sum()

    # excluding stage = 0, the event separation screen
    durations = durations.query("stage > 0")

    # selecting first stage where reading is most important
    data = durations.query("stage == 1").groupby(["personal_id", "wave"]).median()
    for q in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
        data[f"quickest_{int(q * 100)}perc"] = (
            data["duration_in_s"].groupby("wave").transform(lambda x: x < x.quantile(q))
        )
    data = data.drop(columns=["duration_in_s"])

    # Add recurring pattern indicator
    data["has_rec_pattern"] = (
        df_rec_patt_dist["dist_to_rec_patterns_always_lot_or_aex"] == 0
    )

    # Add indicator whether completed full elicitation (all 7 matching probs observed)
    data = data.join(
        matching_probs.groupby(["personal_id", "wave"]).count()[
            "baseline_matching_prob_midp"
        ]
        == 7
    ).rename(columns={"baseline_matching_prob_midp": "completed_elicitation"})
    data["completed_elicitation"] = data["completed_elicitation"].fillna(False)

    # Calculate valid choice as used in Ambig-beliefs Paper
    data["valid_choice"] = data["completed_elicitation"] & (
        ~data["quickest_15perc"] | ~data["has_rec_pattern"]
    )

    # Throw out subject with strange answer pattern (see email by Christian to Miquelle)
    data = data.drop([(848721, "temp")])

    # Save file
    data.to_pickle(produces)
