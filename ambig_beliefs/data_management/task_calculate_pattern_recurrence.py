import numpy as np
import pandas as pd
import pytask

from config import OUT_DATA


def concat_lists(x):
    li = []
    for i in range(7):
        li += list(x[i])
    return np.array(li)


def dist_to_rec_patterns(s, patterns):
    rec_all = [
        7 * [0, 0, 0, 0],
        7 * [0, 0, 0, 1],
        7 * [0, 0, 1, 0],
        7 * [0, 0, 1, 1],
        7 * [0, 1, 0, 0],
        7 * [0, 1, 0, 1],
        7 * [0, 1, 1, 0],
        7 * [0, 1, 1, 1],
        7 * [1, 0, 0, 0],
        7 * [1, 0, 0, 1],
        7 * [1, 0, 1, 0],
        7 * [1, 0, 1, 1],
        7 * [1, 1, 0, 0],
        7 * [1, 1, 0, 1],
        7 * [1, 1, 1, 0],
        7 * [1, 1, 1, 1],
    ]
    rec_t5 = [
        7 * [0, 0, 0, 0],
        7 * [1, 1, 1, 1],
        7 * [0, 1, 1, 1],
        7 * [0, 1, 0, 1],
        7 * [0, 1, 1, 0],
    ]
    rec_o = [7 * [0, 0, 0, 0], 7 * [1, 1, 1, 1]]
    if patterns == "all_rec":
        rec = rec_all
    elif patterns == "top5_freq":
        rec = rec_t5
    elif patterns == "always_lot_or_aex":
        rec = rec_o
    dists = [sum(np.abs(s - np.array(p))) for p in rec]
    return min(dists)


@pytask.mark.depends_on(OUT_DATA / "choices_prepared.pickle")
@pytask.mark.produces(OUT_DATA / "dist_to_rec_patterns.pickle")
def task_calculate_pattern_recurrence(depends_on, produces):
    choices = pd.read_pickle(depends_on)
    waves = choices.index.get_level_values(level="wave").unique()
    concat_these = []
    for w in waves:
        dist = pd.DataFrame(index=choices.index.levels[0])
        dist["wave"] = w
        choices_w = choices.xs(w, level="wave")
        bin_seqs = (
            choices_w.groupby(["personal_id", "aex_event"])["choice"]
            .apply(lambda x: tuple(x.values))
            .unstack()
        )
        # padd length 3 seqeuences so they are minimally different from
        # length 4 sequences with closest matching prob
        bin_seqs_padded = bin_seqs.copy()
        bin_seqs_padded = bin_seqs_padded.applymap(
            lambda x: (1, 1, 0, 0) if x == (1, 1, 0) else x
        )
        bin_seqs_padded = bin_seqs_padded.applymap(
            lambda x: (0, 0, 1, 1) if x == (0, 0, 1) else x
        )
        bin_seqs_padded = bin_seqs_padded.dropna().applymap(lambda x: np.array(x))
        bin_seqs_padded["combined"] = bin_seqs_padded.apply(concat_lists, axis=1)
        for variant in ["all_rec", "top5_freq", "always_lot_or_aex"]:
            d = bin_seqs_padded["combined"].apply(
                lambda x: dist_to_rec_patterns(x, variant)
            )
            dist[f"dist_to_rec_patterns_{variant}"] = d
        concat_these.append(dist)
    dist = pd.concat(concat_these, axis=0, sort=False)
    dist = dist.reset_index().set_index(["personal_id", "wave"]).sort_index()
    dist.to_pickle(produces)
