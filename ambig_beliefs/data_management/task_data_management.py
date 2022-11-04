import numpy as np
import pandas as pd
import pytask

from config import IN_DATA_LISS
from config import IN_SPECS_LISS
from config import OUT_DATA
from config import OUT_DATA_LISS


def calculate_neoadditive_tests(baseline_probs):
    """
    Calculates a df with tests of the neoadditive model, the Goldstein Einhorn
    model and the subjective expected utility model
    """

    def log_odds_transform(p):
        o = p / (1 - p)
        lo = np.log(o)
        return lo

    def transform_to_alpha(x):
        return np.exp(x / 2)

    neoadditive_goldstein_tests = pd.DataFrame(
        index=baseline_probs.groupby(["personal_id", "wave"]).first().index
    )
    list_of_partitions = [["1", "1c"], ["2", "2c"], ["3", "3c"], ["1", "2", "3"]]
    lin_cols = []
    lo_cols = []
    for part in list_of_partitions:
        col_name_lin = "+".join(part)
        val_lin = (
            (
                baseline_probs.loc[
                    (slice(None), slice(None), part), "baseline_matching_prob_midp"
                ]
                / 100
            )
            .groupby(["personal_id", "wave"])
            .sum()
        )
        neoadditive_goldstein_tests[col_name_lin] = val_lin

        if part != ["1", "2", "3"]:
            col_name_lo = "+".join(part) + "_lo"
            lin_cols.append(col_name_lin)
            lo_cols.append(col_name_lo)
            val_lo = transform_to_alpha(
                log_odds_transform(
                    baseline_probs.loc[
                        (slice(None), slice(None), part), "baseline_matching_prob_midp"
                    ]
                    / 100
                )
                .groupby(["personal_id", "wave"])
                .sum()
            )
            neoadditive_goldstein_tests[col_name_lo] = val_lo

    # reorder cols
    neoadditive_goldstein_tests = neoadditive_goldstein_tests[
        ["1+2+3"] + lin_cols + lo_cols
    ]
    # create mean absolute deviations
    neoadditive_goldstein_tests["mad_from_1"] = neoadditive_goldstein_tests[
        ["1+2+3"] + lin_cols
    ].apply(lambda x: np.abs(x - 1).mean(), axis=1)
    neoadditive_goldstein_tests["mad_of_sums"] = neoadditive_goldstein_tests[
        lin_cols
    ].mad(axis=1)
    neoadditive_goldstein_tests["mad_of_lo_alphas"] = neoadditive_goldstein_tests[
        lo_cols
    ].mad(axis=1)

    return neoadditive_goldstein_tests


def calculate_set_monotonicity_violations(
    baseline_probs, choice_properties, no_zero_event
):
    """
    Creates a dataframe with measures of set-monotonicity violations.
    """
    from scipy.optimize import minimize

    def find_min_l2dist_to_remove_subsetv(m):
        """
        Finds the minmal L2 distance the matching probabilities need to be moved from their
        interval midpoints, and the component wise L1 shifts, to eliminate
        set-monotonicity violations. The solution is characterised by the optimisation problem:

        m = vector of matching probabilities ordered as 0, 1, 2, 3, 1c, 2c, 3c
        theta = vector of matching probabilities free of set-monotonicity violations to be found

        choose theta so as to
            min ||m - theta||
            s.t. theta in [0, 100] and
            s.t. 8 inequality constraints characterising the set-monotonicity violations

        min L2 distance = value of objective function at solution
        component wise L1 shift =  |m - theta*|

        :param np.array m: vector of matching probabilities ordered as 0, 1, 2, 3, 1c, 2c, 3c
        """

        def squared_distance_to_m(theta, m):
            diff = theta - m
            return np.dot(diff, diff)

        bnds = ((0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100))
        # superset subset pairs: ['0_1', '1c_2', '1c_3', '2c_0', '2c_1', '2c_3', '3c_1', '3c_2']
        # events:              0, 1, 2, 3, 1c, 2c, 3c
        # argument positions:  0, 1, 2, 3,  4,  5,  6
        # inequality constraint need to be put into format: x[i] - x[j] for x[i] >= x[j]
        # jacobian is derivative of inequality constraint wrt all elements of x

        ineq_cons = (
            {
                "type": "ineq",  # 0 greater than 1
                "fun": lambda x: np.array([x[0] - x[1]]),
                "jac": lambda x: np.array([1, -1, 0, 0, 0, 0, 0]),
            },
            {
                "type": "ineq",  # 1c (4) greater than 2
                "fun": lambda x: np.array([x[4] - x[2]]),
                "jac": lambda x: np.array([0, 0, -1, 0, 1, 0, 0]),
            },
            {
                "type": "ineq",  # 1c (4) greater than 3
                "fun": lambda x: np.array([x[4] - x[3]]),
                "jac": lambda x: np.array([0, 0, 0, -1, 1, 0, 0]),
            },
            {
                "type": "ineq",  # 2c (5)  greater than 0
                "fun": lambda x: np.array([x[5] - x[0]]),
                "jac": lambda x: np.array([-1, 0, 0, 0, 0, 1, 0]),
            },
            {
                "type": "ineq",  # 2c (5) greater than 1
                "fun": lambda x: np.array([x[5] - x[1]]),
                "jac": lambda x: np.array([0, -1, 0, 0, 0, 1, 0]),
            },
            {
                "type": "ineq",  # 2c (5) greater than 3
                "fun": lambda x: np.array([x[5] - x[3]]),
                "jac": lambda x: np.array([0, 0, 0, -1, 0, 1, 0]),
            },
            {
                "type": "ineq",  # 3c (6) greater than 1
                "fun": lambda x: np.array([x[6] - x[1]]),
                "jac": lambda x: np.array([0, -1, 0, 0, 0, 0, 1]),
            },
            {
                "type": "ineq",  # 3c (6) greater 2
                "fun": lambda x: np.array([x[6] - x[2]]),
                "jac": lambda x: np.array([0, 0, -1, 0, 0, 0, 1]),
            },
        )
        solution = minimize(
            squared_distance_to_m,
            m,
            args=(m),
            method="SLSQP",
            bounds=bnds,
            constraints=ineq_cons,
            options={"maxiter": 1e4},
        )
        return solution

    def b_is_subset_of_a(a, b):
        if b == pd.Interval(0, 0, closed="neither"):
            return True
        else:
            lowerb_inside = b.left >= a.left
            upperb_inside = b.right <= a.right
            return lowerb_inside & upperb_inside

    def get_subsets(event_details):
        """
        Find all pairs of superset - subset based on interval definitions
        """

        event_to_its_subsets = {}

        for a_event in event_details.index:
            event_to_its_subsets[a_event] = []
            a_first_interval = event_details.loc[a_event, "aex_interval_1"]
            a_second_interval = event_details.loc[a_event, "aex_interval_2"]
            for b_event in event_details.index:
                b_first_interval = event_details.loc[b_event, "aex_interval_1"]
                b_second_interval = event_details.loc[b_event, "aex_interval_2"]

                if a_event != b_event:
                    c11 = b_is_subset_of_a(a=a_first_interval, b=b_first_interval)
                    c12 = b_is_subset_of_a(a=a_second_interval, b=b_first_interval)
                    c1 = c11 or c12

                    c21 = b_is_subset_of_a(a=a_first_interval, b=b_second_interval)
                    c22 = b_is_subset_of_a(a=a_first_interval, b=b_second_interval)
                    c2 = c21 or c22

                    if c1 & c2:
                        event_to_its_subsets[a_event].append(b_event)

        return event_to_its_subsets

    # Don't use zero Event if specified
    if no_zero_event:
        choice_properties = choice_properties.loc[
            choice_properties["aex_event"] != "0"
        ].copy()

    # load data
    event_details = (
        choice_properties.xs(2, level="wave")
        .groupby("aex_event")[["aex_interval_1", "aex_interval_2"]]
        .first()
    )
    # Identify all pairs of events that such that one is the superset of the other
    event_to_its_subsets = get_subsets(event_details)
    subsetcond_test_pairs = []
    for event in event_to_its_subsets.keys():
        if event_to_its_subsets[event]:
            superset = event
            for subset in event_to_its_subsets[event]:
                pair = f"{superset}_{subset}"
                subsetcond_test_pairs.append(pair)

    indicators = [
        "midp_dist_to_no_vio",
        "min_dist_to_no_vio",
        "midp_dist_is_vio",
        "min_dist_is_vio",
    ]
    col_ind = pd.MultiIndex.from_product(
        [subsetcond_test_pairs, indicators], names=["superset_subset", "indicators"]
    )

    set_monotonicity_violations = pd.DataFrame(
        index=baseline_probs.groupby(["personal_id", "wave"]).first().index,
        columns=col_ind,
    )

    pair = subsetcond_test_pairs[0]
    superset = pair.split("_")[0]
    subset = pair.split("_")[1]

    for pair in subsetcond_test_pairs:
        superset = pair.split("_")[0]
        subset = pair.split("_")[1]

        data = baseline_probs.loc[
            (slice(None), slice(None), [superset, subset]),
        ].unstack()

        data_midp = data["baseline_matching_prob_midp"]
        data_leftb = data["baseline_matching_prob_leftb"]
        data_rightb = data["baseline_matching_prob_rightb"]

        diff_midp = data_midp[subset] - data_midp[superset]
        # if positive, that's a violation. Otherwise there is no violation.
        midp_dist_to_no_vio = diff_midp.where(diff_midp > 0, 0)

        # take smallest value for the subset, the largest for the superset
        diff_min = data_leftb[subset] - data_rightb[superset]
        min_dist_to_no_vio = diff_min.where(diff_min > 0, 0)

        set_monotonicity_violations.loc[
            :, (pair, "midp_dist_to_no_vio")
        ] = midp_dist_to_no_vio
        set_monotonicity_violations.loc[:, (pair, "midp_dist_is_vio")] = (
            midp_dist_to_no_vio > 0
        )
        set_monotonicity_violations.loc[
            :, (pair, "min_dist_to_no_vio")
        ] = min_dist_to_no_vio
        set_monotonicity_violations.loc[:, (pair, "min_dist_is_vio")] = (
            min_dist_to_no_vio > 0
        )

    # Computing aggregates over all event pairs
    mean_dist_midp = set_monotonicity_violations.loc[
        :, (slice(None), "midp_dist_to_no_vio")
    ].mean(axis=1)
    max_dist_midp = set_monotonicity_violations.loc[
        :, (slice(None), "midp_dist_to_no_vio")
    ].max(axis=1)
    mean_dist_min = set_monotonicity_violations.loc[
        :, (slice(None), "min_dist_to_no_vio")
    ].mean(axis=1)
    max_dist_min = set_monotonicity_violations.loc[
        :, (slice(None), "min_dist_to_no_vio")
    ].max(
        axis=1
    )  # noqa
    argmax_dist_min = set_monotonicity_violations.loc[
        (max_dist_min > 0), (slice(None), "min_dist_to_no_vio")
    ].idxmax(axis=1)
    sum_errors_midp = set_monotonicity_violations.loc[
        :, (slice(None), "midp_dist_is_vio")
    ].sum(
        axis=1
    )  # noqa
    sum_errors_min = set_monotonicity_violations.loc[
        :, (slice(None), "min_dist_is_vio")
    ].sum(
        axis=1
    )  # noqa

    set_monotonicity_violations[("total", "mean_midp_dist_to_no_vio")] = mean_dist_midp
    set_monotonicity_violations[("total", "max_midp_dist_to_no_vio")] = max_dist_midp
    set_monotonicity_violations[("total", "mean_min_dist_to_no_vio")] = mean_dist_min
    set_monotonicity_violations[("total", "max_min_dist_to_no_vio")] = max_dist_min
    set_monotonicity_violations[
        ("total", "pair_at_which_max_min_dist_to_no_vio")
    ] = argmax_dist_min.apply(lambda x: x[0])
    set_monotonicity_violations[("total", "sum_midp_dist_is_vio")] = sum_errors_midp
    set_monotonicity_violations[("total", "sum_min_dist_is_vio")] = sum_errors_min
    set_monotonicity_violations[("total", "any_midp_has_vio")] = sum_errors_midp > 0
    set_monotonicity_violations[("total", "any_min_has_vio")] = sum_errors_min > 0

    # add l2 distance
    events = ["0", "1", "2", "3", "1c", "2c", "3c"]
    m_matrix = (
        baseline_probs["baseline_matching_prob_midp"]
        .unstack("aex_event")[events]
        .dropna()
    )
    for row in m_matrix.index:
        m = m_matrix.loc[row]
        solution = find_min_l2dist_to_remove_subsetv(m)
        min_l2_distance = solution.fun**0.5
        component_wise_l1_shift = (m - solution.x).abs()
        set_monotonicity_violations.loc[
            row, ("total", "min_l2_midp_dist_to_no_vio")
        ] = (min_l2_distance if solution.success else np.nan)
        for e in events:
            l1_shift = component_wise_l1_shift[e] if solution.success else np.nan
            set_monotonicity_violations.loc[
                row, ("total", f"{e}_l1_shift_to_min_l2_midp_dist_sol")
            ] = l1_shift
    return set_monotonicity_violations


def prepare_for_analysis(data):
    """
    Apply sample restrictions and transform some variables for analysis.
    """
    data = data.copy()
    data["p"] = data["lottery_p_win"] / 100
    if "choice" in data:
        data["choice"] = data["choice"] == "AEX"
        data = data.loc[~data["choice_final"]]

        # Drop individuals who didn't finish the questionnaire
        def finished_questionnaire(df):
            query = (
                "aex_event == '3c' & lottery_p_win.isin([1, 5, 20, 40, 60, 80, 95, 99])"
            )
            return df.query(query)["choice_string"].count() > 0

        sel = data.groupby(["personal_id", "wave"]).apply(finished_questionnaire)
        index_raw = data.reset_index().set_index(["personal_id", "wave"]).index
        data = data[sel[index_raw].values]
    return data


def generate_event_durations(file_name_dur, file_name, question_type="aex"):
    """
    Generate durations DataFrame containing the duration of all parts.
    """

    # load all data
    path_to_raw = IN_DATA_LISS / "xxx-ambiguous-beliefs" / file_name_dur
    df = pd.read_stata(path_to_raw)
    df.rename(columns={"nomem_encr": "personal_id"}, inplace=True)
    df["personal_id"] = df["personal_id"].astype("int")
    path_to_renaming_file = IN_SPECS_LISS / "xxx-ambiguous-beliefs_renaming.csv"
    old_coln_to_new_coln = (
        pd.read_csv(path_to_renaming_file, sep=";")
        .set_index(file_name)["new_name"]
        .to_dict()
    )
    # dictionaries for remapping
    suffix_to_lott_risk = {
        1: 50,
        2: 90,
        3: 10,
        4: 95,
        5: 70,
        6: 30,
        7: 5,
        8: 99,
        9: 80,
        10: 60,
        11: 40,
        12: 20,
        13: 1,
    }

    lott_risk_to_stages = {
        50: 1,
        90: 2,
        10: 2,
        95: 3,
        70: 3,
        30: 3,
        5: 3,
        99: 4,
        80: 4,
        60: 4,
        40: 4,
        20: 4,
        1: 4,
    }
    if question_type == "aex":
        part_to_start_screen = {
            "Part1": 0,
            "Part2": 0,
            "Part3": 0,
            "Part4": 0,
            "Part5": 0,
            "Part6": 0,
            "Part7": 0,
        }
        part_to_event = {
            "Part1": "0",
            "Part2": "1",
            "Part3": "2",
            "Part4": "3",
            "Part5": "1c",
            "Part6": "2c",
            "Part7": "3c",
        }
        # extracting things from question column
        is_about_event = df["question"].apply(
            lambda x: "keuze" in x and "clim" not in x
        )
        is_separation_screen_between_events = df["question"].apply(
            lambda x: "Part" in x and "clim" not in x
        )
    elif question_type == "temp":
        part_to_start_screen = {
            "Part1 clim": 0,
            "Part2 clim": 0,
            "Part3 clim": 0,
            "Part4 clim": 0,
            "Part5 clim": 0,
            "Part6 clim": 0,
            "Part7 clim": 0,
        }
        part_to_event = {
            "Part1 clim": "0",
            "Part2 clim": "1",
            "Part3 clim": "2",
            "Part4 clim": "3",
            "Part5 clim": "1c",
            "Part6 clim": "2c",
            "Part7 clim": "3c",
        }
        # extracting things from question column
        is_about_event = df["question"].apply(lambda x: "keuze" in x and "clim" in x)
        is_separation_screen_between_events = df["question"].apply(
            lambda x: "Part" in x and "clim" in x
        )
    else:
        raise Exception("question_type must be aex or temp")

    question_identifier_liss = df.loc[is_about_event, "question"].apply(
        lambda x: x.split(",")[0]
    )  # extract keuze_1_1,help_1_1 from values like keuze_1_1,help_1_1
    question_identifier = question_identifier_liss.map(old_coln_to_new_coln)
    event_identifier = question_identifier.apply(
        lambda x: x.split("_")[1][1:]
    )  # extract 0 from values like event_e0_1
    node_identifier = question_identifier.apply(
        lambda x: x.split("_")[2]
    )  # extract 1 from values like event_e0_1
    extra_question = question_identifier.apply(
        lambda x: True if len(x.split("_")) > 3 else False
    )  # extract 1 from entries likeevent_e1_12_1
    lott_risk = node_identifier.astype("int").map(suffix_to_lott_risk)
    df["aex_event"] = event_identifier
    df["lottery_p_win"] = lott_risk
    df["is_extra_question"] = extra_question
    df["event_separation_screen"] = df.loc[
        is_separation_screen_between_events, "question"
    ]
    df["stage"] = lott_risk.map(lott_risk_to_stages)
    df["stage"] = df["stage"].fillna(
        df["event_separation_screen"].map(part_to_start_screen)
    )  # adding separation screen
    df["aex_event"] = df["aex_event"].fillna(
        df["event_separation_screen"].map(part_to_event)
    )
    df.dropna(subset=["aex_event", "stage"], inplace=True)
    df.set_index(["personal_id", "aex_event", "stage"], inplace=True)
    df.rename(columns={"difference": "duration_in_s"}, inplace=True)
    df["duration_in_s"] = pd.to_numeric(df["duration_in_s"], errors="coerce")
    df = df[["duration_in_s", "lottery_p_win"]]

    return df


def run_data_management():

    # Load basic data sets (created in liss-data)
    choices = pd.read_pickle(OUT_DATA_LISS / "ambiguous_beliefs" / "choices.pickle")
    choice_properties = pd.read_pickle(
        OUT_DATA_LISS / "ambiguous_beliefs" / "choice_properties.pickle"
    )
    baseline_matching_probs = pd.read_pickle(
        OUT_DATA_LISS / "ambiguous_beliefs" / "baseline_matching_probs.pickle"
    )

    # Prepare choices and choice_properties file
    choices_prepared = prepare_for_analysis(choices)
    choice_prop_prepared = prepare_for_analysis(choice_properties)

    # Calculate set-monotonicity violations and neoadditive tests
    model_tests = calculate_neoadditive_tests(baseline_matching_probs)

    no_zero_event = False
    set_monotonicity_violations = calculate_set_monotonicity_violations(
        baseline_matching_probs, choice_properties, no_zero_event=no_zero_event
    )

    # ToDo: update wave
    waves_spec = {
        1: {
            "timestamps_file": "wave-1/L_gaudecker2018_1_timestamps_p_do_not_use.dta",
            "data_file": "L_gaudecker2018_1_6p.dta",
        },
        2: {
            "timestamps_file": "wave-2/L_gaudecker2018_2_timestamps_p_do_not_use.dta",
            "data_file": "L_gaudecker2018_2_6p.dta",
        },
        3: {
            "timestamps_file": "wave-3/L_gaudecker2019_wave3_timestamps_p_do_not_use.dta",
            "data_file": "L_gaudecker2019_3_6p.dta",
        },
        4: {
            "timestamps_file": "wave-4/timestamps_L_gaudecker2019_4p_do_not_use.dta",
            "data_file": "L_gaudecker2019_4_6p.dta",
        },
        5: {
            "timestamps_file": "wave-5/timestamps_L_gaudecker2020_5p_do_not_use.dta",
            "data_file": "L_gaudecker2020_5_6p.dta",
        },
        6: {
            "timestamps_file": "wave-6/L_gaudecker2020_6_timestamps_p_do_not_use.dta",
            "data_file": "L_gaudecker2020_6_6p.dta",
        },
        7: {
            "timestamps_file": "wave-7/L_gaudecker2021_7_timestamps_p_do_not_use.dta",
            "data_file": "L_gaudecker2021_7_6p.dta",
        },
    }

    waves_to_merge = {"durations": []}

    for wave, ind_wave_spec in waves_spec.items():
        file_name = ind_wave_spec["data_file"]
        file_name_dur = ind_wave_spec["timestamps_file"]
        question_type = "aex" if wave == 4 else "temp"
        if wave == 4:
            for question_type in ["aex", "temp"]:
                w = 4 if question_type == "aex" else "temp"
                durations = generate_event_durations(
                    file_name_dur, file_name, question_type
                )
                durations["wave"] = w
                waves_to_merge["durations"].append(durations)
        else:
            durations = generate_event_durations(file_name_dur, file_name, "aex")
            durations["wave"] = wave
            waves_to_merge["durations"].append(durations)

    # # Merge waves
    files = {
        "choices_prepared": choices_prepared,
        "choice_prop_prepared": choice_prop_prepared,
        "set_monotonicity_violations": set_monotonicity_violations,
        "model_tests": model_tests,
    }
    for description, contents in waves_to_merge.items():
        single_wave_index_cols = list(contents[0].index.names)
        merged_waves_index_cols = single_wave_index_cols + ["wave"]
        dframe = pd.concat(contents, axis=0, sort=True)
        dframe = dframe.reset_index().set_index(merged_waves_index_cols).sort_index()
        files[description] = dframe

    # Save the files
    for name, value in files.items():
        value.to_pickle(OUT_DATA / (name + ".pickle"))


DEPENDS_ON = [
    OUT_DATA_LISS / "ambiguous_beliefs" / "choices.pickle",
    OUT_DATA_LISS / "ambiguous_beliefs" / "choice_properties.pickle",
    OUT_DATA_LISS / "ambiguous_beliefs" / "baseline_matching_probs.pickle",
]

PRODUCES = [
    OUT_DATA / "durations.pickle",
    OUT_DATA / "set_monotonicity_violations.pickle",
    OUT_DATA / "model_tests.pickle",
    OUT_DATA / "choice_prop_prepared.pickle",
    OUT_DATA / "choices_prepared.pickle",
]


@pytask.mark.depends_on(DEPENDS_ON)
@pytask.mark.produces(PRODUCES)
def task_data_management(depends_on, produces):
    run_data_management()
