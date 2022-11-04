"""
Makes a dataframe collecting various individual characteristics
"""
import numpy as np
import pandas as pd
import pytask

from ambig_beliefs.final.utils_final import std_normalise
from config import OUT_DATA
from config import OUT_DATA_LISS


def get_historical_returns():
    """
    Calculates historical frequencies of events about the 6 month ahead aex return with data
    between May 1 1999 and May 1 2019
    """
    aex_returns = pd.read_pickle(OUT_DATA / "aex_returns.pickle")
    start = pd.Timestamp(year=1999, month=5, day=1)
    end = pd.Timestamp(year=2019, month=5, day=1) - pd.DateOffset(n=6, months=1)
    date_range = pd.date_range(start=start, end=end)
    returns = aex_returns.loc[date_range]["aex_6m_ahead_return"]

    hf_0 = (returns > 0).mean()
    hf_1 = (returns > 0.1).mean()
    hf_2 = (returns < -0.05).mean()
    hf_3 = (returns.between(-0.05, 0.1, inclusive="both")).mean()
    assert np.isclose(hf_1 + hf_2 + hf_3, 1), "historical frequencies don't add up to 1"
    return {"0": hf_0, "1": hf_1, "2": hf_2, "3": hf_3}


def make_background_vars(df):
    """
    Extracts and modifies needed vars from the liss background data
    """
    df = df.groupby("personal_id").first()

    df["has_pos_net_income"] = df["net_income"].dropna() > 0
    background_vars = [
        "female",
        "age",
        "has_pos_net_income",
        "dom_situation",
        "civil_status",
        "hh_children",
        "location_urban",
        "origin",
        # "net_income",
        "edu",
        # "hh_id",
        "hh_position",
    ]

    return df[background_vars]


def make_personality_vars(df):
    """
    Extracts and modifies needed vars from the liss personality
    """
    df = df.groupby("personal_id").first()

    df["optimism_pessimsm"] = df["optimism"] - df["pessimism"]
    # Select variables that are used later
    personality_vars = [
        "extraversion",
        "agreeableness",
        "conscientiousness",
        "neuroticism",
        "openness",
        "optimism",
        "pessimism",
        "optimism_pessimsm",
    ]

    return df[personality_vars]


def make_historical_performance_vars(df):
    """
    Extracts probability of historical performance vars, extent to which individuals
    agree future is like past, and explanations of this and their financial advice.
    """
    df_out = pd.DataFrame(index=df.index)
    no_check_probs = [f"hist_perf_e{i}_nocheck" for i in range(1, 4)]
    probs = [f"hist_perf_e{i}" for i in range(1, 4)]
    agree_in_dutch_to_category = {
        "helemaal oneens": 1,
        "oneens": 2,
        "beetje oneens": 3,
        "beetje eens": 4,
        "eens": 5,
        "helemaal eens": 6,
    }
    for i in range(len(no_check_probs)):
        no_check_prob = df[no_check_probs[i]] / 100
        checked_prob = df[probs[i]] / 100
        combined_prob = checked_prob.where(checked_prob.notnull(), no_check_prob)
        df_out[no_check_probs[i]] = no_check_prob
        df_out[probs[i]] = checked_prob
        df_out[probs[i] + "_combined"] = combined_prob
    df_out["hist_perf_e0_nocheck"] = df["hist_perf_e0_nocheck"] / 100
    # calculate residuals with respect to actual returns

    hist_perf_vars = [
        "hist_perf_e0_nocheck",
        "hist_perf_e1_combined",
        "hist_perf_e2_combined",
        "hist_perf_e3_combined",
    ]
    for v in hist_perf_vars:
        df_out[v] = df_out[v].where((df_out[v] <= 1) & (df_out[v] >= 0))
    actual_returns = get_historical_returns()
    for i, v in enumerate(hist_perf_vars):
        actual_return = actual_returns[str(i)]
        resid = df_out[v] - actual_return
        df_out[v + "_resid"] = resid

    df_out["jf_less_hf_avg_abs_dev"] = (
        df_out[
            [
                "hist_perf_e0_nocheck_resid",
                "hist_perf_e1_combined_resid",
                "hist_perf_e2_combined_resid",
                "hist_perf_e3_combined_resid",
            ]
        ]
        .dropna()
        .abs()
        .mean(axis=1)
    )
    df_out["hist_perf_sum_of_nocheck_events"] = (
        df_out[["hist_perf_e1_nocheck", "hist_perf_e2_nocheck", "hist_perf_e3_nocheck"]]
        .dropna()
        .sum(axis=1)
    )
    df_out["hist_perf_has_additivity_vio"] = (
        df_out["hist_perf_sum_of_nocheck_events"]
        .dropna()
        .apply(lambda x: not np.isclose(x, 1))
        .astype("float")
    )
    df_out["hist_perf_has_set_mono_vio"] = (
        df_out["hist_perf_e0_nocheck"] < df_out["hist_perf_e1_combined"]
    ).astype("float")
    df_out.loc[
        df_out["hist_perf_e1_combined"].isna(), "hist_perf_has_set_mono_vio"
    ] = np.nan

    df_out["hist_perf_response_error"] = (
        df_out[["hist_perf_has_additivity_vio", "hist_perf_has_set_mono_vio"]]
        .dropna()
        .sum(axis=1)
        > 0
    ).astype(float)
    df_out["agree_future_like_past"] = df["agree_future_like_past"].map(
        agree_in_dutch_to_category
    )
    df_out["agree_future_like_past_2"] = df["agree_future_like_past_2"].map(
        agree_in_dutch_to_category
    )
    df_out["why_agree_future_like_past"] = df["why_agree_future_like_past"]
    df_out["financial_adv"] = df["financial_adv"]
    return df_out


def translate_comments(comments):
    """
    Returns a dictionary translating each comment and a list with comments that could
    not be translated.
    """
    from googletrans import Translator

    translator = Translator()
    dutch_to_english = {}
    untranslated_comments = []
    for dutch in comments:
        try:
            english = translator.translate(dutch).text
            dutch_to_english[dutch] = english
        except Exception:
            untranslated_comments.append(dutch)
    return dutch_to_english, untranslated_comments


def make_financial_info(assets, housing, bg):
    """
    Collect all financial variables from different liss dataframes
    """
    raw_asset_vars = [
        "wealth_excl_housing",
        "total_financial_assets",
        "risky_financial_assets",
        "has_rfa_wide_def",
        # "has_rfa",
    ]
    raw_housing_vars = [
        "home_price",
        "home_remaining_mortgage",
        "sec_home_price",
        "sec_home_remaining_mort",
    ]
    raw_income_vars = ["net_income", "gross_income"]

    # set risky assets to missing if total financial assets is missing
    assets.loc[
        assets["total_financial_assets"].isna(), "risky_financial_assets"
    ] = np.nan

    financial_info = assets[raw_asset_vars].join(housing[raw_housing_vars], how="outer")
    financial_info = financial_info.join(bg[raw_income_vars], how="outer")
    financial_info["housing_part_of_wealth"] = (
        financial_info["home_price"]
        - financial_info["home_remaining_mortgage"]
        + financial_info["sec_home_price"]
        - financial_info["sec_home_remaining_mort"]
    )
    # ToDo: housing: fillna(0) ?
    financial_info["wealth"] = (
        financial_info["wealth_excl_housing"] + financial_info["housing_part_of_wealth"]
    )
    return financial_info


def make_clean_within_household_panel(main_sample, bg):
    """
    Construct a dataframe mapping individuals in our sample to their household partner
    if they have one. If seemingly multiple partners are available, choose the one
    closest in age.
    """
    bg = bg.copy().reset_index()
    bg_main_sample = bg.query(f"personal_id in {main_sample}")
    bg_main_sample = bg_main_sample[
        ["personal_id", "age", "female", "hh_position", "year", "hh_id"]
    ].dropna()
    bg_main_sample[["year", "hh_id"]] = bg_main_sample[["year", "hh_id"]].astype("int")
    clean_within_household_links = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            bg_main_sample.set_index(["personal_id", "year"])
            .index.drop_duplicates()
            .to_numpy()
        )
    )
    clean_within_household_links.index.names = ["personal_id", "year"]
    bg_partners = bg.query(
        "hh_position in ['Household head', 'Unwedded partner', 'Wedded partner']"
    )
    within_household_links = pd.merge(
        left=bg_main_sample,
        right=bg_partners[
            ["year", "hh_id", "personal_id", "age", "female", "hh_position"]
        ],
        how="left",
        left_on=["year", "hh_id"],
        right_on=["year", "hh_id"],
    )
    idx_to_pids = (
        within_household_links.groupby(["personal_id_x", "year"])["personal_id_y"]
        .unique()
        .to_dict()
    )

    # if there's no link or one link it's all good
    # if there is more than one link in a given year, establish link to person closest in age
    clean_within_household_links["personal_id_partner"] = np.nan
    for idx in clean_within_household_links.index:
        pids = list(idx_to_pids[idx])
        if idx[0] in pids:
            pids.remove(idx[0])
        if len(pids) == 1:
            clean_within_household_links.loc[idx, "personal_id_partner"] = pids[0]
        elif len(pids) > 1:
            link_data = within_household_links.query(
                f"personal_id_x == {idx[0]} & year == {idx[1]} &"
                f" personal_id_y != {idx[0]} & personal_id_y in {pids}"
            )[["age_x", "personal_id_y", "age_y"]]
            # idxmin() returns type error, need to use to_frame()
            idxmin = (link_data["age_x"] - link_data["age_y"]).abs().to_frame().idxmin()
            best_id = link_data.loc[idxmin, "personal_id_y"].values
            clean_within_household_links.loc[idx, "personal_id_partner"] = best_id
    return clean_within_household_links


def link_financial_info_w_household_panel(financial_info, household_panel):
    """
    Link household panel of people and their partner if they have one with financial variables
    """
    linked_financials = household_panel.join(financial_info)
    linked_financials = pd.merge(
        left=linked_financials,
        right=financial_info,
        how="left",
        left_on=["personal_id_partner", "year"],
        right_index=True,
        suffixes=("", "_partner"),
    )
    return linked_financials


def make_combined_vars(df, financial_vars):
    """
    Sum variables within household. If nan for person in our sample,
    combined variable also nan. Partner variables set to 0 if nan. Numerical combined variables
    divided by 1.7 if there is a partner.
    """
    # financial_vars_partner = [v + "_partner" for v in financial_vars]
    for v in financial_vars:
        val = df[v] + df[v + "_partner"].fillna(0)
        if "has_rfa" in v:
            val = (val.dropna() > 0).astype("float")
        df[v + "_com"] = val
        df[v + "_comnoadj"] = val
        df[v + "_indadj"] = df[v]
        if v in [
            "risky_financial_assets",
            "total_financial_assets",
            "wealth",
            "net_income",
        ]:
            df[v + "_com"] = df[v + "_com"].where(
                df["personal_id_partner"].isnull(), df[v + "_com"] / np.sqrt(2)
            )
            df[v + "_indadj"] = df[v + "_indadj"].where(
                df["personal_id_partner"].isnull(), df[v + "_indadj"] / np.sqrt(2)
            )
            # df[v + "_com"] = df[v + "_com"] / np.sqrt(df["hh_members"])
            df[v + "_comnoadj"] = df[v + "_comnoadj"].where(
                df["personal_id_partner"].isnull(), df[v + "_comnoadj"] / 1
            )
    return df


def aggregate(df, financial_vars, main_sample):
    """
    Aggregate panel into crosssection using only individual data, household summed data,
    the last observation, or the mean observation. Fractions are computed before the
    last/mean operation. Booleans afterwards.
    """
    aggregated = pd.DataFrame(index=main_sample)
    for agg_level in ["ind", "indadj", "com", "comnoadj"]:
        if agg_level == "ind":
            variabs = financial_vars
        elif agg_level == "indadj":
            variabs = [v + "_indadj" for v in financial_vars]
        elif agg_level == "com":
            variabs = [v + "_com" for v in financial_vars]
        elif agg_level == "comnoadj":
            variabs = [v + "_comnoadj" for v in financial_vars]

        temp_reduced = df[variabs].copy()
        temp_reduced.columns = financial_vars
        temp_reduced["frac_of_tfa_in_rfa"] = temp_reduced[
            "risky_financial_assets"
        ] / temp_reduced["total_financial_assets"].where(
            temp_reduced["total_financial_assets"] > 0
        )

        # Winsorize at 0 and 1
        temp_reduced.loc[
            temp_reduced["frac_of_tfa_in_rfa"] < 0, "frac_of_tfa_in_rfa"
        ] = 0
        temp_reduced.loc[
            temp_reduced["frac_of_tfa_in_rfa"] > 1, "frac_of_tfa_in_rfa"
        ] = 1

        grouped = temp_reduced.groupby("personal_id")
        for years in ["last", "first", "mean"]:

            # Idea: Prioritize observations in which both partners are observed
            if years == "last":
                data = grouped.last()
            elif years == "first":
                data = grouped.first()
            elif years == "mean":
                data = grouped.mean()
            data["has_rfa"] = (data["risky_financial_assets"].dropna() > 0).astype(
                "float"
            )
            data["frac_of_tfa_in_rfa_cond_any"] = data["frac_of_tfa_in_rfa"].copy()
            data.loc[
                data["frac_of_tfa_in_rfa_cond_any"] <= 0, "frac_of_tfa_in_rfa_cond_any"
            ] = np.nan

            data.columns = [f"{c}_{agg_level}_{years}" for c in data.columns]
            aggregated = aggregated.join(data)

    return aggregated


def aggregate_financial_variables(assets, housing, bg, main_sample):
    """
    Makes individual, combined, last, mean versions for financial variables. critical_vars
    are those that need to be available for individuals in our main main_sample
    in order for a year-observation to be used.
    """
    financial_info = make_financial_info(assets, housing, bg)
    clean_within_household_panel = make_clean_within_household_panel(main_sample, bg)
    linked_financials = link_financial_info_w_household_panel(
        financial_info, clean_within_household_panel
    )

    # critical_vars = ["risky_financial_assets", "total_financial_assets"]
    financial_vars = [
        "has_rfa_wide_def",
        # "has_rfa",
        "risky_financial_assets",
        "total_financial_assets",
        "wealth",
        "net_income",
    ]
    # financial_vars_partner = [v + "_partner" for v in financial_vars]
    temp = linked_financials.copy()  # .dropna(subset=critical_vars).copy()
    temp = make_combined_vars(temp, financial_vars)

    aggregated = aggregate(temp, financial_vars, main_sample)
    aggregated.index.name = "personal_id"
    return aggregated


def make_core_questionnaire_vars(main_sample):
    """
    Make all variables that originate from the core questionnaires.
    """

    bg = pd.read_pickle(OUT_DATA_LISS / "background.pickle").query(
        "year >= 2018 & year <= 2021"
    )
    bg_vars = make_background_vars(bg)
    assets = pd.read_pickle(OUT_DATA_LISS / "assets.pickle").query(
        "year >= 2018 & year <= 2021"
    )
    housing = pd.read_pickle(OUT_DATA_LISS / "housing.pickle").query(
        "year >= 2018 & year <= 2021"
    )

    pers_raw = pd.read_pickle(OUT_DATA_LISS / "personality.pickle").query(
        "year >= 2018 & year <= 2021"
    )
    pers = make_personality_vars(pers_raw)
    agg_financial_vars = aggregate_financial_variables(assets, housing, bg, main_sample)

    # Join data sets
    df = bg_vars.loc[main_sample].join(agg_financial_vars).join(pers)

    return df


def make_ellsberg_vars(df):
    def params_in_triangle(ambig_av, ll_insen):
        cond_1 = 0 <= ll_insen <= 1
        cond_2 = -ll_insen <= 2 * ambig_av <= ll_insen
        return cond_1 & cond_2

    df.rename(
        columns={"ambig_av": "ambig_av_els", "ll_insens": "ll_insen_els"}, inplace=True
    )
    df["ambig_av_els"] *= 1 / 2
    df["paras_in_triangle_els"] = df[["ambig_av_els", "ll_insen_els"]].apply(
        lambda x: params_in_triangle(
            ambig_av=x["ambig_av_els"], ll_insen=x["ll_insen_els"]
        ),
        axis=1,
    )
    has_extra_incentive = df["group"] == "extra incentive"
    ambig_choices = [
        i
        for i in df.columns
        if any([j in i for j in ["q1", "q2", "q3", "q4", "q5"]]) and "choice" in i
    ]
    a = df[ambig_choices] == "Box O"
    b = df[ambig_choices] == "Box B"
    not_always_indifferent = a.any(axis=1) | b.any(axis=1)

    ells_durs = [
        i
        for i in df.columns
        if "dur" in i and all(j not in i for j in ["example", "duration"])
    ]
    ells_durs = [
        i
        for i in df.columns
        if "dur" in i and "example" not in i and "duration" not in i
    ]
    not_too_quick = df[ells_durs].mean(axis=1) > 3
    df["els_param_restriction"] = (
        has_extra_incentive & not_always_indifferent & not_too_quick
    )

    return df[
        [
            "ambig_av_els",
            "ll_insen_els",
            "paras_in_triangle_els",
            "els_param_restriction",
        ]
    ]


def make_num_risk(collected_data, pref_num):
    # Add responses in ambiguity wave an pref numeracy wave
    num_risk = pd.concat(
        [
            collected_data[
                [
                    "fin_numeracy",
                    "prob_numeracy",
                    "basic_numeracy",
                    "general_risk_q",
                    "quantitative_risk_q",
                ]
            ],
            pref_num[
                [
                    "basic_numeracy",
                    "general_risk_q",
                    "quantitative_risk_q",
                    "loc_fa_score",
                    # "ce_20pct_for_300eur_90pct",
                    # "reached_bad_node",
                    # "reached_bad_node_error",
                ]
            ],
        ]
    )

    # Take mean if two observations exist
    num_risk = num_risk.groupby("personal_id").mean()

    # Calculate numeracy index
    numeracy_colls = [
        "prob_numeracy",
        "fin_numeracy",
        "basic_numeracy",
    ]
    for c in numeracy_colls:
        num_risk[c] = std_normalise(num_risk[c])

    num_risk["numeracy_index"] = num_risk[numeracy_colls].mean(axis=1, skipna=False)

    # calculate some variables
    num_risk["risk_aversion_quant"] = -num_risk["quantitative_risk_q"]
    num_risk["risk_aversion_qual"] = -num_risk["general_risk_q"]

    # Weights taken from Falk et al.
    num_risk["risk_taking_index"] = (
        std_normalise(num_risk["quantitative_risk_q"]) * 0.472_998_5
        + std_normalise(num_risk["general_risk_q"]) * 0.527_001_5
    )
    num_risk["risk_aversion_index"] = std_normalise(num_risk["risk_taking_index"] * -1)

    return num_risk


def make_climate_change_qualitative_q(df):
    """
    Maps the two qualitative climate change questions from wave 4 to numerical scores.
    """
    temp = pd.DataFrame(index=df.index)

    knowledge_answers_to_scores = {
        "1 zeer slecht": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5 zeer goed": 4,
    }
    assert set(df["climate_change_knowledge"].dropna()) == set(
        knowledge_answers_to_scores.keys()
    )
    temp["understands_climate_change"] = (
        df["climate_change_knowledge"].map(knowledge_answers_to_scores).astype(float)
        / 4
    )
    threat_answers_to_scores = {
        "helemaal eens": 5,
        "beetje oneens": 4,
        "beetje eens": 3,
        "eens": 2,
        "oneens": 1,
        "helemaal oneens": 0,
    }
    assert set(df["climate_change_threat"].dropna()) == set(
        threat_answers_to_scores.keys()
    )
    temp["threatened_by_climate_change"] = (
        df["climate_change_threat"].map(threat_answers_to_scores).astype(float) / 5
    )

    return temp


DEPENDS_ON = {
    "ambiguity": OUT_DATA_LISS / "ambiguity.pickle",
    "assets": OUT_DATA_LISS / "assets.pickle",
    "housing": OUT_DATA_LISS / "housing.pickle",
    "ambiguous_beliefs": OUT_DATA_LISS / "ambiguous_beliefs.pickle",
    "background": OUT_DATA_LISS / "background.pickle",
    "personality": OUT_DATA_LISS / "personality.pickle",
    "pref_numeracy": OUT_DATA_LISS / "pref_numeracy.pickle",
    "aex_returns": OUT_DATA / "aex_returns.pickle",
}

PRODUCES = OUT_DATA / "individual.pickle"


@pytask.mark.depends_on(DEPENDS_ON)
@pytask.mark.produces(PRODUCES)
def task_make_individual_chars(depends_on, produces):
    collected_data = pd.read_pickle(depends_on["ambiguous_beliefs"])
    participated = collected_data["start_time"].unstack(level=1).notnull().any(axis=1)
    main_sample = list(participated[participated].index)
    collected_data = collected_data.reset_index().set_index("personal_id")
    pref_num = pd.read_pickle(depends_on["pref_numeracy"]).droplevel("year")

    # Numeracy and risk aversion variables
    num_risk = make_num_risk(collected_data, pref_num)

    # data from wave 3
    w3 = collected_data.query("wave == 3").copy()
    hist_perf = make_historical_performance_vars(w3)

    # make ellsberg variables
    ellsberg = pd.read_pickle(depends_on["ambiguity"]).droplevel("year")
    ellsberg = make_ellsberg_vars(ellsberg)

    # make qualitative climate change q
    w4 = collected_data.query("wave == 4").copy()
    cc_qualitative_q = make_climate_change_qualitative_q(w4)

    # data from core liss panel
    core_data = make_core_questionnaire_vars(main_sample)

    # merging everything
    individual = (
        core_data.join(ellsberg).join(hist_perf).join(num_risk).join(cc_qualitative_q)
    )

    # Save the file
    individual.to_pickle(produces)
