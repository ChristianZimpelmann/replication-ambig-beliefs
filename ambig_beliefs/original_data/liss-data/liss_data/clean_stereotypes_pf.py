import numpy as np
import pandas as pd
import yaml

from config import IN_SPECS_LISS
from liss_data.cleaning_helpers import replace_values
from liss_data.cleaning_helpers import set_types_file
from liss_data.data_checks import general_data_checks
from liss_data.utils_liss_data import clean_background_vars


def clean_stereotypes_pf(data):
    """
    Data cleaning for fin literacy questionnaire.
    """

    with open(IN_SPECS_LISS / "xyy-stereotypes-pf_replacing.yaml") as file:
        replace_dict = yaml.full_load(file)

    rename_df = pd.read_csv(
        IN_SPECS_LISS / "xyy-stereotypes-pf_renaming.csv",
        sep=";",
    )

    data = replace_values(data, replace_dict, rename_df)

    # Calculate stereotypes_vars
    for var in [
        "self_assessment_greedy",
        "self_assessment_gambler",
        "self_assessment_selfish",
        "other_stocks_greedy",
        "other_stocks_gambler",
        "other_stocks_selfish",
        "other_nostocks_greedy",
        "other_nostocks_gambler",
        "other_nostocks_selfish",
        "response_stocks_greedy",
        "response_stocks_gambler",
        "response_stocks_selfish",
        "response_nostocks_greedy",
        "response_nostocks_gambler",
        "response_nostocks_selfish",
    ]:
        data[var] = data[var].astype(float)

    # Add new variables which exclude I dont know
    for var in [
        "confidence_stock_knowledge",
        "zero_sum_belief_stock",
        "belief_no_skill_succ_stock",
        "no_talk_money",
    ]:
        data[f"{var}_incl_dont_know"] = data[var]
        data[var] = data[var].replace({"I do not know": np.nan}).astype(float)

    # Clean demographic variables
    data["net_income_incl_cat"] = data["net_income_incl_cat"].astype(float)
    data["gross_income_incl_cat"] = data["gross_income_incl_cat"].astype(float)
    data = clean_background_vars(data)

    # Set types of variables using renaming file.
    data = set_types_file(
        panel=data,
        rename_df=rename_df,
        cat_sep="|",
        int_to_float=True,
        bool_to_float=True,
        few_int_to_cat=False,
    )

    # Check some consistency in the data.
    _check_stereotypes_pf(data)

    return data


def _check_stereotypes_pf(panel):
    """Check some of the data in the ambigous_beliefs database.

    Args:
        panel(pandas.DataFrame): The data frame to be checked.
    """
    out = panel.copy()
    general_data_checks(out)
