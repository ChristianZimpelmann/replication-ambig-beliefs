from pathlib import Path

from pytask import console

# Hotfix for pytask issue #271
console.legacy_windows = True


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


IN_DATA_LISS = ROOT / "ambig_beliefs" / "original_data" / "liss-data" / "data"
IN_SPECS_LISS = (
    ROOT / "ambig_beliefs" / "original_data" / "liss-data" / "liss_data_specs"
)
OUT_DATA_LISS = OUT / "liss-data"
OUT_DATA_CORONA_PREP = OUT / "data" / "liss-prep"


OUT_DATA_CORONA_INSTALL = OUT / "data" / "liss-data-covid-19"
IN_SPECS_CORONA = (
    ROOT / "ambig_beliefs" / "original_data" / "liss-data" / "corona_prep_specs"
)
OUT_TABLES_CORONA = OUT / "data" / "corona-tables"

IN_DATA = ROOT / "ambig_beliefs" / "original_data"
TESTS = ROOT / "ambig_beliefs" / "tests"
IN_MODEL_CODE = ROOT / "ambig_beliefs" / "model_code"
IN_MODEL_SPECS = ROOT / "ambig_beliefs" / "model_specs"
OUT_DATA = OUT / "data"
OUT_ANALYSIS = OUT / "analysis"
OUT_UNDER_GIT = ROOT / "out_under_git"
OUT_FIGURES = OUT / "figures"
OUT_TABLES = OUT / "tables"
OUT_PAPER = OUT / "paper"

# LISS preparation, only used inside that project
FILE_FORMATS_LISS = ["pickle"]
CORONA_PREP_LISS = False
CORONA_INSTALL = False

# "Estimate the main models, takes a couple of days."
ESTIMATE = True
REDUCED_N_AGENTS = False
MODEL_NAMES_ESTIMATION = [
    "event_level_temp",
    "event_level_w1",
    "event_level_w2",
    "event_level_w3",
    "event_level_w4",
    "event_level_w5",
    "event_level_w6",
    "event_level_w7",    
    "event_level_temp_unrestricted_above_sigma",
    "event_level_w1_unrestricted_above_sigma",
    "event_level_w2_unrestricted_above_sigma",
    "event_level_w3_unrestricted_above_sigma",
    "event_level_w4_unrestricted_above_sigma",
    "event_level_w5_unrestricted_above_sigma",
    "event_level_w6_unrestricted_above_sigma",
    "event_level_w7_unrestricted_above_sigma",   
    "event_level_temp_all_obs",
    "event_level_w2_all_obs",
    "event_level_w3_all_obs",
    "event_level_w4_all_obs",
    "event_level_w5_all_obs",
    "event_level_w6_all_obs",
    "event_level_w7_all_obs",
    "event_level_het_prob_2_7",
    "event_level_het_prob_2_7_all_obs",
    "event_level_het_prob_2_7_unrestricted_above_sigma",
]

MODES = [
    "estimate",
    # "monte_carlo"
]

OPT_SETTINGS_NAMES = [
    # "opt_lattice",
    # "opt_buckshot",
    "opt_diff_evolution",
    "opt_diff_evolution_large",
    # "opt_diff_evolution_str_termination",
]

MODEL_SPECS = {
    "event_level_het_prob_2_7": {
        "est_model_name": "event_level_het_prob_2_7",
        "restrictions": "completed_at_least_2_waves_with_sensible_choices_excl1",
        "indices_params": False,
        "k_groups": ["k3", "k4", "k5", "k8"],
        "asset_calc": "_com_mean_inc_ind_mean_assets",
        "wbw_models": [f"event_level_w{i+1}" for i in range(1, 7)],
        "climate_model": "event_level_temp",
    },
    "event_level_het_prob_2_7_unrestricted_above_sigma": {
        "est_model_name": "event_level_het_prob_2_7_unrestricted_above_sigma",
        "restrictions": "completed_at_least_2_waves_with_sensible_choices_excl1",
        "indices_params": False,
        "k_groups": ["k4"],
        "asset_calc": "_com_mean_inc_ind_mean_assets",
        "wbw_models": [
            f"event_level_w{i+1}_unrestricted_above_sigma" for i in range(1, 7)
        ],
        "climate_model": "event_level_temp_unrestricted_above_sigma",
    },
    "event_level_het_prob_2_7_all_obs": {
        "est_model_name": "event_level_het_prob_2_7_all_obs",
        "restrictions": "completed_any_wave_excl1",
        "indices_params": False,
        "k_groups": ["k4"],
        "asset_calc": "_com_mean_inc_ind_mean_assets",
        "wbw_models": [f"event_level_w{i+1}_all_obs" for i in range(1, 7)],
        "climate_model": "event_level_temp_all_obs",
    },
    "event_level_het_prob_2_7_balanced_panel": {
        "est_model_name": "event_level_het_prob_2_7",
        "restrictions": "completed_at_least_6_waves_with_sensible_choices_excl1",
        "indices_params": False,
        "k_groups": ["k4"],
        "asset_calc": "_com_mean_inc_ind_mean_assets",
        "wbw_models": [f"event_level_w{i+1}" for i in range(1, 7)],
        "climate_model": "event_level_temp",
    },
    "indices_mean": {
        "restrictions": "completed_at_least_2_waves_with_sensible_choices_excl1",
        "indices_params": True,
        "k_groups": ["k3", "k4", "k5"],
        "asset_calc": "_com_mean_inc_ind_mean_assets",
        "wbw_models": list(range(1, 7)),
        "climate_model": "temp",
        "indices_mean": True,
    },
    "indices_single_waves": {
        "restrictions": "completed_at_least_2_waves_with_sensible_choices_excl1",
        "indices_params": True,
        "k_groups": ["k3", "k4", "k5"],
        "asset_calc": "_com_mean_inc_ind_mean_assets",
        "wbw_models": list(range(1, 7)),
        "climate_model": "temp",
        "indices_mean": False,
    },
}

NAMES_MAIN_SPEC = ["event_level_het_prob_2_7"]
NAMES_ROBUSTNESS_SPEC = [
    "event_level_het_prob_2_7_unrestricted_above_sigma",
    "event_level_het_prob_2_7_all_obs",
    "event_level_het_prob_2_7_balanced_panel",
]
NAMES_INDICES_SPEC = [
    "indices_mean",
    "indices_single_waves",
]

K_MAX = [15, 8]

BASIC_CONTROLS = [
    "age_groups",
    "female",
    "edu",
    "net_income_groups",
    "total_financial_assets_groups",
    "risk_aversion_index",
    "numeracy_index",
]
