"""
Makes violinplots comparing distributions of parameters across aex and temperature domains
using wave 4 estimates.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytask
import seaborn as sns

from ambig_beliefs.final.utils_final import put_reg_sample_together
from config import MODEL_SPECS
from config import NAMES_MAIN_SPEC as pooled_models
from config import OUT_DATA
from config import OUT_FIGURES
from config import OUT_UNDER_GIT
from config import ROOT

param_to_name = {
    "ambig_av": r"$\alpha$",
    "ll_insen": r"$\ell$",
    "theta": r"$\sigma$",
}
param_to_lim = {"ambig_av": (-0.5, 0.5), "ll_insen": (0, 1), "theta": (0, 0.6)}


def add_vertical_line(ax, x, source, col, ls="--"):
    "Adds vertical line going up to the y value of the plotted density"
    data_x, data_y = ax.get_lines()[0].get_data()
    data_x_temp, data_y_temp = ax.get_lines()[1].get_data()

    x_pos = np.abs(data_x - x).argmin()
    if source == "aex":
        ax.plot([data_x[x_pos], data_x[x_pos]], [0, data_y[x_pos]], color=col, ls=ls)
    else:
        ax.plot(
            [data_x[x_pos], data_x[x_pos]], [0, data_y_temp[x_pos]], color=col, ls=ls
        )


PARAMETRIZATION = {}
for m in pooled_models:
    wbw_models = MODEL_SPECS[m]["wbw_models"]
    climate_model = MODEL_SPECS[m]["climate_model"]
    depends_on = {
        "utils_final": ROOT / "ambig_beliefs" / "final" / "utils_final.py",
        "individual": OUT_DATA / "individual.pickle",
        "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
    }
    depends_on.update(
        {
            mod: OUT_UNDER_GIT / mod / "opt_diff_evolution" / "results.pickle"
            for mod in wbw_models + [climate_model] + [m]
        }
    )
    PARAMETRIZATION[m] = {
        "depends_on": depends_on,
        "produces": {f"domain_dists_{m}": OUT_FIGURES / m / "domain_dists.pdf"},
        "wbw_models": wbw_models,
        "climate_model": climate_model,
    }

for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=m)
    def task_fig_dists_across_domains(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        m=m,
        wbw_models=kwargs["wbw_models"],
        climate_model=kwargs["climate_model"],
        asset_calc=MODEL_SPECS[m]["asset_calc"],
        param_to_name=param_to_name,
        param_to_lim=param_to_lim,
    ):
        m = str(m)
        models = wbw_models + [climate_model]
        df = put_reg_sample_together(
            in_path_dict=depends_on,
            asset_calc=asset_calc,
            restrictions="personal_id.notnull()",
            models=models + [m],
        )

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Superimposed density + quantiles plots
        for j, param in enumerate(["ambig_av", "ll_insen", "theta"]):
            temp = df[param].dropna().reset_index().query("wave in [4, 'temp']")
            temp["Domain"] = temp["wave"].map({4: "Aex", "temp": "Climate"})
            data_aex = temp.query("Domain == 'Aex'")[param]
            data_temp = temp.query("Domain == 'Climate'")[param]

            aex_col = "blue"
            temp_col = "orange"
            colors = [aex_col, temp_col]

            sns.kdeplot(
                x=param,
                hue="Domain",
                data=temp,
                ax=ax[j],
                palette=colors,
                hue_order=["Aex", "Climate"],
            )
            ax[j].set_xlim(param_to_lim[param])
            ax[j].get_legend().set_title("")

            data_x, data_y = ax[j].get_lines()[0].get_data()
            data_x_temp, data_y_temp = ax[j].get_lines()[1].get_data()

            median_aex = data_aex.median()
            upper_quartile_aex = data_aex.quantile(0.75)
            lower_quartile_aex = data_aex.quantile(0.25)

            median_temp = data_temp.median()
            upper_quartile_temp = data_temp.quantile(0.75)
            lower_quartile_temp = data_temp.quantile(0.25)

            add_vertical_line(ax[j], median_aex, "aex", aex_col, ls="-.")
            add_vertical_line(ax[j], median_temp, "temp", temp_col, ls="-.")

            add_vertical_line(ax[j], lower_quartile_aex, "aex", aex_col)
            add_vertical_line(ax[j], lower_quartile_temp, "temp", temp_col)

            add_vertical_line(ax[j], upper_quartile_aex, "aex", aex_col)
            add_vertical_line(ax[j], upper_quartile_temp, "temp", temp_col)

            ax[j].set_ylabel("")
            ax[j].set_xlabel(param_to_name[param], fontsize=16)
            ax[j].tick_params(labelsize=15)

            if j > 0:
                ax[j].get_legend().remove()

        plt.tight_layout()
        path_out = produces[f"domain_dists_{m}"]
        fig.savefig(path_out)
        fig.clear()
