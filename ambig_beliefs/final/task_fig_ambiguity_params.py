import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from ambig_beliefs.final.utils_final import merge_model_results
from config import MODEL_SPECS
from config import NAMES_MAIN_SPEC
from config import NAMES_ROBUSTNESS_SPEC
from config import OUT_DATA
from config import OUT_FIGURES
from config import OUT_UNDER_GIT


def plot_ambig_joint_distribution(kind, data, path_dict):

    g = sns.jointplot(x="ambig_av", y="ll_insen", kind=kind, data=data)
    g.ax_joint.set_ylim(-0.5, 1.2)
    g.ax_joint.set_xlim(-0.6, 0.6)
    if (data["ll_insen"] < 0).any():
        # indicate where triangle would be if not all observations are in it
        g.ax_joint.plot([0, 0.5], [0, 1], color="black", ls="--")
        g.ax_joint.plot([0, -0.5], [0, 1], color="black", ls="--")
        g.ax_joint.plot([-0.5, 0.5], [1, 1], color="black", ls="--")

    g.ax_joint.set_xlabel(r"$\alpha^{AEX}$", fontsize=20)
    g.ax_joint.set_ylabel(r"$\ell^{AEX}$", fontsize=20)
    g.ax_joint.tick_params(labelsize=13)
    plt.tight_layout()

    g.savefig(path_dict[f"joint_distr_ambig_{kind}"])


def plot_joint_distribution(x, y, kind, data, path_dict):
    var_to_plot_params = {
        "ambig_av": {"lims": [-0.6, 0.6], "label": r"$\alpha^{AEX}$"},
        "ll_insen": {"lims": [-0.5, 1.2], "label": r"$\ell^{AEX}$"},
        "theta": {"lims": [0, 2], "label": r"$\sigma^{AEX}$"},
    }
    g = sns.jointplot(x=x, y=y, kind=kind, data=data)

    g.ax_joint.set_xlim(var_to_plot_params[x]["lims"])
    g.ax_joint.set_ylim(var_to_plot_params[y]["lims"])

    g.ax_joint.set_xlabel(var_to_plot_params[x]["label"], fontsize=25)
    g.ax_joint.set_ylabel(var_to_plot_params[y]["label"], fontsize=25)
    g.ax_joint.tick_params(labelsize=15)
    plt.tight_layout()

    g.savefig(
        path_dict[f"joint_distr_{x}_{y}_{kind}"],
        format="pdf",
    )


def plot_error_distribution(df, path_dict):
    fig, ax = plt.subplots()

    df["is_hyper"] = df["ll_insen"] < 0
    if df["is_hyper"].any():
        df["is_hyper"] = (df["is_hyper"]).map(
            {True: r"$\ell < 0$ (hypers.)", False: r"$\ell \in [0, 1]$"}
        )
        sns.violinplot(
            y="is_hyper", x="theta", data=df, palette="muted", orient="h", ax=ax
        )
    else:
        sns.violinplot(x="theta", data=df, palette="muted", orient="h", ax=ax)
    ax.set_ylabel("")
    ax.set_xlabel(r"$\sigma^{AEX}$", fontsize=20)
    ax.tick_params(labelsize=13)
    ax.set_xlim(0)
    plt.tight_layout()
    fig.savefig(path_dict["dist_error_kde"], format="pdf")


PARAMETRIZATION = {}
for m in NAMES_MAIN_SPEC + NAMES_ROBUSTNESS_SPEC:
    est_model_name = MODEL_SPECS[m]["est_model_name"]

    depends_on = {
        "sample_restrictions": OUT_DATA / "sample_restrictions.pickle",
        est_model_name: OUT_UNDER_GIT
        / est_model_name
        / "opt_diff_evolution"
        / "results.pickle",
    }
    produces = {
        "joint_distr_ambig_kde": OUT_FIGURES / m / "joint_distr_ambig_kde.pdf",
        "joint_distr_ambig_av_theta_kde": OUT_FIGURES
        / m
        / "joint_distr_ambig_av_theta_kde.pdf",
        "joint_distr_theta_ll_insen_kde": OUT_FIGURES
        / m
        / "joint_distr_theta_ll_insen_kde.pdf",
        "dist_error_kde": OUT_FIGURES / m / "dist_error_kde.pdf",
    }
    PARAMETRIZATION[m] = {
        "depends_on": depends_on,
        "produces": produces,
        "est_model_name": est_model_name,
    }


for m, kwargs in PARAMETRIZATION.items():

    @pytask.mark.task(id=m)
    def task_fig_ambiguity_params(
        depends_on=kwargs["depends_on"],
        produces=kwargs["produces"],
        m=m,
        est_model_name=kwargs["est_model_name"],
        restrictions=MODEL_SPECS[m]["restrictions"],
    ):
        restrictions = "&".join(restrictions.split(","))
        sample_restrictions = pd.read_pickle(depends_on["sample_restrictions"])

        results = merge_model_results(
            models=[est_model_name], in_path_dict=depends_on
        ).droplevel(level="wave")
        df = results.join(sample_restrictions).query(restrictions)

        plot_ambig_joint_distribution("kde", df, produces)
        plot_error_distribution(df, produces)

        plot_joint_distribution("theta", "ll_insen", "kde", df, produces)
        plot_joint_distribution("ambig_av", "theta", "kde", df, produces)
