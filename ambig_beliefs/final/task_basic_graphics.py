import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns

from config import IN_DATA
from config import OUT_FIGURES


def slope_marker(origin, slope, size_frac=0.1, pad_frac=0.1, ax=None, invert=False):
    """Plot triangular slope marker labeled with slope.

    Parameters
    ----------
    origin : (x, y)
        tuple of x, y coordinates for the slope
    slope : float or (rise, run)
        the length of the slope triangle
    size_frac : float
        the fraction of the xaxis length used to determine the size of the slope
        marker. Should be less than 1.
    pad_frac : float
        the fraction of the slope marker used to pad text labels. Should be less
        than 1.
    invert : bool
        Normally, the slope marker is below a line for positive slopes and above
        a line for negative slopes; `invert` flips the marker.
    """
    if ax is None:
        ax = plt.gca()

    if np.iterable(slope):
        rise, run = slope
        slope = float(rise) / run
    else:
        rise = run = None

    x0, y0 = origin
    xlim = ax.get_xlim()
    dx_linear = size_frac * (xlim[1] - xlim[0])
    dx_decades = size_frac * (np.log10(xlim[1]) - np.log10(xlim[0]))

    if invert:
        dx_linear = -dx_linear
        dx_decades = -dx_decades

    dx = dx_linear
    # x_run = _text_position(x0, dx/2.)
    # x_rise = _text_position(x0+dx, pad_frac * dx)

    dy = dx_linear * slope
    # y_run = _text_position(y0, -(pad_frac * dy))
    # y_rise = _text_position(y0, dy/2.)

    # x_pad = pad_frac * dx
    # y_pad = pad_frac * dy
    # va = "top" if y_pad > 0 else "bottom"
    # ha = "left" if x_pad > 0 else "right"
    # if rise is not None:
    #    ax.text(x_run, y_run, str(run), va=va, ha='center')
    #    ax.text(x_rise, y_rise, str(rise), ha=ha, va='center')
    # else:
    #    ax.text(x_rise, y_rise, str(slope), ha=ha, va='center')

    ax.add_patch(_slope_triangle(origin, dx, dy))


def _slope_triangle(
    origin, dx, dy, ec="gray", fc="0.8", fill=False, ls="-", lw=3, **poly_kwargs
):
    """Return Polygon representing slope.
      /|
     / | dy
    /__|
     dx
    """
    verts = [np.asarray(origin)]
    verts.append(verts[0] + (dx, 0))
    verts.append(verts[0] + (dx, dy))
    return plt.Polygon(verts, ec=ec, fc=fc, fill=fill, ls=ls, lw=lw, **poly_kwargs)


def make_neoadditive_fun_plot(ambig_av, ll_insen, path_dict, short_vertical=False):

    fig, ax = plt.subplots(figsize=(8, 8))
    p = np.linspace(0, 1, 1000)

    ax.plot(p, p, color="gray", lw=1, ls="--")
    if short_vertical:
        gap = 0.02
        if ambig_av > 0:
            y_h = 0.5 - gap
            y_l = 0.5 - ambig_av + gap
        else:
            y_h = 0.5 - ambig_av - gap
            y_l = 0.5 + gap
        ax.plot([0.5, 0.5], [y_l, y_h], color="gray", lw=4, ls="-")
        # ax.axvline(x=0.5, ymin=0.22, ymax=0.48, color="gray", lw=2, ls="-")
    if ambig_av is not None:
        # ambig_av = ambig_av / 2
        m = (ll_insen / 2) - ambig_av + (1 - ll_insen) * p
        title = (
            f"likelihood insens.$ = {ll_insen:3.2g}$, ambiguity av.$ = {ambig_av:3.2g}$"
        )
        ax.plot(p, m, color=sns.color_palette()[0], lw=6)

        # position of slope marker x: 0.45, y: use the line and subtract a little bit
        slope_marker(
            (0.8, (ll_insen / 2) - ambig_av + (1 - ll_insen) * 0.8 - 0.05),
            (1 - ll_insen, 1),
            ax=ax,
        )
    else:
        title = "likelihood insens.$ = 0$, ambiguity av.$ = 0$"
    title = " "

    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel(r"Subjective probability $\Pr_\mathrm{subj}(E)$", fontsize=25)
    ax.set_ylabel("Decision weight $W(E)$", fontsize=25)
    ax.set_title(title, fontsize=25)

    ax.tick_params(labelsize=20)

    # fig.tight_layout()
    if ambig_av is None:
        file_name = "neo_additive_illustration_empty"
    else:
        _aa = int(ambig_av * 100)
        _ll = int(ll_insen * 100)
        file_name = f"neo_additive_illustration_aa_{_aa}_ll_{_ll}"
    fig.savefig(path_dict[file_name], format="pdf")
    plt.close("all")


def make_valid_parameters_plot(ambig_av, ll_insen, path_dict):

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.fill([-1, 0, 1], [1, 0, 1], "#ff7f0e", alpha=0.5)
    ax.fill([-0.5, 0, 0.5], [1, 0, 1], "#ff7f0e", alpha=0.5)
    if ambig_av is not None:
        # ambig_av = ambig_av / 2
        ax.plot(
            ambig_av, ll_insen, marker="X", markersize=20, color=sns.color_palette()[0]
        )
    # ax.set_ylim(-0.5, 1.5)
    labelsize = 25
    ax.set_xlabel(r"Ambiguity aversion", fontsize=labelsize)
    ax.set_ylabel(r"Likelihood insensitivity", fontsize=labelsize)
    ax.tick_params(labelsize=20)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([0, 0.5, 1])
    # title = (
    #     "$"
    #     + r"\alpha =  "
    #     + str(np.round(ambig_av, 2))
    #     + r", \ell = "
    #     + str(np.round(ll_insen, 2))
    #     + "$"
    # )
    # ax.set_title(title, fontsize=25)
    ax.set_title("Valid Ambiguity Parameters", fontsize=25)
    ax.set_title("", fontsize=25)

    # plt.gca().set_aspect(1 / 2, adjustable="box")
    # fig.tight_layout()
    if ambig_av is None:
        file_name = "valid_parameters_empty"
    else:
        file_name = (
            f"valid_parameters_aa_{int(ambig_av * 100)}_ll_{int(ll_insen * 100)}"
        )

    fig.savefig(path_dict[file_name], format="pdf")
    plt.close("all")


def make_val_of_investment_plot(path_in, path_out):
    aex_data = pd.read_csv(path_in)
    # change to next month
    aex_data["Date"] = pd.to_datetime(aex_data["Date"]) + pd.DateOffset(months=1)
    aex_data = aex_data.set_index("Date")

    w1 = pd.Timestamp(year=2018, month=5, day=3)
    w2 = pd.Timestamp(year=2018, month=11, day=1)
    w3 = pd.Timestamp(year=2019, month=5, day=1)
    scaling = aex_data["Close"].loc[w1]
    thousand_euro_investment = (aex_data["Close"] / scaling) * 1000

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(thousand_euro_investment, label="", lw=7)
    ax.axvline(x=w1, ls="--", color="black", label="wave 1")
    ax.axvline(x=w2, ls="--", color="black", label="wave 2")
    ax.axvline(x=w3, ls="--", color="black", label="wave 3")

    plt.legend(fontsize=20)
    ax.set_ylim(800, 1200)
    ax.set_xlim(
        pd.Timestamp(year=2018, month=1, day=1), pd.Timestamp(year=2019, month=9, day=1)
    )
    ax.set_xlabel("")
    ax.set_ylabel("€", fontsize=18)
    ax.set_title("Value of €1000 investment into AEX", fontsize=25)

    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator((1, 5, 11)))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("\n%Y"))
    ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%b"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    ax.tick_params(axis="both", which="both", labelsize=18)
    fig.tight_layout()
    fig.savefig(path_out, format="pdf")
    plt.close("all")


def make_events_plot(path_dict):
    def add_jagged_edge(ax, x, y_top, marker, col):
        top_jagged_offset = 0.02
        jagged_gap = 0.013
        for j in range(4):
            ax.scatter(
                x,
                y_top - top_jagged_offset - jagged_gap * j,
                marker=marker,
                color=col,
                s=250,
            )

    def events_plot(file_num, colors, path_dict):
        fig, ax = plt.subplots(figsize=(12, 7))

        max_val = 1175
        min_val = 875
        event_to_support = {
            "0": (1000, max_val),
            "1": (1100, max_val),
            "2": (min_val, 950),
            "3": (950, 1100),
            "1c": (min_val, 1100),
            "2c": (950, max_val),
            "3c": (min_val, 950, 1100, max_val),
        }

        event_labels = [
            "$E_0$",
            "$E_1$",
            "$E_2$",
            "$E_3$",
            "$E_1^C$",
            "$E_2^C$",
            "$E_3^C$",
        ]
        # alphas = [1, 1, 1, 1, 0.25, 0.25,  0.25]

        barwidth = 0.08  # yrange = 0 to 1
        gap = 0.025
        gap_0 = 0.02
        start = 0.85
        i = 0
        for _event, support in event_to_support.items():
            top = start - (gap + barwidth) * i - (i == 1) * gap_0
            bottom = top - barwidth
            if len(support) == 2:
                l, u = support
                ax.axvspan(
                    xmin=l,
                    xmax=u,
                    ymin=bottom,
                    ymax=top,
                    color=colors[i],
                    label=event_labels[i],
                    capstyle="projecting",
                )
                if i in [0, 1, 5]:
                    x_pos = u
                    marker_type = ">"
                    add_jagged_edge(ax, x_pos, top, marker_type, colors[i])
                elif i in [2, 4]:
                    x_pos = l
                    marker_type = "<"
                    add_jagged_edge(ax, x_pos, top, marker_type, colors[i])
            else:
                l, u, l1, u1 = support

                ax.axvspan(
                    xmin=l,
                    xmax=u,
                    ymin=bottom,
                    ymax=top,
                    color=colors[i],
                    label=event_labels[i],
                )
                ax.axvspan(
                    xmin=l1, xmax=u1, ymin=bottom, ymax=top, color=colors[i], label=""
                )
                add_jagged_edge(ax, l, top, "<", colors[i])
                add_jagged_edge(ax, u1, top, ">", colors[i])

            i += 1
        ax.set_xlim(850, 1200)
        ax.set_ylim(0, 1)
        ax.axvline(1000, color="black", ls="--")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=18)
        plt.yticks([])
        ax.set_xlabel("Value of investment into AEX in 6 months", fontsize=18)
        ax.tick_params(labelsize=16)
        fig.tight_layout()
        fig.savefig(path_dict["aex_events_" + str(file_num)])
        plt.close("all")

    def events_plot_paper(colors, path_out):
        fig, ax = plt.subplots(figsize=(10, 10))

        max_val = 1175
        min_val = 875
        event_to_support = {
            "0": (1000, max_val),
            "1": (1100, max_val),
            "2": (min_val, 950),
            "3": (950, 1100),
            "dummy": (0, 0),
            "1c": (min_val, 1100),
            "2c": (950, max_val),
            "3c": (min_val, 950, 1100, max_val),
        }
        event_labels = [
            r"$E^{AEX}_0: Y_{t+6} \in (1000, \infty)$",
            r"$E^{AEX}_1: Y_{t+6} \in (1100, \infty]$",
            r"$E^{AEX}_2: Y_{t+6} \in (-\infty, 950)$",
            r"$E^{AEX}_3: Y_{t+6} \in [950, 1100]$",
            " ",
            r"$E^{AEX}_{1, C}: Y_{t+6} \in (-\infty, 1100]$",
            r"$E^{AEX}_{2, C}: Y_{t+6} \in [950, \infty)$",
            r"$E^{AEX}_{3, C}: Y_{t+6} \in (-\infty, 950) \cup (1100, \infty)$",
        ]

        barwidth = 0.08  # yrange = 0 to 1
        gap = 0.025
        gap_0 = 0.02
        top = 0.955
        i = 0
        for event, support in event_to_support.items():
            if event == "dummy":
                bottom = top
            else:
                top -= gap + barwidth
                if i == 1:
                    top -= gap_0
                bottom = top - barwidth
            if len(support) == 2:
                l, u = support
                ax.axvspan(
                    xmin=l,
                    xmax=u,
                    ymin=bottom,
                    ymax=top,
                    color=colors[i],
                    label=event_labels[i],
                    capstyle="projecting",
                )
                if i in [0, 1, 6]:
                    x_pos = u
                    marker_type = ">"
                    add_jagged_edge(ax, x_pos, top, marker_type, colors[i])
                elif i in [2, 5]:
                    x_pos = l
                    marker_type = "<"
                    add_jagged_edge(ax, x_pos, top, marker_type, colors[i])
            else:
                l, u, l1, u1 = support

                ax.axvspan(
                    xmin=l,
                    xmax=u,
                    ymin=bottom,
                    ymax=top,
                    color=colors[i],
                    label=event_labels[i],
                )
                ax.axvspan(
                    xmin=l1, xmax=u1, ymin=bottom, ymax=top, color=colors[i], label=""
                )
                add_jagged_edge(ax, l, top, "<", colors[i])
                add_jagged_edge(ax, u1, top, ">", colors[i])

            i += 1
        ax.set_xlim(850, 1200)
        ax.set_ylim(0, 1)
        ax.axvline(1000, color="black", ls="--")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=18, ncol=2)
        plt.yticks([])
        ax.set_xlabel("Value of 1000 EUR investment into AEX in 6 months", fontsize=18)
        ax.tick_params(labelsize=16)
        fig.tight_layout()
        fig.savefig(path_out)
        plt.close("all")

    # Loop over colors to be able to put in events one after another
    greens = sns.color_palette("Paired")[2:4][::-1]
    reds = sns.color_palette("Paired")[4:6][::-1]
    grays = ["darkgray", "lightgray"]

    colors = ["lightblue"] + [
        greens[0],
        reds[0],
        grays[0],
        greens[1],
        reds[1],
        grays[1],
    ]
    colors_temp = ["white", "white", "white", "white", "white", "white", "white"]
    for number, event_added in enumerate([1, 4, 2, 5, 3, 6, 0]):
        colors_temp[event_added] = colors[event_added]
        events_plot(number, colors_temp, path_dict=path_dict)

    # Make one more graphic that shows one composite event and the corresponding single events.
    colors_comp = ["white", "white", reds[0], grays[0], greens[1], "white", "white"]
    events_plot(file_num="composite", colors=colors_comp, path_dict=path_dict)

    # Make one more graphic that shows one composite event and the corresponding single events.
    colors_set_m = [reds[0], reds[1], grays[1], grays[1], grays[1], grays[1], grays[1]]
    events_plot(file_num="set_monotonicity", colors=colors_set_m, path_dict=path_dict)

    colors = ["lightblue"] + [
        greens[0],
        reds[0],
        grays[0],
        "white",
        greens[1],
        reds[1],
        grays[1],
    ]

    events_plot_paper(colors, path_out=path_dict["aex_events_paper"])


def median_parameter_illustration(ambig_av, ll_insen, path_out):

    fig, ax = plt.subplots(figsize=(8, 8))
    p = np.linspace(0.12, 0.88, 2)
    p_dots = np.array([0.25, 0.5, 0.75])

    tau_1 = 1 - ll_insen
    tau_0 = 0.5 - ambig_av - tau_1 / 2

    m = tau_0 + tau_1 * p
    m_dots = tau_0 + tau_1 * p_dots

    # Start with vertical lines (go in background).
    for counter, p_dot in enumerate(p_dots):
        m_dot = m_dots[counter]
        shrink = np.abs(m_dot - p_dot) * 0.05
        if m_dot > p_dot:
            y_upper = m_dot - shrink
            y_lower = p_dot + shrink
        else:
            y_upper = p_dot - shrink
            y_lower = m_dot + shrink
        ax.plot([p_dot, p_dot], [y_lower, y_upper], color="gray", linestyle="dotted")

    ax.plot(p, m, color="gray")
    ax.scatter(p_dots, m_dots, color="black", marker="x", s=125, zorder=10)
    ax.plot(p, p, color="black", lw=1)

    ax.set_ylim(0.1, 0.9)
    ax.set_xlim(0.1, 0.9)

    ax.set_xlabel(r"$\Pr_\mathrm{subj}(E)$", fontsize=20)
    ax.set_ylabel("$W(E)$", fontsize=20)

    ax.annotate(
        f"$W\\left( E; \\alpha={ambig_av}, \\ell={ll_insen}\\right)$",
        xy=(0.625, tau_0 + tau_1 * 0.625),
        xycoords="data",
        xytext=(0.87, 0.4),
        textcoords="data",
        arrowprops={"color": "gray", "shrink": 0.05, "width": 1},
        horizontalalignment="right",
        verticalalignment="top",
        size=20,
        color="gray",
    )
    ax.set_xticks(p_dots)
    ax.set_yticks(np.linspace(0.2, 0.8, 7))
    ax.tick_params(labelsize=20)
    plt.tight_layout()
    fig.savefig(path_out)
    plt.close("all")


def neo_additive_illustration(ambig_av, ll_insen, path_out):

    fig, ax = plt.subplots(figsize=(8, 8))
    p = np.linspace(0.02, 0.98, 1000)

    ax.plot(p, p, color="black", lw=1)

    tau_1 = 1 - ll_insen
    tau_0 = 0.5 - ambig_av - tau_1 / 2

    m = tau_0 + tau_1 * p
    ax.plot(p, m, color="black", lw=6)

    p_intersect = -tau_0 / (tau_1 - 1)

    ax.fill_between(p, p, m, where=p < p_intersect, facecolor="green", alpha=0.2)
    ax.fill_between(p, p, m, where=p > p_intersect, facecolor="red", alpha=0.2)

    # Add vertical line.
    p_dot = 0.5
    m_dot = tau_0 + tau_1 * p_dot
    shrink = np.abs(m_dot - p_dot) * 0.05
    if m_dot > p_dot:
        y_upper = m_dot - shrink
        y_lower = p_dot + shrink
    else:
        y_upper = p_dot - shrink
        y_lower = m_dot + shrink
    ax.plot([p_dot, p_dot], [y_lower, y_upper], color="gray", linestyle="dotted")

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    ax.set_xticks(np.linspace(0.1, 0.9, 9))
    ax.set_yticks(np.linspace(0.1, 0.9, 9))

    ax.set_xlabel(r"$\Pr_\mathrm{subj}(E)$", fontsize=20)
    ax.set_ylabel("$W(E)$", fontsize=20)

    ax.tick_params(labelsize=20)

    plt.tight_layout()
    fig.savefig(path_out)
    plt.close("all")


DEPENDS_ON = IN_DATA / "aex" / "aex_yahoo.csv"
PRODUCES = {
    "neo_additive_illustration_empty": OUT_FIGURES
    / "neo_additive_illustration_empty.pdf",
    "neo_additive_illustration_aa_0_ll_0": OUT_FIGURES
    / "neo_additive_illustration_aa_0_ll_0.pdf",
    "neo_additive_illustration_aa_0_ll_40": OUT_FIGURES
    / "neo_additive_illustration_aa_0_ll_40.pdf",
    "neo_additive_illustration_aa_0_ll_80": OUT_FIGURES
    / "neo_additive_illustration_aa_0_ll_80.pdf",
    "neo_additive_illustration_aa_-30_ll_80": OUT_FIGURES
    / "neo_additive_illustration_aa_-30_ll_80.pdf",
    "neo_additive_illustration_aa_30_ll_80": OUT_FIGURES
    / "neo_additive_illustration_aa_30_ll_80.pdf",
    "valid_parameters_empty": OUT_FIGURES / "valid_parameters_empty.pdf",
    "valid_parameters_aa_0_ll_0": OUT_FIGURES / "valid_parameters_aa_0_ll_0.pdf",
    "valid_parameters_aa_0_ll_40": OUT_FIGURES / "valid_parameters_aa_0_ll_40.pdf",
    "valid_parameters_aa_0_ll_80": OUT_FIGURES / "valid_parameters_aa_0_ll_80.pdf",
    "valid_parameters_aa_-30_ll_80": OUT_FIGURES / "valid_parameters_aa_-30_ll_80.pdf",
    "valid_parameters_aa_30_ll_80": OUT_FIGURES / "valid_parameters_aa_30_ll_80.pdf",
    "aex_and_waves": OUT_FIGURES / "aex_and_waves.pdf",
    "aex_events_0": OUT_FIGURES / "aex_events_0.pdf",
    "aex_events_1": OUT_FIGURES / "aex_events_1.pdf",
    "aex_events_2": OUT_FIGURES / "aex_events_2.pdf",
    "aex_events_3": OUT_FIGURES / "aex_events_3.pdf",
    "aex_events_4": OUT_FIGURES / "aex_events_4.pdf",
    "aex_events_5": OUT_FIGURES / "aex_events_5.pdf",
    "aex_events_6": OUT_FIGURES / "aex_events_6.pdf",
    "aex_events_paper": OUT_FIGURES / "aex_events_paper.pdf",
    "aex_events_composite": OUT_FIGURES / "aex_events_composite.pdf",
    "aex_events_set_monotonicity": OUT_FIGURES / "aex_events_set_monotonicity.pdf",
    "median_parameter_illustration": OUT_FIGURES / "median_parameter_illustration.pdf",
    "neo_additive_illustration": OUT_FIGURES / "neo_additive_illustration.pdf",
}


@pytask.mark.depends_on(DEPENDS_ON)
@pytask.mark.produces(PRODUCES)
def task_basic_graphics(depends_on, produces):
    # Build neoadditive fun and triangular of valid points for different parameters

    parameters = [
        {"ambig_av": None, "ll_insen": None, "short_vertical": False},
        {"ambig_av": 0.0, "ll_insen": 0.0, "short_vertical": False},
        {"ambig_av": 0.0, "ll_insen": 0.4, "short_vertical": False},
        {"ambig_av": 0.0, "ll_insen": 0.8, "short_vertical": False},
        {"ambig_av": -0.3, "ll_insen": 0.8, "short_vertical": True},
        {"ambig_av": 0.3, "ll_insen": 0.8, "short_vertical": True},
    ]
    for parameter_pair in parameters:
        make_neoadditive_fun_plot(
            ambig_av=parameter_pair["ambig_av"],
            ll_insen=parameter_pair["ll_insen"],
            short_vertical=parameter_pair["short_vertical"],
            path_dict=produces,
        )
        make_valid_parameters_plot(
            ambig_av=parameter_pair["ambig_av"],
            ll_insen=parameter_pair["ll_insen"],
            path_dict=produces,
        )
    # make_neoadditive_fun_plot(tau_1=0.1, tau_2=0.5)
    # make_neoadditive_fun_plot(tau_1=0.4, tau_2=0.5)
    # make_neoadditive_fun_plot(tau_1=0.3, tau_2=0.1)
    # make_neoadditive_fun_plot(tau_1=0, tau_2=1)
    # make_neoadditive_fun_plot(tau_1=0.1, tau_2=0.7)

    make_val_of_investment_plot(depends_on, produces["aex_and_waves"])
    make_events_plot(produces)

    median_parameter_illustration(
        0.028,
        0.6,
        path_out=produces["median_parameter_illustration"],
    )

    neo_additive_illustration(
        0.1,
        0.6,
        path_out=produces["neo_additive_illustration"],
    )
