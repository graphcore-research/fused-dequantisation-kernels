# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Adapted from https://github.com/graphcore-research/optimal-weight-formats/blob/main/paper/plot_utils.py

"""Generate paper-ready plots"""

import logging
import subprocess
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = sns.color_palette("Dark2")
SEQ_PALETTE = sns.color_palette("flare", as_cmap=True)
DISPLAY_NAMES = {}


def display_name(s: str) -> str:
    return DISPLAY_NAMES.get(s, s)


def configure(disable_tex_for_debug_speed: bool = False) -> None:
    """Place at the start of the notebook, to set up defaults."""
    print(
        "Recommend (Ubuntu):\n"
        "  sudo apt-get install cm-super dvipng fonts-cmu texlive-latex-extra"
    )
    logging.getLogger("matplotlib.texmanager").setLevel(logging.WARNING)
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    sns.set_palette(PALETTE)
    font_name = "CMU Serif"
    matplotlib.rcParams.update(
        {
            # Fonts
            "font.family": "serif",
            "font.serif": [font_name],
            "text.usetex": not disable_tex_for_debug_speed,
            # General
            "figure.figsize": (8, 3),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.edgecolor": "none",
        }
    )
    try:
        matplotlib.font_manager.findfont(
            font_name, rebuild_if_missing=True, fallback_to_default=False
        )
    except ValueError as e:
        print(
            f"Couldn't find font {font_name!r}.\nOn Ubuntu:\n"
            "  sudo apt install fonts-cmu\n"
            "  rm ~/.cache/matplotlib/fontlist-*.json\n"
            "  (restart kernel)\n"
            f"  (original error: {e!r})"
        )


def tidy(figure: matplotlib.figure.Figure) -> None:
    figure.tight_layout()

    for ax in figure.axes:
        for label in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            label.set_text(display_name(label.get_text()))

    for legend in filter(None, [ax.legend_ for ax in figure.axes] + figure.legends):
        title = legend.get_title()
        title.set_text(display_name(title.get_text()))
        for text in legend.get_texts():
            text.set_text(display_name(text.get_text()))


def fmt_latex_booktabs(df: pd.DataFrame, cols: dict[str, str]) -> str:
    """Format as a booktabs table."""

    def fmt_value(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3g}"
        else:
            return str(v)

    s = r"\begin{tabular}" + "{" + "l" * len(cols) + "}" + r" \toprule"
    s += "\n  " + " & ".join(map(fmt_value, cols.values())) + r" \\\midrule"
    for _, row in df.iterrows():
        s += "\n  " + " & ".join(fmt_value(row[col]) for col in cols) + r" \\"
    s += "\n" + r"\bottomrule"
    s += "\n" + r"\end{tabular}"
    return s


# Paper sync

OVERLEAF = Path(__file__).parent / "paper"


def _check_overleaf_cloned() -> bool:
    if not OVERLEAF.exists():
        warnings.warn(f"Repository not found at {OVERLEAF}, disabling save-and-push")
        return False
    return True


def push_to_paper() -> None:
    for git_cmd in [
        "add figures/",
        "commit -m 'Update figures' --quiet",
        "pull --rebase --quiet",
        "push --quiet",
    ]:
        cmd = f"git -C {OVERLEAF} {git_cmd}"
        # print(f"$ {cmd}", file=sys.stderr)
        if subprocess.call(cmd, shell=True):
            print(f"Error running {cmd!r} -- aborting")
            return


def save(name: str, push: bool = True) -> None:
    """Save and push a figure to the paper."""
    if _check_overleaf_cloned():
        plt.savefig(OVERLEAF / "figures" / f"{name}.pdf", bbox_inches="tight")
        if push:
            push_to_paper()
