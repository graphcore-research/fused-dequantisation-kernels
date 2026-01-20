# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Adapted from https://github.com/graphcore-research/optimal-weight-formats/blob/main/paper/plot_utils.py

"""Generate paper-ready plots"""

import inspect
import logging
import math
import re
import subprocess
import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PALETTE = sns.color_palette("Dark2")
SEQ_PALETTE = sns.color_palette("flare", as_cmap=True)
DISPLAY_NAMES = {
    "k": "$k$",
    "tflops": "TFLOP/s",
    "gb_s": "GB/s",
    "tokens_s": "Tokens/s",
    "impl": "Implementation",
    "element_bits": "Element Bits",
    "parameters": "Parameter Count",
}


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


def fmt_latex_booktabs(
    df: pd.DataFrame,
    cols: dict[str, str],
    align: str | None = None,
    nan_str: str = "$-$",
) -> str:
    """Format as a booktabs table."""

    def fmt_value(v: Any) -> str:
        if isinstance(v, int):
            return f"${v}$"
        elif isinstance(v, float):
            if math.isnan(v):
                return nan_str
            return f"${v:.3g}$"
        else:
            return str(v)

    if align is not None:
        assert len(align) == len(cols), "align length must match number of columns"

    col_titles = [fmt_value(value or display_name(key)) for key, value in cols.items()]

    s = r"\begin{tabular}" + "{" + (align or "l" * len(cols)) + "}" + r" \toprule"
    s += "\n  " + " & ".join(col_titles) + r" \\\midrule"
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
        "add code/ figures/ tables/",
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


def save_table(name: str, df: pd.DataFrame, push: bool = True, **args: Any) -> str:
    if _check_overleaf_cloned():
        (OVERLEAF / "tables" / f"{name}.tex").write_text(fmt_latex_booktabs(df, **args))
        if push:
            push_to_paper()


def save_code(fn: Callable[..., Any], push: bool = True) -> None:
    body = inspect.getsource(fn).splitlines()[1:]
    body = [re.sub(r"^    ", "", x) for x in body]
    body = [x for x in body if "# IGNORE" not in x]
    code = "\n".join(body) + "\n"

    if _check_overleaf_cloned():
        (OVERLEAF / "code" / f"{fn.__name__}.py").write_text(code)
        if push:
            push_to_paper()
