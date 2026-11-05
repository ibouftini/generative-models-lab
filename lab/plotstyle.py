"""Shared matplotlib style of the figures (PNG): words in Open Sans, every symbol and number in LaTeX.

Math is written as LaTeX between dollar signs and typeset by matplotlib's mathtext in Computer Modern, the face of the
KaTeX fonts that typeset the text of the course. save() also sets the tick numbers of linear axes in LaTeX, so negative
ticks get a true minus sign.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

plt.rcParams.update({
    "font.family": "Open Sans",
    "font.size": 10,
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
})


def tex_number(v, _pos=None):
    """A tick number in LaTeX: 3200 -> $3{,}200$, -0.5 -> $-0.5$ (true minus sign)."""
    if abs(v - round(v)) < 1e-9:
        text = f"{int(round(v)):,}".replace(",", "{,}")
    else:
        text = f"{v:.6g}"
    return f"${text}$"


def save(fig, path, **kwargs):
    """Typeset the default tick labels of every linear axis in LaTeX, then fig.savefig(path, **kwargs)."""
    for ax in fig.axes:
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == "linear" and type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(FuncFormatter(tex_number))
    fig.savefig(path, **kwargs)
