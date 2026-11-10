"""Helpers for the SVG schematics.

The SVGs are written for inlining in the course pages: shapes carry sch-* classes that the page styles, and tex() writes
<text class="sch-tex">TeX</text>, which the build of the page turns into KaTeX. Opened alone, they show raw TeX labels.
"""
from xml.sax.saxutils import escape

SIZES = ("", "small", "tiny")  # 22, 18 and 12 px, as .sch-tex in post.css


def tex(x, y, source, size="", anchor="start"):
    """A math label placed like <text x y text-anchor>: tex(10, 20, r"Z \\sim \\gamma", "small", "middle")."""
    assert size in SIZES and anchor in ("start", "middle", "end")
    cls = "sch-tex" + (f" sch-tex-{size}" if size else "")
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{escape(source)}</text>'
