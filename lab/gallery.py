"""The figures shown in the README, flattened on white. The originals are black ink on a transparent background, which
would disappear on a dark page.

usage: python -m lab.gallery
"""
from PIL import Image

from . import FIGURES, ROOT

SHOWN = ("chapter1/memorization-steps.png",)


def main():
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    for name in SHOWN:
        image = Image.open(FIGURES / name).convert("RGBA")
        white = Image.new("RGBA", image.size, "white")
        Image.alpha_composite(white, image).convert("RGB").save(out / name.split("/")[1], optimize=True)
        print("written", out / name.split("/")[1])


if __name__ == "__main__":
    main()
