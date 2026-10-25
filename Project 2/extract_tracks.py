"""Digitize the two vector racetracks in Figure 5.5 (book PDF page 134).

This is a one-time asset generator. The training code uses the saved JSON and does
not need the book or pdfplumber at run time.
"""

import json
from pathlib import Path

import pdfplumber


ROOT = Path(__file__).resolve().parent
PDF = ROOT.parent / "RLbook2020.pdf"


def extract(page, name, low_x, high_x):
    rects = [
        r for r in page.rects
        if low_x < r["x0"] < high_x
        and 4.94 < r["width"] < 4.96
        and 4.94 < r["height"] < 4.96
        and 69 < r["top"] < 230
    ]
    stroke_rects = [r for r in rects if r["stroke"]]
    x0 = min(r["x0"] for r in stroke_rects)
    y0 = min(r["top"] for r in stroke_rects)
    spacing = 4.94523

    def grid_xy(rect):
        return (round((rect["x0"] - x0) / spacing),
                round((rect["top"] - y0) / spacing))

    cells = {grid_xy(r) for r in stroke_rects}
    width = max(x for x, _ in cells) + 1
    height = max(y for _, y in cells) + 1
    rows = ["".join("#" if (x, y) in cells else " " for x in range(width))
            for y in range(height)]

    def colored(value):
        return [list(cell) for cell in sorted({
            (grid_xy(r)[0], height - 1 - grid_xy(r)[1])
            for r in rects if r["fill"] and r["non_stroking_color"] == value
        })]

    return {"name": name, "rows_top_down": rows, "start": colored(0.558),
            "finish": colored(0.588)}


def main():
    with pdfplumber.open(PDF) as pdf:
        page = pdf.pages[133]
        tracks = {
            "left": extract(page, "left", 70, 160),
            "right": extract(page, "right", 200, 370),
        }
    (ROOT / "tracks.json").write_text(json.dumps(tracks, indent=2) + "\n", encoding="utf-8")
    for name, track in tracks.items():
        print(name, len(track["rows_top_down"][0]), "x", len(track["rows_top_down"]),
              "cells", sum(row.count("#") for row in track["rows_top_down"]),
              "starts", len(track["start"]), "finish", len(track["finish"]))


if __name__ == "__main__":
    main()
