"""
CineMatch website builder  —  Python + Orange Data Mining
==========================================================

Run this file (or double-click UPDATE_WEBSITE.bat) whenever the team changes:
    data/movies.csv            the movie dataset            (Data Expert)
    data/model_settings.toml   the AI weights               (Data Expert)
    content/site.toml          all text on the website      (Information Researcher)
    template/page.html         the design                   (Designer)

It does three things:
  1. FEATURES   – turns every movie into a row of numbers (genre, mood,
                  language, director, cast, rating), multiplied by the weights.
  2. ORANGE AI  – loads those rows into an Orange `Table`, uses
                  `Orange.distance.Cosine` to measure how alike every pair of
                  movies is, and `Orange.clustering.hierarchical` to group the
                  movies into "families".
  3. WEBSITE    – writes everything into website/index.html, a single file
                  that opens in any browser by double-clicking. No server needed.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MOVIES_CSV = ROOT / "data" / "movies.csv"
SETTINGS = ROOT / "data" / "model_settings.toml"
CONTENT = ROOT / "content" / "site.toml"
TEMPLATE = ROOT / "template" / "page.html"
OUT_DIR = ROOT / "website"

try:
    import numpy as np
    from Orange.data import ContinuousVariable, Domain, StringVariable, Table
    from Orange.distance import Cosine

    ORANGE = True
except ImportError:
    ORANGE = False

# feature group -> field in each movie record
GROUPS = {"genres": "genres", "mood": "mood", "language": "language",
          "director": "director", "cast": "cast"}


# ---------------------------------------------------------------------------
# 1. Read the data
# ---------------------------------------------------------------------------
def split(cell: str) -> list[str]:
    return [v.strip() for v in (cell or "").split("|") if v.strip()]


def to_number(text: str, default=0.0) -> float:
    try:
        return float(str(text).strip())
    except ValueError:
        return default


def load_movies() -> list[dict]:
    # utf-8-sig also accepts files saved by Excel as "CSV UTF-8"
    with open(MOVIES_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("movies.csv is empty.")
    needed = {"title", "year", "genres", "language", "rating"}
    columns = {c.strip().lower() for c in rows[0].keys() if c}
    if not needed <= columns:
        sys.exit(f"movies.csv is missing column(s): {', '.join(sorted(needed - columns))}")

    movies, seen = [], set()
    for line_no, raw in enumerate(rows, start=2):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        title = r.get("title", "")
        if not title:
            continue
        if title.lower() in seen:
            print(f"  ! Skipped duplicate title on line {line_no}: {title}")
            continue
        seen.add(title.lower())
        movies.append({
            "title": title,
            "year": int(to_number(r.get("year"), 0)),
            "genres": split(r.get("genres")),
            "director": split(r.get("director")),
            "cast": split(r.get("lead_cast")),
            "language": r.get("language", ""),
            "rating": round(min(max(to_number(r.get("rating")), 0), 10), 1),
            "runtime": int(to_number(r.get("runtime_min"), 0)),
            "mood": split(r.get("mood")),
            "overview": r.get("overview", ""),
            "poster": r.get("poster_url", "") if r.get("poster_url", "").startswith("http") else "",
        })
    return movies


# ---------------------------------------------------------------------------
# 2. Features
# ---------------------------------------------------------------------------
def build_features(movies: list[dict], weights: dict, ignore=()) -> tuple[list[str], list[list[float]]]:
    names, columns = [], []
    for group, key in GROUPS.items():
        w = 0.0 if group in ignore else float(weights.get(group, 1.0))
        values = sorted({v for m in movies for v in
                         ([m["language"]] if key == "language" else m[key]) if v})
        for value in values:
            names.append(f"{group}={value}")
            col = []
            for m in movies:
                has = value == m["language"] if key == "language" else value in m[key]
                col.append(w if has else 0.0)
            columns.append(col)
    names.append("rating_scaled")
    columns.append([m["rating"] / 10 * float(weights.get("rating", 0.5)) for m in movies])
    rows = [list(r) for r in zip(*columns)]
    return names, rows


# ---------------------------------------------------------------------------
# 3. Orange Data Mining: similarity + clustering
# ---------------------------------------------------------------------------
def orange_table(names, rows, titles):
    domain = Domain([ContinuousVariable(n) for n in names], metas=[StringVariable("title")])
    metas = np.array(titles, dtype=object).reshape(-1, 1)
    return Table.from_numpy(domain, np.array(rows, dtype=float), metas=metas)


def cosine_distance_matrix(names, rows, titles) -> list[list[float]]:
    if ORANGE:
        table = orange_table(names, rows, titles)
        dist = np.nan_to_num(np.asarray(Cosine(table), dtype=float), nan=1.0)
        return dist.tolist()
    # Same maths in plain Python, used only if Orange isn't installed
    norms = [math.sqrt(sum(x * x for x in r)) for r in rows]
    out = []
    for i, a in enumerate(rows):
        line = []
        for j, b in enumerate(rows):
            dot = sum(x * y for x, y in zip(a, b))
            line.append(1.0 - dot / (norms[i] * norms[j]) if norms[i] and norms[j] else 1.0)
        out.append(line)
    return out


def cluster(dist: list[list[float]], k: int) -> list[list[int]]:
    """Group movies into k families with average-linkage hierarchical clustering."""
    k = max(1, min(k, len(dist)))
    if ORANGE:
        try:
            from Orange.clustering.hierarchical import dist_matrix_clustering, leaves, top_clusters
            from Orange.misc import DistMatrix

            tree = dist_matrix_clustering(DistMatrix(np.array(dist)), linkage="average")
            return [[leaf.value.index for leaf in leaves(c)] for c in top_clusters(tree, k)]
        except Exception as e:  # keep the build working even if Orange's API changes
            print(f"  ! Orange clustering unavailable ({e}); using built-in clustering.")
    groups = [[i] for i in range(len(dist))]
    while len(groups) > k:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                d = sum(dist[i][j] for i in groups[a] for j in groups[b]) / (len(groups[a]) * len(groups[b]))
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        groups[a] += groups.pop(b)
    return groups


def name_family(movies: list[dict], members: list[int]) -> str:
    genres = Counter(g for i in members for g in movies[i]["genres"]).most_common(2)
    mood = Counter(m for i in members for m in movies[i]["mood"]).most_common(1)
    label = " & ".join(g for g, _ in genres) or "Mixed"
    return f"{mood[0][0]} {label}" if mood else label


# ---------------------------------------------------------------------------
# 4. Write the website
# ---------------------------------------------------------------------------
SKELETON = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
</head>
<body>
{page}
</body>
</html>
"""


def main() -> None:
    print("CineMatch builder")
    print("  Engine:", "Orange Data Mining" if ORANGE else "plain Python (Orange not installed)")

    movies = load_movies()
    settings = tomllib.loads(SETTINGS.read_text(encoding="utf-8"))
    site = tomllib.loads(CONTENT.read_text(encoding="utf-8"))
    weights = settings.get("weights", {})
    titles = [m["title"] for m in movies]
    print(f"  Loaded {len(movies)} movies")

    names, rows = build_features(movies, weights)
    dist = cosine_distance_matrix(names, rows, titles)
    _, rows_nl = build_features(movies, weights, ignore=("language",))
    dist_nl = cosine_distance_matrix(names, rows_nl, titles)
    print(f"  Compared {len(movies) * (len(movies) - 1) // 2} movie pairs using {len(names)} features")

    k = int(settings.get("clustering", {}).get("families", 6))
    groups = cluster(dist, k)
    families = []
    for fid, members in enumerate(sorted(groups, key=len, reverse=True)):
        for i in members:
            movies[i]["family"] = fid
        families.append({"id": fid, "name": name_family(movies, members), "size": len(members)})
    print(f"  Grouped movies into {len(families)} families")

    sim = lambda d: [[round(max(0.0, 1.0 - x), 3) for x in line] for line in d]
    payload = {
        "site": site,
        "settings": settings,
        "engine": "Orange Data Mining" if ORANGE else "Plain Python (install Orange for the full version)",
        "builtAt": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "featureCount": len(names),
        "movies": movies,
        "families": families,
        "sim": sim(dist),
        "simNoLang": sim(dist_nl),
    }
    data_js = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    template = TEMPLATE.read_text(encoding="utf-8")
    page = template.replace("/*__CINEMATCH_DATA__*/null", data_js)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "index.html").write_text(SKELETON.format(page=page), encoding="utf-8")

    # A copy of the features for the Orange desktop app (for report screenshots)
    with open(OUT_DIR / "features_for_orange.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["title", "family"] + names)
        for m, r in zip(movies, rows):
            w.writerow([m["title"], families[m["family"]]["name"]] + [round(x, 4) for x in r])

    print(f"  Done! Open {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
