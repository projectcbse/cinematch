# 🎬 CineMatch — AI Movie Recommendation Website

CBSE Class XII · Artificial Intelligence (843) · Practical Project
Built with **Python** + **Orange Data Mining**. No server, no Streamlit.

---

## Open the website

Double-click **`OPEN_WEBSITE.html`** (or `website/index.html`).
It opens in Chrome, Edge or any browser, even without Python and without internet.
You can copy the `website` folder to a pen drive and open it on the school computer.

---

## Option: put it on GitHub (live link that updates itself)

1. Create a free account at github.com → **New repository** → name it `cinematch` → **Public** → Create.
2. Click **uploading an existing file**, drag in everything inside the `cinematch` folder, then **Commit changes**.
3. Make sure `.github/workflows/deploy.yml` was uploaded. If it's missing (Mac hides folders starting with a dot),
   click **Add file → Create new file**, type `.github/workflows/deploy.yml` as the name, paste the file's contents, and commit.
4. Go to **Settings → Pages** and set **Source** to **GitHub Actions**.
5. Open the **Actions** tab and wait for the green tick (about 2 minutes).
   Your site is now live at `https://YOUR-USERNAME.github.io/cinematch/`.
6. **Settings → Collaborators → Add people** to invite your group.

After that, anyone can open a file on GitHub (e.g. `data/movies.csv`), click the ✏️ pencil, edit, and
**Commit changes**. GitHub re-runs the Orange AI and the live site updates for everyone in ~2 minutes.

---

## Update the website after changing data or text

1. Save your changes to the CSV / TOML files (see below).
2. Double-click **`UPDATE_WEBSITE.bat`**.
   - First time only: it installs Orange Data Mining (needs internet, a few minutes).
   - It runs the Orange AI, rebuilds `website/index.html` and opens it.

Needs Python 3.11 or 3.12. If Python isn't on PATH, the `.bat` file still finds it.
Manual way: open a terminal in this folder and run `py build_website.py`.

---

## Who edits what

```
cinematch/
├── OPEN_WEBSITE.html          ← double-click to view
├── UPDATE_WEBSITE.bat         ← double-click after any change
├── data/
│   ├── movies.csv             ← DATA EXPERT: the dataset
│   └── model_settings.toml    ← DATA EXPERT: AI weights, number of families
├── content/
│   └── site.toml              ← INFORMATION RESEARCHER: all website text
├── template/
│   └── page.html              ← DESIGNER: layout, colours, fonts (CSS at the top)
├── build_website.py           ← PROTOTYPE BUILDER: Python + Orange AI
└── website/                   ← generated, don't edit by hand
    ├── index.html             ← the finished website
    └── features_for_orange.csv← feature table for the Orange desktop app
```

### Data Expert 📊
Add a row to `data/movies.csv` (Excel → save as **CSV UTF-8**):

| Column | Example | Notes |
|---|---|---|
| title | `Dangal` | must be unique |
| year | `2016` | |
| genres | `Biography\|Drama\|Sport` | separate several with `\|` |
| director | `Nitesh Tiwari` | |
| lead_cast | `Aamir Khan\|Sanya Malhotra` | 2–3 main actors |
| language | `Hindi` | one language |
| rating | `8.3` | 0–10 |
| runtime_min | `161` | |
| mood | `Inspiring\|Emotional` | reuse existing mood words |
| overview | one line | write it in your own words |
| poster_url | *(optional)* | direct image link; blank = coloured card |

Spell genres and moods exactly the same every time (`Sci-Fi`, not `SciFi`).
Please double-check ratings and years; they are approximate.

Experiment for the report: change a weight in `model_settings.toml` (e.g. `mood = 4.0`),
run `UPDATE_WEBSITE.bat`, and screenshot how the recommendations change.

### Information Researcher 📝
Everything on the **About** page, the site name, tagline, school, team names, SDGs and FAQ
lives in `content/site.toml`. Edit the text inside the quotes, save, run `UPDATE_WEBSITE.bat`.

### Tester 🧪
On **Discover**, testers click 👍 *Good pick* / 👎 *Not for me*. Results appear under
**How the AI works → User testing**, with a satisfaction % and a *Copy votes as CSV* button
to paste into Excel. Votes are saved in that browser only, so test on one computer.

---

## How the AI works (for the report & viva)

`build_website.py` uses Orange in two ways:

```python
from Orange.data import Table, Domain, ContinuousVariable
from Orange.distance import Cosine
from Orange.clustering.hierarchical import dist_matrix_clustering, top_clusters

table = Table.from_numpy(domain, features)      # each movie = one row of numbers
distances = Cosine(table)                       # 1. similarity of every pair of movies
tree = dist_matrix_clustering(distances, linkage="average")
families = top_clusters(tree, 6)                # 2. groups movies into "families"
```

1. **Content-based recommendation:** similarity = 1 − cosine distance. 100% means identical features.
2. **Hierarchical clustering:** an unsupervised learning method that groups similar films (no labels needed).

**Orange desktop app screenshots:** open `website/features_for_orange.csv` in Orange Canvas
(set `title` and `family` as *meta*), then connect
`File → Distances (Cosine) → Distance Map` and `Distances → Hierarchical Clustering`.
The `Neighbors` widget gives the same top picks as the website.
