# berlin-bot – OpenStreetMap Berlin (branch `sherif-a1`)

Streamlit app that shows OpenStreetMap (OSM) points of interest in Berlin per postal code (PLZ) or
district (Bezirk) and answers questions about the current view with an LLM chat assistant.
This branch contains **Part A (Tasks 1–4)** of the problem set: complete, tested, ready to deploy.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run main.py        # opens http://localhost:8501
```

The OSM data is **already in `datasets/`** (downloaded 2026-11-24), so the app never calls the Overpass API.
To download it again: `python download_osm.py` (all sources) or `python download_osm.py Pharmacies` (one source).

**Chat assistant:** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and enter your key
(e.g. `OPENAI_API_KEY = "sk-..."`), or type the key into the sidebar. `secrets.toml` is in `.gitignore` –
never commit a key. On Streamlit Community Cloud put the key under *App settings → Secrets*.

**Tests:** `pip install pytest` then `python -m pytest` (17 tests, ~1 s).

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: reads the residents per PLZ and starts the app |
| `config.py` | Data sources (OSM tags), menu options, thresholds, LLM providers |
| `download_osm.py` | Task 1 – downloads one csv per source (osmnx), one point per object |
| `core/methods.py` | Tasks 2–4 – loading, spatial join, indicators, data quality, maps, Streamlit app |
| `core/chat.py` | Chat assistant (provided): context of the current view + LLM call |
| `tests/` | Automated tests of the deterministic code |
| `datasets/` | PLZ / Bezirk polygons, residents per PLZ, downloaded OSM csv files |

## Design decisions

- **One point per object:** OSM ways/relations are polygons (84 % of the playgrounds, 36 % of the
  supermarkets). Their centroid is computed in EPSG:25833 (metres) and converted back to EPSG:4326.
- **Spatial join instead of `addr:postcode`:** points are assigned to PLZ and Bezirk with
  `geopandas.sjoin(predicate="within")`. `addr:postcode` is missing for 17–99 % of the objects and differs
  from the real location for ~15 % of those that have it. Points on a border keep their first match.
- **Empty areas count as 0:** aggregation starts from the polygons (left join), so areas without any
  object are shown – these are the possible gaps.
- **Areas in km² from metres**, never from degrees (total: 892 km², official: 891.8 km²).
- **Few residents:** "per 10,000 residents" is not computed below 500 residents (`min_residents`).
  This affects only PLZ 14053 (Olympiastadion, 139 residents), which would otherwise get extreme values.
- **Residents per Bezirk are approximated:** there is no residents file per district, so every PLZ is
  counted completely in the district of its centre (`sjoin_nearest`). The total stays the same
  (3,291,919), but PLZ areas crossing a district border are not split.

## Data quality (automatic report in the app)

| Source | Objects | Without name | Without subcat | Near-dup. pairs (< 10 m) | Outside Berlin | Postcode = join | Empty PLZ |
|---|---:|---:|---|---:|---:|---:|---:|
| Restaurants | 4,753 | 0.5 % | 20.3 % (cuisine) | 5 | 2 | 86.2 % | 1 |
| Cafes | 2,620 | 1.2 % | 63.7 % (cuisine) | 4 | 0 | 85.3 % | 5 |
| Fast food | 2,870 | 2.0 % | 22.1 % (cuisine) | 4 | 0 | 82.2 % | 2 |
| Pharmacies | 673 | 0.4 % | 97.9 % (brand) | 1 | 0 | 84.0 % | 7 |
| Supermarkets | 1,380 | 0.3 % | 17.8 % (brand) | 1 | 0 | 84.9 % | 2 |
| Playgrounds | 5,560 | 92.6 % | 20.7 % (access) | 0 | 8 | 93.2 % | 0 |
| Charging stations | 1,802 | 85.3 % | 18.8 % (operator) | 6 | 0 | 91.7 % | 11 |

No implausible coordinates in any source.

## Limits

- OSM is crowd-sourced: central districts are mapped more densely, so a low value is no proof of a gap.
- The residents file (3.29 million) is older than the current population of Berlin, so values
  "per 10,000 residents" are approximate.
- The chat assistant only sees the aggregated table and the quality report. Its numbers were correct in
  our tests, but its interpretations were sometimes wrong (e.g. it concluded that missing `brand` tags
  mean pharmacies are under-counted). Always check important answers against the data table.

Data: © OpenStreetMap contributors (ODbL).
