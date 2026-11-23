# -------------------------------
# Installation (needs uv, https://docs.astral.sh/uv/)
uv sync

# -------------------------------
# Step 1: download the OSM data (Task 1), once per data source
uv run download-osm
uv run download-osm Restaurants        # only one source

# -------------------------------
# Step 2: start the Streamlit app
uv run streamlit run src/berlin_bot/app.py

# -------------------------------
# Files (in src/berlin_bot/)
config.py        data sources and options of the pull-down menus
download_osm.py  OSM download (Overpass via osmnx)
app.py           entry point of the app
paths.py         project root, so that "datasets/" resolves from anywhere
core/methods.py  loading, spatial join, indicators, data quality, map, app
datasets/        (project root) PLZ and Bezirk polygons, residents, downloaded OSM csv files
