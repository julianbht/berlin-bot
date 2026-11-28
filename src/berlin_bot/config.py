p = dict()

# -----------------------------------------------------------------------------
# Static datasets (folder "datasets/")
p["geocode"] = "PLZ"
p["file_plz"] = "geodata_berlin_plz.csv"
p["file_dis"] = "geodata_berlin_dis.csv"
p["file_residents"] = "plz_einwohner.csv"
p["file_info"] = "osm_download_info.json"  # written by download_osm.py

# -----------------------------------------------------------------------------
# OSM data sources. The pull-down menu "Data source" is generated from this dict.
#   tags   : OSM tags used for the download (key: value or key: list of values)
#   file   : csv file in "datasets/" written by download_osm.py
#   subcat : OSM tag used for the dependent pull-down menu "Category filter"
#   history: optional; snapshots of the objects on 1 January of each year
#            file       : csv file in "datasets/", one row per object and year
#            start_year : first year; Overpass keeps history only since September 2012
p["sources"] = {
    "Restaurants": {
        "tags": {"amenity": "restaurant"},
        "file": "osm_restaurants.csv",
        "subcat": "cuisine",
    },
    "Cafes": {"tags": {"amenity": "cafe"}, "file": "osm_cafes.csv", "subcat": "cuisine"},
    "Fast food": {
        "tags": {"amenity": "fast_food"},
        "file": "osm_fast_food.csv",
        "subcat": "cuisine",
    },
    "Pharmacies": {
        "tags": {"amenity": "pharmacy"},
        "file": "osm_pharmacies.csv",
        "subcat": "brand",
    },
    "Supermarkets": {
        "tags": {"shop": "supermarket"},
        "file": "osm_supermarkets.csv",
        "subcat": "brand",
    },
    "Playgrounds": {
        "tags": {"leisure": "playground"},
        "file": "osm_playgrounds.csv",
        "subcat": "access",
    },
    "Charging stations": {
        "tags": {"amenity": "charging_station"},
        "file": "osm_charging.csv",
        "subcat": "operator",
    },
    # Hardly any club has a genre tag; wheelchair is the most often filled tag that splits clubs
    "Nightclubs": {
        "tags": {"amenity": "nightclub"},
        "file": "osm_nightclubs.csv",
        "subcat": "wheelchair",
        "history": {"file": "osm_nightclubs_history.csv", "start_year": 2013},
    },
}

# -----------------------------------------------------------------------------
# Downloads: a failed request is repeated; the wait doubles after each try
p["download_attempts"] = 5
p["download_first_wait_s"] = 10

# -----------------------------------------------------------------------------
# Options of the other pull-down menus
p["areas"] = {"Postal code (PLZ)": "PLZ", "District (Bezirk)": "Bezirk"}
p["indicators"] = ["Number", "Number per km2", "Number per 10,000 residents"]
p["maptypes"] = ["Choropleth", "Heatmap", "Point map (clusters)"]
p["colorscales"] = {
    "Neon (purple-pink-yellow)": ["#3a0ca3", "#f72585", "#ffd60a"],
    "Yellow-Red": ["yellow", "red"],
    "White-Blue": ["white", "darkblue"],
    "Green-Purple": ["lightgreen", "purple"],
}

# -----------------------------------------------------------------------------
# Part B: use case "How has Berlin's club scene changed, and where did clubs close?"
# The app opens in a focused nightclub view. All other sources and menus stay available
# behind the switch "Explore all data sources".
p["focus_source"] = "Nightclubs"
p["focus_title"] = "Berlin Club Radar"
p["focus_question"] = (
    "How has Berlin's club scene changed since {year}? Where did clubs close, "
    "and where did new ones open?"
)
p["default_area"] = "District (Bezirk)"  # 12 districts read easier than 190 postal codes

# Plain-language labels for the menus (the values on the left are used in the code)
p["area_labels"] = {"Postal code (PLZ)": "Postal codes", "District (Bezirk)": "Districts"}
p["indicator_labels"] = {
    "Number": "Number of places",
    "Number per km2": "Places per km²",
    "Number per 10,000 residents": "Places per 10,000 residents",
}
p["maptype_labels"] = {
    "Choropleth": "Coloured areas",
    "Heatmap": "Heatmap",
    "Point map (clusters)": "Single places",
}

# Colours: gone / new / still there, and the diverging scale for the net change per area
p["change_colors"] = {"gone": "#ff3b6b", "new": "#2ee6a8", "still there": "#8a8fb5"}
p["net_change_colors"] = ["#ff3b6b", "#2a2a40", "#2ee6a8"]  # loss - no change - gain
p["map_tiles"] = "OpenStreetMap"
p["dark_map"] = True  # OSM tiles are darkened with a CSS filter to fit the dark theme

# -----------------------------------------------------------------------------
# Parameters
p["min_residents"] = 500  # PLZ areas with fewer residents: no per-resident value
p["top_n_category"] = 10  # number of entries in the category filter
p["dup_distance_m"] = 10  # near-duplicate threshold (quality check)
p["crs_geo"] = "EPSG:4326"
p["crs_metric"] = "EPSG:25833"  # ETRS89 / UTM 33N, metres (for area, distance)
p["berlin_lat"] = (52.3, 52.7)  # plausibility box for coordinates
p["berlin_lon"] = (13.0, 13.8)

# -----------------------------------------------------------------------------
# Chat assistant (LLM). Every provider with an OpenAI-compatible API works.
#   secret : name of the API key in .streamlit/secrets.toml or in an environment variable
# Model names change over time: they can also be edited in the sidebar of the app.
p["llm_providers"] = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-haiku-5.5",
        "secret": "OPENROUTER_API_KEY",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "secret": "OPENAI_API_KEY",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "secret": "DEEPSEEK_API_KEY",
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
        "secret": "MISTRAL_API_KEY",
    },
    "Anthropic (Claude)": {
        "base_url": "https://api.anthropic.com/v1/",
        "model": "claude-haiku-4-5-20251001",
        "secret": "ANTHROPIC_API_KEY",
    },
    "Custom (OpenAI-compatible, e.g. local Ollama)": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1",
        "secret": "LLM_API_KEY",
        "editable_url": True,
    },
}
p["llm_default"] = "OpenRouter"
p["llm_max_history"] = 10  # last messages sent to the LLM
p["llm_max_tokens"] = 800  # maximum length of an answer

# -----------------------------------------------------------------------------
pdict = p.copy()
