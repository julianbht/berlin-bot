"""
Downloads one csv file per OSM data source defined in config.py (Task 1).

Usage:
    uv run download-osm                    # all sources
    uv run download-osm Restaurants        # only one source (menu label)

The app never calls the Overpass API itself: run this script once, commit the csv files.
"""

import datetime
import json
import os
import sys

import osmnx as ox
import pandas as pd

from berlin_bot.config import pdict
from berlin_bot.paths import PROJECT_ROOT

# Columns that are kept (if they exist for the downloaded objects)
KEEP = [
    "name",
    "cuisine",
    "brand",
    "operator",
    "access",
    "opening_hours",
    "wheelchair",
    "addr:postcode",
]


def download_source(label, cfg):
    """Downloads all OSM objects of one source and stores them as csv (one point per object)"""
    print("Downloading:", label, cfg["tags"])
    gdf = ox.features_from_place("Berlin, Germany", tags=cfg["tags"])

    # The index is (element type, id). Turn it into normal columns "osm_type" and "osm_id".
    # Hint: gdf.reset_index(); the column names differ between osmnx versions -> use gdf.columns[0], [1]
    gdf = gdf.reset_index()
    gdf = gdf.rename(columns={gdf.columns[0]: "osm_type", gdf.columns[1]: "osm_id"})

    # Nodes are points, ways / relations are polygons. We need ONE point per object:
    # centroid in a projected CRS (pdict["crs_metric"]), then back to lat / lon (pdict["crs_geo"]).
    # A centroid computed in degrees would be distorted, so it is computed in metres.
    centroids = gdf.geometry.to_crs(pdict["crs_metric"]).centroid.to_crs(pdict["crs_geo"])
    gdf["lat"] = centroids.y
    gdf["lon"] = centroids.x

    # Tags do not exist for every object. Add missing columns from KEEP and cfg["subcat"] (filled with
    # pd.NA), so that later code never fails on a missing column.
    # dict.fromkeys drops repeats ("name" is also in KEEP) and keeps the order
    columns = list(
        dict.fromkeys(["osm_type", "osm_id", "name", "lat", "lon", *KEEP, cfg["subcat"]])
    )
    for col in columns:
        if col not in gdf.columns:
            gdf[col] = pd.NA

    df = pd.DataFrame(gdf[columns]).drop_duplicates(subset=["osm_type", "osm_id"])
    os.makedirs("datasets", exist_ok=True)
    df.to_csv("datasets/" + cfg["file"], sep=";", index=False, encoding="utf-8")
    print("  saved", len(df), "objects to datasets/" + cfg["file"])


def write_info(label):
    """Remember the download date per source (shown in the app)"""
    path = "datasets/" + pdict["file_info"]
    info = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
    info[label] = datetime.date.today().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def main():
    os.chdir(PROJECT_ROOT)
    labels = sys.argv[1:] or list(pdict["sources"].keys())
    for label in labels:
        download_source(label, pdict["sources"][label])
        write_info(label)


if __name__ == "__main__":
    main()
