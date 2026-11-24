"""
Downloads one csv file per OSM data source defined in config.py (Task 1).

Usage:
    python download_osm.py                 # all sources
    python download_osm.py Restaurants     # only one source (menu label)

The app never calls the Overpass API itself: run this script once, commit the csv files.
"""

import datetime
import json
import os
import sys

import osmnx as ox
import pandas as pd

from config import pdict

# Columns that are kept (if they exist for the downloaded objects)
KEEP = ["name", "cuisine", "brand", "operator", "access", "opening_hours", "wheelchair", "addr:postcode"]


def download_source(label: str, cfg: dict) -> None:
    """Downloads all OSM objects of one source and stores them as csv (one point per object)"""
    print("Downloading:", label, cfg["tags"])
    # One request to the Overpass API: all objects with these tags inside the boundary of Berlin
    gdf = ox.features_from_place("Berlin, Germany", tags=cfg["tags"])

    # The index is (element type, id), e.g. ("node", 60775323). Turn it into normal columns.
    # The column names differ between osmnx versions -> rename by position
    gdf = gdf.reset_index()
    gdf = gdf.rename(columns={gdf.columns[0]: "osm_type", gdf.columns[1]: "osm_id"})

    # Nodes are points, ways / relations are polygons (e.g. a playground or a supermarket building).
    # We need ONE point per object: the centroid (centre) of the shape, computed in metres
    # (EPSG:25833) because centroids in degrees are slightly wrong, then back to lat / lon (EPSG:4326)
    centroids = gdf.geometry.to_crs(pdict["crs_metric"]).centroid.to_crs(pdict["crs_geo"])
    gdf["lat"] = centroids.y
    gdf["lon"] = centroids.x

    # Tags do not exist for every object (e.g. no pharmacy has "cuisine"). Add missing columns
    # as empty, so that the later code never fails on a missing column
    extra = KEEP + [cfg["subcat"]]
    for col in extra:
        if col not in gdf.columns:
            gdf[col] = pd.NA

    # Select the columns (dict.fromkeys removes duplicates, e.g. "cuisine" in KEEP and as subcat),
    # remove objects that appear twice, save as csv
    cols = list(dict.fromkeys(["osm_type", "osm_id", "name", "lat", "lon", *extra]))
    df = pd.DataFrame(gdf[cols]).drop_duplicates(subset=["osm_type", "osm_id"])
    os.makedirs("datasets", exist_ok=True)
    df.to_csv("datasets/" + cfg["file"], sep=";", index=False, encoding="utf-8")
    print("   saved", len(df), "objects ->", "datasets/" + cfg["file"])


def write_info(label: str) -> None:
    """Remember the download date per source (shown in the app)"""
    path = "datasets/" + pdict["file_info"]
    info = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
    info[label] = datetime.date.today().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    labels = sys.argv[1:] or list(pdict["sources"].keys())
    for label in labels:
        download_source(label, pdict["sources"][label])
        write_info(label)
