"""
Downloads one csv file per OSM data source defined in config.py (Task 1).

Usage:
    uv run download-osm                    # all sources
    uv run download-osm Restaurants        # only one source (menu label)

The app never calls the Overpass API itself: run this script once, commit the csv files.
Sources with a "history" entry also get one snapshot of their objects on 1 January of each year.
"""

import datetime
import json
import os
import sys
import time
from collections.abc import Callable

import osmnx as ox
import pandas as pd
import requests
from osmnx._errors import ResponseStatusCodeError

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

# osmnx itself retries 429, 502 and 504 responses; these are the failures it passes on
RETRY_ERRORS = (requests.RequestException, ResponseStatusCodeError)


def with_backoff[T](action: Callable[[], T], description: str) -> T:
    """Runs action; after a failed request waits first_wait_s, then twice as long each time"""
    attempts = pdict["download_attempts"]
    wait_s = pdict["download_first_wait_s"]
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except RETRY_ERRORS as error:
            if attempt == attempts:
                raise
            print(f"  {description} failed ({error})")
            print(f"  try {attempt + 1} of {attempts} in {wait_s} s")
            time.sleep(wait_s)
            wait_s *= 2
    raise AssertionError("unreachable")


def fetch_points(cfg) -> pd.DataFrame:
    """Downloads all OSM objects of one source (one point per object)"""
    gdf = with_backoff(
        lambda: ox.features_from_place("Berlin, Germany", tags=cfg["tags"]), "Overpass request"
    )

    # The index is (element type, id). Turn it into normal columns "osm_type" and "osm_id".
    # Hint: gdf.reset_index(); the column names differ between osmnx versions
    # -> use gdf.columns[0], [1]
    gdf = gdf.reset_index()
    gdf = gdf.rename(columns={gdf.columns[0]: "osm_type", gdf.columns[1]: "osm_id"})

    # Nodes are points, ways / relations are polygons. We need ONE point per object:
    # centroid in a projected CRS (pdict["crs_metric"]), then back to lat / lon (pdict["crs_geo"]).
    # A centroid computed in degrees would be distorted, so it is computed in metres.
    centroids = gdf.geometry.to_crs(pdict["crs_metric"]).centroid.to_crs(pdict["crs_geo"])
    gdf["lat"] = centroids.y
    gdf["lon"] = centroids.x

    # Tags do not exist for every object. Add missing columns from KEEP and cfg["subcat"]
    # (filled with pd.NA), so that later code never fails on a missing column.
    # dict.fromkeys drops repeats ("name" is also in KEEP) and keeps the order
    columns = list(
        dict.fromkeys(["osm_type", "osm_id", "name", "lat", "lon", *KEEP, cfg["subcat"]])
    )
    for col in columns:
        if col not in gdf.columns:
            gdf[col] = pd.NA

    return pd.DataFrame(gdf[columns]).drop_duplicates(subset=["osm_type", "osm_id"])


def save(df: pd.DataFrame, file: str) -> None:
    os.makedirs("datasets", exist_ok=True)
    df.to_csv("datasets/" + file, sep=";", index=False, encoding="utf-8")


def download_source(label, cfg):
    """Stores the current objects of one source as csv"""
    print("Downloading:", label, cfg["tags"])
    df = fetch_points(cfg)
    save(df, cfg["file"])
    print("  saved", len(df), "objects to datasets/" + cfg["file"])


def download_history(label, cfg):
    """Stores the objects of one source on 1 January of each year as csv, with a column year.

    osmnx caches every finished request, so a run that fails continues at the failed year.
    """
    history = cfg["history"]
    years = range(history["start_year"], datetime.date.today().year + 1)
    print(f"Downloading history: {label}, {years[0]}-{years[-1]}")
    default_settings = ox.settings.overpass_settings
    frames = []
    try:
        for year in years:
            # [date:...] makes Overpass answer with the map as it was at that moment
            ox.settings.overpass_settings = (
                f'[out:json][timeout:{{timeout}}][date:"{year}-01-01T00:00:00Z"]{{maxsize}}'
            )
            df = fetch_points(cfg)
            df.insert(0, "year", year)
            frames.append(df)
            print(f"  {year}: {len(df)} objects")
    finally:
        ox.settings.overpass_settings = default_settings
    df = pd.concat(frames, ignore_index=True)
    save(df, history["file"])
    print(f"  saved {len(df)} rows ({len(years)} years) to datasets/{history['file']}")


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
        cfg = pdict["sources"][label]
        download_source(label, cfg)
        if "history" in cfg:
            download_history(label, cfg)
        write_info(label)


if __name__ == "__main__":
    main()
