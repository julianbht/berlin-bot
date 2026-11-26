"""
Downloads one csv file per OSM data source defined in config.py (Task 1).

Usage:
    uv run download-osm                    # all sources
    uv run download-osm Restaurants        # only one source (menu label)

The app never calls the Overpass API itself: run this script once, commit the csv files.
The source named in pdict["history"] also gets its yearly counts from the ohsome API.
"""

import datetime
import json
import os
import sys

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests

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

    df = pd.DataFrame(gdf[columns]).drop_duplicates(subset=["osm_type", "osm_id"])
    os.makedirs("datasets", exist_ok=True)
    df.to_csv("datasets/" + cfg["file"], sep=";", index=False, encoding="utf-8")
    print("  saved", len(df), "objects to datasets/" + cfg["file"])


def area_feature_collection(file, key, simplify_m):
    """Polygons of one area unit as GeoJSON FeatureCollection, with the area name as feature id"""
    df = pd.read_csv("datasets/" + file, sep=";", dtype={key: str})
    geometry = gpd.GeoSeries.from_wkt(df["geometry"], crs=pdict["crs_geo"])
    gdf = gpd.GeoDataFrame(df[[key]], geometry=geometry).dissolve(by=key, as_index=False)
    gdf["geometry"] = gdf.to_crs(pdict["crs_metric"]).simplify(simplify_m).to_crs(pdict["crs_geo"])
    features = [
        {"type": "Feature", "id": area, "properties": {}, "geometry": geom.__geo_interface__}
        for area, geom in zip(gdf[key], gdf.geometry, strict=True)
    ]
    return {"type": "FeatureCollection", "features": features}


def download_history(cfg):
    """Stores the number of objects per PLZ and Bezirk on 1 January of each year (long format)"""
    print("Downloading history:", cfg["filter"])
    # ohsome refuses timestamps after its latest data, so the last year is 1 January of this year
    time = f"{cfg['start_year']}-01-01/{datetime.date.today().year}-01-01/P1Y"
    frames = []
    for unit, file in [("PLZ", pdict["file_plz"]), ("Bezirk", pdict["file_dis"])]:
        boundaries = area_feature_collection(file, unit, cfg["simplify_m"])
        response = requests.post(
            cfg["url"],
            data={"bpolys": json.dumps(boundaries), "time": time, "filter": cfg["filter"]},
            timeout=600,
        )
        response.raise_for_status()
        rows = [
            (unit, group["groupByObject"], int(entry["timestamp"][:4]), int(entry["value"]))
            for group in response.json()["groupByResult"]
            for entry in group["result"]
        ]
        frames.append(pd.DataFrame(rows, columns=["unit", "area", "year", "count"]))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv("datasets/" + cfg["file"], sep=";", index=False, encoding="utf-8")
    years = sorted(df["year"].unique())
    print(f"  saved {len(df)} rows ({years[0]}-{years[-1]}) to datasets/{cfg['file']}")


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
        if label == pdict["history"]["source"]:
            download_history(pdict["history"])
        write_info(label)


if __name__ == "__main__":
    main()
