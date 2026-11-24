"""Tests for the deterministic part of the app (loading, spatial join, indicators, data quality).

Run:  pip install pytest   then   python -m pytest
"""
import os

import geopandas as gpd
import pandas as pd
import pytest

from config import pdict
from core import methods as m


# -----------------------------------------------------------------------------
# Fixtures: static datasets (loaded once)
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def plz():
    return m.load_areas(pdict["file_plz"], "PLZ")


@pytest.fixture(scope="module")
def dis():
    return m.load_areas(pdict["file_dis"], "Bezirk")


@pytest.fixture(scope="module")
def residents():
    return m.preprop_resid(pd.read_csv("datasets/" + pdict["file_residents"]))


def make_points(rows):
    """Small GeoDataFrame from (name, lat, lon, extra-dict) tuples"""
    df = pd.DataFrame([{"name": n, "lat": la, "lon": lo, **extra} for n, la, lo, extra in rows])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=pdict["crs_geo"])


BRANDENBURG_GATE = (52.5163, 13.3777)       # PLZ 10117, Bezirk Mitte


# -----------------------------------------------------------------------------
# Task 1: downloaded files
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("label", list(pdict["sources"].keys()))
def test_osm_file_exists_and_is_in_berlin(label):
    path = "datasets/" + pdict["sources"][label]["file"]
    assert os.path.exists(path), "run: python download_osm.py \"" + label + "\""
    pts = m.load_osm_points(pdict["sources"][label]["file"])
    assert len(pts) > 0
    # x = longitude (~13.4), y = latitude (~52.5): catches swapped coordinates
    assert pts.geometry.x.between(13.0, 13.8).all()
    assert pts.geometry.y.between(52.3, 52.7).all()


# -----------------------------------------------------------------------------
# Task 2: areas, spatial join, indicators
# -----------------------------------------------------------------------------
def test_areas_are_unique(plz, dis):
    assert len(plz) == 190 and plz["PLZ"].is_unique
    assert len(dis) == 12 and dis["Bezirk"].is_unique
    assert plz.crs == pdict["crs_geo"]


def test_spatial_join_known_point(plz, dis):
    pts = make_points([("Gate", *BRANDENBURG_GATE, {})])
    pts = m.assign_area(m.assign_area(pts, plz, "PLZ"), dis, "Bezirk")
    assert pts.loc[0, "PLZ"] == 10117
    assert pts.loc[0, "Bezirk"] == "Mitte"


def test_spatial_join_point_outside_berlin(plz):
    pts = make_points([("Potsdam", 52.40, 13.06, {})])
    assert pd.isna(m.assign_area(pts, plz, "PLZ").loc[0, "PLZ"])


def test_spatial_join_keeps_one_row_per_point(plz):
    pts = m.load_osm_points(pdict["sources"]["Restaurants"]["file"])
    assert len(m.assign_area(pts, plz, "PLZ")) == len(pts)


def test_aggregate_keeps_empty_areas(plz, residents):
    pts = m.assign_area(make_points([("Gate", *BRANDENBURG_GATE, {})]), plz, "PLZ")
    res = m.residents_by_area(residents, plz, None, "PLZ")
    agg = m.aggregate(pts, plz, "PLZ", "Number", res)
    assert len(agg) == 190                         # all areas, also the empty ones
    assert (agg["count"] == 0).sum() == 189
    assert agg.loc[agg["PLZ"] == 10117, "count"].item() == 1


def test_area_in_km2_not_degrees(plz, residents):
    res = m.residents_by_area(residents, plz, None, "PLZ")
    agg = m.aggregate(make_points([("x", *BRANDENBURG_GATE, {})]).assign(PLZ=10117), plz, "PLZ", "Number", res)
    assert agg["area_km2"].sum() == pytest.approx(891.8, abs=5)    # official area of Berlin


def test_per_residents_undefined_for_small_areas(plz, residents):
    res = m.residents_by_area(residents, plz, None, "PLZ")
    agg = m.aggregate(make_points([("x", *BRANDENBURG_GATE, {})]).assign(PLZ=10117), plz, "PLZ",
                      "Number per 10,000 residents", res)
    small = agg["Einwohner"] < pdict["min_residents"]
    assert small.any() and agg.loc[small, "value"].isna().all()
    assert agg.loc[~small, "value"].notna().all()


def test_residents_bezirk_sum_equals_plz_sum(plz, dis, residents):
    by_plz = m.residents_by_area(residents, plz, dis, "PLZ")
    by_dis = m.residents_by_area(residents, plz, dis, "Bezirk")
    assert len(by_dis) == 12
    assert by_dis["Einwohner"].sum() == by_plz["Einwohner"].sum()


# -----------------------------------------------------------------------------
# Category filter (menu 2)
# -----------------------------------------------------------------------------
def test_filter_multi_valued_tags():
    pts = make_points([("A", 52.5, 13.4, {"cuisine": "italian;pizza"}),
                       ("B", 52.5, 13.4, {"cuisine": "pizza"}),
                       ("C", 52.5, 13.4, {"cuisine": None})])
    assert m.category_options(pts, "cuisine") == ["pizza", "italian"]
    assert list(m.filter_category(pts, "cuisine", "italian")["name"]) == ["A"]
    assert len(m.filter_category(pts, "cuisine", "pizza")) == 2
    assert len(m.filter_category(pts, "cuisine", "All")) == 3
    assert m.category_options(pts, "no_such_column") == []


# -----------------------------------------------------------------------------
# Task 3: data quality report
# -----------------------------------------------------------------------------
def test_quality_report(plz):
    pts = make_points([("Apotheke", 52.5163, 13.3777, {"addr:postcode": "10117"}),
                       ("Apotheke", 52.51633, 13.37772, {"addr:postcode": "10115"}),   # ~4 m away
                       (None, 52.40, 13.06, {"addr:postcode": None})])                # outside
    pts = m.assign_area(pts, plz, "PLZ")
    q = dict(zip(*m.quality_report(pts, plz).T.values))
    assert q["Number of objects"] == "3"
    assert q["Without name"].startswith("1 ")
    assert q["Near-duplicates (same name, < 10 m)"] == "1 pairs"
    assert q["Outside of all Berlin PLZ areas"].startswith("1 ")
    assert q["addr:postcode = PLZ of spatial join"].startswith("1 of 2")
    assert q["PLZ areas without any object"] == "189 of 190"
