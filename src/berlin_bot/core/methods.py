import json
import os

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st

from berlin_bot.config import pdict as CFG  # only for CRS and thresholds
from berlin_bot.core import HelperTools as ht
from berlin_bot.core import chat


# =============================================================================
# 1) Loading
# =============================================================================
@st.cache_data
def load_areas(file: str, key: str) -> gpd.GeoDataFrame:
    """Polygons (PLZ or Bezirk) from WKT csv -> GeoDataFrame (one row per area, EPSG:4326)"""
    df = pd.read_csv("datasets/" + file, sep=";")
    geometry = gpd.GeoSeries.from_wkt(df["geometry"], crs=CFG["crs_geo"])
    gdf = gpd.GeoDataFrame(df[[key]], geometry=geometry)
    # An area split over several rows becomes one (Multi)Polygon
    return gdf.dissolve(by=key, as_index=False)


@st.cache_data
def load_osm_points(file: str) -> gpd.GeoDataFrame:
    """OSM csv (written by download_osm.py) -> GeoDataFrame of Points (EPSG:4326)"""
    df = pd.read_csv("datasets/" + file, sep=";", dtype={"addr:postcode": str})
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    geometry = gpd.points_from_xy(df["lon"], df["lat"], crs=CFG["crs_geo"])
    return gpd.GeoDataFrame(df, geometry=geometry)


@ht.timer
def preprop_resid(dfr):
    """Preprocessing plz_einwohner.csv: only PLZ and residents, only Berlin range"""
    d = dfr.loc[:, ["plz", "einwohner"]].rename(columns={"plz": "PLZ", "einwohner": "Einwohner"})
    return d[(d["PLZ"] > 10000) & (d["PLZ"] < 14200)].reset_index(drop=True)


# =============================================================================
# 2) Spatial join and category filter
# =============================================================================
def assign_area(
    gdf_points: gpd.GeoDataFrame, gdf_areas: gpd.GeoDataFrame, key: str
) -> gpd.GeoDataFrame:
    """Spatial join: adds column <key> (PLZ / Bezirk) to every point. NaN = outside of all areas.
    Points exactly on a border can match twice -> only the first match is kept."""
    joined = gpd.sjoin(gdf_points, gdf_areas[[key, "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined.drop(columns="index_right")


def _split_values(series):
    """'italian;pizza' -> ['italian', 'pizza'] (empty list if missing)"""
    return (
        series.fillna("")
        .astype(str)
        .str.split(";")
        .apply(lambda v: [x.strip() for x in v if x.strip()])
    )


def category_options(gdf, col, n=10):
    """The n most frequent values of a (multi-valued) attribute, for the dependent menu"""
    # TODO: use _split_values, explode, value_counts. Return [] if the column does not exist.
    raise NotImplementedError


def filter_category(gdf, col, category):
    """Keeps objects that have <category> among their values (cuisine=italian;pizza counts for both)"""
    # TODO: "All" -> return gdf unchanged
    raise NotImplementedError


# =============================================================================
# 3) Indicators
# =============================================================================
def residents_by_area(
    df_res: pd.DataFrame, gdf_plz: gpd.GeoDataFrame, gdf_dis: gpd.GeoDataFrame, key: str
) -> pd.DataFrame:
    """Residents per PLZ or per Bezirk (columns <key>, Einwohner).

    Residents are only known per PLZ. For a Bezirk, every PLZ counts in full for the Bezirk that
    holds its centre. A PLZ that crosses a Bezirk border is not split, so Bezirk totals are an
    approximation."""
    if key == "PLZ":
        return pd.DataFrame(gdf_plz[["PLZ"]]).merge(df_res, on="PLZ", how="left")
    if key != "Bezirk":
        raise ValueError(f"Unknown area key: {key}")

    plz_metric = gdf_plz.to_crs(CFG["crs_metric"])
    plz_centres = gpd.GeoDataFrame(plz_metric[["PLZ"]], geometry=plz_metric.centroid)
    dis_metric = gdf_dis[["Bezirk", "geometry"]].to_crs(CFG["crs_metric"])
    # Nearest, not within: a centre can fall just outside every Bezirk polygon
    plz_to_dis = gpd.sjoin_nearest(plz_centres, dis_metric, how="left")
    plz_to_dis = plz_to_dis[~plz_to_dis.index.duplicated(keep="first")]
    with_res = pd.DataFrame(plz_to_dis[["PLZ", "Bezirk"]]).merge(df_res, on="PLZ", how="left")
    return with_res.groupby("Bezirk", as_index=False)["Einwohner"].sum()


def aggregate(
    gdf_points: gpd.GeoDataFrame,
    gdf_areas: gpd.GeoDataFrame,
    key: str,
    indicator: str,
    residents: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """One row per area (also areas without objects = 0) with columns count, area_km2, Einwohner, value"""
    # Points outside all areas have key NaN, which made the column float: cast to the area dtype
    area_keys = gdf_points[key].dropna().astype(gdf_areas[key].dtype)
    counts = area_keys.value_counts().rename("count")

    agg = gdf_areas[[key, "geometry"]].merge(counts, left_on=key, right_index=True, how="left")
    agg["count"] = agg["count"].fillna(0).astype(int)
    agg["area_km2"] = agg.geometry.to_crs(CFG["crs_metric"]).area / 1e6
    agg = agg.merge(residents[[key, "Einwohner"]], on=key, how="left")

    if indicator == "Number":
        agg["value"] = agg["count"].astype(float)
    elif indicator == "Number per km2":
        agg["value"] = agg["count"] / agg["area_km2"]
    elif indicator == "Number per 10,000 residents":
        # Few residents give extreme ratios (a PLZ with 50 residents and 2 cafes -> 400): no value
        enough_residents = agg["Einwohner"] >= CFG["min_residents"]
        agg["value"] = (agg["count"] / agg["Einwohner"] * 10_000).where(enough_residents)
    else:
        raise ValueError(f"Unknown indicator: {indicator}")
    return agg


# =============================================================================
# 4) Data quality
# =============================================================================
def _count_and_share(count: int, total: int) -> str:
    """'12 (3.4 %)'; the share is left out when there is nothing to divide by"""
    if total == 0:
        return str(count)
    return f"{count} ({count / total * 100:.1f} %)"


def _is_missing(series: pd.Series) -> pd.Series:
    """True for NaN and for empty or blank strings"""
    return series.isna() | (series.astype(str).str.strip() == "")


def _near_duplicates(gdf_points: gpd.GeoDataFrame) -> int:
    """Number of objects that have another object with the same name closer than dup_distance_m"""
    named = gdf_points.loc[~_is_missing(gdf_points["name"]), ["name", "geometry"]]
    named = named.to_crs(CFG["crs_metric"])
    pairs = gpd.sjoin(named, named, predicate="dwithin", distance=CFG["dup_distance_m"])
    # The self-join also pairs every object with itself
    pairs = pairs[
        (pairs.index != pairs["index_right"]) & (pairs["name_left"] == pairs["name_right"])
    ]
    return pairs.index.nunique()


def quality_report(
    gdf_points: gpd.GeoDataFrame, gdf_areas: gpd.GeoDataFrame, subcat: str | None = None
) -> pd.DataFrame:
    """Automatic checks of one OSM file. Expects column 'PLZ' (from assign_area). Returns DataFrame."""
    total = len(gdf_points)
    checks: list[tuple[str, str]] = [("Number of objects", str(total))]

    without_name = int(_is_missing(gdf_points["name"]).sum())
    checks.append(("Without name", _count_and_share(without_name, total)))
    if subcat is not None and subcat in gdf_points.columns:
        without_subcat = int(_is_missing(gdf_points[subcat]).sum())
        checks.append((f"Without {subcat}", _count_and_share(without_subcat, total)))

    near_duplicates = _near_duplicates(gdf_points)
    checks.append(
        (
            f"Near-duplicates (same name, < {CFG['dup_distance_m']} m)",
            _count_and_share(near_duplicates, total),
        )
    )

    outside = int(gdf_points["PLZ"].isna().sum())
    checks.append(("Outside Berlin (no PLZ area)", _count_and_share(outside, total)))

    lat_min, lat_max = CFG["berlin_lat"]
    lon_min, lon_max = CFG["berlin_lon"]
    plausible = gdf_points.geometry.y.between(lat_min, lat_max) & gdf_points.geometry.x.between(
        lon_min, lon_max
    )
    checks.append(("Implausible coordinates", _count_and_share(int((~plausible).sum()), total)))

    has_postcode = ~_is_missing(gdf_points["addr:postcode"])
    checks.append(("With addr:postcode", _count_and_share(int(has_postcode.sum()), total)))
    # Only points that have both a postcode and a PLZ from the join can be compared
    comparable = gdf_points[has_postcode & gdf_points["PLZ"].notna()]
    joined_plz = comparable["PLZ"].astype(int).astype(str)
    matches = int((comparable["addr:postcode"].str.strip() == joined_plz).sum())
    checks.append(
        ("addr:postcode equals PLZ of the join", _count_and_share(matches, len(comparable)))
    )

    plz_with_objects = gdf_points["PLZ"].dropna().astype(int)
    empty_areas = int((~gdf_areas["PLZ"].isin(plz_with_objects)).sum())
    checks.append(("PLZ areas without any object", _count_and_share(empty_areas, len(gdf_areas))))

    return pd.DataFrame(checks, columns=["Check", "Result"])


# =============================================================================
# 5) Map
# =============================================================================
def make_map(gdf_points, gdf_agg, maptype, key="PLZ", colors=("yellow", "red"), caption="Value"):
    """Folium map: choropleth (areas), heatmap or marker clusters (single points)"""
    m = folium.Map(location=[52.52, 13.40], zoom_start=10, tiles="OpenStreetMap")

    if maptype == "Choropleth":
        # TODO: LinearColormap(colors, vmin, vmax) from the column "value"; areas with value NaN -> grey.
        #       ONE folium.GeoJson layer (gdf_agg.to_json()) with style_function and tooltip. Add the legend.
        pass
    elif maptype == "Heatmap":
        # TODO: HeatMap([[lat, lon], ...])        <- Folium wants [lat, lon]
        pass
    else:  # Point map (clusters)
        # TODO: MarkerCluster with one marker per point, name as tooltip
        pass
    return m


# =============================================================================
# 6) Streamlit app
# =============================================================================
@st.cache_data(show_spinner="Loading data and assigning areas ...")
def prepare_source(label, pdict):
    """Loads one OSM source, assigns PLZ and Bezirk (spatial join) and runs the quality check"""
    cfg = pdict["sources"][label]
    plz = load_areas(pdict["file_plz"], "PLZ")
    dis = load_areas(pdict["file_dis"], "Bezirk")
    pts = load_osm_points(cfg["file"])
    pts = assign_area(pts, plz, "PLZ")
    pts = assign_area(pts, dis, "Bezirk")
    return pts, plz, dis, quality_report(pts, plz, cfg["subcat"])


def _download_date(pdict, label):
    path = "datasets/" + pdict["file_info"]
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f).get(label, "unknown")
    return "unknown"


@ht.timer
def make_streamlit_app(pdict, df_residents):
    """Makes Streamlit App: OSM data of Berlin, controlled by pull-down menus"""
    st.set_page_config(page_title="OpenStreetMap Berlin", layout="wide")
    st.title("OpenStreetMap Berlin: Places and Supply")

    # ---- Menu 1: data source ------------------------------------------------
    src = st.sidebar.selectbox("Data source", list(pdict["sources"].keys()))
    cfg = pdict["sources"][src]
    if not os.path.exists("datasets/" + cfg["file"]):
        st.error(
            "File datasets/" + cfg["file"] + ' not found. Run:  uv run download-osm "' + src + '"'
        )
        return
    pts, plz, dis, quality = prepare_source(src, pdict)

    # ---- Menu 2: category filter (depends on menu 1) ------------------------
    # TODO: options = ["All"] + category_options(...); use key="cat_" + src so the menu resets with the source
    cat = "All"

    # ---- Menus 3-5 ----------------------------------------------------------
    area_label = st.sidebar.selectbox("Area unit", list(pdict["areas"].keys()))
    ind = st.sidebar.selectbox("Indicator", pdict["indicators"])
    mtype = st.sidebar.selectbox("Map type", pdict["maptypes"])
    # optional menu 6: colour scale (only for the choropleth)

    # TODO: key (PLZ / Bezirk) and the matching polygons, filter by category, stop with st.warning if empty,
    #       residents_by_area, aggregate

    # TODO: key figures with st.metric: objects shown, areas without any object, area with the highest value
    # TODO: map with st_folium(m, width=1000, height=600, returned_objects=[])
    # TODO: st.caption with "© OpenStreetMap contributors" and _download_date(pdict, src)
    # TODO: expander "Top 3 areas and data table" with st.dataframe and st.download_button (CSV)
    # TODO: expander "Data quality" showing the quality DataFrame

    # ---- Chat assistant (LLM) - provided, see core/chat.py -------------------
    # Needs the variables src, cat, area_label, ind, mtype, key, pts_f, agg, quality from your code above.
    ctx = chat.build_context(src, cat, area_label, ind, mtype, key, pts_f, agg, quality)
    chat.render_chat(ctx, pdict)
