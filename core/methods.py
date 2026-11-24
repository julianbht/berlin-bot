"""Loading, spatial join, indicators, data quality, map and Streamlit app (Tasks 2-4)."""

import json
import os

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

import core.chat as chat
import core.HelperTools as ht
from config import pdict as CFG  # only for CRS and thresholds


# =============================================================================
# 1) Loading
# =============================================================================
@st.cache_data
def load_areas(file: str, key: str) -> gpd.GeoDataFrame:
    """Polygons (PLZ or Bezirk) from WKT csv -> GeoDataFrame (one row per area, EPSG:4326)"""
    # Read the csv: one column with the area name/number, one column "geometry" as WKT text
    df = pd.read_csv("datasets/" + file, sep=";")

    # Turn the WKT text ("POLYGON ((13.37 52.53, ...))") into real shapes and set lat/lon as CRS
    gdf = gpd.GeoDataFrame(df[[key]], geometry=gpd.GeoSeries.from_wkt(df["geometry"]), crs=CFG["crs_geo"])

    # If an area appears in several rows, merge its polygons into one row
    return gdf.dissolve(by=key, as_index=False)[[key, "geometry"]]


@st.cache_data
def load_osm_points(file: str) -> gpd.GeoDataFrame:
    """OSM csv (written by download_osm.py) -> GeoDataFrame of Points (EPSG:4326)"""
    # addr:postcode is read as text, otherwise pandas turns "10115" into 10115.0
    df = pd.read_csv("datasets/" + file, sep=";", dtype={"addr:postcode": str}, low_memory=False)

    # Objects without coordinates cannot be placed on the map
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Build one point per row. Order matters: x = longitude, y = latitude
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CFG["crs_geo"])


@ht.timer
def preprop_resid(dfr: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing plz_einwohner.csv: only PLZ and residents, only Berlin range"""
    d = dfr.loc[:, ["plz", "einwohner"]].rename(columns={"plz": "PLZ", "einwohner": "Einwohner"})
    return d[(d["PLZ"] > 10000) & (d["PLZ"] < 14200)].reset_index(drop=True)


# =============================================================================
# 2) Spatial join and category filter
# =============================================================================
def assign_area(gdf_points: gpd.GeoDataFrame, gdf_areas: gpd.GeoDataFrame, key: str) -> gpd.GeoDataFrame:
    """Spatial join: adds column <key> (PLZ / Bezirk) to every point. NaN = outside of all areas.
    Points exactly on a border can match twice -> only the first match is kept."""
    # For every point: in which polygon does it lie? "left" keeps points that lie in no polygon (NaN)
    joined = gpd.sjoin(gdf_points, gdf_areas[[key, "geometry"]], how="left", predicate="within")

    # A point exactly on a border matches two areas -> keep only the first match (one row per point)
    joined = joined[~joined.index.duplicated(keep="first")]

    # sjoin adds a helper column with the row number of the polygon - not needed
    joined = joined.drop(columns="index_right")

    # PLZ is an integer; NaN (outside Berlin) would turn it into a float -> nullable integer "Int64"
    if pd.api.types.is_integer_dtype(gdf_areas[key]):
        joined[key] = joined[key].astype("Int64")
    return joined


def _split_values(series: pd.Series) -> pd.Series:
    """'italian;pizza' -> ['italian', 'pizza'] (empty list if missing)"""
    return series.fillna("").astype(str).str.split(";").apply(lambda v: [x.strip() for x in v if x.strip()])


def category_options(gdf: gpd.GeoDataFrame, col: str, n: int = 10) -> list[str]:
    """The n most frequent values of a (multi-valued) attribute, for the dependent menu"""
    # Some sources have no such tag at all -> no options
    if col not in gdf.columns:
        return []

    # "italian;pizza" counts for both: split into lists, one row per value, then count
    values = _split_values(gdf[col]).explode().dropna()
    return values.value_counts().head(n).index.tolist()


def filter_category(gdf: gpd.GeoDataFrame, col: str, category: str) -> gpd.GeoDataFrame:
    """Keeps objects that have <category> among their values (cuisine=italian;pizza counts for both)"""
    # "All" (or a missing column) means: no filter
    if category == "All" or col not in gdf.columns:
        return gdf

    # Keep a row if the chosen value is one of its values
    mask = _split_values(gdf[col]).apply(lambda values: category in values)
    return gdf[mask]


# =============================================================================
# 3) Indicators
# =============================================================================
def residents_by_area(
    df_res: pd.DataFrame, gdf_plz: gpd.GeoDataFrame, gdf_dis: gpd.GeoDataFrame | None, key: str
) -> pd.DataFrame:
    """Residents per PLZ or per Bezirk (columns <key>, Einwohner)."""
    # PLZ: the residents file already has one number per PLZ -> simple merge
    if key == "PLZ":
        return gdf_plz[["PLZ"]].merge(df_res, on="PLZ", how="left")[["PLZ", "Einwohner"]]

    # Bezirk: there is no residents file per district -> APPROXIMATION:
    # every PLZ is counted completely in the district that contains (or is nearest to) its centre,
    # although some PLZ areas cross district borders.

    # Centre of every PLZ, computed in metres (EPSG:25833) - centroids in degrees are slightly wrong
    centres = gdf_plz.to_crs(CFG["crs_metric"])
    centres["geometry"] = centres.geometry.centroid
    centres = centres.merge(df_res, on="PLZ", how="left")

    # Assign every centre to the nearest district polygon (distance 0 if it lies inside)
    districts = gdf_dis.to_crs(CFG["crs_metric"])[["Bezirk", "geometry"]]
    joined = gpd.sjoin_nearest(centres, districts, how="left")
    joined = joined[~joined.index.duplicated(keep="first")]

    # Sum the residents of all PLZ that belong to the same district
    return joined.groupby("Bezirk", as_index=False)["Einwohner"].sum()


def aggregate(
    gdf_points: gpd.GeoDataFrame, gdf_areas: gpd.GeoDataFrame, key: str, indicator: str, residents: pd.DataFrame
) -> gpd.GeoDataFrame:
    """One row per area (also areas without objects = 0) with columns count, area_km2, Einwohner, value"""
    # Number of points per area
    counts = gdf_points.groupby(key).size().rename("count").reset_index()

    # Start from the POLYGONS (left merge), so areas without any object stay in the table with 0
    agg = gdf_areas[[key, "geometry"]].merge(counts, on=key, how="left")
    agg["count"] = agg["count"].fillna(0).astype(int)

    # Area in km2: convert to metres first (EPSG:25833), never compute areas in degrees
    agg["area_km2"] = agg.to_crs(CFG["crs_metric"]).area / 1e6
    agg = agg.merge(residents, on=key, how="left")

    # The selected indicator goes into the column "value" (used by map, table and chat)
    if indicator == "Number":
        agg["value"] = agg["count"].astype(float)
    elif indicator == "Number per km2":
        agg["value"] = agg["count"] / agg["area_km2"]
    else:  # "Number per 10,000 residents"
        # Areas with very few residents would get extreme values -> not defined (NaN, grey on the map)
        enough = agg["Einwohner"] >= CFG["min_residents"]
        agg["value"] = np.where(enough, agg["count"] / agg["Einwohner"] * 10000, np.nan)
    return agg


# =============================================================================
# 4) Data quality
# =============================================================================
def quality_report(
    gdf_points: gpd.GeoDataFrame, gdf_areas: gpd.GeoDataFrame, subcat: str | None = None
) -> pd.DataFrame:
    """Automatic checks of one OSM file. Expects column 'PLZ' (from assign_area). Returns DataFrame."""
    n = len(gdf_points)
    if n == 0:
        return pd.DataFrame({"Check": ["Number of objects"], "Result": ["0"]})

    def pct(part: float, total: float) -> str:
        """Share as text, e.g. '12.5 %'"""
        return f"{100 * part / total:.1f} %" if total else "-"

    def missing(col: str) -> pd.Series:
        """True where a tag is missing or empty (also True for every row if the column does not exist)"""
        if col not in gdf_points.columns:
            return pd.Series(True, index=gdf_points.index)
        return gdf_points[col].isna() | (gdf_points[col].astype(str).str.strip() == "")

    rows = [("Number of objects", str(n))]

    # Missing tags: objects without a name, and without the tag used by the category filter
    no_name = missing("name")
    rows.append(("Without name", f"{int(no_name.sum())} ({pct(no_name.sum(), n)})"))
    if subcat:
        no_sub = missing(subcat)
        rows.append((f"Without '{subcat}'", f"{int(no_sub.sum())} ({pct(no_sub.sum(), n)})"))

    # Near-duplicates: same name and closer than dup_distance_m metres (probably mapped twice).
    # Work in metres: draw a circle (buffer) around every named point and look for other points inside it
    named = gdf_points.loc[~no_name, ["name", "geometry"]].to_crs(CFG["crs_metric"])
    buffers = named.copy()
    buffers["geometry"] = buffers.geometry.buffer(CFG["dup_distance_m"])
    pairs = gpd.sjoin(named, buffers, how="inner", predicate="within")
    # Count every pair once (index < index_right) and only if both have the same name
    pairs = pairs[(pairs.index < pairs["index_right"]) & (pairs["name_left"] == pairs["name_right"])]
    rows.append((f"Near-duplicates (same name, < {CFG['dup_distance_m']} m)", f"{len(pairs)} pairs"))

    # Points outside Berlin: the spatial join found no PLZ polygon for them
    outside = gdf_points["PLZ"].isna()
    rows.append(("Outside of all Berlin PLZ areas", f"{int(outside.sum())} ({pct(outside.sum(), n)})"))

    # Implausible coordinates: outside a rough box around Berlin (e.g. swapped lat/lon)
    lat_ok = gdf_points["lat"].between(*CFG["berlin_lat"])
    lon_ok = gdf_points["lon"].between(*CFG["berlin_lon"])
    implausible = ~(lat_ok & lon_ok)
    lat_min, lat_max = CFG["berlin_lat"]
    lon_min, lon_max = CFG["berlin_lon"]
    rows.append(
        (
            f"Implausible coordinates (outside lat {lat_min}-{lat_max}, lon {lon_min}-{lon_max})",
            str(int(implausible.sum())),
        )
    )

    # addr:postcode (typed by volunteers) compared with the PLZ found by the spatial join
    has_pc = ~missing("addr:postcode")
    rows.append(("With addr:postcode", f"{int(has_pc.sum())} ({pct(has_pc.sum(), n)})"))
    if has_pc.any():
        pc = pd.to_numeric(gdf_points.loc[has_pc, "addr:postcode"], errors="coerce")  # text -> number
        join = gdf_points.loc[has_pc, "PLZ"].astype(float)
        same = int((pc == join).sum())
        rows.append(
            ("addr:postcode = PLZ of spatial join", f"{same} of {int(has_pc.sum())} ({pct(same, has_pc.sum())})")
        )

    # PLZ areas that contain no object at all (possible gaps - or just not mapped)
    empty = int((~gdf_areas["PLZ"].isin(gdf_points["PLZ"].dropna())).sum())
    rows.append(("PLZ areas without any object", f"{empty} of {len(gdf_areas)}"))

    return pd.DataFrame(rows, columns=["Check", "Result"])


# =============================================================================
# 5) Map
# =============================================================================
def make_map(
    gdf_points: gpd.GeoDataFrame,
    gdf_agg: gpd.GeoDataFrame,
    maptype: str,
    key: str = "PLZ",
    colors: tuple[str, ...] | list[str] = ("yellow", "red"),
    caption: str = "Value",
) -> folium.Map:
    """Folium map: choropleth (areas), heatmap or marker clusters (single points)"""
    m = folium.Map(location=[52.52, 13.40], zoom_start=10, tiles="OpenStreetMap")

    if maptype == "Choropleth":
        # Colour scale from the smallest to the largest value
        values = gdf_agg["value"].dropna()
        vmin = float(values.min()) if len(values) else 0.0
        vmax = float(values.max()) if len(values) else 1.0
        if vmax == vmin:  # all areas equal -> avoid an empty scale
            vmax = vmin + 1
        cmap = LinearColormap(list(colors), vmin=vmin, vmax=vmax, caption=caption)

        # Only the columns needed for colour and tooltip, rounded for display
        data = gdf_agg[[key, "count", "area_km2", "Einwohner", "value", "geometry"]].copy()
        data["area_km2"] = data["area_km2"].round(2)
        data["value"] = data["value"].round(2)

        def style(feature: dict) -> dict:
            """Fill colour of one area: from the colour scale, grey if the value is not defined (NaN)"""
            v = feature["properties"]["value"]
            return {
                "fillColor": cmap(v) if v is not None else "lightgrey",
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.7,
            }

        # ONE GeoJson layer with all areas; the tooltip shows the numbers when hovering over an area
        tooltip = folium.GeoJsonTooltip(
            fields=[key, "count", "Einwohner", "value"], aliases=[key, "Objects", "Residents", caption]
        )
        folium.GeoJson(data.to_json(), style_function=style, tooltip=tooltip).add_to(m)
        cmap.add_to(m)  # legend
    elif maptype == "Heatmap":
        # Folium expects [lat, lon] - the opposite order of GeoPandas (x = lon, y = lat)
        HeatMap(gdf_points[["lat", "lon"]].values.tolist(), radius=12).add_to(m)
    else:  # Point map (clusters)
        # One marker per object; nearby markers are grouped into clusters when zoomed out
        cluster = MarkerCluster().add_to(m)
        for lat, lon, name in zip(gdf_points["lat"], gdf_points["lon"], gdf_points["name"], strict=True):
            label = name if isinstance(name, str) and name.strip() else "(no name)"
            folium.Marker([lat, lon], tooltip=label).add_to(cluster)
    return m


# =============================================================================
# 6) Streamlit app
# =============================================================================
@st.cache_data(show_spinner="Loading data and assigning areas ...")
def prepare_source(
    label: str, pdict: dict
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Loads one OSM source, assigns PLZ and Bezirk (spatial join) and runs the quality check"""
    cfg = pdict["sources"][label]
    plz = load_areas(pdict["file_plz"], "PLZ")
    dis = load_areas(pdict["file_dis"], "Bezirk")
    pts = load_osm_points(cfg["file"])
    pts = assign_area(pts, plz, "PLZ")
    pts = assign_area(pts, dis, "Bezirk")
    return pts, plz, dis, quality_report(pts, plz, cfg["subcat"])


def _download_date(pdict: dict, label: str) -> str:
    path = "datasets/" + pdict["file_info"]
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f).get(label, "unknown")
    return "unknown"


@ht.timer
def make_streamlit_app(pdict: dict, df_residents: pd.DataFrame) -> None:
    """Makes Streamlit App: OSM data of Berlin, controlled by pull-down menus"""
    st.set_page_config(page_title="OpenStreetMap Berlin", layout="wide")
    st.title("OpenStreetMap Berlin: Places and Supply")

    # ---- Menu 1: data source ------------------------------------------------
    src = st.sidebar.selectbox("Data source", list(pdict["sources"].keys()))
    cfg = pdict["sources"][src]
    if not os.path.exists("datasets/" + cfg["file"]):
        st.error("File datasets/" + cfg["file"] + ' not found. Run:  python download_osm.py "' + src + '"')
        return
    pts, plz, dis, quality = prepare_source(src, pdict)

    # ---- Menu 2: category filter (depends on menu 1) ------------------------
    # Options = "All" + the 10 most frequent values of the source's tag (e.g. cuisine).
    # key="cat_" + src gives every source its own menu, so it resets when the source changes
    subcat = cfg["subcat"]
    options = ["All"] + category_options(pts, subcat, pdict["top_n_category"])
    cat = st.sidebar.selectbox("Category filter (" + subcat + ")", options, key="cat_" + src)

    # ---- Menus 3-5 ----------------------------------------------------------
    area_label = st.sidebar.selectbox("Area unit", list(pdict["areas"].keys()))
    ind = st.sidebar.selectbox("Indicator", pdict["indicators"])
    mtype = st.sidebar.selectbox("Map type", pdict["maptypes"])
    # Optional menu 6: colour scale (only used by the choropleth, greyed out otherwise)
    scale = st.sidebar.selectbox("Colour scale", list(pdict["colorscales"].keys()), disabled=(mtype != "Choropleth"))

    # ---- Data for the current selection -------------------------------------
    key = pdict["areas"][area_label]  # "PLZ" or "Bezirk"
    areas = plz if key == "PLZ" else dis
    pts_f = filter_category(pts, subcat, cat)  # points after the category filter
    if len(pts_f) == 0:
        st.warning("No objects for this selection.")
        return
    residents = residents_by_area(df_residents, plz, dis, key)
    agg = aggregate(pts_f, areas, key, ind, residents)  # one row per area with "value"

    # ---- Key figures --------------------------------------------------------
    n_empty = int((agg["count"] == 0).sum())
    best = agg.dropna(subset=["value"]).sort_values("value", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Objects shown", len(pts_f))
    c2.metric("Areas without any object", f"{n_empty} of {len(agg)}")
    if len(best):
        c3.metric(f"Highest value ({ind})", f"{best.iloc[0][key]} ({best.iloc[0]['value']:.2f})")
    # Tell the user about the approximation of residents per district
    if key == "Bezirk" and ind == "Number per 10,000 residents":
        st.caption("Residents per district are approximated: every PLZ is counted in the district of its centre.")

    # ---- Map ----------------------------------------------------------------
    m = make_map(pts_f, agg, mtype, key, pdict["colorscales"][scale], ind)
    st_folium(m, width=1000, height=600, returned_objects=[])  # returned_objects=[]: no rerun on map clicks
    st.caption("© OpenStreetMap contributors (ODbL) · data downloaded: " + _download_date(pdict, src))

    # ---- Table and data quality ---------------------------------------------
    with st.expander("Top 3 areas and data table"):
        # Table sorted by value (areas without value at the end), with CSV download
        tbl = agg.drop(columns="geometry")[[key, "count", "area_km2", "Einwohner", "value"]]
        tbl = tbl.sort_values("value", ascending=False, na_position="last").round(2)
        top3 = tbl.dropna(subset=["value"]).head(3)
        st.write("**Top 3:** " + ", ".join(f"{r[key]} ({r['value']:.2f})" for _, r in top3.iterrows()))
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.download_button(
            "Download CSV",
            tbl.to_csv(index=False).encode("utf-8"),
            file_name="osm_" + src.lower().replace(" ", "_") + "_" + key + ".csv",
            mime="text/csv",
        )
    with st.expander("Data quality"):
        st.dataframe(quality, hide_index=True, width="stretch")

    # ---- Chat assistant (LLM) - provided, see core/chat.py -------------------
    # Needs the variables src, cat, area_label, ind, mtype, key, pts_f, agg, quality from your code above.
    ctx = chat.build_context(src, cat, area_label, ind, mtype, key, pts_f, agg, quality)
    chat.render_chat(ctx, pdict)
