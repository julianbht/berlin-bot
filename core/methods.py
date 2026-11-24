import os
import json

import numpy                         as np
import pandas                        as pd
import geopandas                     as gpd
import folium
from folium.plugins                  import HeatMap, MarkerCluster
from branca.colormap                 import LinearColormap
import streamlit                     as st
from streamlit_folium                import st_folium

import core.HelperTools              as ht
import core.chat                     as chat
from config                          import pdict as CFG      # only for CRS and thresholds


# =============================================================================
# 1) Loading
# =============================================================================
@st.cache_data
def load_areas(file, key):
    """Polygons (PLZ or Bezirk) from WKT csv -> GeoDataFrame (one row per area, EPSG:4326)"""
    df  = pd.read_csv("datasets/" + file, sep=";")
    gdf = gpd.GeoDataFrame(df[[key]], geometry=gpd.GeoSeries.from_wkt(df["geometry"]), crs=CFG["crs_geo"])
    # one row per area (if an area consists of several rows, merge their polygons)
    return gdf.dissolve(by=key, as_index=False)[[key, "geometry"]]


@st.cache_data
def load_osm_points(file):
    """OSM csv (written by download_osm.py) -> GeoDataFrame of Points (EPSG:4326)"""
    df = pd.read_csv("datasets/" + file, sep=";", dtype={"addr:postcode": str}, low_memory=False)
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    # x = longitude, y = latitude
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CFG["crs_geo"])


@ht.timer
def preprop_resid(dfr):
    """Preprocessing plz_einwohner.csv: only PLZ and residents, only Berlin range"""
    d = dfr.loc[:, ["plz", "einwohner"]].rename(columns={"plz": "PLZ", "einwohner": "Einwohner"})
    return d[(d["PLZ"] > 10000) & (d["PLZ"] < 14200)].reset_index(drop=True)


# =============================================================================
# 2) Spatial join and category filter
# =============================================================================
def assign_area(gdf_points, gdf_areas, key):
    """Spatial join: adds column <key> (PLZ / Bezirk) to every point. NaN = outside of all areas.
    Points exactly on a border can match twice -> only the first match is kept."""
    joined = gpd.sjoin(gdf_points, gdf_areas[[key, "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]          # border points: keep first match
    joined = joined.drop(columns="index_right")
    if pd.api.types.is_integer_dtype(gdf_areas[key]):                # PLZ: integer, NaN allowed
        joined[key] = joined[key].astype("Int64")
    return joined


def _split_values(series):
    """'italian;pizza' -> ['italian', 'pizza'] (empty list if missing)"""
    return series.fillna("").astype(str).str.split(";").apply(lambda v: [x.strip() for x in v if x.strip()])


def category_options(gdf, col, n=10):
    """The n most frequent values of a (multi-valued) attribute, for the dependent menu"""
    if col not in gdf.columns:
        return []
    values = _split_values(gdf[col]).explode().dropna()
    return values.value_counts().head(n).index.tolist()


def filter_category(gdf, col, category):
    """Keeps objects that have <category> among their values (cuisine=italian;pizza counts for both)"""
    if category == "All" or col not in gdf.columns:
        return gdf
    mask = _split_values(gdf[col]).apply(lambda values: category in values)
    return gdf[mask]


# =============================================================================
# 3) Indicators
# =============================================================================
def residents_by_area(df_res, gdf_plz, gdf_dis, key):
    """Residents per PLZ or per Bezirk (columns <key>, Einwohner)."""
    if key == "PLZ":
        return gdf_plz[["PLZ"]].merge(df_res, on="PLZ", how="left")[["PLZ", "Einwohner"]]

    # Bezirk: APPROXIMATION - every PLZ is assigned completely to the district that contains
    # (or is nearest to) its centre, although some PLZ areas cross district borders.
    centres = gdf_plz.to_crs(CFG["crs_metric"])
    centres["geometry"] = centres.geometry.centroid
    centres = centres.merge(df_res, on="PLZ", how="left")
    districts = gdf_dis.to_crs(CFG["crs_metric"])[["Bezirk", "geometry"]]
    joined = gpd.sjoin_nearest(centres, districts, how="left")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined.groupby("Bezirk", as_index=False)["Einwohner"].sum()


def aggregate(gdf_points, gdf_areas, key, indicator, residents):
    """One row per area (also areas without objects = 0) with columns count, area_km2, Einwohner, value"""
    # start from the polygons -> areas without objects keep count 0
    counts = gdf_points.groupby(key).size().rename("count").reset_index()
    agg = gdf_areas[[key, "geometry"]].merge(counts, on=key, how="left")
    agg["count"] = agg["count"].fillna(0).astype(int)

    # area in square kilometres, computed in metres (never in degrees)
    agg["area_km2"] = agg.to_crs(CFG["crs_metric"]).area / 1e6
    agg = agg.merge(residents, on=key, how="left")

    if indicator == "Number":
        agg["value"] = agg["count"].astype(float)
    elif indicator == "Number per km2":
        agg["value"] = agg["count"] / agg["area_km2"]
    else:   # "Number per 10,000 residents" - not defined for areas with very few residents
        enough = agg["Einwohner"] >= CFG["min_residents"]
        agg["value"] = np.where(enough, agg["count"] / agg["Einwohner"] * 10000, np.nan)
    return agg


# =============================================================================
# 4) Data quality
# =============================================================================
def quality_report(gdf_points, gdf_areas, subcat=None):
    """Automatic checks of one OSM file. Expects column 'PLZ' (from assign_area). Returns DataFrame."""
    n = len(gdf_points)
    if n == 0:
        return pd.DataFrame({"Check": ["Number of objects"], "Result": ["0"]})

    def pct(part, total):
        return "{:.1f} %".format(100 * part / total) if total else "-"

    def missing(col):
        """True where a tag is missing or empty"""
        if col not in gdf_points.columns:
            return pd.Series(True, index=gdf_points.index)
        return gdf_points[col].isna() | (gdf_points[col].astype(str).str.strip() == "")

    rows = [("Number of objects", str(n))]

    # missing tags
    no_name = missing("name")
    rows.append(("Without name", "{} ({})".format(int(no_name.sum()), pct(no_name.sum(), n))))
    if subcat:
        no_sub = missing(subcat)
        rows.append(("Without '" + subcat + "'", "{} ({})".format(int(no_sub.sum()), pct(no_sub.sum(), n))))

    # near-duplicates: same name and closer than dup_distance_m metres (computed in metres)
    named = gdf_points.loc[~no_name, ["name", "geometry"]].to_crs(CFG["crs_metric"])
    buffers = named.copy()
    buffers["geometry"] = buffers.geometry.buffer(CFG["dup_distance_m"])
    pairs = gpd.sjoin(named, buffers, how="inner", predicate="within")
    pairs = pairs[(pairs.index < pairs["index_right"]) & (pairs["name_left"] == pairs["name_right"])]
    rows.append(("Near-duplicates (same name, < {} m)".format(CFG["dup_distance_m"]),
                 "{} pairs".format(len(pairs))))

    # location checks
    outside = gdf_points["PLZ"].isna()
    rows.append(("Outside of all Berlin PLZ areas", "{} ({})".format(int(outside.sum()), pct(outside.sum(), n))))
    lat_ok = gdf_points["lat"].between(*CFG["berlin_lat"])
    lon_ok = gdf_points["lon"].between(*CFG["berlin_lon"])
    implausible = ~(lat_ok & lon_ok)
    rows.append(("Implausible coordinates (outside lat {}-{}, lon {}-{})".format(
                 *CFG["berlin_lat"], *CFG["berlin_lon"]), str(int(implausible.sum()))))

    # addr:postcode compared with the PLZ of the spatial join
    has_pc = ~missing("addr:postcode")
    rows.append(("With addr:postcode", "{} ({})".format(int(has_pc.sum()), pct(has_pc.sum(), n))))
    if has_pc.any():
        pc   = pd.to_numeric(gdf_points.loc[has_pc, "addr:postcode"], errors="coerce")
        join = gdf_points.loc[has_pc, "PLZ"].astype(float)
        same = int((pc == join).sum())
        rows.append(("addr:postcode = PLZ of spatial join", "{} of {} ({})".format(
                     same, int(has_pc.sum()), pct(same, has_pc.sum()))))

    # areas without any object
    empty = int((~gdf_areas["PLZ"].isin(gdf_points["PLZ"].dropna())).sum())
    rows.append(("PLZ areas without any object", "{} of {}".format(empty, len(gdf_areas))))

    return pd.DataFrame(rows, columns=["Check", "Result"])


# =============================================================================
# 5) Map
# =============================================================================
def make_map(gdf_points, gdf_agg, maptype, key="PLZ", colors=("yellow", "red"), caption="Value"):
    """Folium map: choropleth (areas), heatmap or marker clusters (single points)"""
    m = folium.Map(location=[52.52, 13.40], zoom_start=10, tiles="OpenStreetMap")

    if maptype == "Choropleth":
        values = gdf_agg["value"].dropna()
        vmin = float(values.min()) if len(values) else 0.0
        vmax = float(values.max()) if len(values) else 1.0
        if vmax == vmin:                                   # all areas equal -> avoid an empty scale
            vmax = vmin + 1
        cmap = LinearColormap(list(colors), vmin=vmin, vmax=vmax, caption=caption)

        data = gdf_agg[[key, "count", "area_km2", "Einwohner", "value", "geometry"]].copy()
        data["area_km2"] = data["area_km2"].round(2)
        data["value"]    = data["value"].round(2)

        def style(feature):
            v = feature["properties"]["value"]
            return {"fillColor": cmap(v) if v is not None else "lightgrey",   # NaN -> grey
                    "color": "black", "weight": 0.5, "fillOpacity": 0.7}

        folium.GeoJson(data.to_json(), style_function=style,
                       tooltip=folium.GeoJsonTooltip(fields=[key, "count", "Einwohner", "value"],
                                                     aliases=[key, "Objects", "Residents", caption])
                       ).add_to(m)
        cmap.add_to(m)                                     # legend
    elif maptype == "Heatmap":
        HeatMap(gdf_points[["lat", "lon"]].values.tolist(), radius=12).add_to(m)   # [lat, lon]
    else:   # Point map (clusters)
        cluster = MarkerCluster().add_to(m)
        for lat, lon, name in zip(gdf_points["lat"], gdf_points["lon"], gdf_points["name"]):
            label = name if isinstance(name, str) and name.strip() else "(no name)"
            folium.Marker([lat, lon], tooltip=label).add_to(cluster)
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
        st.error("File datasets/" + cfg["file"] + " not found. Run:  python download_osm.py \"" + src + "\"")
        return
    pts, plz, dis, quality = prepare_source(src, pdict)

    # ---- Menu 2: category filter (depends on menu 1) ------------------------
    subcat  = cfg["subcat"]
    options = ["All"] + category_options(pts, subcat, pdict["top_n_category"])
    cat = st.sidebar.selectbox("Category filter (" + subcat + ")", options, key="cat_" + src)

    # ---- Menus 3-5 ----------------------------------------------------------
    area_label = st.sidebar.selectbox("Area unit", list(pdict["areas"].keys()))
    ind        = st.sidebar.selectbox("Indicator", pdict["indicators"])
    mtype      = st.sidebar.selectbox("Map type", pdict["maptypes"])
    # optional menu 6: colour scale (only for the choropleth)
    scale      = st.sidebar.selectbox("Colour scale", list(pdict["colorscales"].keys()),
                                      disabled=(mtype != "Choropleth"))

    # ---- Data for the current selection --------------------------------------
    key   = pdict["areas"][area_label]                     # "PLZ" or "Bezirk"
    areas = plz if key == "PLZ" else dis
    pts_f = filter_category(pts, subcat, cat)
    if len(pts_f) == 0:
        st.warning("No objects for this selection.")
        return
    residents = residents_by_area(df_residents, plz, dis, key)
    agg       = aggregate(pts_f, areas, key, ind, residents)

    # ---- Key figures ----------------------------------------------------------
    n_empty = int((agg["count"] == 0).sum())
    best    = agg.dropna(subset=["value"]).sort_values("value", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Objects shown", len(pts_f))
    c2.metric("Areas without any object", "{} of {}".format(n_empty, len(agg)))
    if len(best):
        c3.metric("Highest value (" + ind + ")",
                  "{} ({:.2f})".format(best.iloc[0][key], best.iloc[0]["value"]))
    if key == "Bezirk" and ind == "Number per 10,000 residents":
        st.caption("Residents per district are approximated: every PLZ is counted in the district of its centre.")

    # ---- Map --------------------------------------------------------------------
    m = make_map(pts_f, agg, mtype, key, pdict["colorscales"][scale], ind)
    st_folium(m, width=1000, height=600, returned_objects=[])
    st.caption("© OpenStreetMap contributors (ODbL) · data downloaded: " + _download_date(pdict, src))

    # ---- Table and data quality ------------------------------------------------
    with st.expander("Top 3 areas and data table"):
        tbl = agg.drop(columns="geometry")[[key, "count", "area_km2", "Einwohner", "value"]]
        tbl = tbl.sort_values("value", ascending=False, na_position="last").round(2)
        st.write("**Top 3:** " + ", ".join("{} ({:.2f})".format(r[key], r["value"])
                                           for _, r in tbl.dropna(subset=["value"]).head(3).iterrows()))
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.download_button("Download CSV", tbl.to_csv(index=False).encode("utf-8"),
                           file_name="osm_" + src.lower().replace(" ", "_") + "_" + key + ".csv",
                           mime="text/csv")
    with st.expander("Data quality"):
        st.dataframe(quality, hide_index=True, width="stretch")

    # ---- Chat assistant (LLM) - provided, see core/chat.py -------------------
    # Needs the variables src, cat, area_label, ind, mtype, key, pts_f, agg, quality from your code above.
    ctx = chat.build_context(src, cat, area_label, ind, mtype, key, pts_f, agg, quality)
    chat.render_chat(ctx, pdict)
