import json
import os

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

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


def category_options(gdf: gpd.GeoDataFrame, col: str, n: int = 10) -> list[str]:
    """The n most frequent values of a (multi-valued) attribute, for the dependent menu"""
    if col not in gdf.columns:
        return []
    values = _split_values(gdf[col]).explode().dropna()
    return values.value_counts().head(n).index.tolist()


def filter_category(gdf: gpd.GeoDataFrame, col: str, category: str) -> gpd.GeoDataFrame:
    """Keeps objects that have <category> among their values
    (cuisine=italian;pizza counts for both)"""
    if category == "All":
        return gdf
    has_category = _split_values(gdf[col]).apply(lambda values: category in values)
    return gdf[has_category]


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
    """One row per area (also areas without objects = 0)
    with columns count, area_km2, Einwohner, value"""
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
    """Automatic checks of one OSM file. Expects column 'PLZ' (from assign_area).
    Returns DataFrame."""
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
def _choropleth(
    m: folium.Map,
    gdf_agg: gpd.GeoDataFrame,
    key: str,
    colors: list[str] | tuple[str, ...],
    caption: str,
) -> None:
    """One GeoJson layer coloured by the column "value", plus the legend"""
    values = gdf_agg["value"].dropna()
    vmin = float(values.min()) if len(values) else 0.0
    vmax = float(values.max()) if len(values) else 1.0
    if vmax <= vmin:  # one distinct value: the colormap needs a range
        vmax = vmin + 1
    colormap = LinearColormap(list(colors), vmin=vmin, vmax=vmax, caption=caption)

    def style(feature: dict) -> dict:
        value = feature["properties"]["value"]
        fill = "lightgrey" if value is None else colormap(value)
        return {"fillColor": fill, "color": "grey", "weight": 0.5, "fillOpacity": 0.7}

    shown = gdf_agg[[key, "count", "value", "geometry"]].copy()
    shown["value"] = shown["value"].round(2)
    folium.GeoJson(
        shown.to_json(),
        style_function=style,
        tooltip=folium.GeoJsonTooltip(
            fields=[key, "count", "value"], aliases=[key, "Objects", caption]
        ),
    ).add_to(m)
    colormap.add_to(m)


def make_map(
    gdf_points: gpd.GeoDataFrame,
    gdf_agg: gpd.GeoDataFrame,
    maptype: str,
    key: str = "PLZ",
    colors: list[str] | tuple[str, ...] = ("yellow", "red"),
    caption: str = "Value",
) -> folium.Map:
    """Folium map: choropleth (areas), heatmap or marker clusters (single points)"""
    m = folium.Map(location=[52.52, 13.40], zoom_start=10, tiles="OpenStreetMap")
    # Folium wants [lat, lon], GeoPandas stores (x = lon, y = lat)
    lat_lon = list(zip(gdf_points.geometry.y, gdf_points.geometry.x, strict=True))

    if maptype == "Choropleth":
        _choropleth(m, gdf_agg, key, colors, caption)
    elif maptype == "Heatmap":
        HeatMap(lat_lon).add_to(m)
    elif maptype == "Point map (clusters)":
        cluster = MarkerCluster().add_to(m)
        for location, name in zip(lat_lon, gdf_points["name"], strict=True):
            tooltip = None if pd.isna(name) else str(name)
            folium.Marker(location, tooltip=tooltip).add_to(cluster)
    else:
        raise ValueError(f"Unknown map type: {maptype}")
    return m


CHANGE_COLORS = {"gone": "#d62728", "new": "#2ca02c", "still there": "#7f7f7f"}


def compare_points(
    pts_then: gpd.GeoDataFrame, pts_now: gpd.GeoDataFrame
) -> dict[str, gpd.GeoDataFrame]:
    """Splits objects into gone (only then), new (only now) and still there (both), by OSM id.
    An object that a mapper deleted and drew again gets a new id and counts as gone and new."""
    ids_then = set(zip(pts_then["osm_type"], pts_then["osm_id"], strict=True))
    ids_now = set(zip(pts_now["osm_type"], pts_now["osm_id"], strict=True))
    in_now = [i in ids_now for i in zip(pts_then["osm_type"], pts_then["osm_id"], strict=True)]
    in_then = [i in ids_then for i in zip(pts_now["osm_type"], pts_now["osm_id"], strict=True)]
    return {
        "gone": pts_then[[not x for x in in_now]],
        "new": pts_now[[not x for x in in_then]],
        "still there": pts_now[in_then],
    }


def make_change_map(groups: dict[str, gpd.GeoDataFrame]) -> folium.Map:
    """One coloured circle per object: gone, new or still there"""
    m = folium.Map(location=[52.52, 13.40], zoom_start=11, tiles="OpenStreetMap")
    # Grey first, so that the coloured circles lie on top
    for status in ["still there", "gone", "new"]:
        gdf = groups[status]
        color = CHANGE_COLORS[status]
        for lat, lon, name in zip(gdf.geometry.y, gdf.geometry.x, gdf["name"], strict=True):
            label = status if pd.isna(name) else f"{name} ({status})"
            folium.CircleMarker(
                [lat, lon], radius=6, color=color, fill=True, fill_opacity=0.8, tooltip=label
            ).add_to(m)
    return m


# =============================================================================
# 6) Streamlit app
# =============================================================================
@st.cache_data(show_spinner="Loading data and assigning areas ...")
def prepare_source(label, pdict, year: int | None = None):
    """Loads one OSM source, assigns PLZ and Bezirk (spatial join) and runs the quality check.
    With a year: the objects on 1 January of that year, from the history file of the source."""
    cfg = pdict["sources"][label]
    plz = load_areas(pdict["file_plz"], "PLZ")
    dis = load_areas(pdict["file_dis"], "Bezirk")
    if year is None:
        pts = load_osm_points(cfg["file"])
    else:
        pts = load_osm_points(cfg["history"]["file"])
        pts = pts[pts["year"] == year].reset_index(drop=True)
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
def make_streamlit_app(pdict: dict, df_residents: pd.DataFrame) -> None:
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

    # ---- Year and comparison (only sources with history) --------------------
    year: int | None = None
    compare = False
    if "history" in cfg:
        history_years = sorted(load_osm_points(cfg["history"]["file"])["year"].unique())
        chosen = st.sidebar.select_slider(
            "Year (state on 1 January)",
            [*[int(y) for y in history_years], "Today"],
            value="Today",
            key=f"year_{src}",
        )
        if chosen != "Today":
            year = int(chosen)
            compare = st.sidebar.checkbox(
                "Compare with today", help="Shows which places are gone, new or still there"
            )
    pts, plz, dis, quality = prepare_source(src, pdict, year)
    src_shown = src if year is None else f"{src} on 1 January {year}"

    # ---- Menu 2: category filter (depends on menu 1) ------------------------
    subcat = cfg["subcat"]
    options = ["All"] + category_options(pts, subcat, pdict["top_n_category"])
    # The key makes the menu reset to "All" when the data source changes
    cat = st.sidebar.selectbox(f"Category filter ({subcat})", options, key="cat_" + src)

    # ---- Menus 3-6 ----------------------------------------------------------
    area_label = st.sidebar.selectbox("Area unit", list(pdict["areas"].keys()))
    ind = st.sidebar.selectbox("Indicator", pdict["indicators"])
    if compare:
        mtype = f"Change map ({year} compared with today)"
    else:
        mtype = st.sidebar.selectbox("Map type", pdict["maptypes"])
    colors = list(pdict["colorscales"].values())[0]
    if mtype == "Choropleth":
        scale = st.sidebar.selectbox("Colour scale", list(pdict["colorscales"].keys()))
        colors = pdict["colorscales"][scale]

    key = pdict["areas"][area_label]
    areas = plz if key == "PLZ" else dis
    pts_f = filter_category(pts, subcat, cat)
    if pts_f.empty:
        st.warning(f"No objects for {src} with {subcat} = {cat}.")
        return
    residents = residents_by_area(df_residents, plz, dis, key)
    agg = aggregate(pts_f, areas, key, ind, residents)

    # ---- Key figures --------------------------------------------------------
    col_objects, col_empty, col_top = st.columns(3)
    col_objects.metric("Objects shown", len(pts_f))
    empty_areas = int((agg["count"] == 0).sum())
    col_empty.metric("Areas without any object", f"{empty_areas} of {len(agg)}")
    if agg["value"].notna().any():
        top = agg.loc[agg["value"].idxmax()]
        value_text = f"{top['value']:.0f}" if ind == "Number" else f"{top['value']:.2f}"
        col_top.metric(f"Highest value ({ind})", f"{top[key]}: {value_text}")
    else:
        col_top.metric(f"Highest value ({ind})", "no value")

    # ---- Map ----------------------------------------------------------------
    if compare:
        pts_now = filter_category(prepare_source(src, pdict)[0], subcat, cat)
        groups = compare_points(pts_f, pts_now)
        columns = st.columns(len(groups))
        for column, (status, gdf) in zip(columns, groups.items(), strict=True):
            column.metric(f"{status.capitalize()} since {year}", len(gdf))
        m = make_change_map(groups)
        legend = " · ".join(
            f"<span style='color:{color}'>●</span> {status}"
            for status, color in CHANGE_COLORS.items()
        )
        st.markdown(legend, unsafe_allow_html=True)
    else:
        m = make_map(pts_f, agg, mtype, key=key, colors=colors, caption=ind)
    st_folium(m, width=1000, height=600, returned_objects=[])
    caption = "© OpenStreetMap contributors · data downloaded on " + _download_date(pdict, src)
    if year is not None:
        caption += (
            " · OSM history shows when places were mapped or removed in OSM,"
            " which can lag behind real openings and closings"
        )
    st.caption(caption)

    # ---- Tables -------------------------------------------------------------
    table = pd.DataFrame(agg.drop(columns="geometry")).sort_values(
        "value", ascending=False, na_position="last"
    )
    with st.expander("Top 3 areas and data table"):
        st.dataframe(table.head(3), hide_index=True)
        st.dataframe(table, hide_index=True)
        st.download_button(
            "Download table (CSV)",
            table.to_csv(index=False, sep=";"),
            file_name=f"{src_shown}_{key}.csv".replace(" ", "_").lower(),
            mime="text/csv",
        )
    with st.expander("Data quality"):
        st.caption("Checks over the whole data source, without the category filter.")
        st.dataframe(quality, hide_index=True)

    # ---- Chat assistant (LLM) - provided, see core/chat.py -------------------
    # Needs the variables src, cat, area_label, ind, mtype, key, pts_f, agg, quality
    # from your code above.
    ctx = chat.build_context(src_shown, cat, area_label, ind, mtype, key, pts_f, agg, quality)
    chat.render_chat(ctx, pdict)
