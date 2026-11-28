import datetime
import json
import os

import altair as alt
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
# Darkens only the base map tiles, so the coloured areas and circles keep their colours
_DARK_TILES = (
    ".leaflet-tile-pane { filter: invert(1) hue-rotate(180deg) brightness(.85) "
    "contrast(.9) saturate(.6); } .leaflet-container { background: #0b0b16; }"
)
# Legend of the colour scale (branca draws it as svg text) and its size on phones
_LEGEND = (
    ".legend text, .legend .caption {{ fill: {c} !important; color: {c} !important; }}"
    "@media (max-width: 640px) {{ .legend {{ transform: scale(.62); "
    "transform-origin: top right; }} }}"
)


def _is_dark() -> bool:
    """True when the app shows the dark theme (set in make_streamlit_app)"""
    return st.session_state.get("bb_dark", True)


def _base_map(location: list[float], zoom_start: int) -> folium.Map:
    """Empty folium map with OSM tiles; in dark mode the tiles are darkened"""
    m = folium.Map(location=location, zoom_start=zoom_start, tiles=CFG["map_tiles"])
    dark = _is_dark() and CFG["dark_map"]
    css = (_DARK_TILES if dark else "") + _LEGEND.format(c="#e6e6f5" if dark else "#1d1733")
    m.get_root().header.add_child(folium.Element(f"<style>{css}</style>"))
    return m


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
        fill = "#3a3a55" if value is None else colormap(value)
        return {"fillColor": fill, "color": "#55557a", "weight": 0.5, "fillOpacity": 0.75}

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
    m = _base_map([52.52, 13.40], 10)
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


CHANGE_COLORS: dict[str, str] = CFG["change_colors"]


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
    m = _base_map([52.51, 13.40], 11)
    # Grey first, so that the coloured circles lie on top
    for status in ["still there", "gone", "new"]:
        gdf = groups[status]
        color = CHANGE_COLORS[status]
        radius = 5 if status == "still there" else 7
        for lat, lon, name in zip(gdf.geometry.y, gdf.geometry.x, gdf["name"], strict=True):
            label = status if pd.isna(name) else f"{name} ({status})"
            folium.CircleMarker(
                [lat, lon],
                radius=radius,
                color=color,
                weight=2,
                fill=True,
                fill_opacity=0.75,
                tooltip=label,
            ).add_to(m)
    return m


def change_by_area(
    groups: dict[str, gpd.GeoDataFrame], gdf_areas: gpd.GeoDataFrame, key: str
) -> gpd.GeoDataFrame:
    """One row per area with the number of objects then, now, gone, new and the net change"""
    out = gdf_areas[[key, "geometry"]].copy()
    for status in ["gone", "new", "still there"]:
        keys = groups[status][key].dropna().astype(gdf_areas[key].dtype)
        counts = keys.value_counts().rename(status)
        out = out.merge(counts, left_on=key, right_index=True, how="left")
        out[status] = out[status].fillna(0).astype(int)
    out["then"] = out["gone"] + out["still there"]
    out["now"] = out["new"] + out["still there"]
    out["net"] = out["now"] - out["then"]
    return out


def make_net_change_map(gdf_change: gpd.GeoDataFrame, key: str, year: int) -> folium.Map:
    """Choropleth of the net change per area on a diverging scale: red = loss, green = gain"""
    m = _base_map([52.51, 13.40], 10)
    limit = max(int(gdf_change["net"].abs().max()), 1)  # symmetric, so 0 is the middle colour
    loss, middle, gain = CFG["net_change_colors"]
    if not _is_dark():
        middle = "#efeaf7"  # the dark middle colour looks muddy on the light map
    colormap = LinearColormap(
        [loss, middle, gain],
        vmin=-limit,
        vmax=limit,
        caption=f"Net change of clubs since {year}",
    )

    def style(feature: dict) -> dict:
        net = feature["properties"]["net"]
        return {"fillColor": colormap(net), "color": "#55557a", "weight": 0.6, "fillOpacity": 0.8}

    shown = gdf_change[[key, "then", "now", "gone", "new", "net", "geometry"]]
    folium.GeoJson(
        shown.to_json(),
        style_function=style,
        tooltip=folium.GeoJsonTooltip(
            fields=[key, "then", "now", "gone", "new", "net"],
            aliases=[key, f"Clubs {year}", "Clubs today", "Gone", "New", "Net change"],
        ),
    ).add_to(m)
    colormap.add_to(m)
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


# ---- Look and feel ----------------------------------------------------------
# Colours and fonts of the whole app come from .streamlit/config.toml. This CSS only adds the
# header, the cards and the mobile layout that the theme cannot do.
_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

.block-container { padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1250px; }

/* Header with gradient title */
.bb-hero {
    padding: 1.6rem 1.8rem; border-radius: 20px; margin-bottom: 1.2rem;
    background:
        radial-gradient(circle at 10% 0%, var(--bb-glow-pink), transparent 45%),
        radial-gradient(circle at 90% 100%, var(--bb-glow-cyan), transparent 45%),
        var(--bb-hero-bg);
    border: 1px solid var(--bb-line);
}
.bb-hero .bb-kicker {
    font-size: .8rem; letter-spacing: .14em; text-transform: uppercase;
    color: var(--bb-cyan); font-weight: 600;
}
.bb-hero h1 {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: 2.6rem; line-height: 1.1; margin: .3rem 0 .5rem 0; padding: 0;
    display: inline-block;
    background: var(--bb-title);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.bb-hero p { color: var(--bb-muted); font-size: 1.05rem; margin: 0; }

/* Key figure cards */
div[data-testid="stMetric"] {
    background: var(--bb-card); box-shadow: var(--bb-shadow);
    border-radius: 16px; padding: .9rem 1rem;
}
div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }

/* Legend chips above the map */
.bb-legend { display: flex; flex-wrap: wrap; gap: .5rem; margin: .2rem 0 .8rem 0; }
.bb-chip {
    display: inline-flex; align-items: center; gap: .45rem; padding: .3rem .8rem;
    border-radius: 999px; font-size: .9rem; background: var(--bb-chip);
    border: 1px solid var(--bb-line);
}
.bb-dot { width: .7rem; height: .7rem; border-radius: 50%; display: inline-block; }

/* Callout boxes */
.bb-note {
    padding: .8rem 1rem; border-radius: 12px; margin: .4rem 0 1rem 0;
    background: var(--bb-note); border-left: 4px solid var(--bb-cyan); color: var(--bb-text);
}

/* Rounded map */
iframe { border-radius: 16px; }

/* Phones: smaller header, less padding, tabs scroll sideways */
@media (max-width: 640px) {
    .block-container { padding: 4.2rem .8rem 2rem .8rem; }
    .bb-hero { padding: 1.1rem 1.1rem; border-radius: 16px; }
    .bb-hero h1 { font-size: 1.8rem; }
    .bb-hero p { font-size: .95rem; }
    div[data-testid="stMetric"] { padding: .7rem .8rem; }
    /* Key figures as a 2 x 2 grid instead of four rows */
    .st-key-kpis div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .6rem; }
    .st-key-kpis div[data-testid="stColumn"] {
        flex: 1 1 calc(50% - .6rem) !important; min-width: calc(50% - .6rem) !important;
    }
    div[data-baseweb="tab-list"] { overflow-x: auto; flex-wrap: nowrap; }
}
</style>
"""

# Colours of the CSS above, one set per theme
_THEME_VARS = {
    "dark": {
        "hero-bg": "linear-gradient(135deg, #1c1236 0%, #12122a 100%)",
        "glow-pink": "rgba(247,37,133,.35)",
        "glow-cyan": "rgba(76,201,240,.30)",
        "title": "linear-gradient(90deg, #ff2e88, #c77dff 55%, #4cc9f0)",
        "cyan": "#4cc9f0",
        "text": "#d8d8ee",
        "muted": "#c9c9e3",
        "line": "rgba(255,255,255,.09)",
        "card": "linear-gradient(160deg, rgba(255,255,255,.06), rgba(255,255,255,.02))",
        "shadow": "none",
        "chip": "rgba(255,255,255,.06)",
        "note": "rgba(76,201,240,.08)",
    },
    "light": {
        "hero-bg": "linear-gradient(135deg, #ffffff 0%, #f6f0ff 100%)",
        "glow-pink": "rgba(247,37,133,.18)",
        "glow-cyan": "rgba(76,201,240,.22)",
        "title": "linear-gradient(90deg, #e0176f, #8a2be2 55%, #0b8fbf)",
        "cyan": "#0b8fbf",
        "text": "#2a2445",
        "muted": "#4a4466",
        "line": "rgba(60,30,120,.10)",
        "card": "#ffffff",
        "shadow": "0 4px 18px rgba(80,40,160,.08)",
        "chip": "#ffffff",
        "note": "rgba(11,143,191,.08)",
    },
}


def _style(dark: bool) -> str:
    """The CSS of the app with the colour variables of the current theme"""
    colors = _THEME_VARS["dark" if dark else "light"]
    variables = " ".join(f"--bb-{name}: {value};" for name, value in colors.items())
    return _STYLE.replace("<style>", f"<style>:root {{ {variables} }}", 1)


def _theme_toggle() -> bool:
    """Dark mode switch in the sidebar. Streamlit has no function to change the theme from
    Python, so the switch stores the choice where Streamlit keeps it (the same as the menu
    Settings > Theme) and reloads the page. Returns True for dark mode."""
    dark = st.context.theme.type != "light"  # None on the very first run: dark is the default
    wanted = st.sidebar.toggle("🌙 Dark mode", value=dark)
    if wanted != dark:
        choice = "Dark" if wanted else "Light"
        st.html(
            "<script>"
            "const key = `stActiveTheme-${window.location.pathname}-v2`;"
            f"window.localStorage.setItem(key, JSON.stringify('{choice}'));"
            "window.location.reload();"
            "</script>",
            unsafe_allow_javascript=True,
        )
    return dark


def _hero(kicker: str, title: str, text: str) -> None:
    """Header card with a gradient title"""
    st.html(
        f"<div class='bb-hero'><div class='bb-kicker'>{kicker}</div>"
        f"<h1>{title}</h1><p>{text}</p></div>"
    )


def _legend(items: list[tuple[str, str]]) -> None:
    """Row of coloured chips, items = [(colour, text), ...]"""
    chips = "".join(
        f"<span class='bb-chip'><span class='bb-dot' style='background:{color}'></span>"
        f"{text}</span>"
        for color, text in items
    )
    st.html(f"<div class='bb-legend'>{chips}</div>")


def _note(text: str) -> None:
    st.html(f"<div class='bb-note'>{text}</div>")


def _show_map(m: folium.Map, key: str) -> None:
    """Folium map that fills the width of the page, on desktop and on phones"""
    st_folium(m, height=560, use_container_width=True, returned_objects=[], key=key)


@ht.timer
def make_streamlit_app(pdict: dict, df_residents: pd.DataFrame) -> None:
    """Makes Streamlit App: OSM data of Berlin.

    By default the app opens in the focused view for the Part B use case (Berlin nightclubs over
    time). The switch "Explore all data sources" in the sidebar shows the full Part A dashboard."""
    st.set_page_config(
        page_title=pdict["focus_title"] + " · OpenStreetMap Berlin",
        page_icon="🪩",
        layout="wide",
    )
    st.sidebar.markdown(f"## 🪩 {pdict['focus_title']}")
    dark = _theme_toggle()
    st.session_state["bb_dark"] = dark
    st.html(_style(dark))
    explore = st.sidebar.toggle(
        "Explore all data sources",
        value=False,
        help="Shows all OSM sources (restaurants, pharmacies, ...) and all menus of the dashboard.",
    )
    st.sidebar.divider()

    if explore:
        _explore_view(pdict, df_residents)
    else:
        _focus_view(pdict, df_residents)


def _trend_chart(trend_df: pd.DataFrame, year: int) -> alt.Chart:
    """Clubs per year as a glowing line; the y axis starts near the data, not at 0, so the
    change is visible. The chosen starting year is marked."""
    base = alt.Chart(trend_df).encode(
        x=alt.X("Year:O", axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y("Clubs:Q", scale=alt.Scale(zero=False, nice=True), title="Clubs"),
        tooltip=["Year", "Clubs"],
    )
    gradient = alt.Gradient(
        gradient="linear",
        stops=[
            alt.GradientStop(color="rgba(255,46,136,0.45)", offset=0),
            alt.GradientStop(color="rgba(255,46,136,0.0)", offset=1),
        ],
        x1=1,
        x2=1,
        y1=0,
        y2=1,
    )
    area = base.mark_area(color=gradient, interpolate="monotone")
    line = base.mark_line(color="#ff2e88", strokeWidth=3, interpolate="monotone")
    points = base.mark_circle(size=70, color="#4cc9f0" if _is_dark() else "#0b8fbf", opacity=1)
    start = (
        alt.Chart(trend_df[trend_df["Year"] == year])
        .mark_rule(color="#ffd60a", strokeDash=[4, 4], strokeWidth=2)
        .encode(x="Year:O")
    )
    return (area + line + points + start).properties(height=360)


# ---- Focused view: nightclubs over time (Part B) -------------------------------
def _focus_view(pdict: dict, df_residents: pd.DataFrame) -> None:
    """Answers one question: how has the number of clubs changed, and where?"""
    src = pdict["focus_source"]
    cfg = pdict["sources"][src]
    if not os.path.exists("datasets/" + cfg["history"]["file"]):
        st.error(f'History of {src} not found. Run:  uv run download-osm "{src}"')
        return

    # Counts per year from the history file. The snapshot of 1 January of this year is almost
    # the same as today, so it is not offered as a starting year.
    history = load_osm_points(cfg["history"]["file"])
    per_year = history.groupby("year").size()
    this_year = datetime.date.today().year
    years = [int(y) for y in per_year.index if int(y) < this_year]

    # ---- Only two menus: starting year and area unit ----------------------------
    year = st.sidebar.select_slider(
        "Compare today with",
        years,
        value=years[0],
        help="State of OpenStreetMap on 1 January of that year.",
    )
    area_label = st.sidebar.segmented_control(
        "Show changes per",
        list(pdict["areas"].keys()),
        default=pdict["default_area"],
        format_func=lambda a: pdict["area_labels"][a],
        required=True,
        key="focus_area",
    )
    st.sidebar.caption(
        "Tip: switch on *Explore all data sources* above for the other places and menus."
    )

    pts_then, plz, dis, _ = prepare_source(src, pdict, year)
    pts_now, _, _, quality = prepare_source(src, pdict)
    groups = compare_points(pts_then, pts_now)
    key = pdict["areas"][area_label]
    areas = plz if key == "PLZ" else dis
    change = change_by_area(groups, areas, key)
    n_then, n_now = len(pts_then), len(pts_now)
    n_gone, n_new = len(groups["gone"]), len(groups["new"])

    _hero(
        "OpenStreetMap · Berlin nightlife",
        pdict["focus_title"],
        pdict["focus_question"].format(year=year),
    )

    # ---- Key figures --------------------------------------------------------------
    c1, c2, c3, c4 = st.container(key="kpis").columns(4)
    c1.metric(
        "Clubs today",
        n_now,
        delta=f"{n_now - n_then:+d} since {year}",
        border=True,
    )
    c2.metric(f"Clubs on 1 Jan {year}", n_then, border=True)
    c3.metric(
        "Gone",
        n_gone,
        delta=f"{-n_gone} places",
        border=True,
        help="In OSM in that year, not in OSM today (closed, or removed from the map).",
    )
    c4.metric(
        "New",
        n_new,
        delta=f"+{n_new} places",
        border=True,
        help="In OSM today, not in OSM in that year (opened, or mapped later).",
    )

    tab_map, tab_trend, tab_area, tab_data = st.tabs(
        ["🗺️ Map", "📈 Trend", f"🏙️ Per {pdict['area_labels'][area_label].lower()[:-1]}", "🔎 Data"]
    )

    # ---- Map: every club, coloured by what happened to it ---------------------
    with tab_map:
        _legend(
            [
                (CHANGE_COLORS["gone"], f"Gone since {year} · {n_gone}"),
                (CHANGE_COLORS["new"], f"New since {year} · {n_new}"),
                (CHANGE_COLORS["still there"], f"Still there · {len(groups['still there'])}"),
            ]
        )
        _show_map(make_change_map(groups), key="focus_map")

    # ---- Trend: clubs per year --------------------------------------------------
    with tab_trend:
        peak_year = int(per_year.idxmax())
        peak = int(per_year.max())
        _note(
            f"<b>Peak:</b> {peak} clubs in {peak_year}. "
            f"<b>Today:</b> {n_now} clubs, that is {n_now - peak:+d} compared with the peak."
        )
        trend_df = pd.DataFrame({"Year": per_year.index.astype(int), "Clubs": per_year.values})
        st.altair_chart(_trend_chart(trend_df, year), use_container_width=True)
        st.caption(
            "Number of nightclubs in OpenStreetMap on 1 January of each year. "
            f"The dashed line marks {year}."
        )

    # ---- Per area: where clubs were lost or gained --------------------------------
    with tab_area:
        losers = change.sort_values("net").head(1)
        winners = change.sort_values("net", ascending=False).head(1)
        if len(losers) and losers["net"].iloc[0] < 0:
            text = f"<b>Biggest loss:</b> {losers[key].iloc[0]} ({losers['net'].iloc[0]:+d})"
            if winners["net"].iloc[0] > 0:
                text += (
                    f" &nbsp;·&nbsp; <b>Biggest gain:</b> {winners[key].iloc[0]}"
                    f" ({winners['net'].iloc[0]:+d})"
                )
            _note(text)
        _show_map(make_net_change_map(change, key, year), key="focus_net_map")

        table = (
            pd.DataFrame(change.drop(columns="geometry"))
            .loc[lambda d: (d["then"] > 0) | (d["now"] > 0)]
            .sort_values("net")
        )
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_order=[key, "then", "now", "gone", "new", "net"],
            column_config={
                key: st.column_config.TextColumn(pdict["area_labels"][area_label][:-1]),
                "then": st.column_config.NumberColumn(f"Clubs {year}"),
                "now": st.column_config.NumberColumn("Clubs today"),
                "gone": st.column_config.NumberColumn("Gone"),
                "new": st.column_config.NumberColumn("New"),
                "net": st.column_config.NumberColumn("Net change", format="%+d"),
            },
        )

    # ---- Data: names, download, quality ---------------------------------------------
    with tab_data:
        col_gone, col_new = st.columns(2)
        for column, status in [(col_gone, "gone"), (col_new, "new")]:
            names = groups[status][["name", key]].dropna(subset=["name"]).sort_values("name")
            column.markdown(f"**{status.capitalize()} since {year}** ({len(groups[status])})")
            column.dataframe(names, hide_index=True, use_container_width=True, height=280)
        st.download_button(
            "Download table per area (CSV)",
            table.to_csv(index=False, sep=";"),
            file_name=f"nightclubs_change_{year}_{key}.csv".lower(),
            mime="text/csv",
        )
        with st.expander("Data quality (clubs today)"):
            st.dataframe(quality, hide_index=True, use_container_width=True)

    st.caption(
        "© OpenStreetMap contributors · data downloaded on "
        + _download_date(pdict, src)
        + " · OSM history shows when places were mapped or removed in OSM, which can lag behind"
        " real openings and closings."
    )

    # ---- Chat assistant: the table it gets holds the net change per area ------------------
    residents = residents_by_area(df_residents, plz, dis, key)
    agg = aggregate(pts_now, areas, key, "Number", residents)
    agg["value"] = agg[key].map(change.set_index(key)["net"]).astype(float)
    ctx = chat.build_context(
        f"{src}: change from 1 January {year} to today",
        "All",
        area_label,
        f"Net change of clubs since {year} (clubs today minus clubs in {year}); "
        f"in total {n_then} then, {n_now} today, {n_gone} gone, {n_new} new",
        "Change map",
        key,
        pts_now,
        agg,
        quality,
    )
    chat.render_chat(ctx, pdict)


# ---- Full dashboard: all sources and menus (Part A) ---------------------------
def _explore_view(pdict: dict, df_residents: pd.DataFrame) -> None:
    """The Part A dashboard with plain-language labels"""
    # ---- Menu 1: data source ------------------------------------------------
    src = st.sidebar.selectbox("Places", list(pdict["sources"].keys()))
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
    cat = st.sidebar.selectbox(f"Filter by {subcat}", options, key="cat_" + src)

    # ---- Menus 3-6 ----------------------------------------------------------
    area_names = list(pdict["areas"].keys())
    area_label = st.sidebar.selectbox(
        "Area unit",
        area_names,
        index=area_names.index(pdict["default_area"]),
        format_func=lambda a: pdict["area_labels"][a],
    )
    ind = st.sidebar.selectbox(
        "Measure", pdict["indicators"], format_func=lambda i: pdict["indicator_labels"][i]
    )
    if compare:
        mtype = f"Change map ({year} compared with today)"
    else:
        mtype = st.sidebar.selectbox(
            "Map type", pdict["maptypes"], format_func=lambda t: pdict["maptype_labels"][t]
        )
    colors = list(pdict["colorscales"].values())[0]
    if mtype == "Choropleth":
        scale = st.sidebar.selectbox("Colour scale", list(pdict["colorscales"].keys()))
        colors = pdict["colorscales"][scale]

    key = pdict["areas"][area_label]
    areas = plz if key == "PLZ" else dis
    pts_f = filter_category(pts, subcat, cat)
    _hero(
        "OpenStreetMap · Berlin",
        src_shown,
        f"{pdict['indicator_labels'][ind]} per {pdict['area_labels'][area_label].lower()[:-1]}"
        + ("" if cat == "All" else f" · {subcat} = {cat}"),
    )
    if pts_f.empty:
        st.warning(f"No objects for {src} with {subcat} = {cat}.")
        return
    residents = residents_by_area(df_residents, plz, dis, key)
    agg = aggregate(pts_f, areas, key, ind, residents)

    # ---- Key figures --------------------------------------------------------
    col_objects, col_empty, col_top = st.columns(3)
    col_objects.metric("Places shown", len(pts_f), border=True)
    empty_areas = int((agg["count"] == 0).sum())
    col_empty.metric("Areas without any place", f"{empty_areas} of {len(agg)}", border=True)
    if agg["value"].notna().any():
        top = agg.loc[agg["value"].idxmax()]
        value_text = f"{top['value']:.0f}" if ind == "Number" else f"{top['value']:.2f}"
        col_top.metric("Highest value", f"{top[key]}: {value_text}", border=True)
    else:
        col_top.metric("Highest value", "no value", border=True)

    # ---- Map ----------------------------------------------------------------
    if compare:
        pts_now = filter_category(prepare_source(src, pdict)[0], subcat, cat)
        groups = compare_points(pts_f, pts_now)
        _legend(
            [
                (CHANGE_COLORS[status], f"{status.capitalize()} since {year} · {len(gdf)}")
                for status, gdf in groups.items()
            ]
        )
        m = make_change_map(groups)
    else:
        m = make_map(
            pts_f, agg, mtype, key=key, colors=colors, caption=pdict["indicator_labels"][ind]
        )
    _show_map(m, key="map")
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
