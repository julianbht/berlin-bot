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
    # TODO: read "datasets/" + file (sep=";"), geometry via gpd.GeoSeries.from_wkt, set the CRS,
    #       keep only column <key> + geometry, make sure every area appears only once (dissolve)
    df = pd.read_csv("datasets/" + file, sep=";", dtype={key: str})
    geometry = gpd.GeoSeries.from_wkt(df["geometry"], crs=CFG["crs_geo"])
    areas = gpd.GeoDataFrame(df[[key]].copy(), geometry=geometry)
    return areas.dissolve(by=key, as_index=False)


@st.cache_data
def load_osm_points(file):
    """OSM csv (written by download_osm.py) -> GeoDataFrame of Points (EPSG:4326)"""
    # TODO: read the csv (sep=";", keep "addr:postcode" as string), drop rows without lat / lon,
    #       geometry = gpd.points_from_xy(lon, lat)   <- order: x = longitude, y = latitude!
    df = pd.read_csv("datasets/" + file, sep=";", dtype={"addr:postcode": str})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CFG["crs_geo"]
    )


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
    # TODO: gpd.sjoin(..., how="left", predicate="within"), remove duplicated index entries,
    #       drop the helper column "index_right"
    points = gdf_points.drop(columns=[key], errors="ignore")
    joined = gpd.sjoin(points, gdf_areas[[key, "geometry"]], how="left", predicate="within")
    joined = joined.loc[~joined.index.duplicated(keep="first")]
    return joined.drop(columns=["index_right"], errors="ignore")


def _split_values(series):
    """'italian;pizza' -> ['italian', 'pizza'] (empty list if missing)"""
    return series.fillna("").astype(str).str.split(";").apply(lambda v: [x.strip() for x in v if x.strip()])


def category_options(gdf, col, n=10):
    """The n most frequent values of a (multi-valued) attribute, for the dependent menu"""
    # TODO: use _split_values, explode, value_counts. Return [] if the column does not exist.
    if col not in gdf.columns:
        return []
    values = _split_values(gdf[col]).explode().dropna()
    return values.value_counts().head(n).index.tolist()


def filter_category(gdf, col, category):
    """Keeps objects that have <category> among their values (cuisine=italian;pizza counts for both)"""
    # TODO: "All" -> return gdf unchanged
    if category == "All" or col not in gdf.columns:
        return gdf
    return gdf.loc[_split_values(gdf[col]).apply(lambda values: category in values)].copy()


# =============================================================================
# 3) Indicators
# =============================================================================
def residents_by_area(df_res, gdf_plz, gdf_dis, key):
    """Residents per PLZ or per Bezirk (columns <key>, Einwohner)."""
    # TODO: key == "PLZ": merge the PLZ polygons with df_res.
    #       key == "Bezirk": assign every PLZ (by its centre, computed in CFG["crs_metric"]) to a Bezirk
    #       (gpd.sjoin_nearest) and sum the residents per Bezirk. Write down this approximation!
    counts = df_res[["PLZ", "Einwohner"]].copy()
    counts["PLZ"] = counts["PLZ"].astype(str).str.zfill(5)
    counts["Einwohner"] = pd.to_numeric(counts["Einwohner"], errors="coerce")
    counts = counts.groupby("PLZ", as_index=False)["Einwohner"].sum()

    if key == "PLZ":
        return gdf_plz[["PLZ"]].merge(counts, on="PLZ", how="left")

    plz_metric = gdf_plz.to_crs(CFG["crs_metric"])
    centers = gpd.GeoDataFrame(
        plz_metric[["PLZ"]].copy(), geometry=plz_metric.geometry.centroid,
        crs=CFG["crs_metric"]
    )
    districts = gdf_dis[["Bezirk", "geometry"]].to_crs(CFG["crs_metric"])
    assigned = gpd.sjoin_nearest(centers, districts, how="left")
    assigned = assigned.loc[~assigned.index.duplicated(keep="first")]
    assigned = assigned.drop(columns=["index_right", "geometry"], errors="ignore")
    result = assigned.merge(counts, on="PLZ", how="left")
    totals = result.groupby("Bezirk", dropna=True)["Einwohner"].sum(min_count=1).reset_index()
    return gdf_dis[["Bezirk"]].merge(totals, on="Bezirk", how="left")


def aggregate(gdf_points, gdf_areas, key, indicator, residents):
    """One row per area (also areas without objects = 0) with columns count, area_km2, Einwohner, value"""
    # TODO: count points per area; start from the POLYGONS (left merge) so that empty areas get 0
    #       area_km2: area in CFG["crs_metric"] / 1e6
    #       value   : "Number" | "Number per km2" | "Number per 10,000 residents"
    #                 (no value, NaN, if Einwohner < CFG["min_residents"])
    area_counts = gdf_points.groupby(key).size().rename("count").reset_index()
    agg = gdf_areas[[key, "geometry"]].merge(area_counts, on=key, how="left")
    agg["count"] = agg["count"].fillna(0).astype(int)
    agg["area_km2"] = agg.to_crs(CFG["crs_metric"]).geometry.area / 1e6
    agg = agg.merge(residents[[key, "Einwohner"]], on=key, how="left")
    if indicator == "Number":
        agg["value"] = agg["count"].astype(float)
    elif indicator == "Number per km2":
        agg["value"] = agg["count"] / agg["area_km2"].replace(0, np.nan)
    elif indicator == "Number per 10,000 residents":
        denominator = agg["Einwohner"].where(agg["Einwohner"] >= CFG["min_residents"])
        agg["value"] = agg["count"] * 10000 / denominator
    else:
        raise ValueError("Unknown indicator: " + str(indicator))
    return gpd.GeoDataFrame(agg, geometry="geometry", crs=gdf_areas.crs)


# =============================================================================
# 4) Data quality
# =============================================================================
def quality_report(gdf_points, gdf_areas, subcat=None):
    """Automatic checks of one OSM file. Expects column 'PLZ' (from assign_area). Returns DataFrame."""
    # TODO (see Task 5): number of objects, share without name, share without <subcat>,
    #      near-duplicates (same name, closer than CFG["dup_distance_m"] metres - compute in CFG["crs_metric"]),
    #      points outside Berlin (PLZ is NaN), implausible coordinates (CFG["berlin_lat"], CFG["berlin_lon"]),
    #      share of objects with addr:postcode and how often it equals the PLZ of the spatial join,
    #      number of PLZ areas without any object.
    #      Return a DataFrame with the columns "Check" and "Result".
    n = len(gdf_points)
    checks = [("Number of objects", n)]
    names = gdf_points["name"] if "name" in gdf_points else pd.Series(pd.NA, index=gdf_points.index)
    missing_names = names.isna() | names.astype("string").str.strip().eq("").fillna(True)
    checks.append(("Share without name (%)", round(100 * missing_names.mean(), 2) if n else 0.0))

    if subcat:
        values = gdf_points[subcat] if subcat in gdf_points else pd.Series(pd.NA, index=gdf_points.index)
        missing = values.isna() | values.astype("string").str.strip().eq("").fillna(True)
        checks.append(("Share without " + subcat + " (%)", round(100 * missing.mean(), 2) if n else 0.0))

    named = gdf_points.loc[~missing_names].to_crs(CFG["crs_metric"]).copy()
    near_duplicates = set()
    if len(named) > 1:
        from scipy.spatial import cKDTree
        for _, group in named.groupby(names.loc[named.index].astype(str).str.strip().str.casefold()):
            if len(group) < 2:
                continue
            coords = np.column_stack((group.geometry.x.to_numpy(), group.geometry.y.to_numpy()))
            pairs = cKDTree(coords).query_pairs(r=CFG["dup_distance_m"])
            for i, j in pairs:
                near_duplicates.add(group.index[max(i, j)])
    checks.append(("Near-duplicate objects", len(near_duplicates)))
    checks.append(("Points outside Berlin (no PLZ)", int(gdf_points["PLZ"].isna().sum())))
    plausible = (
        gdf_points["lat"].between(*CFG["berlin_lat"])
        & gdf_points["lon"].between(*CFG["berlin_lon"])
    )
    checks.append(("Implausible coordinates", int((~plausible).sum())))

    postcodes = gdf_points.get("addr:postcode", pd.Series(pd.NA, index=gdf_points.index))
    postcodes = postcodes.astype("string").str.strip()
    has_postcode = postcodes.notna() & postcodes.ne("")
    checks.append(("Share with addr:postcode (%)", round(100 * has_postcode.mean(), 2) if n else 0.0))
    comparable = has_postcode & gdf_points["PLZ"].notna()
    agreement = postcodes.loc[comparable].eq(gdf_points.loc[comparable, "PLZ"].astype("string").str.strip())
    checks.append(("addr:postcode matches spatial PLZ (%)",
                   round(100 * agreement.mean(), 2) if comparable.any() else np.nan))
    plz_with_points = set(gdf_points["PLZ"].dropna().astype(str))
    checks.append(("PLZ areas without objects", int((~gdf_areas["PLZ"].astype(str).isin(plz_with_points)).sum())))
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
        valid = gdf_agg["value"].dropna()
        vmin = float(valid.min()) if len(valid) else 0.0
        vmax = float(valid.max()) if len(valid) else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        cmap = LinearColormap(list(colors), vmin=vmin, vmax=vmax, caption=caption)

        def style(feature):
            value = feature["properties"].get("value")
            color = "#bdbdbd" if value is None or pd.isna(value) else cmap(float(value))
            return {"fillColor": color, "color": "#555555", "weight": 0.5, "fillOpacity": 0.7}

        fields = [key, "count", "area_km2", "Einwohner", "value"]
        folium.GeoJson(
            gdf_agg.to_json(), name="Areas", style_function=style,
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=[key, "Objects", "Area (km²)", "Residents", caption])
        ).add_to(m)
        cmap.add_to(m)
    elif maptype == "Heatmap":
        # TODO: HeatMap([[lat, lon], ...])        <- Folium wants [lat, lon]
        coords = gdf_points[["lat", "lon"]].dropna().values.tolist()
        if coords:
            HeatMap(coords).add_to(m)
    else:   # Point map (clusters)
        # TODO: MarkerCluster with one marker per point, name as tooltip
        cluster = MarkerCluster().add_to(m)
        for row in gdf_points.itertuples(index=False):
            name = getattr(row, "name", None)
            tooltip = str(name) if pd.notna(name) and str(name).strip() else "Unnamed object"
            folium.Marker(location=[row.lat, row.lon], tooltip=tooltip).add_to(cluster)
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
    # TODO: options = ["All"] + category_options(...); use key="cat_" + src so the menu resets with the source
    options = ["All"] + category_options(pts, cfg["subcat"], pdict["top_n_category"])
    cat = st.sidebar.selectbox("Category filter", options, key="cat_" + src)

    # ---- Menus 3-5 ----------------------------------------------------------
    area_label = st.sidebar.selectbox("Area unit", list(pdict["areas"].keys()))
    ind        = st.sidebar.selectbox("Indicator", pdict["indicators"])
    mtype      = st.sidebar.selectbox("Map type", pdict["maptypes"])
    # optional menu 6: colour scale (only for the choropleth)

    # TODO: key (PLZ / Bezirk) and the matching polygons, filter by category, stop with st.warning if empty,
    #       residents_by_area, aggregate
    key = pdict["areas"][area_label]
    areas = plz if key == "PLZ" else dis
    pts_f = filter_category(pts, cfg["subcat"], cat)
    if pts_f.empty:
        st.warning("No objects match this category filter.")
        return
    residents = residents_by_area(df_residents, plz, dis, key)
    agg = aggregate(pts_f, areas, key, ind, residents)

    # TODO: key figures with st.metric: objects shown, areas without any object, area with the highest value
    ranked = agg.dropna(subset=["value"]).sort_values("value", ascending=False)
    highest = str(ranked.iloc[0][key]) if not ranked.empty else "N/A"
    a, b, c = st.columns(3)
    a.metric("Objects shown", len(pts_f))
    b.metric("Areas without objects", int((agg["count"] == 0).sum()))
    c.metric("Area with highest value", highest)

    # TODO: map with st_folium(m, width=1000, height=600, returned_objects=[])
    colors = pdict["colorscales"][st.sidebar.selectbox("Colour scale", list(pdict["colorscales"]),
                                                        disabled=mtype != "Choropleth")]
    m = make_map(pts_f, agg, mtype, key=key, colors=colors, caption=ind)
    st_folium(m, width=1000, height=600, returned_objects=[])

    # TODO: st.caption with "© OpenStreetMap contributors" and _download_date(pdict, src)
    st.caption("© OpenStreetMap contributors · OSM download: " + _download_date(pdict, src))

    # TODO: expander "Top 3 areas and data table" with st.dataframe and st.download_button (CSV)
    table = agg.drop(columns="geometry")[[key, "count", "area_km2", "Einwohner", "value"]]
    table = table.sort_values("value", ascending=False, na_position="last")
    with st.expander("Top 3 areas and data table"):
        st.write("Top 3 areas")
        st.dataframe(table.head(3), hide_index=True)
        st.dataframe(table, hide_index=True)
        st.download_button("Download CSV", table.to_csv(index=False, sep=";", encoding="utf-8"),
                           file_name="osm_" + src.lower().replace(" ", "_") + "_aggregated.csv",
                           mime="text/csv")

    # TODO: expander "Data quality" showing the quality DataFrame
    with st.expander("Data quality"):
        st.dataframe(quality, hide_index=True)

    # ---- Chat assistant (LLM) - provided, see core/chat.py -------------------
    # Needs the variables src, cat, area_label, ind, mtype, key, pts_f, agg, quality from your code above.
    ctx = chat.build_context(src, cat, area_label, ind, mtype, key, pts_f, agg, quality)
    chat.render_chat(ctx, pdict)
