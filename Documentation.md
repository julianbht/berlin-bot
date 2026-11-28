# Berlin Bot

## Fundamentals

Here we document briefly some concepts that the tasks later on use. We were unfamiliar with some of the geo data models that are used and summarize them shortly for this reason.

### OSM

OpenStreetMap (OSM) is a free map of the world that volunteers build.

#### Data Model

Everything is an element. An element is something on the map. All map data is made of three element types below.

##### Node

A node is one point with a lat/lon, for example a pharmacy drawn as a pin.

##### Way

A way is a line or an outline made of [nodes](#node), for example a playground drawn as an outline. An open way is a line, like a road. A closed way is an outline, like a building.

##### Relation

A relation groups other elements, for example a large park made of several parts.

##### Tag

Each element has tags, which are key=value pairs such as amenity=pharmacy, name=Linden-Apotheke or brand=dm.

Meaning of tags can be looked up here: <wiki.openstreetmap.org/wiki/Map_features>. The community decides the meaning of each tag.

#### Querying

How to query OSM? The live OSM data is queryable through the Overpass API, a public read-only query service.

Manually: Overpass turbo is a web frontend for Overpass. You can query it by hand at <overpass-turbo.eu>.

Python: `osmnx` library, which also calls the Overpass API.

### GIS

GIS stands for Geographic Information System. It's the general field and the software for storing, analysing and showing geo data.

#### Data Model

The data model of GIS is called the Simple Features Geometry Model.

Simple Features is a standard for storing the shape of geographic objects.

Almost every [GIS](#gis) tool uses it: PostGIS, QGIS, GDAL, Shapely, GeoPandas. A [Polygon](#polygon) means the same thing in each of them.

Three things make it different from [OSM](#osm)'s model:

- Each geometry stores its own coordinates directly. In OSM, a [way](#way) only stores [node](#node) ids, and you have to look up the nodes to get coordinates.
- There are no [tags](#tag) or ids. A geometry is only a shape, so the attributes (name, brand, ...) live in the other columns of the GeoDataFrame.
- It is "simple" because [LineStrings](#linestring) are straight between points (no curves) and [Polygons](#polygon) must not cross themselves.

A geometry is the shape of one object. It is always one of the types below.

##### Point

One coordinate pair, for example a pharmacy.

##### LineString

An ordered list of [Points](#point), for example a road.

##### Polygon

A closed outer ring, with optional inner rings (holes), for example a playground, or a park with a lake cut out.

##### Multi Geometries

MultiPoint, MultiLineString and MultiPolygon are collections of one type, for example a park made of several separate areas.

##### GeometryCollection

A mix of types. Rarely used.

##### WKT

WKT ("Well-Known Text") is the text format that [Simple Features](#data-model-1) defines for writing a geometry as plain text. The PLZ and Bezirk files store each [polygon](#polygon) as a WKT string in the geometry column, for example POLYGON ((13.37 52.53, 13.37 52.53, ...)). A CSV can only hold text, so this is how a shape fits in a CSV cell. gpd.GeoSeries.from_wkt parses those strings back into Shapely [Polygon](#polygon) objects.

### OSM to GIS

- [Node](#node), [way](#way) and [relation](#relation) are OSM's data model.
- [Point](#point) and [Polygon](#polygon) are geometry types from [GIS](#gis).

[OSM](#osm) has its own [data model](#data-model). [GIS](#gis) tools use the [simple features geometry model](#data-model-1), where each object carries its own coordinates.

## A1

```python
gdf = ox.features_from_place("Berlin, Germany", tags=cfg["tags"])
```  

This returns a geodataframe where each row is one [OSM](#osm) element. The index becomes the element type and id, and the [tags](#tag) become columns. Only the shape is converted: `osmnx` turns the [OSM data model](#data-model) into a [Simple Features geometry](#data-model-1) in the geometry column.

| OSM element                                           | becomes geometry     |
|-------------------------------------------------------|----------------------|
| node                                                  | Point                |
| closed way (outline, like a building)                 | Polygon              |
| open way (like a road)                                | LineString           |
| relation (a group, like a park made of several areas) | usually MultiPolygon |

It turns out that in this gdf, some rows end up as [Points](#point) (okay) and some as [Polygons](#polygon) (not okay). The app needs one [Point](#point) per element for the point map and for counting per PLZ.

We don't want the [Polygons](#polygon). So we just convert them to [Points](#point) by their centroid.

What we get we just persist to `.csv` files. But only some columns get saved.

### Data Sources

This is a project specific concept. In `config.py` there is a `p` dict for parameters. The `sources` key lists the data sources we are interested in. Each data source is one entry in the "Data source" pull-down menu of the app, and it has three fields:

- `tags`: which OSM elements to download. For example, for "Restaurants" we accept elements with the [tag](#tag) amenity=restaurant.
- `file`: the `.csv` file in `datasets/` that the download writes to.
- `subcat`: the [tag](#tag) used for the "Category filter" pull-down menu, for example `cuisine` for restaurants or `brand` for pharmacies.

Right now each data source is just one specific [tag](#tag), not a group. But a tag can also take a list of values, for example `{"shop": ["supermarket", "convenience"]}`, so one data source could combine several kinds of elements.

## A2

The task is to design a couple of helper function that work on the data in-memory. We start with all our custom points of interest in Berlin that we downloaded.

First, two data loaders. Then one that assigns each of our custom POI to a a bezirk. Then one function that aggregates population data. The final function brings everything together. It aggregates population and custom POI data.

- load_areas(): bezirk areas from csv (where they are saved as [WKT](#wkt)) into a gdf.
- load_osm_points(): the csvs have lat/lon. needs to be turned into [points](#point) for a gdf.
- assign_area(): joins each point to the [polygon](#polygon) it lies within
- residents_by_area(): for PLZ, it merges the resident counts. For Bezirk, each PLZ counts in full for the Bezirk nearest to its centre, computed in metres.
- aggregate(): counts points per area

| Function          | Input                                        | Output                                               |
|-------------------|----------------------------------------------|------------------------------------------------------|
| load_areas        | PLZ / Bezirk polygon CSVs                    | the area polygons                                    |
| load_osm_points   | a POI CSV                                    | the POIs as Points                                   |
| assign_area       | POIs + polygons                              | POIs with a PLZ and a Bezirk column                  |
| residents_by_area | population CSV (+ polygons, for Bezirk)      | residents per area                                   |
| aggregate         | POIs with areas + polygons + residents       | one row per area: count, area_km2, Einwohner, value  |

Our own decisions:

Few residents: used the existing min_residents = 500. Below that, "per 10,000 residents" gives no value, so the map shows the area grey.

## A3

| Source            | Objects | Without name | Without subcat    | Near-dups | Outside | Postcode = join | Empty PLZ |
|-------------------|---------|--------------|-------------------|-----------|---------|-----------------|-----------|
| Restaurants       | 4750    | 0.4 %        | 20.3 % (cuisine)  | 10        | 2       | 86.2 %          | 1         |
| Cafes             | 2617    | 1.2 %        | 63.6 % (cuisine)  | 8         | 0       | 85.3 %          | 5         |
| Fast food         | 2872    | 2.0 %        | 22.1 % (cuisine)  | 8         | 0       | 82.2 %          | 2         |
| Pharmacies        | 673     | 0.4 %        | 97.9 % (brand)    | 2         | 0       | 84.0 %          | 7         |
| Supermarkets      | 1380    | 0.3 %        | 17.8 % (brand)    | 2         | 0       | 84.9 %          | 2         |
| Playgrounds       | 5560    | 92.6 %       | 20.7 % (access)   | 0         | 8       | 93.2 %          | 0         |
| Charging stations | 1802    | 85.3 %       | 18.8 % (operator) | 9         | 0       | 91.7 %          | 11        |

"Postcode = join" is the share of points where the postcode written in OSM matches the PLZ that the spatial join found.

Each point can have two PLZ values:

- addr:postcode is a [tag](#tag) that a volunteer typed into OSM as part of the address, for example 10623. Many points don't have it.
- PLZ from the join is what assign_area found by checking which PLZ [polygon](#polygon) the point's coordinates lie in.

How the less obvious checks work:

- Near-duplicates: a self spatial join with dwithin 10 m
- Postcode match: this only compares points that have both an addr:postcode and a PLZ from the join.
