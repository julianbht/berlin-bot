# berlin-bot

Streamlit app that maps OpenStreetMap places in Berlin and answers questions about them with an LLM chat assistant.

## Setup

### Dependencies

Install the dependencies (needs [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
```

The chat assistant uses OpenRouter. Copy `.env.example` to `.env` and set your key there:

```bash
cp .env.example .env
```

### Download the OSM data

The app never calls the Overpass API itself. Download the data once.

```bash
uv run download-osm
```

### Run the app

```bash
uv run berlin-bot
```

Then open <http://localhost:8501>. Streamlit options are passed on, for example `uv run berlin-bot --server.port 8502`. After a change in the code, stop the app and start it again.

## Project Layout

```text
berlin-bot/
├── datasets/              PLZ and Bezirk polygons, residents, downloaded OSM csv files
├── src/berlin_bot/
│   ├── config.py          data sources, menu options, LLM providers
│   ├── download_osm.py    OSM download (Task 1)
│   ├── app.py             entry point of the Streamlit app
│   └── core/
│       ├── methods.py     loading, spatial join, indicators, data quality, map, app (Tasks 2-4)
│       └── chat.py        chat assistant: prompt, context, LLM call
├── .env.example           names of the API keys
└── pyproject.toml         dependencies and the scripts download-osm and berlin-bot
```

## Data Quality

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

- addr:postcode is a tag that a volunteer typed into OSM as part of the address, for example 10623. Many points don't have it.
- PLZ from the join is what assign_area found by checking which PLZ polygon the point's coordinates lie in.
