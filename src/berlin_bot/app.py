import os

import pandas as pd
from dotenv import load_dotenv

from berlin_bot.config import pdict
from berlin_bot.core import HelperTools as ht
from berlin_bot.core import methods as m1
from berlin_bot.paths import PROJECT_ROOT

os.chdir(PROJECT_ROOT)
# API keys for the chat from .env (local). On Streamlit Community Cloud they come from st.secrets.
load_dotenv(PROJECT_ROOT / ".env")


# -----------------------------------------------------------------------------
@ht.timer
def main():
    """Main: Generation of Streamlit App for visualizing OpenStreetMap data in Berlin"""

    # residents per PLZ; the OSM data is loaded inside the app, depending on the pull-down menus
    df_residents = pd.read_csv("datasets/" + pdict["file_residents"], sep=",", decimal=".")
    df_residents = m1.preprop_resid(df_residents)

    m1.make_streamlit_app(pdict, df_residents)


if __name__ == "__main__":
    main()
