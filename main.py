import os

# works locally and on Streamlit Community Cloud (no absolute path needed)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import pdict
from core import HelperTools as ht
from core import methods as m1


# -----------------------------------------------------------------------------
@ht.timer
def main() -> None:
    """Main: Generation of Streamlit App for visualizing OpenStreetMap data in Berlin"""

    # Residents per PLZ (whole Germany) -> keep only PLZ and residents of Berlin.
    # The OSM data is loaded inside the app, depending on the pull-down menus
    df_residents = pd.read_csv("datasets/" + pdict["file_residents"], sep=",", decimal=".")
    df_residents = m1.preprop_resid(df_residents)

    m1.make_streamlit_app(pdict, df_residents)


if __name__ == "__main__":
    main()
