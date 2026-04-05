import pandas as pd
import numpy as np

def prepare_bin_visuals(df, feature, summary_table):
    """
    Cria um dataframe com:
    - intervalo (min, max)
    - woe
    - % eventos
    - % não-eventos
    - total por bin
    """
    viz = summary_table.copy()

    viz["interval_str"] = viz["interval"].apply(lambda x: f"{round(x[0],2)} — {round(x[1],2)}")
    viz["pct_events"] = viz["sum"] / viz["sum"].sum()
    viz["pct_non_events"] = viz["non_events"] / viz["non_events"].sum()
    viz["total"] = viz["sum"] + viz["non_events"]

    return viz