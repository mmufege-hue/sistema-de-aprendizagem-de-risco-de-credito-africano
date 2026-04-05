import pandas as pd
import numpy as np

def compute_woe_iv(df, feature, target="default", bins=10):
    """
    Calcula WOE e IV para uma variável numérica ou categórica.
    df: dataframe
    feature: nome da variável
    target: variável de default (binária)
    bins: número de bins se for numérica
    """

    data = df[[feature, target]].copy()

    # Binning numérico
    if pd.api.types.is_numeric_dtype(data[feature]):
        data["bin"] = pd.qcut(data[feature], q=bins, duplicates="drop")
    else:
        data["bin"] = data[feature]

    # Tabela auxiliar
    grouped = data.groupby("bin")[target].agg(
        events="sum",
        total="count"
    )
    grouped["non_events"] = grouped["total"] - grouped["events"]

    # Totais
    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()

    # Proporções
    grouped["perc_events"] = grouped["events"] / total_events
    grouped["perc_non_events"] = grouped["non_events"] / total_non_events

    # WOE
    grouped["woe"] = np.log((grouped["perc_non_events"] + 0.0001) /
                            (grouped["perc_events"] + 0.0001))

    # IV
    grouped["iv_component"] = (grouped["perc_non_events"] - grouped["perc_events"]) * grouped["woe"]
    iv = grouped["iv_component"].sum()

    return grouped.reset_index(), iv