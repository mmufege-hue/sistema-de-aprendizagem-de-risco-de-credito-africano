import pandas as pd
import numpy as np

def build_woe_mapping(summary_table, feature):
    """
    Constrói um dicionário:
    { (min, max): woe_value }
    onde 'min' e 'max' vêm do ChiMerge.
    """
    mapping = []

    for i in range(len(summary_table)):
        row = summary_table.iloc[i]
        interval = row["interval"]
        woe = row["woe"]
        mapping.append({"interval": interval, "woe": woe})

    return mapping


def apply_woe(df, feature, mapping):
    """
    Aplica a transformação WOE a uma variável numérica com base nos intervalos.
    """
    woe_values = []

    for value in df[feature]:
        applied = False

        for m in mapping:
            low, high = m["interval"]
            if low <= value <= high:
                woe_values.append(m["woe"])
                applied = True
                break

        if not applied:
            woe_values.append(np.nan)

    return np.array(woe_values)