import pandas as pd
import numpy as np

def stress_pd(pd_baseline, inflation=0, gdp=0, fx=0):
    """
    Aplica choques macroeconómicos sobre PD baseline.
    Fórmula simplificada de elasticidade IFRS 9.
    """
    return pd_baseline * (1
        + 0.6 * inflation      # peso inflação
        - 0.5 * gdp            # peso PIB
        + 0.4 * fx             # peso cambial
    )

def compute_ecl(pd, lgd, ead):
    return pd * lgd * ead

def run_ifrs9_stress_test(df, model_preds, scenarios):
    """
    scenarios = {
       "BASE": {"inflation":0, "gdp":0, "fx":0},
       "ADVERSO": {"inflation":0.05, "gdp":-0.02, "fx":0.10},
       "SEVERO": {"inflation":0.12, "gdp":-0.05, "fx":0.25}
    }
    """

    results = []

    for scenario_name, params in scenarios.items():

        stressed_pd = stress_pd(
            pd_baseline=model_preds,
            inflation=params["inflation"],
            gdp=params["gdp"],
            fx=params["fx"]
        )

        stressed_pd = np.clip(stressed_pd, 0.0001, 0.95)

        df_s = df.copy()
        df_s["PD"] = stressed_pd
        df_s["LGD"] = df_s.get("LGD", 0.45)
        df_s["EAD"] = df_s.get("EAD", 1000)

        df_s["ECL"] = compute_ecl(df_s["PD"], df_s["LGD"], df_s["EAD"])

        results.append((scenario_name, df_s))

    return results