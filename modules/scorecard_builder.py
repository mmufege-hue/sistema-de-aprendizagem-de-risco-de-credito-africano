import pandas as pd
import numpy as np

def build_scorecard(coef_df, woe_tables, base_score=600, pdo=50):
    """
    Gera scorecard estilo bancário:
    - coef_df: coeficientes do modelo WOE
    - woe_tables: dict → var_name : tabela WOE com intervalos
    """

    scorecard = []

    # Odds base
    odds0 = 1  # baseline
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(odds0)

    for var, coef in zip(coef_df["variavel"], coef_df["coef"]):

        var_name = var.replace("_woe", "")
        woe_table = woe_tables[var_name]

        for _, row in woe_table.iterrows():
            points = -(coef * row["woe"] * factor)

            scorecard.append({
                "Variável": var_name,
                "Intervalo": row["interval"],
                "WOE": round(row["woe"], 4),
                "Coeficiente": round(coef, 4),
                "Pontos": round(points, 2)
            })

    scorecard_df = pd.DataFrame(scorecard)

    return scorecard_df, offset, factor


def score_individual(df_row, coef_df, offset, factor):
    """
    Calcula score final de um cliente.
    """

    total = offset

    for _, row in coef_df.iterrows():
        var = row["variavel"]
        coef = row["coef"]
        woe_value = df_row[var]
        total += -(coef * woe_value * factor)

    return total