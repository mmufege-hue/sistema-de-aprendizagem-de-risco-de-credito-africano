import numpy as np
import pandas as pd

def scoreband(pd):
    """
    Cria bandas de risco estilo banco.
    """
    if pd < 0.05: return "A"
    if pd < 0.10: return "B"
    if pd < 0.20: return "C"
    if pd < 0.35: return "D"
    return "E"


def decision_rules(pd, africa_score, proxy_income, georisk):
    """
    Motor principal de decisão baseado em políticas definidas.
    """

    reasons = []

    # 1. Hard rejects
    if africa_score < 20:
        return "REJECT", ["AfricaScore muito baixo"]

    if pd > 0.45:
        return "REJECT", ["PD demasiado alto"]

    if georisk < 0.3:
        reasons.append("Zona geográfica de alto risco")

    # 2. Limites de crédito sugeridos
    if africa_score >= 80:
        limit = proxy_income * 0.5
    elif africa_score >= 60:
        limit = proxy_income * 0.3
    elif africa_score >= 40:
        limit = proxy_income * 0.15
    else:
        limit = proxy_income * 0.05

    # 3. Decisão automática
    if pd < 0.08 and africa_score >= 50:
        return "APPROVE", reasons + [f"Limite sugerido: {limit:.0f}"]

    if 0.08 <= pd < 0.16 and africa_score >= 40:
        return "REVIEW", reasons + [f"Revisão manual — limite sugerido {limit:.0f}"]

    if pd >= 0.16:
        return "REJECT", reasons + ["PD elevado"]

    return "REVIEW", reasons


def process_decisions(df):
    """
    Aplica o motor de decisão linha a linha.
    """
    decisions = []
    scorebands_list = []
    reasons_list = []

    for _, row in df.iterrows():
        pd = row["PD"]
        africa = row["africa_score"]
        income = row["proxy_income"]
        geo = row["georisk"]

        sb = scoreband(pd)
        dec, reason = decision_rules(pd, africa, income, geo)

        decisions.append(dec)
        scorebands_list.append(sb)
        reasons_list.append(reason)

    df["decision"] = decisions
    df["scoreband"] = scorebands_list
    df["decision_reasons"] = reasons_list

    return df