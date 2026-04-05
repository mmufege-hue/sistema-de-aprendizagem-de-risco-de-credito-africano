import numpy as np
import pandas as pd

def generate_alternative_variables(df):
    """
    Cria variáveis alternativas para mercados africanos
    onde os clientes podem não ter histórico bancário.
    """

    n = len(df)

    # 1. Mobile Behavior Score (0–100)
    df["mobile_score"] = np.random.normal(60, 15, n).clip(0, 100)

    # 2. Proxy Income (estimativa de renda baseada no uso)
    df["proxy_income"] = (
        df["mobile_score"] * np.random.uniform(8, 15)
        + np.random.normal(0, 30)
    ).clip(50, 2500)

    # 3. Consumo / Top-ups mensais
    df["topups_month"] = np.random.gamma(2, 20, n).clip(1, 500)

    # 4. Estabilidade do telemóvel (trocas recentes = risco)
    df["phone_changes_12m"] = np.random.binomial(3, 0.15, n)

    # 5. Georisk (0 = risco alto, 1 = risco baixo)
    df["georisk"] = np.random.choice(
        [0.2, 0.4, 0.6, 0.8],
        size=n,
        p=[0.2, 0.3, 0.3, 0.2]
    )

    # 6. Volatilidade nos topups (instabilidade financeira)
    df["topup_volatility"] = np.random.uniform(0.1, 1, n)

    return df


def build_africa_score(df):
    """
    Score alternativo baseado em variáveis não tradicionais.
    """

    # Normalizar algumas variáveis
    proxy_income_norm = df["proxy_income"] / df["proxy_income"].max()
    topups_norm = df["topups_month"] / df["topups_month"].max()

    score = (
        0.30 * df["mobile_score"] / 100 +
        0.25 * proxy_income_norm +
        0.20 * df["georisk"] +
        0.15 * topups_norm -
        0.10 * df["phone_changes_12m"] / df["phone_changes_12m"].max() -
        0.10 * df["topup_volatility"]
    )

    df["africa_score"] = (score * 100).clip(0, 100)

    return df