import shap
import numpy as np
import pandas as pd


def compute_shap_values(model, X):
    """
    Calcula os valores SHAP para um modelo de regressão logística.
    """
    explainer = shap.LinearExplainer(model, X, feature_dependence="independent")
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def explain_decision_with_rules(row, policy):
    """
    Explica a decisão do motor com base em regras da política.
    """

    reasons = []

    if row["africa_score"] < policy["min_africa_score"]:
        reasons.append("AfricaScore abaixo do mínimo definido pela política")

    if row["PD"] > policy["max_pd"]:
        reasons.append("PD acima do limite permitido")

    if row["proxy_income"] < policy["min_proxy_income"]:
        reasons.append("Renda estimada abaixo do mínimo permitido")

    if row["georisk"] < policy["min_georisk"]:
        reasons.append("Georisk abaixo do limite mínimo")

    # Thresholds de aprovação
    if row["PD"] < policy["approve_threshold"]:
        reasons.append("PD encontra-se na zona de aprovação automática")
    elif row["PD"] < policy["review_threshold"]:
        reasons.append("PD dentro da zona de revisão manual")
    else:
        reasons.append("PD na zona de rejeição")

    return reasons