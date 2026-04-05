import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def train_logit_woe(df, target="default"):
    """
    Treina regressão logística usando apenas variáveis transformadas em WOE.
    Retorna o modelo, AUC, KS e o dataframe de coeficientes.
    """

    # Selecionar somente variáveis WOE
    woe_features = [col for col in df.columns if col.endswith("_woe")]

    X = df[woe_features]
    y = df[target]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Probabilidades previstas
    preds = model.predict_proba(X)[:, 1]

    # AUC
    auc = roc_auc_score(y, preds)

    # KS → separação entre as curvas
    fpr, tpr, thresholds = roc_curve(y, preds)
    ks = max(tpr - fpr)

    # tabela de coeficientes
    coef_df = pd.DataFrame({
        "variavel": woe_features,
        "coef": model.coef_[0]
    }).sort_values(by="coef", ascending=False)

    return model, preds, auc, ks, coef_df