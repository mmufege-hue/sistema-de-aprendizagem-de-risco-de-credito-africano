import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

def train_logit_raw(df, target="default"):
    """
    Treina um modelo com variáveis raw (não-WOE).
    """
    raw_features = [
        col for col in df.columns
        if col not in [target] and not col.endswith("_woe")
    ]

    X = df[raw_features]
    y = df[target]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)
    fpr, tpr, _ = roc_curve(y, preds)
    ks = max(tpr - fpr)

    coef_df = pd.DataFrame({
        "variável": raw_features,
        "coef": model.coef_[0],
        "tipo": "RAW"
    })

    return model, preds, auc, ks, coef_df


def train_logit_woe(df, target="default"):
    """
    Treina um modelo com variáveis WOE.
    """
    woe_features = [col for col in df.columns if col.endswith("_woe")]

    X = df[woe_features]
    y = df[target]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)
    fpr, tpr, _ = roc_curve(y, preds)
    ks = max(tpr - fpr)

    coef_df = pd.DataFrame({
        "variável": woe_features,
        "coef": model.coef_[0],
        "tipo": "WOE"
    })

    return model, preds, auc, ks, coef_df