import numpy as np

def recalc_pd(model, woe_vars_dict):
    """
    Recalcula PD quando o utilizador altera variáveis.
    woe_vars_dict = {"renda_woe": x, "idade_woe": y, ...}
    """
    X = np.array([list(woe_vars_dict.values())])
    return model.predict_proba(X)[0][1]


def simulate_limit(proxy_income, africa_score, policy):
    """
    Recalcula limite sugerido com base nos sliders.
    """
    if africa_score >= 80:
        return proxy_income * policy["limit_multiplier_high"]
    elif africa_score >= 60:
        return proxy_income * policy["limit_multiplier_mid"]
    else:
        return proxy_income * policy["limit_multiplier_low"]