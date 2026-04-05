def decision_dynamic(pd, africa_score, proxy_income, georisk, policy):

    reasons = []

    # HARD REJECTS
    if africa_score < policy["min_africa_score"]:
        return "REJECT", ["AfricaScore abaixo da política"]

    if pd > policy["max_pd"]:
        return "REJECT", ["PD acima do limite permitido"]

    if proxy_income < policy["min_proxy_income"]:
        return "REJECT", ["Renda estimada insuficiente"]

    if georisk < policy["min_georisk"]:
        reasons.append("Zona geográfica de risco elevado")

    # LIMITE DE CRÉDITO
    if africa_score >= 80:
        limit = proxy_income * policy["limit_multiplier_high"]
    elif africa_score >= 60:
        limit = proxy_income * policy["limit_multiplier_mid"]
    else:
        limit = proxy_income * policy["limit_multiplier_low"]

    # APROVAÇÃO DIRETA
    if pd < policy["approve_threshold"]:
        return "APPROVE", reasons + [f"Limite sugerido: {limit:.0f}"]

    # REVISÃO
    if pd < policy["review_threshold"]:
        return "REVIEW", reasons + [f"Revisão manual — limite sugerido: {limit:.0f}"]

    # REJEIÇÃO FINAL
    return "REJECT", reasons + ["PD elevado mesmo após critérios de mitigação"]