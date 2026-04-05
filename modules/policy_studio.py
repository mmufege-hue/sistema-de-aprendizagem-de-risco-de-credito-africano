def build_policy(
    min_africa_score=30,
    max_pd=0.35,
    min_proxy_income=100,
    min_georisk=0.3,
    approve_threshold=0.08,
    review_threshold=0.16,
    limit_multiplier_high=0.5,
    limit_multiplier_mid=0.3,
    limit_multiplier_low=0.1
):
    """
    Cria um dicionário de políticas de crédito configuráveis pelo utilizador.
    """

    return {
        "min_africa_score": min_africa_score,
        "max_pd": max_pd,
        "min_proxy_income": min_proxy_income,
        "min_georisk": min_georisk,
        "approve_threshold": approve_threshold,
        "review_threshold": review_threshold,
        "limit_multiplier_high": limit_multiplier_high,
        "limit_multiplier_mid": limit_multiplier_mid,
        "limit_multiplier_low": limit_multiplier_low,
    }