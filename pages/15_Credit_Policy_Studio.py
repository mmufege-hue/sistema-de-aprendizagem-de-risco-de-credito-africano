import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from modules.model_logit_woe import train_logit_woe
from modules.africa_mode import generate_alternative_variables, build_africa_score
from modules.policy_studio import build_policy
from modules.decision_engine_dynamic import decision_dynamic

st.title("🎛️ Credit Policy Studio")

df = st.session_state.get("dataset")

if df is None:
    st.error("Carrega ou gera um dataset primeiro.")
    st.stop()
    raise SystemExit(0)

# -----------------------------
# POLÍTICAS CONFIGURÁVEIS
# -----------------------------
st.subheader("⚙️ Definir Políticas de Crédito")

min_africa = st.slider("AfricaScore mínimo para aprovação", 0, 100, 30)
max_pd = st.slider("PD máximo permitido", 0.01, 0.60, 0.35)
min_income = st.slider("Renda mínima estimada (proxy income)", 50, 3000, 100)
min_georisk = st.slider("Georisk mínimo", 0.0, 1.0, 0.3)

approve_threshold = st.slider("Aprovar automaticamente se PD < ", 0.01, 0.20, 0.08)
review_threshold = st.slider("Revisão manual se PD < ", 0.05, 0.40, 0.16)

lm_high = st.slider("Limite (%) para AfricaScore > 80", 0.1, 1.0, 0.5)
lm_mid = st.slider("Limite (%) para AfricaScore > 60", 0.05, 0.6, 0.3)
lm_low = st.slider("Limite (%) restante", 0.01, 0.3, 0.1)

policy = build_policy(
    min_africa_score=min_africa,
    max_pd=max_pd,
    min_proxy_income=min_income,
    min_georisk=min_georisk,
    approve_threshold=approve_threshold,
    review_threshold=review_threshold,
    limit_multiplier_high=lm_high,
    limit_multiplier_mid=lm_mid,
    limit_multiplier_low=lm_low
)

st.success("✅ Política construída!")

# -----------------------------
# TREINAR MODELO PARA PD
# -----------------------------
model, preds, auc, ks, coef_df = train_logit_woe(df)
df["PD"] = preds

# Garantir AfricaScore
if "africa_score" not in df.columns:
    df = generate_alternative_variables(df)
    df = build_africa_score(df)
    st.session_state["dataset"] = df

# -----------------------------
# EXECUTAR POLÍTICA
# -----------------------------
if st.button("Aplicar Política ao Dataset"):

    decisions = []
    reasons_all = []

    for _, row in df.iterrows():
        dec, rs = decision_dynamic(
            pd=row["PD"],
            africa_score=row["africa_score"],
            proxy_income=row["proxy_income"],
            georisk=row["georisk"],
            policy=policy
        )
        decisions.append(dec)
        reasons_all.append(rs)

    df["decision"] = decisions
    df["reasons"] = reasons_all

    st.write("### Resultados")
    st.dataframe(df.head())

    # Gráfico das decisões
    st.subheader("📊 Distribuição das Decisões")
    fig, ax = plt.subplots()
    df["decision"].value_counts().plot(kind="bar", color=["green", "orange", "red"], ax=ax)
    st.pyplot(fig)

    st.success("✅ Política aplicada com sucesso!")
    st.session_state["dataset"] = df
