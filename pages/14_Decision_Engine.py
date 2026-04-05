import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from modules.model_logit_woe import train_logit_woe
from modules.africa_mode import build_africa_score, generate_alternative_variables
from modules.decision_engine import process_decisions

st.title("⚖️ Motor de Decisão — Approve / Reject Engine")

df = st.session_state.get("dataset")

if df is None:
    st.error("Gera primeiro um dataset com WOE e AfricaScore.")
    st.stop()
    raise SystemExit(0)

# 1. Garantir que AfricaScore existe
required_cols = ["africa_score", "proxy_income", "georisk"]

if not all(col in df.columns for col in required_cols):
    st.warning("Variáveis alternativas não encontradas. Gerando automaticamente...")
    df = generate_alternative_variables(df)
    df = build_africa_score(df)
    st.session_state["dataset"] = df

# 2. Treinar modelo para obter PD
st.subheader("📈 Treinar Modelo (PD Baseline)")
model, preds, auc, ks, coef_df = train_logit_woe(df)
df["PD"] = preds

st.write(f"AUC: {auc:.3f} | KS: {ks:.3f}")

# 3. Executar motor de decisão
if st.button("Executar Motor de Decisão"):

    df_decision = process_decisions(df)

    st.success("✅ Decisões processadas!")

    st.write(df_decision[[
        "PD", "africa_score", "proxy_income",
        "georisk", "scoreband", "decision", "decision_reasons"
    ]].head())

    # 4. Gráfico das decisões
    st.subheader("📊 Distribuição das Decisões")

    fig, ax = plt.subplots()
    df_decision["decision"].value_counts().plot(kind="bar", ax=ax)
    ax.set_ylabel("Nº de clientes")
    st.pyplot(fig)

    # guardar no estado
    st.session_state["dataset"] = df_decision
