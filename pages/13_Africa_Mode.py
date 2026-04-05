import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from modules.africa_mode import generate_alternative_variables, build_africa_score

st.title("🌍 AFRICA MODE — Scoring Alternativo para Clientes Sem Histórico")

df = st.session_state.get("dataset")

if df is None:
    st.error("Gera primeiro um dataset na aba 'Gerar Dataset'.")
    st.stop()
    raise SystemExit(0)

# -------------------------------
# Gerar variáveis alternativas
# -------------------------------

if st.button("Gerar Variáveis Alternativas"):
    df = generate_alternative_variables(df)
    st.success("✅ Variáveis alternativas geradas!")
    st.dataframe(df.head())
    st.session_state["dataset"] = df

# -------------------------------
# Gerar score AFRICA
# -------------------------------
if st.button("Calcular Africa Score"):
    df = build_africa_score(df)
    st.success("✅ Africa Score criado!")
    st.dataframe(df[["mobile_score", "proxy_income", "georisk", "africa_score"]].head())
    st.session_state["dataset"] = df

    # -------------------------------
    # GRÁFICOS
    # -------------------------------
    st.subheader("📊 Distribuição do Africa Score")

    fig, ax = plt.subplots()
    ax.hist(df["africa_score"], bins=20, color="green")
    ax.set_xlabel("Africa Score")
    ax.set_ylabel("Quantidade")
    st.pyplot(fig)

    st.subheader("📈 Relação entre Proxy Income e Africa Score")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["proxy_income"], df["africa_score"], alpha=0.3)
    ax2.set_xlabel("Proxy Income")
    ax2.set_ylabel("Africa Score")
    st.pyplot(fig2)

    st.subheader("🗺️ Georisk vs Africa Score")
    fig3, ax3 = plt.subplots()
    ax3.boxplot([
        df[df["georisk"]==g]["africa_score"]
        for g in sorted(df["georisk"].unique())
    ])
    ax3.set_xticklabels(sorted(df["georisk"].unique()))
    ax3.set_xlabel("Georisk")
    ax3.set_ylabel("Africa Score")
    st.pyplot(fig3)
