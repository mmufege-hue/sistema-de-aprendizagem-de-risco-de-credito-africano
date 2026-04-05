import streamlit as st
import pandas as pd
from modules.woe_iv import compute_woe_iv

st.title("📊 WOE & IV — Análise de Poder Preditivo")

df = st.session_state.get("dataset")

if df is None:
    st.error("⚠️ Primeiro gera um dataset na aba 'Gerar Dataset' e executa a app com 'streamlit run'.")
    st.stop()
    raise SystemExit(0)

st.write("### Escolha uma variável para calcular WOE & IV")
feature = st.selectbox("Variável", df.columns[df.columns != "default"])

bins = st.slider("Número de bins (só para variáveis numéricas)", 3, 20, 10)

if st.button("Calcular"):
    grouped, iv = compute_woe_iv(df, feature, bins=bins)

    st.write("### Tabela de bins com WOE e IV")
    st.dataframe(grouped)

    st.metric("IV Total", f"{iv:.4f}")

    # Interpretação automática
    if iv < 0.02:
        strength = "Inútil"
    elif iv < 0.1:
        strength = "Fraca"
    elif iv < 0.3:
        strength = "Média"
    elif iv < 0.5:
        strength = "Forte"
    else:
        strength = "SUSPEITA (pode haver leakage)"

    st.write(f"**Interpretação:** {strength}")
