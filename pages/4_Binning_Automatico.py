import streamlit as st
import pandas as pd
from modules.chimerge import chimerge_binning

st.title("🧩 Binning Automático — ChiMerge")

df = st.session_state.get("dataset")

if df is None:
    st.error("Gera primeiro um dataset na aba 'Gerar Dataset' e executa a app com 'streamlit run'.")
    st.stop()
    raise SystemExit(0)

feature = st.selectbox("Escolhe a variável numérica:", df.columns[df.columns != "default"])

max_bins = st.slider("Número máximo de bins:", 3, 15, 6)
level = st.selectbox("Nível de significância (quanto maior, menos merges):",
                     ["90%", "95%", "99%"])
conf_map = {"90%": 0.90, "95%": 0.95, "99%": 0.99}

if st.button("Executar ChiMerge"):
    intervals, summary = chimerge_binning(
        df=df,
        feature=feature,
        target="default",
        max_bins=max_bins,
        confidence=conf_map[level]
    )

    st.subheader("📌 Intervalos finais dos bins:")
    st.write(intervals)

    st.subheader("📊 Resumo dos bins:")
    st.dataframe(summary)

    st.success("✅ ChiMerge executado com sucesso!")
