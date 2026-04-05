import streamlit as st
import pandas as pd
import numpy as np

from modules.chimerge import chimerge_binning
from modules.woe_iv import compute_woe_iv
from modules.woe_transform import build_woe_mapping, apply_woe

st.title("🔄 Transformação WOE — Criar Variável Nova")

df = st.session_state.get("dataset")

if df is None:
    st.error("Gera primeiro um dataset na aba 'Gerar Dataset' e executa a app com 'streamlit run'.")
    st.stop()
    raise SystemExit(0)

feature = st.selectbox("Escolhe a variável para transformar em WOE:",
                       df.columns[df.columns != "default"])

max_bins = st.slider("Número máximo de bins:", 3, 12, 6)

if st.button("Executar Transformação WOE"):

    # 1) Binning automático (ChiMerge)
    intervals, summary = chimerge_binning(
        df=df,
        feature=feature,
        target="default",
        max_bins=max_bins
    )

    st.subheader("📌 Intervalos detectados:")
    st.write(intervals)

    # 2) Calcular WOE/IV nos bins finais
    # Usamos os bins e eventos obtidos no summary
    summary["perc_events"] = summary["sum"] / summary["sum"].sum()
    summary["perc_non_events"] = summary["non_events"] / summary["non_events"].sum()

    summary["woe"] = (summary["perc_non_events"] / summary["perc_events"]).apply(lambda x: np.log(x))

    st.subheader("📊 Tabela com WOE:")
    st.write(summary)

    # 3) Criar mapping → aplicar ao dataset
    mapping = build_woe_mapping(summary, feature)

    df[f"{feature}_woe"] = apply_woe(df, feature, mapping)

    st.success(f"✅ Variável '{feature}_woe' criada com sucesso!")

    st.write(df[[feature, f"{feature}_woe", "default"]].head())

    st.session_state["dataset"] = df
