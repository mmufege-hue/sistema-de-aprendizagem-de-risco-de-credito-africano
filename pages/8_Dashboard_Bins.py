import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modules.chimerge import chimerge_binning
from modules.bin_visuals import prepare_bin_visuals

st.title("📊 Dashboard dos Bins — WOE, Eventos e Distribuição")

# -----------------------
# Verificar dataset
# -----------------------
df = st.session_state.get("dataset")

if df is None:
    st.error("Gera primeiro um dataset na aba 'Gerar Dataset'.")
    st.stop()
    raise SystemExit(0)

feature = st.selectbox(
    "Escolhe a variável numérica:",
    [col for col in df.columns if col not in ["default"] and not col.endswith("_woe")]
)

max_bins = st.slider("Número máximo de bins:", 3, 15, 6)

# -----------------------
# Executar ChiMerge
# -----------------------
if st.button("Gerar Dashboard"):
    intervals, summary = chimerge_binning(
        df=df,
        feature=feature,
        target="default",
        max_bins=max_bins
    )

    # Calcular WOE e estatísticas
    summary["perc_events"] = summary["sum"] / summary["sum"].sum()
    summary["perc_non_events"] = summary["non_events"] / summary["non_events"].sum()
    summary["woe"] = np.log(summary["perc_non_events"] / summary["perc_events"])
    summary["interval"] = intervals

    viz = prepare_bin_visuals(df, feature, summary)

    st.subheader("📌 Tabela dos Bins")
    st.dataframe(viz)

    # ===========================
    # GRÁFICO 1 — WOE POR BIN
    # ===========================
    st.subheader("📈 WOE por Bin")

    fig, ax = plt.subplots()
    ax.bar(viz["interval_str"], viz["woe"], color="orange")
    ax.set_xlabel("Intervalo")
    ax.set_ylabel("WOE")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ===========================
    # GRÁFICO 2 — % EVENTOS POR BIN
    # ===========================
    st.subheader("📉 Percentual de Default por Bin")

    fig2, ax2 = plt.subplots()
    ax2.plot(viz["interval_str"], viz["pct_events"], marker="o")
    ax2.set_xlabel("Intervalo")
    ax2.set_ylabel("% de eventos (default)")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # ===========================
    # GRÁFICO 3 — DISTRIBUIÇÃO DE CLIENTES
    # ===========================
    st.subheader("📊 Distribuição de Clientes por Bin")

    fig3, ax3 = plt.subplots()
    ax3.bar(viz["interval_str"], viz["total"], color="blue")
    ax3.set_xlabel("Intervalo")
    ax3.set_ylabel("Nº de clientes")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.success("✅ Dashboard gerado!")
