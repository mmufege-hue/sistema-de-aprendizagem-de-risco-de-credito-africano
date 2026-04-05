import streamlit as st
import matplotlib.pyplot as plt

from modules.compare_models import train_logit_raw, train_logit_woe
from sklearn.metrics import roc_curve

st.title("⚔️ Comparação Completa: Modelo Raw vs Modelo WOE")

# Validar dataset
df = st.session_state.get("dataset")

if df is None:
    st.error("Gera primeiro um dataset na aba 'Gerar Dataset'.")
    st.stop()
    raise SystemExit(0)

# Validar se há variáveis WOE
woe_vars = [c for c in df.columns if c.endswith("_woe")]
if len(woe_vars) == 0:
    st.error("Nenhuma variável WOE encontrada. Cria variáveis WOE primeiro.")
    st.stop()

if st.button("Comparar Modelos"):

    # -----------------------------
    # MODELO RAW
    # -----------------------------
    raw_model, raw_preds, raw_auc, raw_ks, raw_coef = train_logit_raw(df)
    # -----------------------------
    # MODELO WOE
    # -----------------------------
    woe_model, woe_preds, woe_auc, woe_ks, woe_coef = train_logit_woe(df)

    # ---------------------------------------------------------------
    # METRICAS GLOBAIS
    # ---------------------------------------------------------------
    st.subheader("📊 Métricas de Performance")

    col1, col2 = st.columns(2)
    col1.metric("AUC (Raw)", f"{raw_auc:.3f}")
    col2.metric("AUC (WOE)", f"{woe_auc:.3f}")

    col1, col2 = st.columns(2)
    col1.metric("KS (Raw)", f"{raw_ks:.3f}")
    col2.metric("KS (WOE)", f"{woe_ks:.3f}")

    # ---------------------------------------------------------------
    # CURVAS ROC
    # ---------------------------------------------------------------
    st.subheader("📉 Curvas ROC — Lado a Lado")

    fpr_raw, tpr_raw, _ = roc_curve(df["default"], raw_preds)
    fpr_woe, tpr_woe, _ = roc_curve(df["default"], woe_preds)

    fig, ax = plt.subplots()
    ax.plot(fpr_raw, tpr_raw, label=f"Raw (AUC={raw_auc:.3f})")
    ax.plot(fpr_woe, tpr_woe, label=f"WOE (AUC={woe_auc:.3f})", color="orange")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    st.pyplot(fig)

    # ---------------------------------------------------------------
    # COEFICIENTES
    # ---------------------------------------------------------------
    st.subheader("🧠 Coeficientes dos Modelos")

    st.write("### Modelo Raw")
    st.dataframe(raw_coef.sort_values(by="coef", ascending=False))

    st.write("### Modelo WOE")
    st.dataframe(woe_coef.sort_values(by="coef", ascending=False))

    # ---------------------------------------------------------------
    # INTERPRETAÇÃO AUTOMÁTICA
    # ---------------------------------------------------------------
    st.subheader("🤖 Interpretação Automática")

    melhor_modelo = "WOE" if woe_auc > raw_auc else "RAW"

    st.write(f"""
    **Modelo com melhor performance geral:**  
    ✅ **{melhor_modelo}**

    - O modelo WOE tende a ser mais estável e monotónico.  
    - O modelo Raw pode ter performance pior por falta de binning.  
    - A vantagem do WOE é especialmente visível no KS.
    """)

    st.success("✅ Comparação concluída com sucesso!")
