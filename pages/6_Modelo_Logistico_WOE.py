import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from modules.model_logit_woe import train_logit_woe
from sklearn.metrics import roc_curve

st.title("📈 Modelo Logístico — Variáveis WOE")

# VALIDAR SE O DATASET EXISTE
df = st.session_state.get("dataset")

if df is None:
    st.error("⚠️ Primeiro gere um dataset e aplique transformações WOE.")
    st.stop()
    raise SystemExit(0)

# VALIDAR SE EXISTEM VARIÁVEIS WOE
woe_vars = [c for c in df.columns if c.endswith("_woe")]

if len(woe_vars) == 0:
    st.error("⚠️ Nenhuma variável WOE encontrada. Transforme algumas variáveis primeiro.")
    st.stop()

st.write("### Variáveis WOE encontradas:")
st.write(woe_vars)

if st.button("Treinar Modelo"):

    model, preds, auc, ks, coef_df = train_logit_woe(df)

    st.success("✅ Modelo treinado com sucesso!")

    st.metric("AUC", f"{auc:.3f}")
    st.metric("KS", f"{ks:.3f}")

    # Tabela de coeficientes
    st.subheader("📌 Coeficientes do Modelo")
    st.dataframe(coef_df)

    # Interpretação automática
    def interpret_coef(value):
        if value > 0:
            return "Aumenta o risco de default"
        return "Diminui o risco de default"

    coef_df["interpretação"] = coef_df["coef"].apply(interpret_coef)
    st.write("### 🧠 Interpretação automática dos coeficientes:")
    st.dataframe(coef_df)

    # Curva ROC
    st.subheader("📉 Curva ROC")
    fpr, tpr, _ = roc_curve(df["default"], preds)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
    ax.legend()
    st.pyplot(fig)
