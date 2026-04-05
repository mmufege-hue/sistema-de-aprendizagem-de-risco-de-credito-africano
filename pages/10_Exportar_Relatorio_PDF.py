import streamlit as st
import pandas as pd
from modules.pdf_report import create_pdf_report
from modules.compare_models import train_logit_woe
from modules.psi import calculate_psi
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

st.title("📄 Exportar Relatório PDF — End-to-End")

df = st.session_state.get("dataset")

if df is None:
    st.error("Gera primeiro um dataset.")
    st.stop()
    raise SystemExit(0)

# Verificar WOE
woe_vars = [c for c in df.columns if c.endswith("_woe")]
if len(woe_vars) == 0:
    st.error("Cria variáveis WOE primeiro.")
    st.stop()

# --------------------------
# Upload nova população (PSI)
# --------------------------
uploaded = st.file_uploader("Carrega nova população para PSI (CSV)")

if uploaded:
    new_df = pd.read_csv(uploaded)
    st.success("Nova população carregada!")

if st.button("Gerar PDF"):

    # ----------------------------
    # Modelo WOE
    # ----------------------------
    model, preds, auc, ks, coef_df = train_logit_woe(df)

    # ROC
    fpr, tpr, _ = roc_curve(df["default"], preds)
    roc_fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1],"--")
    ax.legend()

    # PSI
    if uploaded:
    # Usar a mesma variável selecionada para o PSI
    psi_table, psi_val = calculate_psi(
        base_values=df[feature],
        new_values=new_df[feature],
        bins=10
    )
    else:
        psi_table = pd.DataFrame()
        psi_val = 0.0

    # WOE table (todas as variáveis com _woe)
    woe_table = df[[c for c in df.columns if c.endswith("_woe")]]

    # bins table — opcional (placeholder)
    bins_table = pd.DataFrame()

    # Criar PDF
    create_pdf_report(
        filename="Relatorio_Modelo_Credito.pdf",
        woe_table=woe_table.head(20),
        bins_table=bins_table,
        coef_table=coef_df,
        auc=auc,
        ks=ks,
        roc_fig=roc_fig,
        psi_table=psi_table,
        psi_value=psi_val
    )

    with open("Relatorio_Modelo_Credito.pdf", "rb") as f:
        st.download_button("📥 Download Relatório PDF", f, file_name="Relatorio_Modelo_Credito.pdf")

    st.success("✅ PDF gerado com sucesso!")
