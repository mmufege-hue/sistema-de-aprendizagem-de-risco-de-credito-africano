import streamlit as st
import pandas as pd
import numpy as np
from modules.model_logit_woe import train_logit_woe
from modules.scorecard_builder import build_scorecard, score_individual
from modules.chimerge import chimerge_binning
from modules.woe_transform import build_woe_mapping, apply_woe
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

st.title("📊 Scorecard Exporter — Excel & PDF")

df = st.session_state.get("dataset")

if df is None:
    st.error("Gera primeiro um dataset com WOE.")
    st.stop()
    raise SystemExit(0)

# -----------------------------
# TREINAR MODELO
# -----------------------------
model, preds, auc, ks, coef_df = train_logit_woe(df)

# -----------------------------
# GERAR TABELAS WOE POR VARIÁVEL
# -----------------------------
woe_vars = [c for c in df.columns if c.endswith("_woe")]
woe_tables = {}

for var in woe_vars:
    name = var.replace("_woe", "")
    intervals, summary = chimerge_binning(df, name, "default")
    summary["interval"] = intervals
    summary["perc_events"] = summary["sum"] / summary["sum"].sum()
    summary["perc_non_events"] = summary["non_events"] / summary["non_events"].sum()
    summary["woe"] = (summary["perc_non_events"] / summary["perc_events"]).apply(lambda x: np.log(x))
    woe_tables[name] = summary[["interval", "woe"]]

# -----------------------------
# SCORECARD
# -----------------------------
scorecard_df, offset, factor = build_scorecard(coef_df, woe_tables)

st.subheader("✅ Scorecard Gerado")
st.dataframe(scorecard_df)

# -----------------------------
# EXPORTAR EXCEL
# -----------------------------
if st.button("📥 Download Excel"):
    scorecard_df.to_excel("Scorecard.xlsx", index=False)
    with open("Scorecard.xlsx", "rb") as f:
        st.download_button("Baixar Scorecard.xlsx", f, "Scorecard.xlsx")

# -----------------------------
# EXPORTAR PDF
# -----------------------------
if st.button("📄 Exportar PDF"):

    pdf = canvas.Canvas("Scorecard.pdf", pagesize=letter)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, 750, "Scorecard de Crédito — CreditMasterLab")
    pdf.setFont("Helvetica", 10)

    y = 700
    for _, row in scorecard_df.iterrows():
        text = f"{row['Variável']} | {row['Intervalo']} | WOE={row['WOE']} | Coef={row['Coeficiente']} | Pontos={row['Pontos']}"
        pdf.drawString(40, y, text)
        y -= 14
        if y < 60:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y = 750

    pdf.save()

    with open("Scorecard.pdf", "rb") as f:
        st.download_button("Baixar Scorecard.pdf", f, "Scorecard.pdf")

st.success("✅ Scorecard pronto!")
