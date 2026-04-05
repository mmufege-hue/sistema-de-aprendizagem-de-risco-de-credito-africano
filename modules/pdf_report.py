from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import pandas as pd
import matplotlib.pyplot as plt

def save_plot(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches="tight")

def df_to_table(pdf, df, x=40, y=650):
    data = [df.columns.to_list()] + df.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.black),
    ]))
    table.wrapOn(pdf, 50, 50)
    table.drawOn(pdf, x, y)

def create_pdf_report(
    filename,
    woe_table,
    bins_table,
    coef_table,
    auc,
    ks,
    roc_fig,
    psi_table,
    psi_value
):
    pdf = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # -----------------------------
    # PAGE 1 — Summary
    # -----------------------------
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, 750, "Relatório de Modelo de Crédito — CreditMasterLab")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(40, 720, f"AUC: {auc:.3f}")
    pdf.drawString(40, 700, f"KS: {ks:.3f}")
    pdf.drawString(40, 680, f"PSI: {psi_value:.3f}")

    pdf.showPage()

    # -----------------------------
    # PAGE 2 — WOE Table
    # -----------------------------
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, 750, "Tabela WOE")
    df_to_table(pdf, woe_table.head(20))
    pdf.showPage()

    # -----------------------------
    # PAGE 3 — Bins Table
    # -----------------------------
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, 750, "Tabela de Bins (ChiMerge)")
    df_to_table(pdf, bins_table.head(20))
    pdf.showPage()

    # -----------------------------
    # PAGE 4 — Coeficientes
    # -----------------------------
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, 750, "Coeficientes do Modelo (WOE)")
    df_to_table(pdf, coef_table)
    pdf.showPage()

    # -----------------------------
    # PAGE 5 — PSI Table
    # -----------------------------
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, 750, "PSI — Population Stability Index")
    df_to_table(pdf, psi_table)
    pdf.showPage()

    # -----------------------------
    # PAGE 6 — ROC Curve
    # -----------------------------
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, 750, "Curva ROC")

    roc_fig.savefig("roc_temp.png", dpi=300, bbox_inches="tight")
    pdf.drawImage("roc_temp.png", 50, 270, width=500, height=400)

    pdf.showPage()
    pdf.save()