import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from modules.chimerge import chimerge_binning
from modules.woe_transform import build_woe_mapping, apply_woe
from modules.model_logit_woe import train_logit_woe
from modules.psi import calculate_psi
from modules.pdf_report import create_pdf_report


def run_pipeline(df, new_df=None, max_bins=6):

    original_df = df.copy()

    # ==========================
    # 1. CHIMERGE + WOE
    # ==========================
    numeric_vars = [
        col for col in df.columns
        if col not in ["default"] and df[col].dtype != "object"
    ]

    woe_tables = {}
    bins_tables = {}

    for var in numeric_vars:
        intervals, summary = chimerge_binning(
            df=df, feature=var, target="default", max_bins=max_bins
        )

        summary["perc_events"] = summary["sum"] / summary["sum"].sum()
        summary["perc_non_events"] = summary["non_events"] / summary["non_events"].sum()
        summary["woe"] = np.log(summary["perc_non_events"] / summary["perc_events"])
        summary["interval"] = intervals

        mapping = build_woe_mapping(summary, var)
        df[f"{var}_woe"] = apply_woe(df, var, mapping)

        woe_tables[var] = summary[["interval", "woe"]]
        bins_tables[var] = summary

    # ==========================
    # 2. MODELO LOGÍSTICO WOE
    # ==========================
    model, preds, auc, ks, coef_df = train_logit_woe(df)

    fpr, tpr, _ = roc_curve(df["default"], preds)

    roc_fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.legend()

    # ==========================
    # 3. PSI (opcional)
    # ==========================
    if new_df is not None:
        psi_table, psi_value = calculate_psi(
            base_values=df[[c for c in df.columns if c.endswith("_woe")][0]],
            new_values=new_df[[c for c in new_df.columns if c.endswith("_woe")][0]],
            bins=10
        )
    else:
        psi_table = pd.DataFrame()
        psi_value = 0.0

    # ==========================
    # 4. PDF
    # ==========================
    create_pdf_report(
        filename="Pipeline_Final.pdf",
        woe_table=pd.concat(woe_tables),
        bins_table=pd.concat(bins_tables),
        coef_table=coef_df,
        auc=auc,
        ks=ks,
        roc_fig=roc_fig,
        psi_table=psi_table,
        psi_value=psi_value
    )

    return df, auc, ks, psi_value