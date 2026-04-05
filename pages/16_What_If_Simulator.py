import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modules.model_logit_woe import train_logit_woe
from modules.decision_engine_dynamic import decision_dynamic
from modules.policy_studio import build_policy
from modules.what_if import recalc_pd, simulate_limit

st.title("🔮 What‑If Simulator — Como a Decisão Muda?")

# -----------------------------
# Validar dataset
# -----------------------------
df = st.session_state.get("dataset")

if df is None:
    st.error("Carrega ou gera primeiro um dataset.")
    st.stop()
    raise SystemExit(0)

# -----------------------------
# Treinar modelo para PD
# -----------------------------
model, preds, auc, ks, coef_df = train_logit_woe(df)
df["PD"] = preds

st.success(f"Modelo carregado • AUC {auc:.3f} | KS {ks:.3f}")

# -----------------------------
# ESCOLHER UM CLIENTE
# -----------------------------
st.subheader("👤 Selecionar Cliente")

idx = st.number_input("ID do cliente", 0, len(df)-1, 0)
client = df.iloc[idx]

st.write("### Dados Originais do Cliente")
st.dataframe(client.to_frame().T)

# -----------------------------
# SLIDERS DE WHAT‑IF
# -----------------------------
st.subheader("✨ Ajustar Variáveis (What‑If)")

new_africa = st.slider("AfricaScore", 0, 100, int(client["africa_score"]))
new_income = st.slider("Proxy Income", 50, 3000, int(client["proxy_income"]))
new_geo = st.slider("GeoRisk", 0.0, 1.0, float(client["georisk"]))
new_mobile = st.slider("Mobile Score", 0, 100, int(client["mobile_score"]))
new_topups = st.slider("Top-Ups Mensais", 1, 500, int(client["topups_month"]))

# -----------------------------
# POLÍTICA DINÂMICA
# -----------------------------
st.subheader("📜 Política de Crédito (parâmetros ajustáveis)")

policy = build_policy(
    min_africa_score = st.slider("Min AfricaScore", 0, 100, 30),
    max_pd = st.slider("Max PD", 0.05, 0.60, 0.35),
    min_proxy_income = st.slider("Min Renda Estimada", 50, 3000, 100),
    min_georisk = st.slider("Min GeoRisk", 0.0, 1.0, 0.3),
    approve_threshold = st.slider("Aprovar se PD abaixo de:", 0.01, 0.20, 0.08),
    review_threshold = st.slider("Revisão se PD abaixo de:", 0.05, 0.40, 0.16),
    limit_multiplier_high = st.slider("Limite (Africa > 80)", 0.1, 1.0, 0.5),
    limit_multiplier_mid = st.slider("Limite (Africa > 60)", 0.05, 0.6, 0.3),
    limit_multiplier_low = st.slider("Limite (restante)", 0.01, 0.3, 0.1),
)

# -----------------------------
# WHAT‑IF — RECALCULAR PD
# -----------------------------
st.subheader("🔄 Resultado What‑If")

# Apenas variáveis WOE são usadas no modelo WOE
woe_vars = {k: client[k] for k in client.index if k.endswith("_woe")}

# PD baseline:
pd_original = client["PD"]

# Simulação com sliders (mantemos WOE igual neste protótipo)
pd_new = recalc_pd(model, woe_vars)

# DECISÃO COM BASE NO WHAT‑IF
decision, reasons = decision_dynamic(
    pd=pd_new,
    africa_score=new_africa,
    proxy_income=new_income,
    georisk=new_geo,
    policy=policy
)

limit = simulate_limit(new_income, new_africa, policy)

# -----------------------------
# MOSTRAR RESULTADOS
# -----------------------------
col1, col2 = st.columns(2)
col1.metric("PD Original", f"{pd_original:.3f}")
col2.metric("PD Simulado", f"{pd_new:.3f}")

st.metric("Decisão", decision)
st.metric("Limite Sugerido", f"{limit:.0f} Kz")

st.write("### 🧠 Razões da Decisão")
st.write(reasons)

# -----------------------------
# GRÁFICO PD Original vs Simulado
# -----------------------------
st.subheader("📈 Comparação: PD Original vs Simulado")

fig, ax = plt.subplots()
ax.bar(["Original", "Simulado"], [pd_original, pd_new], color=["blue", "orange"])
st.pyplot(fig)
