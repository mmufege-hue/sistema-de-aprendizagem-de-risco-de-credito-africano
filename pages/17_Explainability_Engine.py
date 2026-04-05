import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from modules.model_logit_woe import train_logit_woe
from modules.policy_studio import build_policy
from modules.decision_engine_dynamic import decision_dynamic
from modules.explainability import compute_shap_values, explain_decision_with_rules


st.title("🧠 Explainability Engine — SHAP + Regras de Política")

df = st.session_state.get("dataset")

if df is None:
    st.error("Precisas gerar ou carregar um dataset antes.")
    st.stop()
    raise SystemExit(0)

# -----------------------------
# Treinar modelo para PD
# -----------------------------
st.subheader("Treinar Modelo WOE")
model, preds, auc, ks, coef_df = train_logit_woe(df)
df["PD"] = preds

st.success(f"Modelo carregado — AUC {auc:.3f} | KS {ks:.3f}")

# -----------------------------
# Selecionar cliente
# -----------------------------
st.subheader("Selecionar Cliente para Explicação")

idx = st.number_input("ID do Cliente:", 0, len(df)-1, 0)
client = df.iloc[idx]

st.write("### Dados do Cliente")
st.dataframe(client.to_frame().T)

# -----------------------------
# Construir política automática
# -----------------------------
st.subheader("Política de crédito utilizada")

policy = build_policy()

st.json(policy)

# -----------------------------
# SHAP Values
# -----------------------------
st.subheader("📌 Explicação do Modelo (SHAP)")

# Apenas variáveis WOE vão para o modelo
woe_vars = [c for c in df.columns if c.endswith("_woe")]
X = df[woe_vars]

explainer, shap_values = compute_shap_values(model, X)

# gráfico SHAP waterfall para cliente selecionado
fig, ax = plt.subplots(figsize=(8, 5))
shap.waterfall_plot(
    shap.Explanation(values=shap_values[idx],
                     base_values=explainer.expected_value,
                     feature_names=woe_vars,
                     data=X.iloc[idx].values),
    max_display=10
)
st.pyplot(fig)

# -----------------------------
# DECISÃO DO MOTOR + REGRAS
# -----------------------------
st.subheader("📌 Explicação do Motor de Decisão (Regras)")

decision, reasons = decision_dynamic(
    pd=client["PD"],
    africa_score=client["africa_score"],
    proxy_income=client["proxy_income"],
    georisk=client["georisk"],
    policy=policy
)

st.metric("Decisão Final", decision)
st.write("### 🧠 Razões:")
for r in reasons:
    st.write("- ", r)

# -----------------------------
# EXPLICAÇÃO DAS REGRAS
# -----------------------------
st.subheader("📘 Explicação Detalhada da Política")

rule_reasons = explain_decision_with_rules(client, policy)
for r in rule_reasons:
    st.write("- ", r)
