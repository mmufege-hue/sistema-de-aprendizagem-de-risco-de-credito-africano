import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from modules.model_logit_woe import train_logit_woe
from modules.stress_ifrs9 import run_ifrs9_stress_test

st.title("🌪️ IFRS 9 — Stress Test Macroeconómico")

# -------------------------
# Validar dataset
# -------------------------
df = st.session_state.get("dataset")

if df is None:
    st.error("Gera primeiro um dataset com variáveis WOE")
    st.stop()
    raise SystemExit(0)

# -------------------------
# Treinar modelo WOE
# -------------------------
st.subheader("📈 Treinar Modelo WOE para obter PD baseline")

model, preds, auc, ks, coef_df = train_logit_woe(df)
df["PD_baseline"] = preds

st.success(f"Modelo treinado — AUC {auc:.3f} | KS {ks:.3f}")

# -------------------------
# Configurar Cenários
# -------------------------
st.subheader("🌍 Definir Cenários Macroeconómicos")

base_infl = st.number_input("Inflação (BASE)", value=0.00)
adv_infl = st.number_input("Inflação (ADVERSO)", value=0.05)
sev_infl = st.number_input("Inflação (SEVERO)", value=0.12)

base_gdp = st.number_input("PIB (BASE)", value=0.00)
adv_gdp = st.number_input("PIB (ADVERSO)", value=-0.02)
sev_gdp = st.number_input("PIB (SEVERO)", value=-0.05)

base_fx = st.number_input("Câmbio (BASE)", value=0.00)
adv_fx = st.number_input("Câmbio (ADVERSO)", value=0.10)
sev_fx = st.number_input("Câmbio (SEVERO)", value=0.25)

scenarios = {
    "BASE": {"inflation": base_infl, "gdp": base_gdp, "fx": base_fx},
    "ADVERSO": {"inflation": adv_infl, "gdp": adv_gdp, "fx": adv_fx},
    "SEVERO": {"inflation": sev_infl, "gdp": sev_gdp, "fx": sev_fx},
}

# -------------------------
# BOTÃO — Rodar Stress Test
# -------------------------
if st.button("Rodar Stress Test IFRS 9"):

    results = run_ifrs9_stress_test(df, preds, scenarios)

    st.success("✅ Stress test concluído!")

    # Mostrar ECL agregado por cenário
    scenario_names = []
    total_ecls = []

    for name, scenario_df in results:
        scenario_names.append(name)
        total_ecls.append(scenario_df["ECL"].sum())

    # -------------------------
    # Gráfico de barras ECL total
    # -------------------------
    st.subheader("💰 ECL Total por Cenário")

    fig, ax = plt.subplots()
    ax.bar(scenario_names, total_ecls, color=["blue", "orange", "red"])
    ax.set_ylabel("ECL Total")
    st.pyplot(fig)

    # Mostrar tabelas detalhadas
    for name, scenario_df in results:
        st.write(f"### 🧾 Cenário: {name}")
        st.dataframe(scenario_df[["PD", "LGD", "EAD", "ECL"]].head())
