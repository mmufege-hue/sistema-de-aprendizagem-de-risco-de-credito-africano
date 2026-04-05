import streamlit as st
import pandas as pd
import numpy as np

st.title("🏗️ Gerar Dataset — CreditMasterLab")

# Inicialização do estado
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

st.write("Use o botão abaixo para gerar um dataset de risco de crédito.")

n = st.slider("Número de clientes:", 500, 20000, 2000)

if st.button("Gerar Dataset"):
    renda = np.random.normal(1500, 500, n)
    idade = np.random.randint(18, 70, n)
    atraso30 = np.random.binomial(1, 0.15, n)

    prob_default = (
        0.02 +
        0.0002 * (40 - idade) +
        0.00015 * (2000 - renda) +
        0.25 * atraso30
    )

    default = np.random.binomial(1, np.clip(prob_default, 0.01, 0.7))

    df = pd.DataFrame({
        "renda": renda,
        "idade": idade,
        "atraso30": atraso30,
        "default": default
    })

    st.session_state["dataset"] = df

    st.success("✅ Dataset gerado com sucesso!")
    st.dataframe(df.head())
