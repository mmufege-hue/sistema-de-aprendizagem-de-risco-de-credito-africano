import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

st.set_page_config(page_title="CreditMasterLab", layout="wide")

st.title("💳 CreditMasterLab — Versão Inicial")
st.write("Laboratório interativo para aprender Risco de Crédito com profundidade.")

# ------------------------------------------
# SIDEBAR
# ------------------------------------------
menu = st.sidebar.selectbox(
    "Menu",
    [
        "📘 Teoria Base",
        "🏗️ Gerar Dataset",
        "📊 Modelar Scoring",
        "🔍 Explicar Modelo"
    ]
)

# ------------------------------------------
# 1. TEORIA BASE
# ------------------------------------------
if menu == "📘 Teoria Base":
    st.header("📘 Fundamentos de Risco de Crédito")

    teoria = st.selectbox(
        "Escolhe um tema:",
        ["Probabilidade de Default (PD)",
         "WOE & Information Value",
         "Regressão Logística",
         "Métricas: AUC, KS"]
    )

    if teoria == "Probabilidade de Default (PD)":
        st.subheader("Probabilidade de Default (PD)")
        st.write("""
        PD é a probabilidade de um cliente entrar em incumprimento num
        determinado horizonte de tempo. É a base de qualquer modelo de scoring.
        """)

    if teoria == "WOE & Information Value":
        st.subheader("WOE & IV")
        st.write("""
        WOE transforma variáveis categóricas/numéricas em bins com poder
        preditivo mais estável.  
        IV mede a força da variável para prever default.
        """)

# ------------------------------------------
# 2. GERAR DATASET
# ------------------------------------------
elif menu == "🏗️ Gerar Dataset":
    st.header("🏗️ Gerador Automático de Datasets de Crédito")

    n = st.slider("Número de clientes:", 500, 20000, 2000)

    if st.button("Gerar"):
        renda = np.random.normal(1500, 500, n)
        idade = np.random.randint(18, 70, n)
        atraso30 = np.random.binomial(1, 0.15, n)

        # default com lógica simples
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

        st.write(df.head())

        st.session_state["dataset"] = df
        st.success("✅ Dataset gerado e guardado em memória!")

# ------------------------------------------
# 3. MODELAR SCORING
# ------------------------------------------
elif menu == "📊 Modelar Scoring":
    st.header("📊 Criar Modelo de Regressão Logística")

    if "dataset" not in st.session_state:
        st.error("Gera primeiro um dataset.")
    else:
        df = st.session_state["dataset"]

        X = df[["renda", "idade", "atraso30"]]
        y = df["default"]

        model = LogisticRegression()
        model.fit(X, y)

        st.success("✅ Modelo treinado!")
        st.write("Coeficientes:")

        coef_df = pd.DataFrame({
            "Variável": X.columns,
            "Coeficiente": model.coef_[0]
        })

        st.write(coef_df)

        # AUC
        preds = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, preds)
        st.metric("AUC do Modelo", f"{auc:.3f}")

# ------------------------------------------
# 4. EXPLICAÇÃO DO MODELO
# ------------------------------------------
elif menu == "🔍 Explicar Modelo":
    st.header("🔍 Explicação Automática dos Coeficientes")

    if "dataset" not in st.session_state:
        st.error("Gera e treina um modelo primeiro.")
    else:
        st.write("""
        Aqui explica-se automaticamente o impacto de cada variável.
        (A versão completa terá SHAP, WOE e gráficos avançados.)
        """)