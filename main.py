import streamlit as st

st.set_page_config(page_title="CreditMasterLab", layout="wide")

# Inicializa estado global
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

st.title("CreditMasterLab — Página Inicial")

st.write("""
Bem-vindo ao Sistema Africano de Aprendizagem de Risco de Crédito.

Use o menu lateral (pastas *pages*) para navegar pelas funcionalidades.
Comece sempre pela aba **Gerar Dataset**.
""")
