import streamlit as st
import pandas as pd

from modules.pipeline import run_pipeline

st.title("🚀 Pipeline Automático End‑to‑End")
st.write("Dataset → ChiMerge → WOE → Modelo → AUC → KS → ROC → PSI → PDF")

# ----------------------------------------------
# Dataset base
# ----------------------------------------------
df = st.session_state.get("dataset")

if df is None:
    st.error("Primeiro gera um dataset na aba 'Gerar Dataset'.")
    st.stop()
    raise SystemExit(0)

# ----------------------------------------------
# Upload nova população para PSI
# ----------------------------------------------
uploaded = st.file_uploader("Carregar nova população (opcional, para PSI)", type=["csv"])

if uploaded:
    new_df = pd.read_csv(uploaded)
    st.success("Nova população carregada!")
else:
    new_df = None

max_bins = st.slider("Número máximo de bins por variável", 3, 12, 6)

# ----------------------------------------------
# BOTÃO PRINCIPAL
# ----------------------------------------------
if st.button("EXECUTAR PIPELINE COMPLETO"):

    st.info("⏳ A executar pipeline… isto pode demorar alguns segundos.")

    df_final, auc, ks, psi = run_pipeline(
        df=df, new_df=new_df, max_bins=max_bins
    )

    st.success("✅ Pipeline concluído com sucesso!")

    # Resultados principais
    st.metric("AUC Final", f"{auc:.3f}")
    st.metric("KS Final", f"{ks:.3f}")
    st.metric("PSI (se aplicável)", f"{psi:.3f}")

    with open("Pipeline_Final.pdf", "rb") as f:
        st.download_button(
            label="📥 Download do Relatório PDF Completo",
            data=f,
            file_name="Pipeline_Final.pdf"
        )

    st.write("✅ Dataset final (com WOE):")
    st.dataframe(df_final.head())
