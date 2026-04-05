import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.psi import calculate_psi

st.title("📈 PSI — Population Stability Index")

st.write("""
O PSI compara a população de treino com uma nova população para detetar drift.
""")

# --------------------------
# Seleção das duas populações
# --------------------------
df = st.session_state.get("dataset")

if df is None:
    st.error("⚠️ Gera primeiro um dataset na aba 'Gerar Dataset'.")
    st.stop()
    raise SystemExit(0)

feature = st.selectbox(
    "Escolhe a variável (crua ou WOE):",
    [col for col in df.columns if col != "default"]
)

st.write("### Carregar segunda população (produção)")
uploaded = st.file_uploader("Carrega um CSV com a nova população")

if uploaded:
    new_df = pd.read_csv(uploaded)
    st.success("Nova população carregada!")

    if feature not in new_df.columns:
        st.error(f"A nova população não contém a variável '{feature}'.")
        st.stop()

    if st.button("Calcular PSI"):
        # valores base
        base_values = df[feature].dropna()
        new_values = new_df[feature].dropna()

        psi_table, psi_total = calculate_psi(
            base_values=base_values,
            new_values=new_values,
            bins=10
        )

        st.subheader("📌 Tabela PSI")
        st.dataframe(psi_table)

        st.metric("PSI Total", f"{psi_total:.3f}")

        # interpretação
        if psi_total < 0.1:
            status = "✅ Estável"
        elif psi_total < 0.25:
            status = "⚠️ Mudança moderada — monitorizar"
        else:
            status = "❌ Instabilidade severa — possível drift"

        st.write(f"### Interpretação: {status}")

        # gráficos
        st.subheader("📊 Distribuição Base vs Nova")

        fig, ax = plt.subplots()
        ax.plot(psi_table["bin"], psi_table["base_perc"], label="Base")
        ax.plot(psi_table["bin"], psi_table["new_perc"], label="Nova")
        ax.set_xlabel("Bins")
        ax.set_ylabel("Proporção")
        ax.legend()
        st.pyplot(fig)
