import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==============================
# CONFIGURAÇÃO DA PÁGINA
# ==============================

st.set_page_config(
    page_title="Predição de Risco de Defasagem",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Predição de Risco de Defasagem Escolar")
st.markdown(
"""
Aplicação de **Machine Learning** para identificar alunos com risco de defasagem.

Modelo treinado com **Random Forest** utilizando indicadores educacionais.
"""
)

# ==============================
# CARREGAR MODELO
# ==============================

modelo = joblib.load("modelo_risco.pkl")
scaler = joblib.load("scaler.pkl")
colunas_modelo = joblib.load("colunas_modelo.pkl")

# ==============================
# SIDEBAR INPUT
# ==============================

st.sidebar.header("📋 Dados do Aluno")

idade = st.sidebar.number_input("Idade", 6, 25, 12)

ida = st.sidebar.slider("IDA", 0.0, 10.0, 6.0)
ieg = st.sidebar.slider("IEG", 0.0, 10.0, 6.0)
ipp = st.sidebar.slider("IPP", 0.0, 10.0, 6.0)
ips = st.sidebar.slider("IPS", 0.0, 10.0, 6.0)
iaa = st.sidebar.slider("IAA", 0.0, 10.0, 6.0)
ipv = st.sidebar.slider("IPV", 0.0, 10.0, 6.0)

inde23 = st.sidebar.slider("INDE 2023", 0.0, 10.0, 6.0)
inde24 = st.sidebar.slider("INDE 2024", 0.0, 10.0, 6.0)

anos_pm = st.sidebar.slider("Anos na Passos Mágicos", 0, 10, 2)

# ==============================
# FEATURE ENGINEERING
# ==============================

trio_principal = (ida + ieg + ipp) / 3

contagem_baixos = sum([
    ida < 5,
    ieg < 5,
    ipp < 5
])

ipp_ida_media = (ipp + ida) / 2

variacao = inde24 - inde23

deterioracao = 1 if variacao < 0 else 0

# ==============================
# DATAFRAME DE INPUT
# ==============================

dados = pd.DataFrame({

    "Idade":[idade],
    "IDA":[ida],
    "IEG":[ieg],
    "IPP":[ipp],
    "IPS":[ips],
    "IAA":[iaa],
    "IPV":[ipv],
    "INDE 23":[inde23],
    "INDE_2024_CLEAN":[inde24],
    "ANOS_NA_PM":[anos_pm],

    "TRIO_PRINCIPAL":[trio_principal],
    "CONTAGEM_BAIXOS":[contagem_baixos],
    "IPP_IDA_MEDIA":[ipp_ida_media],
    "VARIACAO_23_24":[variacao],
    "DETERIORACAO_2024":[deterioracao]

})

# garantir mesmas colunas do modelo
dados = dados.reindex(columns=colunas_modelo, fill_value=0)

# escala
dados_scaled = scaler.transform(dados)

# ==============================
# PREDIÇÃO
# ==============================

if st.button("🔎 Calcular Risco"):

    prob = modelo.predict_proba(dados_scaled)[0][1]
    pred = modelo.predict(dados_scaled)[0]

    col1, col2 = st.columns(2)

    # RESULTADO
    with col1:

        st.subheader("Resultado da Predição")

        if pred == 1:
            st.error(f"⚠️ Risco de Defasagem: {prob:.2%}")
        else:
            st.success(f"✅ Baixo risco: {prob:.2%}")

    # GRÁFICO
    with col2:

        fig, ax = plt.subplots()

        ax.bar(
            ["Baixo risco","Risco"],
            [1-prob, prob]
        )

        ax.set_ylabel("Probabilidade")
        ax.set_title("Distribuição de Probabilidade")

        st.pyplot(fig)

# ==============================
# IMPORTÂNCIA DAS FEATURES
# ==============================

st.subheader("📈 Variáveis Mais Importantes do Modelo")

importancias = modelo.feature_importances_

features = pd.DataFrame({
    "Feature": colunas_modelo,
    "Importância": importancias
}).sort_values(by="Importância", ascending=False).head(10)

fig2, ax2 = plt.subplots()

ax2.barh(
    features["Feature"],
    features["Importância"]
)

ax2.invert_yaxis()
ax2.set_title("Top 10 Features")

st.pyplot(fig2)

# ==============================
# FOOTER
# ==============================

st.markdown("---")
st.markdown(
"""
Projeto de Machine Learning aplicado à educação.

Modelo: Random Forest  
Aplicação construída com Streamlit.
"""
)
