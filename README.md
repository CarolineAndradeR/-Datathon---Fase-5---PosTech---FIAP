# 📊 Predição de Risco de Defasagem Escolar com Machine Learning

## 📌 Visão Geral

Este projeto utiliza **Machine Learning** para identificar **alunos com maior risco de defasagem educacional** a partir de indicadores acadêmicos, comportamentais e psicopedagógicos.

A proposta é apoiar instituições educacionais na **identificação precoce de alunos em risco**, permitindo ações pedagógicas mais rápidas e direcionadas.

O modelo foi desenvolvido em **Python**, utilizando bibliotecas de ciência de dados e o ecossistema de Machine Learning do **Scikit-learn**.

---

# 🎯 Objetivo

Construir um modelo preditivo capaz de:

* Identificar alunos com **maior risco de defasagem**
* Explicar **quais fatores mais influenciam esse risco**
* Disponibilizar uma base para **ferramentas de apoio à decisão educacional**

---

# 📂 Estrutura do Projeto

```
projeto-risco-defasagem/

│
├── data/
│   └── dados_limpos_2024.csv
│
├── notebooks/
│   └── notebook_modelo_risco_defasagem.ipynb
│
├── model/
│   └── modelo_risco.pkl
│
├── app/
│   └── app_streamlit.py
│
├── README.md
└── requirements.txt
```

---

# 📊 Dataset

O dataset contém informações anonimizadas de alunos, incluindo:

### Informações demográficas

* Idade
* Gênero
* Ano de ingresso

### Indicadores acadêmicos

* **IDA** – Indicador de desempenho acadêmico
* **IEG** – Indicador de engajamento
* **IAA** – Indicador de aprendizagem
* **IPS** – Indicador socioemocional
* **IPP** – Indicador psicopedagógico
* **IPV** – Indicador de vulnerabilidade

### Histórico de desempenho

* INDE 2022
* INDE 2023
* INDE 2024

Cada linha representa **um aluno**.

---

# ⚙️ Pipeline de Ciência de Dados

O projeto segue um fluxo padrão de **Machine Learning aplicado**.

## 1️⃣ Análise Exploratória

Foram avaliados:

* distribuição das variáveis
* qualidade dos dados
* correlações entre indicadores educacionais

Correlação relevante observada:

| Indicadores | Correlação |
| ----------- | ---------- |
| IPP × IPV   | ~0.75      |
| IDA × IEG   | ~0.53      |
| IDA × IPV   | ~0.51      |

Isso indica relações importantes entre **desempenho, engajamento e fatores psicopedagógicos**.

---

# 🧠 Feature Engineering

Foram criadas variáveis derivadas para capturar padrões mais complexos.

### TRIO_PRINCIPAL

Combina três indicadores-chave:

```
TRIO_PRINCIPAL = média(IDA, IEG, IPP)
```

Representa um **indicador consolidado de desempenho educacional**.

---

### CONTAGEM_BAIXOS

Conta quantos indicadores estão abaixo de um nível mínimo.

Permite identificar alunos com **dificuldade em múltiplas dimensões**.

---

### IPP_IDA_MEDIA

Média entre:

* desempenho acadêmico
* indicador psicopedagógico

Ajuda a identificar **desequilíbrio entre desempenho e fatores comportamentais**.

---

### VARIACAO_23_24

Mede mudança de desempenho entre:

```
INDE 2023 → INDE 2024
```

Captura **tendência recente de evolução ou deterioração**.

---

### DETERIORACAO_2024

Variável binária que identifica **queda relevante de desempenho no último ano**.

Essa variável é importante porque mudanças recentes costumam preceder a defasagem.

---

# 🎯 Definição da Variável de Risco

A variável alvo do modelo é:

```
RISCO_DEFASAGEM
```

Ela é criada a partir do indicador **IAN**.

O código classifica como risco os alunos que estão **no percentil superior do indicador**:

```python
limite = df['IAN'].quantile(0.88)
df['RISCO_DEFASAGEM'] = (df['IAN'] >= limite)
```

Isso faz com que aproximadamente **11–12% dos alunos sejam classificados como risco**, concentrando o modelo nos casos mais críticos.

---

# 🤖 Modelos Testados

Foram avaliados três algoritmos:

### Regressão Logística

Modelo estatístico simples usado como baseline.

---

### Random Forest

Modelo baseado em **conjunto de árvores de decisão**, capaz de capturar relações não lineares entre variáveis.

Implementado usando **Scikit-learn**.

---

### XGBoost

Algoritmo avançado de boosting amplamente utilizado em competições de ciência de dados.

---

# 📈 Resultados

O modelo **Random Forest apresentou o melhor desempenho**.

### Métrica utilizada

**ROC AUC**

Resultado:

```
ROC AUC = 0.928
```

Esse valor indica **excelente capacidade de separação entre alunos em risco e não risco**.

---

# 📊 Avaliação do Modelo

Foram utilizados dois gráficos principais para avaliação.

## Curva ROC

Mostra a capacidade do modelo de distinguir as duas classes.

Quanto mais próxima do canto superior esquerdo, melhor o desempenho.

---

## Matriz de Confusão

Permite avaliar:

* acertos do modelo
* falsos positivos
* falsos negativos

Isso ajuda a entender **como o modelo erra e acerta nas previsões**.

---

# 🔍 Explicabilidade do Modelo

Para entender **quais fatores influenciam mais a previsão**, foram analisadas:

* Feature Importance
* SHAP Values

Essas técnicas mostram **quais variáveis têm maior impacto na previsão de risco**.

---

# 🚀 Aplicação

O modelo pode ser integrado em uma aplicação web usando:

**Streamlit**

Essa aplicação permite:

* inserir dados de um aluno
* obter previsão de risco
* visualizar explicação da previsão

---

# 💡 Impacto Educacional

Ferramentas como essa podem ajudar instituições a:

* identificar alunos em risco mais cedo
* direcionar suporte pedagógico
* otimizar recursos educacionais
* melhorar retenção e desempenho acadêmico

---

# 🛠 Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* **Scikit-learn**
* **Streamlit**


# 📄 Licença

Este projeto é destinado a fins educacionais e acadêmicos.
