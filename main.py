import streamlit as st
import joblib

# Configuração da página
st.set_page_config(page_title="Triagem Corona Vírus", page_icon="🦠", layout="wide")

# Título e Instruções
st.title("Triagem Corona Vírus")
st.write("Selecione os sintomas do paciente:")
st.write("")

# Seleção de sintomas
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    febre = st.checkbox("Febre")
    cansaco = st.checkbox("Cansaço")

with col2:
    tosse_seca = st.checkbox("Tosse seca")
    espirro = st.checkbox("Espirro")

with col3:
    dores_no_corpo = st.checkbox("Dores no corpo")
    coriza = st.checkbox("Coriza")

with col4:
    dor_de_garganta = st.checkbox("Dor de garganta")
    diarreia = st.checkbox("Diarreia")

with col5:
    dor_de_cabeca = st.checkbox("Dor de cabeça")
    falta_de_ar = st.checkbox("Falta de ar")

# Sintomas do Paciente
# Febre, Cansaço, Tosse, Espirro, Dores no Corpo, Corizando, Dor de Garganta, Diarreia, Dor de CAbeça, Falta de Ar
X = [
    [
        febre,
        cansaco,
        tosse_seca,
        espirro,
        dores_no_corpo,
        coriza,
        dor_de_garganta,
        diarreia,
        dor_de_cabeca,
        falta_de_ar,
    ]
]

# Carregar modelos
gaussian_nb_model = None
dt_classifier_model = None
knn_classifier_model = None

if not gaussian_nb_model or not dt_classifier_model or not knn_classifier_model:
    gaussian_nb_model = joblib.load("./gaussian_nb_model.pkl")
    dt_classifier_model = joblib.load("./dt_classifier_model.pkl")
    knn_classifier_model = joblib.load("./knn_classifier_model.pkl")

# Seleção do modelo
model_list = [dt_classifier_model, knn_classifier_model, gaussian_nb_model]
selected_model = st.selectbox("Selecione um modelo:", model_list)

# Avaliação dos sintomas pelo modelo
result = None

avaliar = st.button("Avaliar")

if avaliar:
    model_result = selected_model.predict(X)
    if model_result == 0:
        result = "Paciente não apresenta sintomas da COVID-19"
    else:
        result = "Recomenda-se fazer o teste da COVID-19"

# Resultado
st.write("")
st.write("Resultado:")
if result:
    st.info(result)
