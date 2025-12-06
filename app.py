import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Cargar el modelo y las columnas guardadas
@st.cache_resource
def load_artifacts():
    modelo = pickle.load(open('modelo_xgboost_final.pkl', 'rb'))
    columnas = pickle.load(open('nombres_columnas.pkl', 'rb'))
    return modelo, columnas

try:
    model, model_columns = load_artifacts()
except FileNotFoundError:
    st.error("No se encontraron los archivos .pkl. Asegúrate de haber ejecutado el PASO 1 y tenerlos en la misma carpeta.")
    st.stop()

# 2. Título y Descripción
st.title("📊 Predicción de Churn (XGBoost Hiperparametrizado)")
st.markdown("""
Esta aplicación utiliza un modelo **XGBoost** optimizado para predecir la probabilidad 
de que un cliente abandone la compañía (Churn).
""")

# 3. Sidebar: Inputs del Usuario
st.sidebar.header("Datos del Cliente")

# --- VARIABLES NUMÉRICAS ---
# Ajusta los rangos (min_value, max_value) según tus datos reales
tenure = st.sidebar.slider("Meses de Permanencia (Tenure)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Cargos Mensuales ($)", min_value=0.0, max_value=150.0, value=50.0)
total_charges = st.sidebar.number_input("Cargos Totales ($)", min_value=0.0, max_value=10000.0, value=500.0)

# --- VARIABLES CATEGÓRICAS ---
# Estas deben coincidir con las opciones originales antes del One-Hot Encoding
gender = st.sidebar.selectbox("Género", ["Female", "Male"])
partner = st.sidebar.selectbox("Tiene Pareja (Partner)", ["Yes", "No"])
dependents = st.sidebar.selectbox("Personas a cargo (Dependents)", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Servicio Telefónico", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Múltiples Líneas", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Seguridad Online", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Backup Online", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Protección de Dispositivo", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Soporte Técnico", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Facturación sin papel", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Método de Pago", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# 4. Botón de Predicción
if st.button("Calcular Probabilidad de Churn"):
    
    # A. Crear un DataFrame con los datos crudos del usuario
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [0], # Asumimos 0 o agrega un checkbox si es relevante
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # B. Preprocesamiento (One-Hot Encoding)
    # Convertimos las categóricas a dummies
    input_data_encoded = pd.get_dummies(input_data)

    # C. ALINEACIÓN DE COLUMNAS (El paso más importante)
    # Reindexamos para que el dataframe tenga EXACTAMENTE las mismas columnas que el entrenamiento
    # Las columnas faltantes se rellenan con 0
    input_data_final = input_data_encoded.reindex(columns=model_columns, fill_value=0)

    # D. Predicción
    prediction = model.predict(input_data_final)
    probability = model.predict_proba(input_data_final)[:, 1][0]

    # 5. Mostrar Resultados
    st.divider()
    st.subheader("Resultados del Modelo")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Probabilidad de Fuga", value=f"{probability:.2%}")
    
    with col2:
        if probability > 0.5:
            st.error("⚠️ ALTO RIESGO: Se predice que el cliente se irá.")
        else:
            st.success("✅ BAJO RIESGO: El cliente probablemente se quede.")

    # Detalle técnico opcional
    with st.expander("Ver datos procesados (para depuración)"):
        st.write(input_data_final)