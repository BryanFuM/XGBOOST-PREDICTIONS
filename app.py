import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Cargar modelo, columnas Y EL ESCALADOR
@st.cache_resource
def load_artifacts():
    modelo = pickle.load(open('modelo_xgboost_final.pkl', 'rb'))
    columnas = pickle.load(open('nombres_columnas.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb')) # <--- NUEVO
    return modelo, columnas, scaler

try:
    model, model_columns, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Faltan archivos .pkl (modelo, columnas o scaler).")
    st.stop()

st.title("📊 Predicción de Churn (XGBoost)")

st.sidebar.header("Datos del Cliente")

# --- INPUTS ---
# Nombres deben coincidir con tu CSV limpio (snake_case)
tenure = st.sidebar.slider("Meses (tenure)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Cargos Mensuales", min_value=0.0, value=50.0)
total_charges = st.sidebar.number_input("Cargos Totales", min_value=0.0, value=500.0)

# Categóricas (Ejemplo resumido, asegúrate de tener todas las que usaste)
gender = st.sidebar.selectbox("Género", ["Female", "Male"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Añadir input para senior_citizen en el sidebar
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])

if st.button("Calcular"):
    
    # A. Crear DataFrame con todas las columnas que espera el modelo (inicializadas a 0)
    input_data_final = pd.DataFrame(columns=model_columns)
    input_data_final.loc[0] = 0  # Inicializar fila con ceros
    
    # B. Escalar columnas numéricas usando DataFrame con nombres (evita warning)
    numeric_df = pd.DataFrame(
        [[tenure, monthly_charges, total_charges]], 
        columns=['tenure', 'monthly_charges', 'total_charges']
    )
    scaled_values = scaler.transform(numeric_df)
    
    input_data_final['tenure'] = scaled_values[0, 0]
    input_data_final['monthly_charges'] = scaled_values[0, 1]
    input_data_final['total_charges'] = scaled_values[0, 2]
    input_data_final['senior_citizen'] = senior_citizen
    
    # C. Asignar valores One-Hot manualmente (las columnas base quedan en 0)
    # Gender: base es Female, solo activamos Male si corresponde
    if gender == "Male":
        input_data_final['gender_Male'] = 1
    
    # Partner: base es No
    if partner == "Yes":
        input_data_final['partner_Yes'] = 1
    
    # Dependents: base es No
    if dependents == "Yes":
        input_data_final['dependents_Yes'] = 1
    
    # Phone Service: base es No
    if phone_service == "Yes":
        input_data_final['phone_service_Yes'] = 1
    
    # Multiple Lines: base es No
    if multiple_lines == "Yes":
        input_data_final['multiple_lines_Yes'] = 1
    elif multiple_lines == "No phone service":
        input_data_final['multiple_lines_No phone service'] = 1
    
    # Internet Service: base es DSL
    if internet_service == "Fiber optic":
        input_data_final['internet_service_Fiber optic'] = 1
    elif internet_service == "No":
        input_data_final['internet_service_No'] = 1
    
    # Online Security: base es No
    if online_security == "Yes":
        input_data_final['online_security_Yes'] = 1
    elif online_security == "No internet service":
        input_data_final['online_security_No internet service'] = 1
    
    # Online Backup: base es No
    if online_backup == "Yes":
        input_data_final['online_backup_Yes'] = 1
    elif online_backup == "No internet service":
        input_data_final['online_backup_No internet service'] = 1
    
    # Device Protection: base es No
    if device_protection == "Yes":
        input_data_final['device_protection_Yes'] = 1
    elif device_protection == "No internet service":
        input_data_final['device_protection_No internet service'] = 1
    
    # Tech Support: base es No
    if tech_support == "Yes":
        input_data_final['tech_support_Yes'] = 1
    elif tech_support == "No internet service":
        input_data_final['tech_support_No internet service'] = 1
    
    # Streaming TV: base es No
    if streaming_tv == "Yes":
        input_data_final['streaming_tv_Yes'] = 1
    elif streaming_tv == "No internet service":
        input_data_final['streaming_tv_No internet service'] = 1
    
    # Streaming Movies: base es No
    if streaming_movies == "Yes":
        input_data_final['streaming_movies_Yes'] = 1
    elif streaming_movies == "No internet service":
        input_data_final['streaming_movies_No internet service'] = 1
    
    # Contract: base es Month-to-month
    if contract == "One year":
        input_data_final['contract_One year'] = 1
    elif contract == "Two year":
        input_data_final['contract_Two year'] = 1
    
    # Paperless Billing: base es No
    if paperless_billing == "Yes":
        input_data_final['paperless_billing_Yes'] = 1
    
    # Payment Method: base es Bank transfer (automatic)
    if payment_method == "Credit card (automatic)":
        input_data_final['payment_method_Credit card (automatic)'] = 1
    elif payment_method == "Electronic check":
        input_data_final['payment_method_Electronic check'] = 1
    elif payment_method == "Mailed check":
        input_data_final['payment_method_Mailed check'] = 1
    
    # D. Predecir
    probability = model.predict_proba(input_data_final)[:, 1][0]
    probability = float(probability)  # Convertir a float nativo de Python

    # Mostrar resultado con escala más detallada
    st.subheader("Resultado de la Predicción")
    
    # Barra de progreso visual
    st.progress(probability)
    st.metric("Probabilidad de Fuga", f"{probability:.2%}")
    
    # Escala de riesgo con umbrales estándar
    if probability > 0.50:
        st.error("🔴 ALTO RIESGO DE CHURN - ¡Acción inmediata requerida!")
    elif probability > 0.30:
        st.warning("🟡 RIESGO MEDIO DE CHURN - Monitorear de cerca")
    else:
        st.success("🟢 BAJO RIESGO DE CHURN - Cliente estable")
    
    # Mostrar factores de riesgo detectados
    with st.expander("📋 Ver factores de riesgo", expanded=True):
        factores_riesgo = []
        factores_proteccion = []
        
        # Factores que AUMENTAN el riesgo
        if tenure < 12:
            factores_riesgo.append("⚠️ Cliente nuevo (menos de 12 meses)")
        if contract == "Month-to-month":
            factores_riesgo.append("⚠️ Contrato mes a mes (sin compromiso)")
        if internet_service == "Fiber optic":
            factores_riesgo.append("⚠️ Fibra óptica (servicio más caro)")
        if online_security == "No":
            factores_riesgo.append("⚠️ Sin seguridad online")
        if tech_support == "No":
            factores_riesgo.append("⚠️ Sin soporte técnico")
        if payment_method == "Electronic check":
            factores_riesgo.append("⚠️ Pago con cheque electrónico")
        if paperless_billing == "Yes":
            factores_riesgo.append("⚠️ Facturación sin papel")
        if monthly_charges > 70:
            factores_riesgo.append(f"⚠️ Cargos mensuales altos (${monthly_charges:.2f})")
        if partner == "No":
            factores_riesgo.append("⚠️ Sin pareja")
        if dependents == "No":
            factores_riesgo.append("⚠️ Sin dependientes")
            
        # Factores que DISMINUYEN el riesgo
        if tenure >= 24:
            factores_proteccion.append("✅ Cliente leal (24+ meses)")
        if contract in ["One year", "Two year"]:
            factores_proteccion.append(f"✅ Contrato {contract}")
        if online_security == "Yes":
            factores_proteccion.append("✅ Con seguridad online")
        if tech_support == "Yes":
            factores_proteccion.append("✅ Con soporte técnico")
        if payment_method in ["Credit card (automatic)", "Bank transfer (automatic)"]:
            factores_proteccion.append("✅ Pago automático")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔴 Factores de Riesgo:**")
            if factores_riesgo:
                for f in factores_riesgo:
                    st.write(f)
            else:
                st.write("Ninguno detectado")
        
        with col2:
            st.markdown("**🟢 Factores de Protección:**")
            if factores_proteccion:
                for f in factores_proteccion:
                    st.write(f)
            else:
                st.write("Ninguno detectado")