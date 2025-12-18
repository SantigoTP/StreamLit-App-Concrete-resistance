import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Configuración de la página
st.set_page_config(page_title="Predicción de Resistencia del Concreto", page_icon="🏗️")

# 2. Cargar el modelo
@st.cache_resource # Esto evita que el modelo se recargue cada vez que mueves un botón
def load_model():
    return joblib.load('modelo_concreto.pkl')

model = load_model()

# 3. Interfaz de usuario
st.title("🏗️ Analizador de Resistencia de Concreto")
st.markdown("""
Esta aplicación predice la **Resistencia a la Compresión (MPa)** del concreto basada en su composición química y edad.
""")

st.sidebar.header("Parámetros de Entrada")

def user_input_features():
    # Creamos sliders y inputs en la barra lateral
    cement = st.sidebar.number_input("Cemento (kg/m³)", 100.0, 600.0, 300.0)
    slag = st.sidebar.number_input("Escoria (Slag)", 0.0, 400.0, 0.0)
    fly_ash = st.sidebar.number_input("Ceniza (Fly Ash)", 0.0, 300.0, 0.0)
    water = st.sidebar.number_input("Agua (kg/m³)", 100.0, 250.0, 180.0)
    superplasticizer = st.sidebar.number_input("Superplastificante", 0.0, 35.0, 0.0)
    coarse_agg = st.sidebar.number_input("Agregado Grueso", 700.0, 1200.0, 900.0)
    fine_agg = st.sidebar.number_input("Agregado Fino", 500.0, 1000.0, 700.0)
    age = st.sidebar.slider("Edad (Días)", 1, 365, 28)
    
    # Aplicamos la transformación logarítmica que requiere tu modelo
    log_age = np.log(age) if age > 0 else 0
    
    data = {
        'cement': cement,
        'blast_furnace_salag': slag,
        'fly_ash': fly_ash,
        'water': water,
        'superplasticizer': superplasticizer,
        'coarse_aggregate': coarse_agg,
        'fine_aggregate': fine_agg,
        'log_age': log_age
    }
    return pd.DataFrame([data]), age

df_input, original_age = user_input_features()

# 4. Mostrar los datos ingresados
st.subheader("Resumen de la Mezcla")
st.write(df_input)

# 5. Realizar Predicción
if st.button("Calcular Resistencia"):
    prediction = model.predict(df_input)[0]
    
    # Mostrar resultado con un diseño llamativo
    st.success(f"### Resistencia Predicha: {prediction:.2f} MPa")
    
    # 6. Gráfico de evolución (Visualización extra)
    st.subheader("Evolución Estimada según la Edad")
    
    # Generamos un rango de edades para ver cómo crecería la resistencia de esta mezcla
    ages = np.array([1, 3, 7, 14, 28, 56, 90, 180, 365])
    log_ages = np.log(ages)
    
    # Creamos un set de datos para cada edad manteniendo el resto constante
    temp_df = pd.concat([df_input] * len(ages), ignore_index=True)
    temp_df['log_age'] = log_ages
    
    curve_preds = model.predict(temp_df)
    
    # Graficamos con Matplotlib
    fig, ax = plt.subplots()
    ax.plot(ages, curve_preds, marker='o', linestyle='-', color='#2c3e50')
    ax.scatter([original_age], [prediction], color='red', s=100, label='Punto Actual', zorder=5)
    ax.set_xlabel("Días")
    ax.set_ylabel("Resistencia (MPa)")
    ax.set_title("Curva de Resistencia Estimada")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)