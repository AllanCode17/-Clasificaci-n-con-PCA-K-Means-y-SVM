import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Reconocimiento de Dígitos MNIST",
    page_icon="🔢",
    layout="wide"
)

# ==========================================
# Estilos CSS Avanzados - Paleta Cyberpunk / Deep Ocean
# ==========================================
st.markdown("""
<style>
    /* Fondo principal */
    .stApp {
        background-color: #0A0E17;
        font-family: 'Segoe UI', -apple-system, sans-serif;
    }
    
    /* Contenedores de Métricas en el Sidebar */
    div[data-testid="stMetric"] {
        background-color: #161B26;
        border: 1px solid #232D3F;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-bottom: 10px;
    }
    
    div[data-testid="stMetricValue"] {
        color: #00F2FE !important;
        font-weight: 700;
        font-size: 1.6rem !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #8E9AA8 !important;
        font-size: 0.85rem !important;
    }

    /* Títulos y textos */
    h1 {
        color: #00F2FE !important;
        font-weight: 800;
    }
    h2, h3 {
        color: #F0F4F8 !important;
        border-bottom: 1px solid #232D3F;
        padding-bottom: 8px;
    }
    .stMarkdown p {
        color: #93A3B6;
    }
    
    /* Cajas de Alertas e información */
    .stAlert {
        background-color: #161B26 !important;
        border: 1px solid #232D3F !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Clasificador MNIST con Reducción de Dimensionalidad (PCA, K-Means & SVM) - Allan Manuel Orellana Orellana 20211920128")
st.markdown("""
Esta aplicación web utiliza **PCA** para reducir las dimensiones de imágenes de dígitos manuscritos, 
**K-Means** para analizar su agrupación en clústeres y una **Máquina de Vectores de Soporte (SVM)** para predecir el dígito correcto.
""")

# ==========================================
# Carga de Recursos y Modelos
# ==========================================
@st.cache_resource
def cargar_modelos_y_datos():
    ruta_pca = "models/pca_digit_recognizer.pkl"
    ruta_kmeans = "models/kmeans_digit_clusters.pkl"
    ruta_svm = "models/svm_digit_classifier.pkl"
    ruta_metadata = "models/model_metadata.json"
    ruta_csv = "outputs/mnist_resultados_pca.csv"
    
    pca = joblib.load(ruta_pca) if os.path.exists(ruta_pca) else None
    kmeans = joblib.load(ruta_kmeans) if os.path.exists(ruta_kmeans) else None
    svm = joblib.load(ruta_svm) if os.path.exists(ruta_svm) else None
    
    metadata = {}
    if os.path.exists(ruta_metadata):
        with open(ruta_metadata, "r") as f:
            metadata = json.load(f)
            
    df_pca = pd.read_csv(ruta_csv) if os.path.exists(ruta_csv) else None
    
    return pca, kmeans, svm, metadata, df_pca

pca_model, kmeans_model, svm_model, metadata, df_pca = cargar_modelos_y_datos()

if pca_model is None or svm_model is None or kmeans_model is None:
    st.error("No se encontraron los modelos en la carpeta models/. Asegúrate de haber subido los archivos .pkl correctamente.")
    st.stop()

# ==========================================
# Panel Lateral de Control y Métricas Globales
# ==========================================
st.sidebar.header("Desempeño del Sistema")

st.sidebar.metric(label="Precisión SVM (Accuracy)", value=f"{metadata.get('accuracy_score_svm', 0.0)*100:.2f}%")
st.sidebar.metric(label="Componentes Entrenados PCA", value=f"{metadata.get('n_components_pca', 0)}")
st.sidebar.metric(label="Varianza Explicada Acumulada", value=f"{metadata.get('varianza_explicada', 0.0)*100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.header("Ajustes de Compresión")

max_componentes = metadata.get("n_components_pca", 35)
n_componentes_seleccionados = st.sidebar.slider(
    "Componentes Principales (PCA)", 
    min_value=2, 
    max_value=max_componentes, 
    value=max_componentes
)

st.sidebar.markdown("---")
st.sidebar.header("Selección de Muestra")

@st.cache_data
def cargar_imagenes_muestra():
    ruta_muestra = "mnist_sample.csv"
    if os.path.exists(ruta_muestra):
        df = pd.read_csv(ruta_muestra)
        if "label" in df.columns:
            X = df.drop(columns=["label"]).values
            y = df["label"].values
            return X, y
        else:
            return df.values, np.zeros(len(df))
    return None, None

X_raw, y_raw = cargar_imagenes_muestra()

if X_raw is not None:
    idx_imagen = st.sidebar.slider("Índice del dataset (0-99)", 0, len(X_raw) - 1, 10)
    imagen_seleccionada = X_raw[idx_imagen]
    etiqueta_real = y_raw[idx_imagen]
else:
    st.sidebar.error("Archivo mnist_sample.csv no encontrado.")
    imagen_seleccionada = np.zeros(784)
    imagen_seleccionada[14*28:15*28] = 255
    etiqueta_real = "Desconocido"

# ==========================================
# Inferencia y Clasificación en Tiempo Real
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Inferencia y Clasificación en Tiempo Real")

col_img, col_pred = st.columns([1, 2.2], gap="large")

with col_img:
    st.markdown("**Imagen Seleccionada del Dataset:**")
    matriz_imagen = imagen_seleccionada.reshape(28, 28)
    
    fig_img, ax_img = plt.subplots(figsize=(2.8, 2.8))
    fig_img.patch.set_facecolor("#0A0E17")
    ax_img.imshow(matriz_imagen, cmap="plasma") 
    ax_img.axis("off")
    st.pyplot(fig_img)
    
    # SOLUCIÓN: Agregado unsafe_allow_html=True para que renderice la etiqueta de color correctamente
    st.markdown(f"Etiqueta Origen: <span style='color:#00F2FE; font-weight:bold;'>{etiqueta_real}</span>", unsafe_allow_html=True)

with col_pred:
    st.markdown("**Análisis Predictivo de los Modelos Enlazados:**")
    
    if imagen_seleccionada.max() > 1.0:
        imagen_normalizada = imagen_seleccionada / 255.0
    else:
        imagen_normalizada = imagen_seleccionada
        
    imagen_input = imagen_normalizada.reshape(1, -1)
    imagen_pca = pca_model.transform(imagen_input)
    
    imagen_pca_filtrada = imagen_pca[:, :n_componentes_seleccionados]
    imagen_pca_completa = np.zeros((1, max_componentes))
    imagen_pca_completa[:, :n_componentes_seleccionados] = imagen_pca_filtrada

    cluster_predicho = kmeans_model.predict(imagen_pca_completa)[0]
    clase_predicha_svm = svm_model.predict(imagen_pca_completa)[0]
    
    st.info(f"Clúster Asignado por K-Means (No Supervisado): Clúster {cluster_predicho}")
    st.success(f"Clasificación Final SVM (Supervisado): Dígito {clase_predicha_svm}")
    
    st.markdown(f"""
    **Explicación del Proceso:** La imagen original de 784 dimensiones (28x28 píxeles) extraída del archivo CSV se comprimió usando los componentes principales de PCA. 
    El algoritmo **K-Means** identificó que sus patrones geométricos corresponden al **Clúster {cluster_predicho}** (agrupación no supervisada). 
    Finalmente, la **SVM** determinó de forma supervisada con una frontera de decisión hiperplana que el número dibujado es un **{clase_predicha_svm}**.
    """)

st.markdown("<hr style='border: 1px solid #232D3F;'>", unsafe_allow_html=True)

# ==========================================
# Estructura de Gráficas Geométricas
# ==========================================
st.subheader("Topología Espacial del Modelo")
col_graph1, col_graph2 = st.columns(2, gap="large")

with col_graph1:
    st.markdown("**Proyección Bidimensional de Datos (PCA)**")
    if df_pca is not None:
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        fig1.patch.set_facecolor("#0A0E17")
        ax1.set_facecolor("#161B26")
        
        sns.scatterplot(
            data=df_pca, x="PC1", y="PC2", hue="label", 
            palette="Spectral", alpha=0.5, ax=ax1, legend="full", edgecolor="none"
        )
        ax1.set_title("Distribución por Clases Reales (Dígitos 0-9)", color="#F0F4F8", fontsize=10)
        ax1.tick_params(colors="#8E9AA8", labelsize=8)
        ax1.xaxis.label.set_color('#8E9AA8')
        ax1.yaxis.label.set_color('#8E9AA8')
        ax1.legend(facecolor="#161B26", edgecolor="#232D3F", labelcolor="#F0F4F8", fontsize=7)
        st.pyplot(fig1)
    else:
        st.info("Sube el archivo outputs/mnist_resultados_pca.csv para ver la proyección 2D.")

with col_graph2:
    st.markdown("**Estructura de Centroides (K-Means)**")
    if df_pca is not None:
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        fig2.patch.set_facecolor("#0A0E17")
        ax2.set_facecolor("#161B26")
        
        sns.scatterplot(
            data=df_pca, x="PC1", y="PC2", hue="cluster", 
            palette="coolwarm", alpha=0.5, ax=ax2, legend="full", edgecolor="none"
        )
        ax2.set_title("Agrupación Visual de Clusters del Espacio PCA", color="#F0F4F8", fontsize=10)
        ax2.tick_params(colors="#8E9AA8", labelsize=8)
        ax2.xaxis.label.set_color('#8E9AA8')
        ax2.yaxis.label.set_color('#8E9AA8')
        ax2.legend(facecolor="#161B26", edgecolor="#232D3F", labelcolor="#F0F4F8", fontsize=7)
        st.pyplot(fig2)
    else:
        st.info("Sube el archivo outputs/mnist_resultados_pca.csv para ver las agrupaciones de K-Means.")
