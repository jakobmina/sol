
import streamlit as st
import sys
import os

# Import simulation modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from physics.sun_3d import generar_sol_metriplectico_3d
from physics.sun import ParametrosMetriplecticos

st.set_page_config(page_title="Sol Metripléctico 3D", layout="wide")

st.title("Simulación del Sol Metripléctico")
st.markdown("Visualización interactiva de la estructura conservativa/disipativa del sistema.")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Parámetros")
    r_core = st.slider("Radio del Núcleo (Hamiltoniano)", 1.0, 8.0, 3.0, help="Define el tamaño de la zona estable conservativa")
    r_halo = st.slider("Radio del Halo (Disipativo)", 2.0, 9.0, 5.5, help="Define el alcance de la zona turbulenta disipativa")
    evap = st.slider("Factor de Evaporación", 0.0, 5.0, 1.0, help="Intensidad del viento de evaporación")
    resolution = st.slider("Resolución del Grid", 50, 200, 100, step=10, help="Resolución de la malla (N x N). Valores altos aumentan la calidad pero pueden causar errores de memoria.")
    
    st.info("""
    **Leyenda:**
    - 🟡 **Núcleo**: Energía conservada
    - 🔵 **Halo**: Disipación de entropía
    - 🌊 **Flujo**: Evaporación de materia
    """)

with col2:
    params = ParametrosMetriplecticos(
        R_core=r_core,
        R_halo=r_halo,
        factor_evaporacion=evap,
        N=resolution  # Use lower resolution for 3D performance
    )
    
    with st.spinner("Generando simulación 3D..."):
        # Generate 3D Plotly figure
        # Reduce frames to 20 for smoother but lighter animation
        fig = generar_sol_metriplectico_3d(params, n_frames=20, show_plot=False)
        st.plotly_chart(fig, use_container_width=True)
