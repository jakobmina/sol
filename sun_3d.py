"""
Visualización 3D Interactiva Animada de un 'Sol Metripléctico':
Núcleo Hamiltoniano denso rodeado por un Halo Disipativo turbulento
que evapora materia entrante.

Inspirado en conceptos de: Jako (Reducción Fragmentada Sistemática Causal, 
Segunda Cuantización Cuasiperiódica y Sistemas Metriplécticos).

CARACTERÍSTICAS:
- Visualización 3D interactiva con Plotly
- Animación temporal mostrando evolución del sistema
- Campo vectorial 3D para flujo de evaporación
- Controles de cámara (rotación, zoom, pan)
- Modulación temporal usando el Operador Áureo O_n(t)
"""

from typing import Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importar física existente del módulo 2D
from physics.sun import (
    ParametrosMetriplecticos,
    calcular_campos_basicos,
    calcular_densidad_core,
    calcular_densidad_halo,
    calcular_densidad_total,
    calcular_campo_evaporacion
)


def operador_aureo(n: float, t: float, phi: float) -> float:
    """
    Operador Áureo O_n(t) para modulación temporal.
    
    Genera modulación cuasiperiódica basada en la razón áurea,
    evitando periodicidad espuria.
    
    Args:
        n: Número de modo (típicamente 1)
        t: Tiempo normalizado [0, 2π]
        phi: Razón áurea (φ ≈ 1.618)
        
    Returns:
        Valor del operador en [-1, 1]
    """
    return np.cos(np.pi * n * t) * np.cos(np.pi * phi * n * t)


def calcular_densidad_temporal(R: np.ndarray, Theta: np.ndarray, t: float,
                                params: ParametrosMetriplecticos) -> np.ndarray:
    """
    Calcula la densidad total con evolución temporal.
    
    La componente Hamiltoniana (núcleo) permanece estable.
    La componente Disipativa (halo) evoluciona con el tiempo.
    
    Args:
        R: campo de distancias radiales [L_char]
        Theta: campo de ángulos polares [radianes]
        t: tiempo normalizado [0, 2π]
        params: parámetros del modelo
        
    Returns:
        Densidad total en el instante t
    """
    phi = params.phi
    
    # Núcleo estable (Hamiltoniano - conservativo)
    densidad_core = calcular_densidad_core(R, params)
    
    # Halo turbulento (Disipativo - evoluciona con el tiempo)
    densidad_halo_base = calcular_densidad_halo(R, Theta, params)
    
    # Modulación temporal usando el Operador Áureo
    modulacion_temporal = 0.5 + 0.5 * operador_aureo(1.0, t, phi)
    
    # El halo se modula con el tiempo, el núcleo permanece estable
    densidad_halo_t = densidad_halo_base * modulacion_temporal
    
    # Combinar componentes
    densidad_total = calcular_densidad_total(densidad_core, densidad_halo_t, params)
    
    return densidad_total


def crear_superficie_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                        nombre: str = "Densidad de Energía") -> go.Surface:
    """
    Crea una superficie 3D para Plotly.
    
    Args:
        X, Y: coordenadas de malla
        Z: valores de altura (densidad)
        nombre: nombre de la traza
        
    Returns:
        Objeto Surface de Plotly
    """
    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Inferno',
        name=nombre,
        showscale=True,
        colorbar=dict(
            title=dict(
                text="ρ [norm]",
                side="right"
            ),
            tickmode="linear",
            tick0=0,
            dtick=0.3
        ),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project=dict(z=True)
            )
        ),
        opacity=0.9
    )
    
    return surface


def crear_campo_vectorial_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                              U: np.ndarray, V: np.ndarray, W: np.ndarray,
                              skip: int = 15) -> go.Cone:
    """
    Crea un campo vectorial 3D usando conos.
    
    Args:
        X, Y, Z: coordenadas de malla
        U, V, W: componentes del vector
        skip: factor de submuestreo para rendimiento
        
    Returns:
        Objeto Cone de Plotly
    """
    # Submuestreo para rendimiento
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]
    Z_sub = Z[::skip, ::skip]
    U_sub = U[::skip, ::skip]
    V_sub = V[::skip, ::skip]
    W_sub = W[::skip, ::skip]
    
    # Calcular magnitud para colorear
    magnitud = np.sqrt(U_sub**2 + V_sub**2 + W_sub**2)
    
    cone = go.Cone(
        x=X_sub.flatten(),
        y=Y_sub.flatten(),
        z=Z_sub.flatten(),
        u=U_sub.flatten(),
        v=V_sub.flatten(),
        w=W_sub.flatten(),
        colorscale=[[0, 'cyan'], [1, 'lightcyan']],
        sizemode="absolute",
        sizeref=0.3,
        name="Flujo de Evaporación",
        showscale=False,
        opacity=0.6
    )
    
    return cone



class SistemaParticulas:
    """
    Sistema de partículas Lagrangiano para visualizar la dinámica metripléctica.
    
    Dinámica:
    a = F_grav + F_drag + F_evap
    """
    def __init__(self, n_particulas: int, params: ParametrosMetriplecticos):
        self.n = n_particulas
        self.params = params
        self.dt = 0.1  # Paso de tiempo
        
        # Estado
        self.r = np.zeros((n_particulas, 3)) # Posición (x, y, z)
        self.v = np.zeros((n_particulas, 3)) # Velocidad (vx, vy, vz)
        
        self.reset()
        
    def reset(self):
        """Inicializa partículas en el Halo."""
        # Distribuir aleatoriamente en el disco del halo
        radios = np.random.uniform(self.params.R_core, self.params.R_halo, self.n)
        angulos = np.random.uniform(0, 2*np.pi, self.n)
        z_spread = 0.2
        
        self.r[:, 0] = radios * np.cos(angulos)
        self.r[:, 1] = radios * np.sin(angulos)
        self.r[:, 2] = np.random.uniform(-z_spread, z_spread, self.n)
        
        # Velocidad inicial: Rotación Kepleriana aproximada + ruido
        v_circular = np.sqrt(10.0 / radios) # Asumiendo GM=10
        self.v[:, 0] = -v_circular * np.sin(angulos)
        self.v[:, 1] = v_circular * np.cos(angulos)
        self.v[:, 2] = np.random.normal(0, 0.1, self.n)
        
    def update(self, t: float):
        """Integra ecuaciones de movimiento (Symplectic Euler)."""
        # Distancias y radios
        r_vec = self.r
        r_mag = np.linalg.norm(r_vec, axis=1, keepdims=True)
        r_mag = np.maximum(r_mag, 0.1) # Evitar singularidad
        
        # --- 1. Gravedad Central (Conservativa) ---
        # F_grav = - GM / r^3 * vec(r)
        GM = 15.0  # Ajustado para visualización
        acc_grav = -GM * r_vec / r_mag**3
        
        # --- 2. Fricción con el Halo (Disipativa) ---
        # F_drag = -gamma * rho * v
        # Aproximamos densidad del halo localmente
        # rho ~ exp(- (r - r_0)^2 / w^2)
        r_0 = self.params.R_core + self.params.offset_halo
        w = self.params.width_halo
        rho_local = np.exp(-0.5 * ((r_mag.flatten() - r_0) / w)**2)
        rho_local = rho_local[:, np.newaxis] # Shape (n, 1)
        
        gamma = 0.8 # Coeficiente de fricción
        acc_drag = -gamma * rho_local * self.v
        
        # --- 3. Viento de Evaporación (Expulsión) ---
        # Solo actúa si están cerca del "anillo verde" (pico del halo)
        # Fuerza radial positiva + componente vertical
        fuerza_evap = 0.0
        # Modulación temporal del viento
        modulacion = 0.5 + 0.5 * operador_aureo(1.0, t, self.params.phi)
        
        # Dirección radial normalizada
        r_norm = r_vec / r_mag
        
        # Componente vertical (arriba/abajo dependiendo de z)
        z_sign = np.sign(r_vec[:, 2:])
        vec_evap = r_norm + 0.5 * z_sign * np.array([0, 0, 1])
        
        acc_evap = 1.5 * modulacion * rho_local * vec_evap * self.params.factor_evaporacion
        
        # --- Integración ---
        acc_total = acc_grav + acc_drag + acc_evap
        
        # Euler Simpléctico
        self.v += acc_total * self.dt
        self.r += self.v * self.dt
        
        # Límites: Reiniciar si escapan o caen al centro
        escapados = r_mag.flatten() > self.params.L
        caidos = r_mag.flatten() < 1.0
        reiniciar = escapados | caidos
        
        if np.any(reiniciar):
            idx = np.where(reiniciar)[0]
            # Reinyectar en el halo exterior
            n_new = len(idx)
            radios = np.random.uniform(self.params.R_halo - 1.0, self.params.R_halo, n_new)
            angulos = np.random.uniform(0, 2*np.pi, n_new)
            
            self.r[idx, 0] = radios * np.cos(angulos)
            self.r[idx, 1] = radios * np.sin(angulos)
            self.r[idx, 2] = np.random.uniform(-0.2, 0.2, n_new)
            
            v_kepler = np.sqrt(10.0 / radios)
            self.v[idx, 0] = -v_kepler * np.sin(angulos)
            self.v[idx, 1] = v_kepler * np.cos(angulos)
            self.v[idx, 2] = 0

    def get_trace(self):
        """Retorna traza Scatter3d para Plotly."""
        return go.Scatter3d(
            x=self.r[:, 0],
            y=self.r[:, 1],
            z=self.r[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=np.linalg.norm(self.v, axis=1), # Color por velocidad
                colorscale='Viridis',
                opacity=0.8
            ),
            name='Partículas Lagrangianas'
        )


def generar_sol_metriplectico_3d(params: ParametrosMetriplecticos | None = None,
                                   n_frames: int = 25,
                                   show_plot: bool = True) -> go.Figure:
    """
    Genera visualización 3D interactiva animada del Sol Metripléctico.
    Incluye sistema de partículas.
    """
    if params is None:
        params = ParametrosMetriplecticos()
    
    print("Generando visualización 3D del Sol Metripléctico...")
    print(f"  - Resolución: {params.N}x{params.N}")
    print(f"  - Frames de animación: {n_frames}")
    
    # --- Cálculos de campos básicos ---
    X, Y, R, Theta = calcular_campos_basicos(params)
    
    # --- Sistema de Partículas ---
    particulas = SistemaParticulas(n_particulas=100, params=params)
    
    # --- Crear frames de animación ---
    frames = []
    tiempos = np.linspace(0, 2*np.pi, n_frames)
    
    # Pre-calcular estado inicial para la figura base (t=0)
    # Hacemos una copia profunda para no afectar la simulación
    import copy
    particulas_0 = copy.deepcopy(particulas)
    
    densidad_0 = calcular_densidad_temporal(R, Theta, 0, params)
    surface_0 = crear_superficie_3d(X, Y, densidad_0)
    
    # Campo vectorial simplificado para no saturar
    capa_base_halo = np.exp(-0.5 * ((R - (params.R_core + params.offset_halo)) / params.width_halo)**2)
    U0, V0 = calcular_campo_evaporacion(capa_base_halo, R, Theta, params)
    W0 = 0.3 * capa_base_halo * operador_aureo(1.0, 0, params.phi)
    cone_0 = crear_campo_vectorial_3d(X, Y, densidad_0, U0, V0, W0, skip=25)
    
    trace_particulas_0 = particulas_0.get_trace()
    
    # Loop de animación
    dt_sim = 0.5 # Avance de tiempo entre frames para las partículas
    
    for i, t in enumerate(tiempos):
        # 1. Evolución de campos escalares
        densidad_t = calcular_densidad_temporal(R, Theta, t, params)
        
        capa_halo = np.exp(-0.5 * ((R - (params.R_core + params.offset_halo)) / params.width_halo)**2)
        U, V = calcular_campo_evaporacion(capa_halo, R, Theta, params)
        W = 0.3 * capa_halo * operador_aureo(1.0, t, params.phi)
        
        surface = crear_superficie_3d(X, Y, densidad_t, f"Densidad t={t:.2f}")
        cone = crear_campo_vectorial_3d(X, Y, densidad_t, U, V, W, skip=25)
        
        # 2. Evolución de partículas
        # Avanzar múltiples pasos pequeños para estabilidad numérica
        for _ in range(3): 
            particulas.update(t)
            
        trace_blob = particulas.get_trace()
        
        frame = go.Frame(
            data=[surface, cone, trace_blob],
            name=f"frame_{i}",
            layout=go.Layout(title_text=f"Sol Metripléctico - t = {t:.2f}")
        )
        frames.append(frame)
    
    # Construir Figura
    fig = go.Figure(data=[surface_0, cone_0, trace_particulas_0], frames=frames)
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text="Sol Metripléctico 3D + Dinámica Lagrangiana",
            x=0.5,
            xanchor='center',
            font=dict(size=16, color='white')
        ),
        scene=dict(
            xaxis=dict(range=[-params.L, params.L], backgroundcolor="rgb(20, 20, 30)"),
            yaxis=dict(range=[-params.L, params.L], backgroundcolor="rgb(20, 20, 30)"),
            zaxis=dict(range=[0, 2.0], backgroundcolor="rgb(20, 20, 30)"),
            aspectmode='cube'
        ),
        paper_bgcolor='rgb(10, 10, 20)',
        plot_bgcolor='rgb(10, 10, 20)',
        font=dict(color='white'),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate", args=[[None], {"mode": "immediate"}])
            ]
        )]
    )
    
    if show_plot:
        fig.show()
    
    return fig


if __name__ == "__main__":
    # Ejecución con parámetros por defecto
    generar_sol_metriplectico_3d()
    
    # Ejemplo: mayor resolución temporal (descomenta para explorar)
    # generar_sol_metriplectico_3d(n_frames=40)
